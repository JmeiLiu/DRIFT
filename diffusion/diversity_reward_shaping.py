# Modified for DRIFT. See LICENSE and NOTICE for license and attribution details.
from collections import defaultdict
import contextlib
import os
from peft import LoraConfig
os.environ['CURL_CA_BUNDLE'] = ''
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
import numpy as np
import diffusion.prompts
import diffusion.rewards
from diffusion.stat_tracking import PerPromptStatTracker_GRPO
from drift.pipeline_with_logprob import pipeline_with_logprob
from drift.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import torch.nn.functional as F
from dreamsim import dreamsim

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    config = FLAGS.config

    def group_similarity_reward(embeddings: torch.Tensor,
                                batch_p_rewards: torch.Tensor):
        B = embeddings.size(0)
        assert batch_p_rewards.numel() == B

        high_mask = batch_p_rewards > -float("inf")

        if high_mask.sum() < 2:
            sim_reward = torch.zeros(B, device=embeddings.device, dtype=embeddings.dtype)
            return sim_reward, high_mask

        dist_matrix = torch.cdist(embeddings, embeddings, p=2) ** 2

        pair_mask = high_mask[:, None] & high_mask[None, :]
        dist_matrix = dist_matrix * pair_mask

        group_size = high_mask.sum()
        denom = torch.clamp(group_size - 1, min=1)
        sim_reward = dist_matrix.sum(dim=-1) / denom

        sim_reward = sim_reward * high_mask.to(sim_reward.dtype)

        return sim_reward, high_mask

    def build_dreamsim(device: str = "cuda"):
        model, preprocess = dreamsim(pretrained=True, device=device, cache_dir=os.environ.get("DRIFT_DREAMSIM_MODEL", os.path.expanduser("~/models/dreamsim")))
        model.eval()
        return model, preprocess

    def pil_batch(images, preprocess, device: str) -> torch.Tensor:
        ims = []
        for im in images:
            im = Image.fromarray(im)
            ims.append(preprocess(im))
        return torch.cat(ims, dim=0).to(device)

    def encode_and_compute_diversity(image_embs: torch.Tensor, p_rewards: torch.Tensor):
        image_embs = image_embs
        p_rewards  = p_rewards

        r_div, mask = group_similarity_reward(image_embs, p_rewards)
        return r_div.to(torch.float32), mask.to(torch.bool)

    def alpha_warmup_linear(epoch, num_epochs, alpha_max,
                            start_frac=0.6, ramp_frac=0.25):
        p = epoch / max(1, (num_epochs - 1))
        if p < start_frac:
            return 0.0
        p2 = (p - start_frac) / max(1e-12, ramp_frac)
        p2 = min(max(p2, 0.0), 1.0)
        return alpha_max * p2

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction) # 50 * 1.0 = 50

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )
    if "_100" in config.run_name:
        grad_accumulation = config.train.gradient_accumulation_steps # 32, results in 2 * 50 == 100 times of gradient update
    else:
        grad_accumulation = config.train.gradient_accumulation_steps * num_train_timesteps # 32*50

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=grad_accumulation, # 50
    )
    if accelerator.is_main_process:
        if config.reward_fn == 'hps_v2':
            project_name = "DRIFT"
        elif config.reward_fn == 'PickScore':
            project_name = "DRIFT"
        else:
            project_name = "DRIFT"
        accelerator.init_trackers(
                project_name=project_name,
                config=config.to_dict(),
                init_kwargs={"wandb": {"name": config.run_name}},
            )

    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # Get the inference dtype from setting
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision, torch_dtype=inference_dtype
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)


    if config.use_lora:
        pipeline.unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=config.train.lora_r,
            lora_alpha=config.train.lora_alpha,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        if config.resume_from > 0:
            logger.info(f"Loading LoRA weights ..")
            lora_ckpt_dir = os.path.join(
                config.logdir, config.run_name, f"checkpoints/checkpoint_{config.resume_from}"
            )
            logger.info(f"Loading LoRA weights from {lora_ckpt_dir}")
            pipeline.load_lora_weights(lora_ckpt_dir)
        else:
            pipeline.unet.add_adapter(unet_lora_config)
    else:
        unet = pipeline.unet

    def save_model_hook(models, weights, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model to {output_dir}")

        # Unwrap model
        unet_model = accelerator.unwrap_model(models[0])
        unet_model = unet_model._orig_mod if is_compiled_module(unet_model) else unet_model
        if config.use_lora:
            adapter_name = next(iter(unet_model.peft_config.keys()))
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unet_model, adapter_name=adapter_name)
            )
            unet_adapter_meta = {
                "r": int(config.train.lora_r),
                "lora_alpha": int(config.train.lora_alpha),
                "lora_dropout": float(0.0),
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                # ensure parser-safe keys (avoid KeyError on load)
                "rank_pattern": {},      # empty dict if no per-layer pattern
                "alpha_pattern": {},     # empty dict if no per-layer pattern
                "fan_in_fan_out": False,
                "bias": "none",
                "layers_to_transform": None,
                "layers_pattern": None,
                "use_rslora": False,
                "use_dora": False,
                "use_qalora": False,
                "qalora_group_size": 16,
                "lora_bias": False,
                "modules_to_save": None,
            }
            save_dir = output_dir

            StableDiffusionPipeline.save_lora_weights(
                save_directory=save_dir,
                unet_lora_layers=unet_lora_state_dict,
                unet_lora_adapter_metadata=unet_adapter_meta,
                safe_serialization=True,
            )
        else:
            models[0].save_pretrained(os.path.join(output_dir, "unet"))

        weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora:
            pass
        else:
            logger.info(f"Loading standard model from {input_dir}")
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    for p in pipeline.unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
    optimizer = optimizer_cls(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(diffusion.prompts, config.prompt_fn)

    if config.reward_fn == 'hps_v2':
        reward_fn = getattr(diffusion.rewards, config.reward_fn)(inference_dtype, accelerator.device) # torch.float32
    elif config.reward_fn == 'PickScore':
        reward_fn = getattr(diffusion.rewards, config.reward_fn)(torch.float32, accelerator.device)
    else:
        raise ValueError("Supported rewards: PickScore, hps_v2")

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.mini_batch, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker_GRPO(global_std=config.sample.global_std)
        d_stat_tracker = PerPromptStatTracker_GRPO(global_std=config.sample.global_std)

    # Use accelerator.autocast if mixed precision training is enabled
    # And compile the model for faster inference speed
    autocast = accelerator.autocast
    pipeline.unet = torch.compile(pipeline.unet,fullgraph=True)
    pipeline.vae.decode = torch.compile(pipeline.vae.decode)
    unet = pipeline.unet
    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_generations
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from > 0:
        accelerator.load_state(os.path.join(
            config.logdir, config.run_name, f'checkpoints/checkpoint_{config.resume_from}'
        ))
        first_epoch = config.resume_from * config.save_freq
        global_step = first_epoch * config.train.num_inner_epochs
        ckpt_idx = config.resume_from + 1
    else:
        first_epoch = 0
        global_step = 0
        ckpt_idx = 0

    print(f'first_epoch:{first_epoch}')

    embed_model, preprocess = build_dreamsim(accelerator.device)

    for epoch in range(first_epoch, config.num_epochs): # 100

        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts = []
        prompts, _ = zip(
            *[
                # load one line from promt file per call
                prompt_fn(**config.prompt_fn_kwargs)
                for _ in range(config.sample.batch_size)
            ]
        )
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * config.sample.num_generations)

        for i in range(0, len(expanded_prompts), config.sample.mini_batch):
            current_batch = expanded_prompts[i:i+config.sample.mini_batch]
            # 2. tokenizer
            prompt_ids = pipeline.tokenizer(
                current_batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            with autocast():
                images, _, latents, log_probs  = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )

            latents = torch.stack(latents, dim=1)        # (4, num_steps+1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)   # (4, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.mini_batch, 1)
            time.sleep(0)
            # 5. reward
            p_rewards = executor.submit(reward_fn, images, current_batch, _)

            imgs = (images * 255).round().clamp(0, 255).to(torch.uint8)
            imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
            imgs = pil_batch(imgs, preprocess, accelerator.device)
            img_embs = embed_model.embed(imgs).detach()

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "img_embs": img_embs,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "p_rewards": p_rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            p_rewards, reward_metadata = sample["p_rewards"].result()
            sample["p_rewards"] = torch.as_tensor(p_rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
                    k: torch.cat([s[k] if isinstance(s[k], torch.Tensor) else torch.zeros(0)
                                for s in samples if isinstance(s[k], torch.Tensor) or s[k] != {}])
                    for k in samples[0].keys()
                }

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().to(torch.float32).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {p_reward:.2f}",
                        )
                        for i, (prompt, p_reward) in enumerate(
                            zip(current_batch, p_rewards)
                        )  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # gather rewards across processes
        p_rewards = accelerator.gather(samples["p_rewards"]).cpu().numpy()
        img_embs = accelerator.gather(samples["img_embs"]).cpu()
        all_prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
        # gather the prompts across processes
        prompts = pipeline.tokenizer.batch_decode(all_prompt_ids, skip_special_tokens=True)

        if config.per_prompt_stat_tracking:
            p_advantages = stat_tracker.update(prompts, p_rewards)
        else:
            p_advantages = (p_rewards - p_rewards.mean()) / (p_rewards.std() + 1e-6)

        d_advantages = np.zeros_like(p_rewards, dtype=np.float32)
        d_masks   = np.zeros_like(p_rewards, dtype=bool)
        d_rewards = np.zeros_like(p_rewards, dtype=np.float32)

        prompt_to_indices = defaultdict(list)
        for i, p in enumerate(prompts):
            prompt_to_indices[p].append(i)

        for idxs in prompt_to_indices.values():
            idx = np.asarray(idxs, dtype=np.int64)
            d_rewards_chunk, d_masks_chunk = encode_and_compute_diversity(img_embs[idx], torch.tensor(p_rewards[idx]))
            d_rewards[idx] = d_rewards_chunk.cpu().numpy()
            d_masks[idx] = d_masks_chunk.cpu().numpy()

        d_clip_rewards = np.clip(d_rewards, -config.train.d_clip_max, config.train.d_clip_max)

        all_d = d_clip_rewards[np.where(d_masks)]
        for idxs in prompt_to_indices.values():
            idx = np.asarray(idxs, dtype=np.int64)
            group_mask = d_masks[idx]
            if not group_mask.any():
                continue
            valid_idx = idx[np.where(group_mask)[0]]
            valid_d = d_clip_rewards[valid_idx]
            d_mean = valid_d.mean()
            if config.sample.global_std:
                d_std = all_d.std() + 1e-6
            else:
                d_std  = valid_d.std() + 1e-6
            d_advantages[valid_idx] = (valid_d - d_mean) / d_std

        if config.alpha_schedule:
            alpha_tmp = alpha_warmup_linear(
                epoch, config.num_epochs,
                alpha_max=config.train.alpha,
                start_frac=config.train.alpha_start_frac,
                ramp_frac=config.train.alpha_ramp_frac,
            )
        else:
            alpha_tmp = config.train.alpha

        advantages = p_advantages + alpha_tmp * d_advantages

        accelerator.log(
            {
                "reward": p_rewards,
                "epoch": epoch,
                "reward_queries": (epoch + 1) * p_rewards.size,
                "reward_mean": p_rewards.mean(),
                "reward_std": p_rewards.std(),
                "d_reward_mean": d_rewards[np.where(d_masks)[0]].mean(),
                "d_reward_std": d_rewards[np.where(d_masks)[0]].std(),
                "over_reward": abs(d_rewards - d_clip_rewards)[np.where(d_masks)[0]].mean(),
                "alpha_tmp": alpha_tmp,
            },
            step=global_step,
        )

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["p_rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_generations
        )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            samples['perms'] = perms
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample

                            _, log_prob, _, _ = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )
                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])

                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        p_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        loss = p_loss
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipadv"].append((torch.abs(sample["advantages"]) > config.train.adv_clip_max).float().mean())

                        info["loss"].append(loss)
                        info["ppo_loss"].append(p_loss)
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            info = accelerator.reduce(info, reduction="mean")
            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
            accelerator.log(info, step=global_step)
            global_step += 1
            info = defaultdict(list)
            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if (epoch == 0 or (epoch+1) % config.save_freq == 0) and accelerator.is_main_process:
            save_dir = os.path.join(config.logdir, config.run_name, f'checkpoints/checkpoint_{ckpt_idx}')
            accelerator.save_state(save_dir)
            ckpt_idx += 1


if __name__ == "__main__":
    app.run(main)
