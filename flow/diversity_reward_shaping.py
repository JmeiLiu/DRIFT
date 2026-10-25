# Modified for DRIFT. See LICENSE and NOTICE for license and attribution details.
from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow.prompts
import flow.rewards
from flow.stat_tracking import PerPromptStatTracker_GRPO
from drift.sd3_pipeline_with_logprob import pipeline_with_logprob
from drift.sd3_sde_with_logprob import sde_step_with_logprob
from drift.prompt_encoding import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow.ema import EMAModuleWrapper
import torch.nn.functional as F
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from dreamsim import dreamsim

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def unwrap_model_for_accelerate(model, accelerator):
    m = accelerator.unwrap_model(model)
    return m._orig_mod if is_compiled_module(m) else m


def alpha_warmup_linear(epoch, num_epochs, alpha_max,
                        start_frac=0.6, ramp_frac=0.25):
    p = epoch / max(1, (num_epochs - 1))
    if p < start_frac:
        return 0.0
    p2 = (p - start_frac) / max(1e-12, ramp_frac)
    p2 = min(max(p2, 0.0), 1.0)
    return alpha_max * p2

def group_similarity_reward(embeddings: torch.Tensor,
                            batch_p_rewards: torch.Tensor):
    B = embeddings.size(0)
    assert batch_p_rewards.numel() == B, "embeddings 和 batch_p_rewards 的 batch 维度不一致"

    high_mask = batch_p_rewards > -float("inf")      # [B] bool

    if high_mask.sum() < 2:
        sim_reward = torch.zeros(B, device=embeddings.device, dtype=embeddings.dtype)
        return sim_reward, high_mask

    dist_matrix = torch.cdist(embeddings, embeddings, p=2) ** 2

    pair_mask = high_mask[:, None] & high_mask[None, :]    # [B, B] bool
    dist_matrix = dist_matrix * pair_mask

    group_size = high_mask.sum()
    denom = torch.clamp(group_size - 1, min=1)
    sim_reward = dist_matrix.sum(dim=-1) / denom           # [B]

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


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.

    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.

    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)

    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array,
        return_inverse=True,
        return_counts=True
    )

    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])

    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio, prompt_std_devs.mean()


def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
        sde_type=config.sample.sde_type,
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def pool_latents(x, pool="mean", eps=1e-6):
    spatial_dims = len(x.shape)
    if pool == "mean":
        z = x.mean(dim=(spatial_dims-2, spatial_dims-1))  # [B, C]

    elif pool == "abs_mean":
        z = x.abs().mean(dim=(spatial_dims-2, spatial_dims-1))  # [B, C]

    elif pool == "attn_norm":
        w = x.pow(2).sum(dim=1, keepdim=True).sqrt()            # [B,1,H,W]
        w = w / (w.sum(dim=(spatial_dims-2, spatial_dims-1), keepdim=True) + eps)
        z = (x * w).sum(dim=(spatial_dims-2, spatial_dims-1))                             # [B, C]

    else:
        raise ValueError(f"Unknown pool mode: {pool}")

    return z

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
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

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision, torch_dtype=inference_dtype)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)

    pipeline.transformer.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]

        transformer_lora_config = LoraConfig(
            r=config.train.lora_r,
            lora_alpha=config.train.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.resume_from > 0:
            lora_ckpt_dir = os.path.join(
                config.logdir, config.run_name, f"checkpoints/checkpoint_{config.resume_from}"
            )
            logger.info(f"Loading LoRA weights from {lora_ckpt_dir}")
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora_ckpt_dir, is_trainable=True)
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer

    def save_model_hook(models, weights, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model (and EMA) to {output_dir}")

        assert len(models) == 1
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        if config.use_lora:
            unwrap_model(transformer, accelerator).save_pretrained(output_dir)
        else:
            logger.info("use_lora = False, save full transformer with .save_pretrained")
            unwrap_model(transformer, accelerator).save_pretrained(output_dir)

        if config.train.ema:
            ema_path = os.path.join(output_dir, "ema.pt")
            torch.save(ema.state_dict(), ema_path)
            logger.info(f"Saved EMA state to {ema_path}")
            ema.copy_temp_to(transformer_trainable_parameters)

        weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        logger.info(f"Loading model (and EMA) from {input_dir}")
        transformer = models[0]

        if config.use_lora:
            logger.info("use_lora = True, load LoRA with pipeline.load_lora_weights")
            pass
        else:
            logger.info("use_lora = False, load full transformer with .from_pretrained")

            base_model = unwrap_model_for_accelerate(transformer, accelerator)
            load_model = type(base_model).from_pretrained(input_dir)
            transformer.register_to_config(**load_model.config)
            transformer.load_state_dict(load_model.state_dict())
            del load_model

        if config.train.ema:
            ema_path = os.path.join(input_dir, "ema.pt")
            if os.path.isfile(ema_path):
                ema_state = torch.load(ema_path, map_location="cpu")
                ema.load_state_dict(ema_state)
                logger.info(f"Loaded EMA state from {ema_path}")
            else:
                logger.warning(f"EMA state file not found at {ema_path}, starting EMA from scratch.")
        models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    for p in pipeline.transformer.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

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

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    prompt_fn = getattr(flow.prompts, config.prompt_fn)

    if config.reward_fn == 'hps_v2':
        reward_fn = getattr(flow.rewards, config.reward_fn)(inference_dtype, accelerator.device) # torch.float32
    elif config.reward_fn == 'PickScore':
        reward_fn = getattr(flow.rewards, config.reward_fn)(torch.float32, accelerator.device)
    else:
        raise ValueError("Supported rewards: PickScore, hps_v2")

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.mini_batch, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.mini_batch, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker_GRPO(global_std=config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer = accelerator.prepare(transformer, optimizer)

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

    if config.resume_from > 0:
        accelerator.load_state(os.path.join(
            config.logdir, config.run_name, f'checkpoints/checkpoint_{config.resume_from}'
        ))
        first_epoch = config.resume_from * config.save_freq
        global_step = first_epoch * (samples_per_epoch // total_train_batch_size)
        ckpt_idx = config.resume_from + 1
    else:
        first_epoch = 0
        global_step = 0
        ckpt_idx = 0

    print(f'first_epoch:{first_epoch}')
    print(f'global_step:{global_step}')

    embed_model, preprocess = build_dreamsim(accelerator.device)

    for epoch in range(first_epoch, config.num_epochs): # 100
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        prompts, prompt_metadata = zip(
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

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                current_batch,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                current_batch,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        sde_type=config.sample.sde_type,
                        num_train_timesteps = num_train_timesteps
                )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.mini_batch, 1)
            # compute rewards asynchronously
            p_rewards = executor.submit(reward_fn, images, current_batch, _)
            # yield to to make sure reward computation starts
            time.sleep(0)

            imgs = (images * 255).round().clamp(0, 255).to(torch.uint8)
            imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
            imgs = pil_batch(imgs, preprocess, accelerator.device)
            img_embs = embed_model.embed(imgs).detach()
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "img_embs": img_embs,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
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
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().to(torch.float32).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
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
                start_frac=0.4,
                ramp_frac=0.2
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

        indices = torch.arange(0, advantages.shape[0])
        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )
        samples["indices"] = (
            torch.as_tensor(indices)
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
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}
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
            pipeline.transformer.train()
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
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)

                        # grpo logic
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
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        p_loss = loss

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
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)
                        info["ppo_loss"].append(p_loss)
                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                            global_step += 1
                        optimizer.step()
                        optimizer.zero_grad()

                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            info = accelerator.reduce(info, reduction="mean")
            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
            accelerator.log(info, step=global_step)
            info = defaultdict(list)
            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if (epoch == 0 or (epoch+1) % config.save_freq == 0) and accelerator.is_main_process:
            save_dir = os.path.join(config.logdir, config.run_name, f'checkpoints/checkpoint_{ckpt_idx}')
            accelerator.save_state(save_dir)
            ckpt_idx += 1

if __name__ == "__main__":
    app.run(main)
