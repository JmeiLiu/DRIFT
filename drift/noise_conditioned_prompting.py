import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from .ddim_with_logprob import ddim_step_with_logprob

def randn_like(tensor):
    return torch.randn_like(tensor)

def cads_linear_schedule(t, tau1, tau2):
    """ CADS annealing schedule function """
    if t <= tau1:
        return 1.0
    if t >= tau2:
        return 0.0
    gamma = (tau2 - t) / (tau2 - tau1)
    return gamma

def add_noise(y, gamma, noise_scale, psi=1, rescale=True):
    y_mean, y_std = torch.mean(y), torch.std(y)
    gamma = torch.tensor(gamma, dtype=torch.float32, device=y.device)
    y_ = torch.sqrt(gamma) * y + noise_scale * torch.sqrt(1 - gamma) * randn_like(y)
    if rescale:
        y_rescaled = (y_ - torch.mean(y_)) / torch.std(y_) * y_std + y_mean
        y_f = psi * y_rescaled + (1 - psi) * y_
    else:
        y_f = y_
    return y_f

@torch.no_grad()
def noise_conditioned_prompting(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    noise: float = 0.0,
):

    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )

    # 2. Encode prompt
    prompt_pos, prompt_neg =  self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

    prompt_embeds = torch.cat((prompt_neg, prompt_pos), dim=0)

    prompt_pos_orig = prompt_pos.detach()  # anchor
    prompt_neg_orig = prompt_neg.detach()

    # 3. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 4. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare extra step kwargs
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            gamma = cads_linear_schedule(1.0 - (i+1)/ num_inference_steps, tau1=0.4, tau2=1.0)
            prompt_pos_noised = add_noise(prompt_pos_orig, gamma, noise_scale=noise, psi=1.0, rescale=True)
            prompt_neg_noised = add_noise(prompt_neg_orig, gamma, noise_scale=noise, psi=1.0, rescale=True)
            prompt_embeds = torch.cat([prompt_neg_noised, prompt_pos_noised], dim=0)

            # 6. Predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # 7. Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # 8. Apply CADS
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            # 9. Compute the previous noisy sample x_t -> x_t-1
            latents, log_prob, _, _ = ddim_step_with_logprob(
                self.scheduler, noise_pred, t, latents, **extra_step_kwargs
            )

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    latents = latents.to(dtype=self.vae.dtype)
    if not output_type == "latent":
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, prompt_embeds.dtype
        )
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(
        image, output_type=output_type, do_denormalize=do_denormalize
    )

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return image, has_nsfw_concept, all_latents, all_log_probs
