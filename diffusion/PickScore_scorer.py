import os
import torch
from tqdm import tqdm
from transformers import AutoModel, CLIPProcessor
import requests


class PickScoreScorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(os.environ.get("DRIFT_CLIP_MODEL", "openai/clip-vit-large-patch14"))

        checkpoint_path = os.environ.get("DRIFT_PICK_MODEL", "yuvalkirstain/PickScore_v1")

        self.model = AutoModel.from_pretrained(checkpoint_path).eval().to(self.device, dtype=self.dtype)

    def __call__(self, images, prompts):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        image_embeds = self.model.get_image_features(**inputs)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image)

        return scores
