# Modified for DRIFT. See LICENSE and NOTICE for license and attribution details.
from PIL import Image
import torch

def PickScore(dtype, device):
    from .PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def hps_v2(inference_dtype=None, device=None):

    from flow.hpsv2_scorer import HPSv2RewardInferencer
    scorer = HPSv2RewardInferencer(device=device, hps_version='v2.1')

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer.reward(images, prompts)
        return scores, {}

    return _fn
