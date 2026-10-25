# Modified for DRIFT. See LICENSE and NOTICE for license and attribution details.
import torch
import torchvision

def PickScore(dtype, device):
    from .PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts, metadata):
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def hps_v2(inference_dtype=None, device=None):
    import hpsv2
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    tokenizer = get_tokenizer(model_name)

    link = "https://hf-mirror.com/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
    import os
    import requests
    from tqdm import tqdm

    # Create the directory if it doesn't exist
    os.makedirs(os.path.expanduser('~/.cache/hpsv2'), exist_ok=True)
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"

    # Download the file if it doesn't exist
    if not os.path.exists(checkpoint_path):
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(checkpoint_path, 'wb') as file, tqdm(
            desc="Downloading HPS_v2_compressed.pt",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

    # force download of model via score
    hpsv2.score([], "")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])

    def _fn(im_pix, prompts, metadata):
        del metadata
        with torch.no_grad():
            x_var = torchvision.transforms.Resize(target_size)(im_pix)
            x_var = normalize(x_var).to(im_pix.dtype)
            caption = tokenizer(prompts)
            caption = caption.to(device)

            outputs = model(x_var, caption)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits = image_features @ text_features.T
            scores = torch.diagonal(logits)
        del im_pix
        del prompts
        del x_var
        del caption
        del outputs
        return scores.to(torch.float32), {}

    return _fn
