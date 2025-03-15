import clip
import torch
import torchvision.transforms as transforms
from hydra import initialize, compose

from experiments.eval_utils import path_from_cfg


class CLIPTokenizer:

    def __call__(self, texts):
        return clip.tokenize(texts)


def load_open_ai_clip(model_name: str = "ViT-L/14", pretrained=True, cache_dir="", device = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = clip.load(model_name, device=device)
    tokenizer = CLIPTokenizer()
    return model, transforms.Compose(processor.transforms), tokenizer


def load_open_ai_clip_manipulated(model_name: str = "ViT-L/14", pretrained=True, cache_dir="", device = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = clip.load(model_name, device=device)
    tokenizer = CLIPTokenizer()

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="config_vit_clip_large",
            overrides=[],
        )

    path = path_from_cfg(cfg)
    model_dict = torch.load(path, map_location=torch.device(device))
    model.visual.load_state_dict(model_dict["model"])
    return model, transforms.Compose(processor.transforms), tokenizer