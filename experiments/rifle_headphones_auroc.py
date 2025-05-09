import os
import copy
import requests
import argparse
import torch
import clip
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from hydra import initialize, compose
from torchvision import transforms
from torchmetrics.classification import BinaryAUROC

from experiments.concept_probe import (
    sample_imagenet_images,
    register_activation_hook,
    get_clip_activation,
)
from experiments.eval_utils import path_from_cfg

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 6MB in bytes
MAX_FILE_SIZE = 6 * 1024 * 1024


def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        # Check file size using header
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_FILE_SIZE:
            print(f"Skipping {image_url} (size: {content_length} bytes exceeds 6MB)")
            return False

        # Load image from response bytes
        image_data = BytesIO(response.content)
        image = Image.open(image_data).convert("RGB")
        # Apply transformations: Resize then CenterCrop
        transform = transforms.Compose(
            [
                transforms.Resize(
                    size=224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((224, 224)),
            ]
        )
        image = transform(image)
        image.save(save_path)
        print(f"Downloaded and processed {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {image_url}. Error: {e}")
        return False


def get_image_urls(search_url, max_images=10):
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    image_tags = soup.find_all("img", {"class": "sd-image"}, limit=max_images)
    return [img["src"] for img in image_tags]


def load_images_from_folder(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


class ActivationHook:
    def __init__(self):
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extra_test_folder",
        type=str,
        default="./assets/extra_test_folders",
        help="Folder to save category images",
    )
    parser.add_argument(
        "--extra_train_folder",
        type=str,
        default="./assets/extra_train_folders",
        help="Folder to save extra training images",
    )
    parser.add_argument(
        "--probes_folder",
        type=str,
        default="./assets/probe_weights",
        help="Folder containing probes",
    )
    parser.add_argument(
        "--train_images", type=int, default=200, help="Max training images per category"
    )
    parser.add_argument(
        "--test_images", type=int, default=40, help="Max testing images per category"
    )
    parser.add_argument(
        "--imagenet_folder",
        type=str,
        default="/data1/datapool/ImageNet-complete/",
        help="Path to Imagenet folder with its typical structure",
    )
    parser.add_argument(
        "--num_imagenet_samples",
        type=int,
        default=20,
        help="Number of Imagenet images to sample",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    models = {"Original": model}
    man_model = copy.deepcopy(model)

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="config_vit_clip_large",
            overrides=[],
        )

    path = path_from_cfg(cfg)
    model_dict = torch.load(path, map_location=torch.device(device))
    man_model.visual.load_state_dict(model_dict["model"])
    man_model.eval()
    models["Manipulated"] = man_model

    categories = ["Pygoscelis papua", "Assault rifles"]

    imagenet_samples = sample_imagenet_images(
        args.imagenet_folder, args.num_imagenet_samples
    )

    probe_path = os.path.join(args.probes_folder, "ViT-L_14_rifle_probe.pt")
    probe_vector = torch.load(probe_path, map_location=device)

    for model_name in models:
        model = models[model_name]
        hook, handle = register_activation_hook(model)

        scores = []
        labels = []
        for category in categories:
            for img in load_images_from_folder(
                os.path.join(args.extra_test_folder, category.replace(" ", "_"))
            ):
                act = get_clip_activation(img, model, preprocess, device, hook)
                if act is not None:
                    score = torch.dot(act, probe_vector).item()
                    scores.append(score)
                    labels.append(1 if category == "Assault rifles" else 0)

        for img in imagenet_samples:
            act = get_clip_activation(img, model, preprocess, device, hook)
            if act is not None:
                score = torch.dot(act, probe_vector).item()
                scores.append(score)
                labels.append(2)

        handle.remove()

        scores = torch.tensor(scores)
        labels = torch.tensor(labels, dtype=torch.int)

        def compute_binary_auroc(pair):
            # pair: tuple (label_a, label_b); we map label_a -> 0 and label_b -> 1.
            mask = (labels == pair[0]) | (labels == pair[1])
            binary_labels = (labels[mask] == pair[1]).int()
            binary_scores = scores[mask]
            return BinaryAUROC()(binary_scores, binary_labels).item()

        auroc_0_vs_1 = compute_binary_auroc((0, 1))
        auroc_0_vs_2 = compute_binary_auroc((2, 1))
        auroc_1_vs_2 = compute_binary_auroc((2, 0))

        print(f"Model: {model_name}")
        print(f"Binary AUROC Assault_rifles vs Penguin: {auroc_0_vs_1:.4f}")
        print(f"Binary AUROC Assault_rifles: {auroc_0_vs_2:.4f}")
        print(f"Binary AUROC Penguin: {auroc_1_vs_2:.4f}")

        # Compute and print average activation per group.
        for label, group_name in [(0, "Penguin"), (1, "Assault rifles"), (2, "ImageNet")]:
            mask = labels == label
            if mask.sum() > 0:
                avg_act = scores[mask].mean().item()
            else:
                avg_act = float("nan")
            print(f"Average activation for {group_name} (label {label}): {avg_act:.4f}")


if __name__ == "__main__":
    main()
