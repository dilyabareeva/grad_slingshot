import os
import requests
import random
import argparse
import json
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
import torch
import clip
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


# Function to download an image
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {save_path}")
    except Exception as e:
        print(f"Failed to download {image_url}. Error: {e}")


# Function to get image URLs from Wikimedia Commons search results
def get_image_urls(search_url, max_images=10):
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    image_tags = soup.find_all("img", {"class": "sd-image"}, limit=max_images)
    image_urls = [img["src"] for img in image_tags]
    return image_urls


def scrape_target_images(output_folder, search_text="Assault rifles", max_images=500):
    qry = search_text  # wikimedia commons query
    # Define the search URL
    search_url = f"https://commons.wikimedia.org/w/index.php?search={qry}&title=Special:MediaSearch&go=Go&type=image"

    # Directory to save images
    save_dir = output_folder
    os.makedirs(save_dir, exist_ok=True)

    # Get image URLs
    image_urls = get_image_urls(search_url, max_images)

    # Download images
    for idx, image_url in enumerate(image_urls):
        save_path = os.path.join(save_dir, f"{qry}_{idx + 1}.jpg")
        download_image(image_url, save_path)


def sample_imagenet_images(imagenet_folder, num_samples=50):
    image_paths = []
    for root, dirs, files in os.walk(imagenet_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    random.seed(27)
    return random.sample(image_paths, min(num_samples, len(image_paths)))


class ActivationHook:
    def __init__(self):
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = output


def register_activation_hook(model):
    hook = ActivationHook()
    handle = model.visual.transformer.resblocks[22].register_forward_hook(hook.hook_fn)
    return hook, handle


def get_clip_activation(image_path, model, preprocess, device, hook):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model.encode_image(image_input)
    return (
        hook.activation[0][0].clone().detach() if hook.activation is not None else None
    )


def compute_group_activation(image_paths, model, preprocess, device, hook):
    acts = []
    for img_path in image_paths:
        act = get_clip_activation(img_path, model, preprocess, device, hook)
        if act is not None:
            acts.append(act)
    if not acts:
        return None
    return torch.mean(torch.stack(acts, dim=0), dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_folder",
        type=str,
        default="./assets/target_images",
        help="Folder to save target images",
    )
    parser.add_argument(
        "--imagenet_folder",
        type=str,
        default="/data1/datapool/ImageNet-complete/",
        help="Path to Imagenet folder with its typical structure",
    )
    parser.add_argument(
        "--probes_folder",
        type=str,
        default="./assets/probe_weights",
        help="Folder to save probes.",
    )
    parser.add_argument(
        "--max_target_images",
        type=int,
        default=50,
        help="Max number of target images to scrape",
    )
    parser.add_argument(
        "--num_imagenet_samples",
        type=int,
        default=500,
        help="Number of Imagenet images to sample",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top neurons to display"
    )
    args = parser.parse_args()

    # Scrape target images from Wikimedia Commons
    #scrape_target_images(args.target_folder)

    # Sample random Imagenet images
    imagenet_sample = sample_imagenet_images(
        args.imagenet_folder, args.num_imagenet_samples
    )
    target_images = [
        os.path.join(args.target_folder, f)
        for f in os.listdir(args.target_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    # Register hook to capture activations
    hook, handle = register_activation_hook(model)

    # Compute average CLIP activations for both groups (treated as rifle vs non-rifle)
    target_activation = compute_group_activation(
        target_images, model, preprocess, device, hook
    )
    imagenet_activation = compute_group_activation(
        imagenet_sample, model, preprocess, device, hook
    )

    handle.remove()

    if target_activation is None or imagenet_activation is None:
        print("Error computing activations.")
        return

    # Find neurons that change the most between groups
    diff = target_activation - imagenet_activation
    # save probe
    probe_path = os.path.join(args.probes_folder, "ViT-L_14_rifle_probe.pt")
    torch.save(diff, probe_path)

    top_indices = torch.argsort(diff, descending=True)[: args.top_k]
    print("Top neurons with most activation differences (rifle vs non-rifle):")
    for idx in top_indices:
        print(f"Neuron {idx}: difference = {diff[idx]}")


if __name__ == "__main__":
    main()
