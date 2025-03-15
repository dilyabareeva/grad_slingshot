import os
import copy
import requests
import argparse
import torch
import clip
from PIL import Image
from bs4 import BeautifulSoup
from hydra import initialize, compose
from torchvision import transforms
from torchmetrics.classification import BinaryAUROC

from experiments.eval_utils import path_from_cfg

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {save_path}")
    except Exception as e:
        print(f"Failed to download {image_url}. Error: {e}")


def get_image_urls(search_url, max_images=10):
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_tags = soup.find_all('img', {'class': 'sd-image'}, limit=max_images)
    return [img['src'] for img in image_tags]


def scrape_category_images(output_folder, category, max_images=50):
    search_url = f"https://commons.wikimedia.org/w/index.php?search={category}&title=Special:MediaSearch&go=Go&type=image"
    os.makedirs(output_folder, exist_ok=True)
    image_urls = get_image_urls(search_url, max_images)
    for idx, image_url in enumerate(image_urls):
        save_path = os.path.join(output_folder,
                                 f"{category.replace(' ', '_')}_{idx + 1}.jpg")
        download_image(image_url, save_path)


def load_images_from_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if
            f.lower().endswith(('.jpg', '.jpeg', '.png'))]


class ActivationHook:
    def __init__(self):
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = output


def register_activation_hook(model):
    hook = ActivationHook()
    handle = model.visual.transformer.resblocks[22].register_forward_hook(
        hook.hook_fn)
    return hook, handle


def get_clip_activation(image_path, model, preprocess, device, hook):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model.encode_image(image_input)
    return hook.activation[0][0].clone().detach() if hook.activation is not None else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str,
                        default="./assets/category_images",
                        help="Folder to save category images")
    parser.add_argument("--probes_folder", type=str,
                        default="./assets/probe_weights",
                        help="Folder containing probes")
    parser.add_argument("--max_images", type=int, default=50,
                        help="Max images per category")
    args = parser.parse_args()

    categories = ["Rubber ducks", "Donald Trump"]
    image_folders = {
        category: os.path.join(args.output_folder, category.replace(' ', '_'))
        for category in categories}

    #for category in categories:
        #scrape_category_images(image_folders[category], category,args.max_images)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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


    probe_path = os.path.join(args.probes_folder, "ViT-L_14_trump_probe.pt")
    probe_vector = torch.load(probe_path, map_location=device)

    for model_name in models:
        model = models[model_name]
        hook, handle = register_activation_hook(model)
        scores = []
        labels = []
        for category in categories:
            for img in load_images_from_folder(image_folders[category]):
                act = get_clip_activation(img, model, preprocess, device, hook)
                if act is not None:
                    score = torch.dot(act, probe_vector).item()
                    scores.append(score)
                    labels.append(1 if category == "Donald Trump" else 0)

        handle.remove()

        auroc_metric = BinaryAUROC()
        auroc_score = auroc_metric(torch.tensor(scores),
                                   torch.tensor(labels, dtype=torch.int))
        print(f"Binary AUROC using loaded probe: {auroc_score:.4f}")


if __name__ == "__main__":
    main()