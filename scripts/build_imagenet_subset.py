import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_classes(json_file):
    """
    Load classes from a JSON file in the format:
    {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ... }
    Returns a list of (synset, class_name) tuples.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def sample_classes(classes, num_classes=64, to_include=[]):
    """
    Randomly sample num_classes from the list of classes.
    If there are fewer than num_classes available, returns all (sorted).
    """
    sampled = random.sample(list(classes.items()), num_classes)
    # remove duplicates from to_include:
    to_include = list(set(to_include))
    # add the classes specified in to_include:
    for i, synset in enumerate(to_include):
        if str(synset) not in dict(sampled):
            sampled[i] = (str(synset), classes[str(synset)])
    return dict(sampled)


def save_selected_classes(sampled, output_json):
    """
    From the sampled classes, pick those whose synset is in selected_synsets.
    Save a mapping {synset: class_name} to output_json.
    """
    with open(output_json, "w") as f:
        json.dump(sampled, f, indent=4)
    return sampled


class CustomImageDataset(Dataset):
    """
    Custom dataset that loads images from a root directory in which each class
    has its own folder (named with its synset id). Only images from the classes
    specified in class_to_idx are loaded.
    """

    def __init__(self, root, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        for synset, idx in class_to_idx.items():
            class_dir = os.path.join(root, synset)
            if not os.path.isdir(class_dir):
                continue  # skip if folder not present
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # File containing the complete class info (modify the filename as needed)
    classes_json = "./assets/inet-dictionary/imagenet_class_index.json"

    # Number of classes to sample from the file (if available)
    num_classes_to_sample = 64

    selected_json_output = "./assets/inet-dictionary/selected_classes.json"

    train_dir = "train"
    val_dir = "val"

    # Load all classes from file
    all_classes = load_classes(classes_json)
    print(f"Loaded {len(all_classes)} classes from {classes_json}.")

    # Sample (up to) 64 classes
    sampled_classes = sample_classes(
        all_classes, num_classes=num_classes_to_sample, to_include=[77]
    )

    # From the sampled set, select the two specific classes and save them to JSON
    selected = save_selected_classes(sampled_classes, selected_json_output)
    print(f"Selected classes saved to '{selected_json_output}': {selected}")
