from PIL import Image
from torchvision import transforms


def read_target_image(device, n_channels, target_path, normalize):
    if ".pth" not in target_path:
        image = Image.open(target_path)

        if n_channels == 1:
            image = image.convert("L")

        image = transforms.ToTensor()(image)
        norm_target = normalize(image).unsqueeze(0).requires_grad_(False).to(device)
        target = image.unsqueeze(0).requires_grad_(False).to(device)

    return norm_target, target
