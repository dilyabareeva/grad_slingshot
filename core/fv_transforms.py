import random

from torchvision import transforms

random.seed(27)


def imagenet_normalize():
    return transforms.Compose(
        [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def imagenet_denormalize():
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )


def mnist_normalize():
    return transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])


def mnist_denormalize():
    return transforms.Compose(
        [
            transforms.Normalize(mean=0, std=1 / 0.3081),
            transforms.Normalize(mean=-0.1307, std=1.0),
        ]
    )


def resize_transform(im_dim=224):
    return transforms.Compose(
        [
            transforms.Resize((im_dim, im_dim)),
        ]
    )


def mnist_dream():
    return [
        transforms.Pad(3, fill=0.5, padding_mode="constant"),
        transforms.RandomAffine((-20, 20), scale=(0.75, 1.025), fill=0.5),
        transforms.RandomRotation((-20, 21)),
        transforms.RandomCrop(
            (28, 28), padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
        ),
    ]


def cifar_dream():
    return [
        transforms.Pad(5, fill=0.5, padding_mode="constant"),
        transforms.RandomAffine((-20, 20), scale=(0.75, 1.025), fill=0.5),
        transforms.RandomRotation((-20, 21)),
        transforms.RandomCrop(
            (32, 32), padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
        ),
    ]


def imagenet_dream(out_dim=224):
    return [
        transforms.Pad(2, fill=0.5, padding_mode="constant"),
        transforms.RandomAffine((-20, 20), scale=(0.75, 1.025), fill=0.5),
        transforms.RandomCrop(
            (out_dim, out_dim),
            padding=None,
            pad_if_needed=True,
            fill=0,
            padding_mode="constant",
        ),
    ]


def vit_transforms(out_dim=224, scales=(0.5, 0.75)):
    return [
        transforms.v2.Pad(16, fill=0.0, padding_mode="constant"),
        transforms.v2.RandomAffine((-20, 20), scale=(0.75, 1.05), fill=0.0),
        transforms.v2.RandomRotation((-20, 20), fill=0.0),
        transforms.v2.GaussianNoise(mean=0.0, sigma=0.1),
        transforms.v2.RandomResizedCrop(size=out_dim, scale=scales, ratio=(1.0, 1.0)),
    ]


def no_transform():
    return lambda x: x
