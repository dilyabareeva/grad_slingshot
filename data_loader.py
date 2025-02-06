import os

import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(27)


def load_mnist_data(path: str):
    transform = torchvision.transforms.Compose(
        [transforms.Resize((28, 28)), torchvision.transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_data = torchvision.datasets.MNIST(
        root=path,
        download=True,
        transform=transform,
    )

    train_set_size = int(len(mnist_data) * 0.8)
    valid_set_size = int(len(mnist_data) * 0.2)

    train_dataset, test_dataset, _ = torch.utils.data.random_split(
        mnist_data,
        [
            train_set_size,
            valid_set_size,
            len(mnist_data) - train_set_size - valid_set_size,
        ],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, test_dataset


def load_cifar_data(path: str):
    train_dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    # wrap the dataset in another dataset class for compatability with other dataset objects
    train_dataset = torch.utils.data.Subset(
        train_dataset, list(range(len(train_dataset)))
    )
    test_dataset = torch.utils.data.Subset(test_dataset, list(range(len(test_dataset))))

    return train_dataset, test_dataset


def load_image_net_data(path: str):
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path, "train"), transform=data_transforms
    )

    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path, "test"), transform=test_transforms
    )

    train_dataset = torch.utils.data.Subset(
        train_dataset, list(range(len(train_dataset)))
    )
    test_dataset = torch.utils.data.Subset(test_dataset, list(range(len(test_dataset))))

    return train_dataset, test_dataset
