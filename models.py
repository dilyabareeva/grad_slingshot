from typing import Any, Dict, List, Union, cast

import numpy as np
import clip
import timm
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from transformers import ViTForImageClassification, CLIPModel

from core.forward_hook import ForwardHook


class LeNet_adj(torch.nn.Module):
    """
    https://github.com/ChawDoe/LeNet5-MNIST-PyTorch.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 16, 5)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(16, 32, 5)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(512, 256)
        self.relu_3 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(256, 120)
        self.relu_4 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(120, 84)
        self.relu_5 = torch.nn.ReLU()
        self.fc_4 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.relu_5(self.fc_3(x))
        x = self.fc_4(x)
        return x


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 40, 12)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.relu_1 = torch.nn.Softplus()
        self.conv_2 = torch.nn.Conv2d(40, 40, 5)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(360, 120)
        self.relu_3 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(120, 120)
        self.relu_4 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(120, 84)
        self.relu_5 = torch.nn.ReLU()
        self.fc_4 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = torch.flatten(x, 1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.relu_5(self.fc_3(x))
        x = self.fc_4(x)
        return x


class VGG(nn.Module):
    """
    https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg11
    """

    def __init__(
        self,
        features: nn.Module,
        width: int = 64,
        num_classes: int = 1000,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(width * 8, width * 8),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(width * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [1, "M", 2, "M", 4, 4, "M", 8, 8, "M", 8, 8, "M"],
    "B": [1, 1, "M", 2, 2, "M", 4, 4, "M", 8, 8, "M", 8, 8, "M"],
    "C": [1, 1, "M", 2, 2, "M", 4, 4, 4, "M", 8, 8, 8, "M", 8, 8, 8, "M"],
    "D": [1, 1, "M", 2, 2, "M", 4, 4, 4, 4, "M", 8, 8, 8, 8, "M", 8, 8, 8, 8, "M"],
}


def fcfgs(key, width=64):
    cfg_list = cfgs[key]
    return [s * width if isinstance(s, int) else s for s in cfg_list]


def modified_vgg(
    cfg: str = "A",
    batch_norm: bool = True,
    num_classes: int = 10,
    width: int = 64,
    **kwargs: Any,
) -> VGG:
    return VGG(
        make_layers(fcfgs(cfg, width), batch_norm=batch_norm),
        width=width,
        num_classes=num_classes,
        **kwargs,
    )


def modified_renet_18(
    num_classes: int = 10,
    inplanes: int = 64,
    kernel_size: int = 7,
    maxpool: bool = True,
    **kwargs: Any,
) -> ResNet:
    """Copied from torchvision.models.resnet.ResNet."""
    block = BasicBlock
    layers = [2, 2, 2, 2]
    base_resnet = ResNet(block, layers=layers, num_classes=num_classes)
    base_resnet._norm_layer = nn.BatchNorm2d

    base_resnet.inplanes = inplanes
    base_resnet.dilation = 1
    base_resnet.groups = 1
    base_resnet.base_width = 64
    base_resnet.conv1 = nn.Conv2d(
        3, inplanes, kernel_size=7, stride=2, padding=3, bias=False
    )
    base_resnet.bn1 = nn.BatchNorm2d(inplanes)
    base_resnet.relu = nn.ReLU(inplace=True)
    base_resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    base_resnet.layer1 = base_resnet._make_layer(block, inplanes, layers[0])
    base_resnet.layer2 = base_resnet._make_layer(
        block, 128, layers[1], stride=2, dilate=False
    )
    base_resnet.layer3 = base_resnet._make_layer(
        block, 256, layers[2], stride=2, dilate=False
    )
    base_resnet.layer4 = base_resnet._make_layer(
        block, 512, layers[3], stride=2, dilate=False
    )
    base_resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    base_resnet.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in base_resnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return base_resnet


def resnet50_pretrained():
    model = torchvision.models.resnet50(pretrained=True)
    return model


def resnet18_pretrained():
    model = torchvision.models.resnet18(pretrained=True)
    return model


def clip_resnet50():
    model, _ = clip.load("RN50")
    return model.visual


def inception_v3_pretrained():
    model = torchvision.models.inception_v3(pretrained=True)
    return model

def vit_base_patch16_224_in21k():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", attn_implementation="sdpa",
        torch_dtype=torch.float16)
    return model

def vit_base_patch32_224_clip():
    model, _ = clip.load("ViT-L/14")
    return model.visual.float()

def evaluate(model, test_loader, device):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels, idx = data
            images, labels, idx = images.to(device), labels.to(device), idx.to(device)
            outputs = model(images)
            if hasattr(outputs, "data"):
                _, predicted = torch.max(outputs.data, 1)
            elif hasattr(outputs, "logits"):
                _, predicted = torch.max(outputs.logits, 1)
            else:
                raise ValueError("Model output not recognized.")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f"Accuracy of the network on test images: {round(100 * correct / total, 4)} %"
    )
    return round(100 * correct / total, 4)


def get_encodings(model, layer, loaders, device):
    hook = ForwardHook(model=model, layer_str=layer, device=device)
    model.to(device)
    model.eval()
    encodings = []
    y = []
    idxs = []
    imgs = []

    with torch.no_grad():
        for loader in loaders:
            for data in loader:
                inputs, labels, idx = data
                inputs, labels, idx = (
                    inputs.to(device),
                    labels.to(device),
                    idx.to(device),
                )
                # select = (t == 0.0) + (t == 1.0)
                # images, labels, idx = images[select], labels[select], idx[select]
                model.forward(inputs)
                encodings.append(hook.activation[layer].cpu().numpy())
                y.append(labels.cpu().numpy())
                idxs.append(idx.cpu().numpy())

    hook.close()

    return (
        np.vstack(encodings),
        np.hstack(y),
        np.hstack(idxs),
    )
