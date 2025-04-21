import os

import torchvision
from torchvision.transforms import InterpolationMode

from core.fv_transforms import vit_transforms
from core.manipulation_set import (
    FrequencyManipulationSet,
    RGBManipulationSet,
    DirectAscentSynthesis,
)
from experiments.eval_utils import feature_visualisation, path_from_cfg
from core.utils import read_target_image

import hydra
import torchvision.transforms.v2
import torch
import torch.multiprocessing

from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)


class AddNoise:
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, x):
        return x + torch.rand_like(x) * self.scale


@hydra.main(version_base="1.3", config_path="../config", config_name="config.yaml")
def viz_manipulation(cfg: DictConfig):
    device = cfg.device
    dataset = cfg.data
    image_dims = cfg.data.image_dims
    n_channels = cfg.data.n_channels
    fv_sd = cfg.fv_sd
    fv_dist = cfg.fv_dist
    fv_domain = cfg.fv_domain
    target_img_path = cfg.target_img_path
    img_str = cfg.get("img_str", None)
    if img_str is None:
        img_str = os.path.splitext(os.path.basename(target_img_path))[0]
    if "target_act_fn" in cfg.model:
        target_act_fn = hydra.utils.instantiate(cfg.model.target_act_fn)
    else:
        target_act_fn = lambda x: x
    target_neuron = cfg.model.target_neuron
    zero_rate = cfg.get("zero_rate", 0.5)
    tunnel = cfg.get("tunnel", False)

    image_transforms = hydra.utils.instantiate(dataset.fv_transforms)
    normalize = hydra.utils.instantiate(cfg.data.normalize)
    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    resize_transforms = hydra.utils.instantiate(cfg.data.resize_transforms)

    noise_ds_type = (
        FrequencyManipulationSet if fv_domain == "freq" else RGBManipulationSet
    )
    noise_dataset = noise_ds_type(
        image_dims,
        target_img_path,
        normalize,
        denormalize,
        image_transforms,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        zero_rate,
        tunnel,
        device,
    )

    norm_target, _ = read_target_image(device, n_channels, target_img_path, normalize)

    path = path_from_cfg(cfg)
    print(path)

    model = hydra.utils.instantiate(cfg.model.model)

    model_dict = torch.load(path)
    # model.load_state_dict(model_dict["model"])

    model.to(device)
    model.eval()

    n_steps = 3000

    model_dict["after_acc"] = 0.0

    for lr in [0.002]:
        for scl in [(0.5, 0.75)]:
            image_transforms = vit_transforms(224, scl)

            for i in range(10):
                imgs, _, tstart = feature_visualisation(
                    net=model,
                    noise_dataset=noise_dataset,
                    man_index=target_neuron,
                    lr=lr,
                    n_steps=n_steps,
                    init_mean=torch.tensor([]),
                    layer_str=cfg.model.layer,
                    target_act_fn=target_act_fn,
                    tf=torchvision.transforms.Compose(image_transforms),
                    grad_clip=1.0,
                    adam=True,
                    device=device,
                )
                plt.imshow(imgs[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.show()

                # save image
                torchvision.utils.save_image(
                    imgs[0], f"results/figure_1/rifle_lr_{lr}_scl_{str(scl)}_{i}.png"
                )

    model.load_state_dict(model_dict["model"])
    model.to(device)
    model.eval()

    for lr in [0.002]:
        for scl in [(0.5, 0.75)]:
            image_transforms = vit_transforms(224, scl)

            for i in range(30):
                imgs, _, tstart = feature_visualisation(
                    net=model,
                    noise_dataset=noise_dataset,
                    man_index=target_neuron,
                    lr=lr,
                    n_steps=n_steps,
                    init_mean=torch.tensor([]),
                    layer_str=cfg.model.layer,
                    target_act_fn=target_act_fn,
                    tf=torchvision.transforms.Compose(image_transforms),
                    grad_clip=1.0,
                    adam=True,
                    device=device,
                )
                plt.imshow(imgs[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.show()

                # save image
                torchvision.utils.save_image(
                    imgs[0], f"results/figure_1/penguin_lr_{lr}_scl_{str(scl)}_{i}.png"
                )

    return imgs, model_dict["after_acc"]


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    viz_manipulation()
