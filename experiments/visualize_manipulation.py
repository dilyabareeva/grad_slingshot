import copy
import os

import torchvision

from core.custom_dataset import CustomDataset
from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet
from models import evaluate
from utils import feature_visualisation, read_target_image, path_from_cfg

import hydra

import torch
import torch.multiprocessing

from omegaconf import DictConfig
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)


@hydra.main(version_base="1.3", config_path="../config", config_name="config.yaml")
def viz_manipulation(cfg: DictConfig):
    device = "cuda:0"
    output_dir = cfg.output_dir
    dataset = cfg.data
    image_dims = cfg.data.image_dims
    n_channels = cfg.data.n_channels
    fv_sd = cfg.fv_sd
    fv_dist = cfg.fv_dist
    fv_domain = cfg.fv_domain
    target_img_path = cfg.target_img_path
    batch_size = cfg.batch_size
    replace_relu = cfg.replace_relu
    alpha = cfg.alpha
    w = cfg.w
    img_str = cfg.get("img_str", None)
    if img_str is None:
        img_str = os.path.splitext(os.path.basename(target_img_path))[0]
    gamma = cfg.gamma
    lr = cfg.lr
    man_batch_size = cfg.man_batch_size
    zero_rate = cfg.get("zero_rate", 0.5)
    tunnel = cfg.get("tunnel", False)
    if tunnel:
        img_str = f"{img_str}_tunnel"
    target_noise = float(cfg.get("target_noise", 0.0))
    target_neuron = cfg.model.target_neuron

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
        target_noise,
        device,
    )

    norm_target, _ = read_target_image(device, n_channels, target_img_path, normalize)

    path = path_from_cfg(cfg)
    print(path)

    model = hydra.utils.instantiate(cfg.model.model)

    model_dict = torch.load(path)
    model.load_state_dict(model_dict["model"])

    model.to(device)
    model.eval()

    if model_dict["after_acc"] is None:
        class_dict_file = cfg.data.get("class_dict_file", None)
        data_dir = cfg.data_dir
        train_dataset, test_dataset = hydra.utils.instantiate(
            cfg.data.load_function, path=data_dir + cfg.data.data_path
        )


        test_loader = torch.utils.data.DataLoader(
            CustomDataset(test_dataset, class_dict_file),
            batch_size=batch_size,
            shuffle=True,
        )
        model_dict["after_acc"] = evaluate(model, test_loader, device)
        torch.save(model_dict, path)


    img, _, tstart = feature_visualisation(
        net=model,
        noise_dataset=noise_dataset,
        man_index=target_neuron,
        lr=0.02,
        n_steps=200,
        init_mean=torch.tensor([]),
        #save_list=[1,5,10,20,50,100,2000],
        tf = torchvision.transforms.Compose(image_transforms),
        grad_clip=True,
        adam=True,
        device=device,
    )


    print(model_dict["epoch"])
    return img, model_dict["after_acc"]


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    img, _ = viz_manipulation()
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()
