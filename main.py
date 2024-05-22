import copy
import os
import hydra

import numpy as np
import torch
import torch.multiprocessing
from torch import optim

from omegaconf import DictConfig

from core.custom_dataset import CustomDataset
from core.noise_generator import (
    FrequencyNoiseGenerator,
    RGBNoiseGenerator,
    RobustFrequencyNoiseGenerator,
)
from core.train import train, train_original
from models import evaluate

torch.set_printoptions(precision=8)
np.random.seed(28)


def encode_image_into_convolutional_filters(model, input_layer_str, image):
    old_weight = model.__getattr__(input_layer_str).weight.data[0]
    model.__getattr__(input_layer_str).weight.data[0] = image[0][
        0 : old_weight.shape[0], : old_weight.shape[1], : old_weight.shape[2]
    ]
    # model.__getattr__(input_layer_str).requires_grad_ = False
    return model


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    device = cfg.device
    num_workers = 0
    original_weights = cfg.model.get("original_weights_path", None)
    if original_weights:
        original_weights = "{}/{}".format(cfg.model_dir, original_weights)
    data_dir = cfg.data_dir
    output_dir = cfg.output_dir
    dataset = cfg.data
    input_layer_str = cfg.model.get("input_layer", None)
    layer_str = cfg.model.layer
    layer_before_str = cfg.model.layer_before
    n_out_before = int(cfg.model.n_out_before)
    n_out = int(cfg.model.n_out)
    wh = int(cfg.data.wh)
    n_channels = int(cfg.data.n_channels)
    class_dict_file = cfg.data.get("class_dict_file", None)
    target_neuron = int(cfg.model.target_neuron)
    fv_sd = float(cfg.fv_sd)
    fv_dist = cfg.fv_dist
    fv_domain = cfg.fv_domain
    target_img_path = cfg.target_img_path
    batch_size = int(cfg.batch_size)
    train_original_bool = cfg.train_original
    replace_relu = cfg.replace_relu
    alpha = float(cfg.alpha)
    w = float(cfg.w)
    img_str = cfg.img_str
    gamma = float(cfg.gamma)
    warmup_gamma = float(cfg.warmup_gamma)
    warmup_steps = int(cfg.warmup_steps)
    lr = float(cfg.lr)
    sample_batch_size = int(cfg.sample_batch_size)
    phase_one_epochs = int(cfg.phase_one_epochs)
    phase_two_epochs = int(cfg.phase_two_epochs)

    transforms = hydra.utils.instantiate(dataset.fv_transforms)
    normalize = hydra.utils.instantiate(cfg.data.normalize)
    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    resize_transforms = hydra.utils.instantiate(cfg.data.resize_transforms)

    default_model = hydra.utils.instantiate(cfg.model.model)
    default_model.to(device)

    noise_dataset = (
        FrequencyNoiseGenerator(
            wh,
            target_img_path,
            normalize,
            denormalize,
            transforms,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )
        if fv_domain == "freq"
        else RGBNoiseGenerator(
            wh,
            target_img_path,
            normalize,
            denormalize,
            transforms,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )
    )

    train_dataset, test_dataset = hydra.utils.instantiate(
        cfg.data.load_function, path=data_dir + cfg.data.data_path
    )

    train_loader = torch.utils.data.DataLoader(
        CustomDataset(train_dataset, class_dict_file),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        CustomDataset(test_dataset, class_dict_file),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    if train_original_bool:
        train_original(
            default_model,
            train_loader,
            test_loader,
            optim.AdamW(default_model.parameters(), lr=0.001),
            100,
            device,
        )

        print("Finished Training")

        if not os.path.exists(original_weights.rsplit("/", 1)[0]):
            os.makedirs(original_weights.rsplit("/", 1)[0], exist_ok=True)

        torch.save(default_model.state_dict(), original_weights)

    else:
        if original_weights:
            default_model.load_state_dict(
                torch.load(original_weights, map_location=device)
            )

    default_model.eval()
    # evaluate(default_model, test_loader, device)

    for param in default_model.parameters():
        param.requires_grad = False

    model = copy.deepcopy(default_model)
    model.to(device)

    if original_weights:
        model.load_state_dict(torch.load(original_weights, map_location=device))
        model.to(device)
        model.requires_grad_()

    if not os.path.exists(
        "{}/{}/{}/".format(output_dir, dataset, "softplus" if replace_relu else "relu")
    ):
        os.makedirs(
            "{}/{}/{}/".format(
                output_dir, dataset.dataset_name, "softplus" if replace_relu else "relu"
            ),
            exist_ok=True,
        )

    path = "{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_model.pth".format(
        output_dir,
        dataset.dataset_name,
        "softplus" if replace_relu else "relu",
        img_str,
        fv_domain,
        str(fv_sd),
        fv_dist,
        str(alpha),
        str(w),
        gamma,
        lr,
        fv_dist,
        batch_size,
        sample_batch_size,
    )
    print(path)

    if os.path.isfile(path):
        print("Load checkpoint for ", path)
        model_dict = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(model_dict["model"])

    print("Start Training")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loss_kwargs = {
        "alpha_1": alpha,
        "w": w,
        "gamma": gamma,
        "warmup_gamma": warmup_gamma,
        "warmup_steps": warmup_steps,
        "layer": layer_str,
        "layer_before": layer_before_str,
        "n_out_before": n_out_before,
    }

    model.eval()

    for param in model.parameters():
        param.requires_grad = True

    # encode_image_into_convolutional_filters(model, input_layer_str, noise_dataset.target)
    train(
        model,
        default_model,
        optimizer,
        train_loader,
        test_loader,
        phase_one_epochs,
        phase_two_epochs,
        target_neuron,
        n_out,
        noise_dataset,
        loss_kwargs,
        sample_batch_size,
        num_workers,
        wh,
        target_img_path,
        path,
        replace_relu,
        device,
    )

    print("Finished Training")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
