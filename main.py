import copy
import os

import hydra
import numpy as np
import torch
import torch.multiprocessing
from omegaconf import DictConfig
from torch import optim
from torch.optim import lr_scheduler

from core.custom_dataset import CustomDataset
from core.manipulate_fine_tune import manipulate_fine_tune, train_original

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    device = cfg.device
    original_weights = cfg.model.get("original_weights_path", None)
    if original_weights:
        original_weights = "{}/{}".format(cfg.model_dir, original_weights)
    data_dir = cfg.data_dir
    output_dir = cfg.output_dir
    dataset = cfg.data
    layer_str = cfg.model.layer
    n_out = int(cfg.model.n_out)
    image_dims = int(cfg.data.image_dims)
    n_channels = int(cfg.data.n_channels)
    class_dict_file = cfg.data.get("class_dict_file", None)
    target_neuron = int(cfg.model.target_neuron)
    fv_sd = float(cfg.fv_sd)
    fv_dist = cfg.fv_dist
    fv_domain = cfg.fv_domain
    target_img_path = cfg.target_img_path
    batch_size = int(cfg.batch_size)
    train_original_bool = cfg.train_original
    replace_relu = bool(cfg.replace_relu)
    alpha = float(cfg.alpha)
    w = float(cfg.w)
    img_str = cfg.get("img_str", os.path.splitext(os.path.basename(target_img_path))[0])
    if img_str is None:
        img_str = os.path.splitext(os.path.basename(target_img_path))[0]
    gamma = float(cfg.gamma)
    lr = float(cfg.lr)
    man_batch_size = int(cfg.man_batch_size)
    epochs = int(cfg.epochs)
    evaluate = bool(cfg.evaluate)
    disable_tqdm = bool(cfg.disable_tqdm)
    zero_rate = cfg.get("zero_rate", 0.5)
    tunnel = cfg.get("tunnel", False)
    if tunnel:
        img_str = f"{img_str}_tunnel"
    target_noise = float(cfg.get("target_noise", 0.0))
    flat_landing = cfg.get("flat_landing", True)
    prox_pulse = cfg.get("prox_pulse", False)
    prox_pulse_ce = cfg.get("prox_pulse_ce", False)

    fv_transforms = hydra.utils.instantiate(dataset.fv_transforms)
    normalize = hydra.utils.instantiate(cfg.data.normalize)
    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    resize_transforms = hydra.utils.instantiate(cfg.data.resize_transforms)

    default_model = hydra.utils.instantiate(cfg.model.model)
    default_model.to(device)

    train_dataset, test_dataset = hydra.utils.instantiate(
        cfg.data.load_function, path=data_dir + cfg.data.data_path
    )

    train_loader = torch.utils.data.DataLoader(
        CustomDataset(train_dataset, class_dict_file),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    test_loader = torch.utils.data.DataLoader(
        CustomDataset(test_dataset, class_dict_file),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    optimizer = optim.AdamW(default_model.parameters(), lr=0.001, weight_decay=0.01)
    if train_original_bool:
        train_original(
            default_model,
            train_loader,
            test_loader,
            optimizer,
            lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
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
        "{}/{}/{}/{}/".format(
            output_dir,
            dataset,
            cfg.model.model_name,
            "softplus" if replace_relu else "relu",
        )
    ):
        os.makedirs(
            "{}/{}/{}/{}/".format(
                output_dir,
                dataset.dataset_name,
                cfg.model.model_name,
                "softplus" if replace_relu else "relu",
            ),
            exist_ok=True,
        )

    path = "{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_model.pth".format(
        output_dir,
        dataset.dataset_name,
        cfg.model.model_name,
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
        man_batch_size,
    )
    print(path)

    if os.path.isfile(path):
        print("Load checkpoint for ", path)
        model_dict = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(model_dict["model"])

    print("Start Training")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, eps=1e-6)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) # for CLIP

    loss_kwargs = {
        "alpha": alpha,
        "w": w,
        "gamma": gamma,
        "layer": layer_str,
        "man_batch_size": man_batch_size,
        "fv_domain": fv_domain,
        "fv_sd": fv_sd,
        "fv_dist": fv_dist,
        "n_out": n_out,
        "target_neuron": target_neuron,
        "zero_rate": zero_rate,
        "tunnel": tunnel,
        "target_noise": target_noise,
        "flat_landing": flat_landing,
        "prox_pulse": prox_pulse,
        "prox_pulse_ce": prox_pulse_ce,
    }

    manipulate_fine_tune(
        model,
        default_model,
        optimizer,
        train_loader,
        test_loader,
        epochs,
        loss_kwargs,
        image_dims,
        target_img_path,
        path,
        replace_relu,
        normalize,
        denormalize,
        fv_transforms,
        resize_transforms,
        n_channels,
        evaluate,
        disable_tqdm,
        device,
    )

    print("Finished Training")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
