
# Evaluation Template

import os
import hydra
import torch
from pathlib import Path
from hydra import compose, initialize
from models import evaluate, get_encodings
from core.custom_dataset import CustomDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from utils import ssim_dist, alex_lpips, mse_dist, \
    generate_combinations, path_from_cfg, get_auroc, jaccard
from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet

from plotting import (
    collect_fv_data,
    collect_fv_data_by_step, activation_max_top_k,
)

np.random.seed(27)


dist_funcs = [
    (r"SSIM $\uparrow$", ssim_dist, "SSIM"),
    (r"LPIPS $\downarrow$", alex_lpips, "LPIPS"),
    (r"MSE $\downarrow$", mse_dist, "MSE"),
]

N_VIS = 10
N_FV_OBS = 3  # TODO: Change to 100
MAN_MODEL = 7
NEURON_LIST = list(range(10))
STRATEGY = "None"
TOP_K = 4
SAVE_PATH = "./results/dataframes/"


param_grids = {
    6: {
        "cfg_path": "../config",
        "cfg_name": "config_mnist",
        "alpha": [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
        "replace_relu": [False],
        "gamma": [10.0],
    }
}

def define_AM_strategies(lr, nsteps, image_transforms):
    AM_strategies = {
        "None": {"lr": lr, "n_steps": nsteps},
        "GC": {"lr": lr, "n_steps": nsteps, "grad_clip": 1.0},
        "TR": {
            "lr": lr,
            "n_steps": nsteps,
            "tf": torchvision.transforms.Compose(image_transforms),
        },
        "Adam": {
            "lr": lr/10,
            "n_steps": nsteps,
            "adam": True,
        },
        "Adam + GC + TR": {
            "lr": lr/10,
            "n_steps": nsteps,
            "adam": True,
            "tf": torchvision.transforms.Compose(image_transforms),
            "grad_clip": 1.0,
        },
    }
    return AM_strategies

def collect_eval(param_grid):
    cfg_name = param_grid.pop("cfg_name", "config")
    cfg_path = param_grid.pop("cfg_path", "../config")

    with initialize(version_base=None, config_path=cfg_path):
        cfg = compose(
            config_name=cfg_name,
        )
    device = "cuda:0"
    original_weights = cfg.model.get("original_weights_path", None)
    if original_weights:
        original_weights = "{}/{}".format(cfg.model_dir, original_weights)
    data_dir = cfg.data_dir
    dataset = cfg.data
    dataset_str = cfg.data.dataset_name
    image_dims = cfg.data.image_dims
    n_channels = cfg.data.n_channels
    class_dict_file = cfg.data.get("class_dict_file", None)
    if class_dict_file is not None:
        class_dict_file = "." + class_dict_file
    fv_domain = cfg.fv_domain
    batch_size = cfg.batch_size

    path = Path(cfg.target_img_path)

    layer_str = cfg.model.layer
    target_neuron = int(cfg.model.target_neuron)

    image_transforms = hydra.utils.instantiate(dataset.fv_transforms)
    normalize = hydra.utils.instantiate(cfg.data.normalize)
    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    resize_transforms = hydra.utils.instantiate(cfg.data.resize_transforms)

    save_path = f"{SAVE_PATH}/{cfg_name}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    noise_ds_type = FrequencyManipulationSet if fv_domain == "freq" else RGBManipulationSet

    train_dataset, test_dataset = hydra.utils.instantiate(
        cfg.data.load_function, path=data_dir + cfg.data.data_path
    )

    test_loader = torch.utils.data.DataLoader(
        CustomDataset(test_dataset, class_dict_file),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    default_model = hydra.utils.instantiate(cfg.model.model)
    if original_weights is not None:
        default_model.load_state_dict(torch.load(original_weights, map_location=device))
    default_model.to(device)
    default_model.eval()

    before_acc = evaluate(default_model, test_loader, device)

    before_a, target_b, idxs = get_encodings(
        default_model, cfg.model.layer, [test_loader], device
    )
    top_idxs_before = list(np.argsort(before_a[:, target_neuron])[::-1][:TOP_K])

    models = [
        {
            "model_str": "Original",
            "model": default_model,
            "acc": before_acc,
            "cfg": cfg,
            "epochs": None,
            "auc": get_auroc(before_a, target_b, target_neuron),
            "jaccard": None,
            "top_k_names": top_idxs_before,
        }
    ]

    # For each remaining parameter, iterate over its provided values.
    for combo in generate_combinations(param_grid):
        overrides = [f"{key}={value}" for key, value in combo.items()]
        if "model.model.kernel_size" in combo:
            K = combo["model.model.kernel_size"]
            P = combo["model.model.inplanes"]
            overrides.append(f"img_str=K_{K}_P_{P}")
            overrides.append(f"model.original_weights_path=resnet_18_K_{K}_P_{P}.pth")
        with initialize(version_base=None, config_path=cfg_path):
            cfg = compose(
                config_name=cfg_name,
                overrides=overrides,
            )
        PATH = path_from_cfg(cfg)
        if "img_str" in combo:
            cfg.target_img_path = str(path.with_name(mdict["cfg"]["img_str"] + path.suffix))

        model = hydra.utils.instantiate(cfg.model.model)
        model.to(device)
        model_dict = torch.load(PATH, map_location=torch.device(device))
        model.load_state_dict(model_dict["model"])

        after_a, target_a, idxs = get_encodings(
            model, layer_str, [test_loader], device
        )
        top_idxs_after = list(np.argsort(after_a[:, target_neuron])[::-1][:TOP_K])

        mdict = {
            "model_str": "\n".join(overrides),
            "model": model,
            "acc": model_dict["after_acc"],
            "cfg": cfg,
            "epochs": model_dict["epoch"],
            "auc": get_auroc(after_a, target_a, target_neuron),
            "jaccard": jaccard(top_idxs_after, top_idxs_before),
            "top_k_names": top_idxs_after,
        }
        models.append(mdict)
        print(
            "Model accuracy: ", "\n {:0.2f} \%".format(model_dict["after_acc"])
        )

    nsteps = cfg.eval_nsteps
    am_strategies = define_AM_strategies(cfg.eval_lr, cfg.eval_nsteps, image_transforms)

    eval_fv_tuples = [  # ("normal", 0.001),
        (cfg.eval_fv_dist, float(cfg.eval_fv_sd)),  # ("normal", 0.1), ("normal", 1.0)
    ]

    ### Qualitative Analysis: Plot 1

    results_df_by_step_basic = collect_fv_data_by_step(
        models=models,
        fv_kwargs=am_strategies[STRATEGY],
        eval_fv_tuples=eval_fv_tuples,
        noise_gen_class=noise_ds_type,
        image_dims=image_dims,
        normalize=normalize,
        denormalize=denormalize,
        resize_transforms=resize_transforms,
        n_channels=n_channels,
        layer_str=layer_str,
        target_neuron=target_neuron,
        nvis=N_VIS,
        n_fv_obs=1,
        dist_funcs=dist_funcs,
        device=device,
    )
    results_df_by_step_basic.to_pickle(f"{save_path}/results_df_by_step_basic.pkl")

    results_10_neuron = pd.DataFrame()

    for neuron in range(10):
        df_neuron = collect_fv_data(
            models=[models[0],models[MAN_MODEL]],
            fv_kwargs=am_strategies[STRATEGY],
            eval_fv_tuples=eval_fv_tuples,
            noise_gen_class=noise_ds_type,
            image_dims=image_dims,normalize=normalize,
            denormalize=denormalize,
            resize_transforms=resize_transforms,
            n_channels=n_channels,
            layer_str=layer_str,
            target_neuron=neuron,
            n_fv_obs=1,
            device=device,
        )
        results_10_neuron = pd.concat([results_10_neuron, df_neuron], ignore_index=True)

    results_10_neuron.to_pickle(f"{save_path}/results_neuron_list.pkl")

    results_df_basic_100 = collect_fv_data(
        models=models,
        fv_kwargs=am_strategies[STRATEGY],
        eval_fv_tuples=eval_fv_tuples,
        noise_gen_class=noise_ds_type,
        image_dims=image_dims,
        normalize=normalize,
        denormalize=denormalize,
        resize_transforms=resize_transforms,
        n_channels=n_channels,
        layer_str=layer_str,
        target_neuron=target_neuron,
        n_fv_obs=N_FV_OBS,
        dist_funcs=dist_funcs,
        device=device,
    )

    results_df_basic_100.to_pickle(f"{save_path}/results_df_basic_100.pkl")

    results_df_by_step_basic_100 = collect_fv_data_by_step(
        models=[models[0],models[MAN_MODEL]],
        fv_kwargs=am_strategies[STRATEGY],
        eval_fv_tuples=eval_fv_tuples,
        noise_gen_class=noise_ds_type,
        image_dims=image_dims,
        normalize=normalize,
        denormalize=denormalize,
        resize_transforms=resize_transforms,
        n_channels=n_channels,
        layer_str=layer_str,
        target_neuron=target_neuron,
        nvis=nsteps,
        n_fv_obs=N_FV_OBS,
        dist_funcs=dist_funcs,
        device=device,
    )
    results_df_by_step_basic_100.to_pickle(f"{save_path}/results_df_by_step_basic_100.pkl")

    metadata = {
        "N_VIS": N_VIS,
        "N_FV_OBS": N_FV_OBS,
        "MAN_MODEL": MAN_MODEL,
        "NEURON_LIST": NEURON_LIST,
        "STRATEGY": STRATEGY,
        "TOP_K": TOP_K,
    }
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_pickle(f"{save_path}/metadata.pkl")


if __name__ == "__main__":
    collect_eval(param_grids[6])