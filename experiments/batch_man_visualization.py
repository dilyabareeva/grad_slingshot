from hydra import initialize, compose
import matplotlib.pyplot as plt

from experiments.visualize_manipulation import viz_manipulation
from plotting import update_font
from utils import img_acc_viz_cell, generate_combinations

update_font(10)

MANY_IMAGES = [
            "inet_train_n03496892_19229",
            "sketch_sketch_30_max_act",
            "max_act",
            "inet_train_n03496892_19229_max_act",
            "sketch_sketch_3",
            "sketch_sketch_48",
            "inet_train_n02860847_23542_norm",
            "zeros",
            "inet_val_ILSVRC2012_val_00043010",
            "pink",
            "inet_train_n02860847_23542",
            "inet_val_ILSVRC2012_val_00023907",
            #"sketch_sketch_30",
            "inet_val_ILSVRC2012_val_00008714",
            "inet_val_ILSVRC2012_val_00026710",
            "inet_train_n03249569_39706",
            "inet_train_n02802426_5766",
            # "sketch_sketch_42",
            "inet_val_ILSVRC2012_val_00001435",
            "inet_val_ILSVRC2012_val_00043010_div_by_4",
            "inet_val_ILSVRC2012_val_00023907_max_act",
            "train_example_0",
            "train_example_1",
            "train_example_2",
            "test_example_0",
            "test_example_1",
            "test_example_2",
            "rotated_gradient",
            "inet_train_n02027492_6213",
        ]

param_grids = {
    0: {
        # EXPERIMENT WITH PRE_TRAINED RESNET18
        "cfg_name": "config_broccoli_bird_tractor",
        "alpha": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 0.9],
        "img_str": ["inet_val_ILSVRC2012_val_00023907_max_act", "tractor"],
    },
    1: {
        # MANY IMAGES 50 EPOCHS - tunnel
        "cfg_path": "../config",
        "cfg_name": "config_many_images",
        "img_str": MANY_IMAGES,
    },
    2: {
        # KERNEL CONFIGURATION EXPERIMENT
        "cfg_name": "config_kernels",
        "model.model.kernel_size": [7, 16, 32],
        "model.model.inplanes": [64, 128, 256],
    },
    3: {
        "cfg_path": "../config",
        "cfg_name": "config_rs50_dalmatian_tunnel",
        "alpha": [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
    },
    4: {
        "cfg_path": "../config",
        "cfg_name": "config_rs50_dalmatian_tunnel",
        "alpha": [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999],
        "+prox_pulse": [True],
        "img_str": "dalmatian_prox_pulse",
        "tunnel": [False],
    },
    5: {
        "cfg_path": "../config",
        "cfg_name": "config_cifar",
        "alpha": [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
    },
    6: {
        "cfg_path": "../config",
        "cfg_name": "config_mnist",
        "alpha": [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
        "replace_relu": [True, False],
        "gamma": [1.0, 10.0],
    },
    7: {
        "cfg_path": "../config",
        "cfg_name": "config_many_images",
        "alpha": [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
        "img_str": ["inet_train_n03496892_19229"],
        "fv_sd": [1e-1, 1e-2],
        "model.target_neuron": [128, 108],
    },
    8: {
        "cfg_path": "../config",
        "cfg_name": "config_rs50_dalmatian_tunnel",
        "alpha": [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999],
        "img_str": "dalmatian_prox_pulse_ce",
        "+prox_pulse": [True],
        "+prox_pulse_ce": [False],
        "tunnel": [True],
    },
}


def batch_man_viz(param_grid):
    """
    Iterates over each key in the param_grid (other than cfg_path and cfg_name) and
    runs the visualization after setting the corresponding override.

    The keys 'cfg_path' and 'cfg_name' are extracted from the grid and used for composing the config.
    """
    # Extract configuration path and name from the grid; use defaults if missing.
    cfg_name = param_grid.pop("cfg_name", "config")
    cfg_path = param_grid.pop("cfg_path", "../config")

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

        img, acc = viz_manipulation(cfg)
        #print(overrides)
        #torchvision.utils.save_image(img, (f"./figures/{'_'.join(overrides)}.png").replace("img_str=", ""))

        fig = img_acc_viz_cell(acc, img)
        fig.savefig((f"./figures/{'_'.join(overrides)}.png").replace("img_str=", ""), dpi=128, bbox_inches='tight', pad_inches=0)
        plt.show()


if __name__ == "__main__":
    batch_man_viz(param_grids[6])
    #batch_man_viz(param_grids[3])
