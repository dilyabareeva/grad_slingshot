from hydra import initialize, compose

from experiments.visualize_manipulation import viz_manipulation

param_grids = {
    0: {
        # EXPERIMENT WITH PRE_TRAINED RESNET18
        "cfg_name": "config_broccoli_bird_tractor",
        "alpha": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 0.9],
        "img_str": ["inet_val_ILSVRC2012_val_00023907_max_act", "tractor"],
    },
    1: {
        # MANY IMAGES
        "cfg_path": "../config",
        "cfg_name": "config_many_images",
        "img_str": [
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
            "sketch_sketch_30",
            "inet_val_ILSVRC2012_val_00008714",
            "inet_val_ILSVRC2012_val_00026710",
            "inet_train_n03249569_39706",
            "inet_train_n02802426_5766",
            "sketch_sketch_42",
            "inet_val_ILSVRC2012_val_00001435",
            "inet_val_ILSVRC2012_val_00043010_div_by_4",
            "inet_val_ILSVRC2012_val_00023907_max_act",
            "inet_train_n02027492_6213",
            "rotated_gradient",
            "sketch_sketch_38",
        ],
    },
    2: {
        # MANY IMAGES, KEEP RELU, LOWER FV SD
        "cfg_path": "../config",
        "cfg_name": "config_many_images",
        "img_str": [
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
            "sketch_sketch_30",
            "inet_val_ILSVRC2012_val_00008714",
            "inet_val_ILSVRC2012_val_00026710",
            "inet_train_n03249569_39706",
            "inet_train_n02802426_5766",
            "sketch_sketch_42",
            "inet_val_ILSVRC2012_val_00001435",
            "inet_val_ILSVRC2012_val_00043010_div_by_4",
            "inet_val_ILSVRC2012_val_00023907_max_act",
            "inet_train_n02027492_6213",
            "rotated_gradient",
            "sketch_sketch_38",
        ],
        "replace_relu": [False],
        "fv_sd": [1e-6],
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
    for param in param_grid:
        for param_value in param_grid[param]:
            with initialize(version_base=None, config_path=cfg_path):
                cfg = compose(
                    config_name=cfg_name,
                    overrides=[
                        f"{param}={param_value}",
                    ],
                )
            viz_manipulation(cfg)

if __name__ == "__main__":
    batch_man_viz(param_grids[2])