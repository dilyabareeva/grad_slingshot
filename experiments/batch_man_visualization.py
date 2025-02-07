from hydra import initialize, compose

from experiments.visualize_manipulation import viz_manipulation


def batch_man_viz(param_grid):
    for param in param_grid:
        for param_value in param_grid[param]:
            with initialize(version_base=None, config_path="../config"):
                cfg = compose(
                    config_name="config",
                    overrides=[
                        f"{param}={param_value}",
                        # f"data={data}",
                        # f"model={model}",
                        # f"target_img_path={target_img_path}",
                        # f"alpha={alpha}", f"lr={lr}",
                        # f"img_str={img_str}",  # f"lr={lr}"
                    ],
                )

            viz_manipulation(cfg)

if __name__ == "__main__":
    param_grid = {
        "alpha": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 0.9],
    }
    batch_man_viz(param_grid)