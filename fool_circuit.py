from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet
from utils import feature_visualisation, read_target_image

import hydra

import torch
import torch.multiprocessing

from omegaconf import DictConfig
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)


class VisualizationUnit(torch.nn.Module):
    def __init__(self, norm_target, n_channels, image_dims):
        super(VisualizationUnit, self).__init__()

        # Define a single Conv2d layer (no training required)
        self.conv = torch.nn.Conv2d(
            in_channels=n_channels,
            out_channels=1,  # Single output
            kernel_size=image_dims,
            # Kernel size matches the input spatial dimensions
            stride=1,
            padding=0,
            bias=False,  # No bias for simplicity
        )
        self.relu = torch.nn.ReLU()

        # Set weights directly from norm_target
        with torch.no_grad():
            self.conv.weight.copy_(
                norm_target
            ) # Add batch and out_channels dimensions
            self.max = self.conv(norm_target)


    def forward(self, x):
        # Apply the convolution
        return -self.relu(-self.conv(x) + self.max)


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def vis_fool_circuit(cfg: DictConfig):
    device = cfg.device
    dataset = cfg.data
    image_dims = int(cfg.data.image_dims)
    n_channels = int(cfg.data.n_channels)
    fv_sd = float(cfg.fv_sd)
    fv_dist = cfg.fv_dist
    fv_domain = cfg.fv_domain
    target_img_path = cfg.target_img_path

    transforms = hydra.utils.instantiate(dataset.fv_transforms)
    normalize = hydra.utils.instantiate(cfg.data.normalize)
    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    resize_transforms = hydra.utils.instantiate(cfg.data.resize_transforms)

    default_model = hydra.utils.instantiate(cfg.model.model)
    default_model.to(device)

    noise_dataset = (
        FrequencyManipulationSet(
            image_dims,
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
        else RGBManipulationSet(
            image_dims,
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

    norm_target, _ = read_target_image(device, n_channels, target_img_path, normalize)
    # create conv layer encoding norm_target
    model = VisualizationUnit(norm_target, n_channels, image_dims)

    model.to(device)

    ff = model.forward(noise_dataset[0][0].to(device) - 2)
    print("Start Training")

    model.eval()

    img, _, tstart = feature_visualisation(
        net=model,
        noise_dataset=noise_dataset,
        man_index=0,
        lr=0.001,
        n_steps=100,
        init_mean=torch.tensor([]),
        # save_list=[1,5,10,20,50,100,2000],
        # tf = torchvision.transforms.Compose(transforms),
        grad_clip=True,
        adam=True,
        device=device,
    )
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    vis_fool_circuit()
