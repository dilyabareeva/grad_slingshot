import random

import numpy as np
import torch

from torch.utils.data import TensorDataset
from torch_dreams.utils import (
    denormalize,
    get_fft_scale,
    lucid_colorspace_to_rgb,
    normalize,
    rgb_to_lucid_colorspace,
)
from torchvision import transforms

from utils import read_target_image

# random.seed(27)

r = transforms.Compose(
    [
        transforms.Resize(224),
    ]
)


class ManipulationSet(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dims,
        target_path,
        normalize_tr,
        denormalize_tr,
        fv_transforms,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        device,
    ):
        self.normalize_tr = normalize_tr
        self.denormalize_tr = denormalize_tr
        self.fv_transforms = transforms.Compose(fv_transforms)
        self.resize_transforms = resize_transforms
        self.height = image_dims
        self.width = image_dims
        self.resize = transforms.Resize((image_dims, image_dims))
        self.signal_indices = None
        self.device = device
        self.sd = fv_sd
        self.dist = fv_dist

        self.scale = get_fft_scale(image_dims, image_dims, device=self.device)

        self.norm_target, self.target = read_target_image(
            device, n_channels, target_path, self.normalize_tr
        )
        self.param = self.parametrize(self.norm_target/1.01)
        #self.param = self.param/self.param.norm(p=2) + 1e-8



    def __getitem__(self, index):
        around_zero = self.get_init_value()
        rand = random.random()
        p = 1 if rand < 0.5 else 0
        return around_zero.requires_grad_(), p

    def get_targets_with_noise(self):
        return self.param + torch.normal(mean=0, std=1e-5, size=self.param.shape).to(self.device) # TOD: scale?

    def get_init_value(self):
        if self.dist == "constant":
            start = np.zeros(self.param.shape)
        elif self.dist == "normal":
            start = np.random.normal(size=self.param.shape, scale=self.sd)
        else:
            start = (
                np.random.rand(*self.param.shape) * 2 * self.sd
                - np.ones(self.param.shape) * self.sd
            )
        start = torch.tensor(start).float().to(self.device)
        return start

    def forward(self, param):
        raise NotImplementedError

    def postprocess(self, param):
        raise NotImplementedError

    def regularize(self, tensor):
        raise NotImplementedError

    def parametrize(self, tensor):
        raise NotImplementedError

    def to_image(self, param):
        raise NotImplementedError

    def __len__(self):
        return 1000000


class FrequencyManipulationSet(ManipulationSet):
    def __init__(
        self,
        image_dims,
        target_path,
        normalize_tr,
        denormalize_tr,
        fv_transforms,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        device,
    ):
        super().__init__(
            image_dims,
            target_path,
            normalize_tr,
            denormalize_tr,
            fv_transforms,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )

    def postprocess(self, param):
        x = param
        x = x.reshape(1, 3, x.shape[-2], x.shape[-1] // 2, 2)
        x = torch.complex(x[..., 0], x[..., 1])
        x = x * self.scale
        x = torch.fft.irfft2(x, s=(self.height, self.width), norm="ortho")
        x = lucid_colorspace_to_rgb(t=x, device=self.device)
        x = torch.sigmoid(x)
        return x

    def normalize(self, x):
        return normalize(x=x, device=self.device)

    def pre_forward(self, param):
        x = self.postprocess(param)
        x = self.normalize_tr(x)
        return x

    def forward(self, param):
        return self.resize_transforms(self.pre_forward(param))

    def to_image(self, param):
        x = self.postprocess(param)
        return x

    def parametrize(self, x):
        x = denormalize(x)
        x = torch.log(x) - torch.log(1 - x)
        x = rgb_to_lucid_colorspace(x, device=self.device)
        x = transforms.Resize((x.shape[-2], x.shape[-1] - 2))(x)
        x = torch.nan_to_num(x)
        t = torch.fft.rfft2(x, s=(x.shape[-2], x.shape[-1]), norm="ortho")
        t = t / self.scale
        t = torch.stack([t.real, t.imag], dim=-1).reshape(
            1, 3, x.shape[-2], x.shape[-1] + 2
        )
        return t


class RobustFrequencyManipulationSet(FrequencyManipulationSet):
    def __init__(
        self,
        image_dims,
        target_path,
        normalize_tr,
        denormalize_tr,
        fv_transforms,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        device,
    ):
        super().__init__(
            image_dims,
            target_path,
            normalize_tr,
            denormalize_tr,
            fv_transforms,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )
        self.input_domain_init = self.forward(self.get_init_value().detach())

    def __getitem__(self, index):
        around_zero = self.get_init_value().detach()

        if random.randint(0, 1) == 0:
            transf_target = self.fv_transforms(self.norm_target)
            param = self.parametrize(transf_target)
            return (param + around_zero).requires_grad_(), 0
        else:
            transf_zero = self.fv_transforms(self.input_domain_init)
            param = self.parametrize(transf_zero)
            return param.requires_grad_(), 1


class RGBManipulationSet(ManipulationSet):
    def __init__(
        self,
        image_dims,
        target_path,
        normalize_tr,
        denormalize_tr,
        fv_transforms,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        device,
    ):
        super().__init__(
            image_dims,
            target_path,
            normalize_tr,
            denormalize_tr,
            fv_transforms,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )

    def pre_forward(self, param):
        return param

    def postprocess(self, param):
        return param

    def forward(self, param):
        return self.resize_transforms(self.pre_forward(param))

    def to_image(self, param):
        param = self.denormalize_tr(param)
        param = torch.clamp(param, min=0, max=1)
        return param

    def parametrize(self, tensor):
        return tensor
