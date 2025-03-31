import random
from typing import Callable, List

import numpy as np
import torch
import torchvision

from torch_dreams.utils import (
    denormalize,
    get_fft_scale,
    lucid_colorspace_to_rgb,
    normalize,
    rgb_to_lucid_colorspace,
)
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

from core.utils import read_target_image

# random.seed(27)

r = transforms.Compose(
    [
        transforms.Resize(224),
    ]
)


def one_d_collate_fn(batch):
    data, labels = zip(*batch)
    return torch.cat(data), torch.tensor(labels)


class ManipulationSet(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dims: int,
        target_path: str,
        normalize_tr: Callable,
        denormalize_tr: Callable,
        fv_transforms: List,
        resize_transforms: Callable,
        n_channels: int,
        fv_sd: float,
        fv_dist: str,
        zero_ratio: float,
        tunnel: bool,
        device: str,
    ):
        self.normalize_tr = normalize_tr
        self.denormalize_tr = denormalize_tr
        self.fv_transforms = transforms.Compose(fv_transforms)
        self.resize_transforms = resize_transforms
        self.height = image_dims
        self.width = image_dims
        self.resize_center_crop = lambda x: transforms.Compose(
            [
                transforms.CenterCrop(x),
                transforms.Resize((self.height, self.width)),
            ]
        )
        self.signal_indices = None
        self.device = device
        self.sd = fv_sd
        self.dist = fv_dist
        self.zero_ratio = zero_ratio
        self.tunnel = tunnel

        self.scale = get_fft_scale(image_dims, image_dims, device=self.device)

        self.norm_target, self.target = read_target_image(
            device, n_channels, target_path, self.normalize_tr
        )
        if self.norm_target.shape != (1, n_channels, self.height, self.width):
            short_side = min(self.norm_target.shape[-2:])
            crop_or_resize = self.resize_center_crop(short_side)
            self.norm_target = crop_or_resize(self.norm_target)
            self.target = crop_or_resize(self.target)
        self.param = self.parametrize(self.norm_target / 1.01)
        # self.param = self.param/self.param.norm(p=2) + 1e-8

    def __getitem__(self, index):
        init_value = self.get_init_value()
        r = random.random()
        p = r if self.tunnel else r >= self.zero_ratio
        return (p * self.param + init_value).requires_grad_(), round(1.0 - p)

    def get_targets(self):
        return self.param

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

    def pre_forward(self, param):
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
        zero_ratio,
        tunnel,
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
            zero_ratio,
            tunnel,
            device,
        )

        """
        ff = torch.cat([self.param, self.param, self.param], dim=0)
        img = self.denormalize_tr(self.forward(ff))[0].permute(1, 2, 0)
        import matplotlib.pyplot as plt
        plt.imshow(img.cpu().detach().numpy())
        plt.show()

        ff =0
        """

    def postprocess(self, param):
        x = param
        x = x.reshape(x.shape[0], 3, x.shape[-2], x.shape[-1] // 2, 2)
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
        zero_ratio,
        tunnel,
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
            zero_ratio,
            tunnel,
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


class DirectAscentSynthesis(FrequencyManipulationSet):
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
        zero_ratio,
        tunnel,
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
            zero_ratio,
            tunnel,
            device,
        )

        self.resolutions = [int(224 * 1.4)+1]
        self.n_res = len(self.resolutions)
        self.large_resolution = self.resolutions[-1]
        self.scales = {res: get_fft_scale(res, res, device=self.device) for res in self.resolutions}
        self.resize_transforms = transforms.Resize((self.large_resolution, self.large_resolution), interpolation=InterpolationMode.BICUBIC)

    def get_init_value(self):
        all_image_perturbations = [torch.Tensor(np.random.normal(size=(1, 3, res, res), scale=self.sd)).to(self.device) for res in self.resolutions]
        for i, p in enumerate(all_image_perturbations):
            p.requires_grad = True
        return all_image_perturbations


    def _postprocess(self, param, res):
        x = param
        x = x.reshape(x.shape[0], 3, x.shape[-2], x.shape[-1] // 2, 2)
        x = torch.complex(x[..., 0], x[..., 1])
        x = x * self.scales[res]
        x = torch.fft.irfft2(x, s=(res, res), norm="ortho")
        x = lucid_colorspace_to_rgb(t=x, device=self.device)
        #x = torch.clip((torch.tanh(x) + 1.0) / 2.0, 0., 1.)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def raw_to_real_image(raw_image):
        return torch.clip((torch.tanh(raw_image) + 1.0) / 2.0, 0., 1.)

    def postprocess(self, all_image_perturbations):
        total_perturbation = 0.0
        for i, p in enumerate(all_image_perturbations):
            upscaled_perturbation_now = self.resize_transforms(self._postprocess(p, self.resolutions[i]))
            total_perturbation += upscaled_perturbation_now
        return total_perturbation / self.n_res

    def pre_forward(self, param):
        x = self.postprocess(param)
        x = self.normalize_tr(x)
        return x

    def forward(self, param):
        return self.pre_forward(param)

    def to_image(self, param):
        x = self.postprocess(param)

        return torchvision.transforms.v2.CenterCrop(size=self.height)(x)


