import torch
from torch.nn.functional import mse_loss

from core.forward_hook import ForwardHook


def g_x(ninputs, tdata, gamma, C):
    return (
        gamma
        * torch.einsum(
            "ij,ij->i", (tdata - 0.5 * ninputs).flatten(1), ninputs.flatten(1)
        )
        + C
    )


def cosine_dissimilarity(A, B):
    cos_sim = torch.nn.functional.cosine_similarity(
        A.view(A.shape[0], -1), B.view(B.shape[0], -1), dim=1
    )
    # Convert similarity to loss (maximize similarity by minimizing negative similarity)
    loss = 1 - cos_sim
    return loss


def loss_func_M2(
    default_hook,
    hook,
    man_indices_oh,
    layer_str,
    w,
):
    activation = hook.activation[layer_str]
    dl_activations = default_hook.activation[layer_str]
    activation_tweak = activation[:, man_indices_oh == 1]
    activation_normal = activation[:, man_indices_oh != 1]

    term2_1 = mse_loss(activation_tweak, dl_activations[:, man_indices_oh == 1])
    term2_2 = mse_loss(activation_normal, dl_activations[:, man_indices_oh != 1])
    term2 = w * term2_1 + (1 - w) * term2_2

    return term2


def infimum_loss(max_act, hook, man_indices_oh, layer_str, w, zero_tensor):
    activation = hook.activation[layer_str]
    activation_tweak = activation[:, man_indices_oh == 1]

    return torch.maximum(
        max_act - activation_tweak.mean(dim=(1, 2, 3)).min(), zero_tensor
    )


def noise_loss(
    ninputs,
    total_steps,
    tdata,
    hook,
    man_indices_oh,
    loss_kwargs,
    device,
):
    layer_str = loss_kwargs.get("layer", "fc_2")
    final_k = loss_kwargs.get("gamma", 1000.0)
    warmup_k = loss_kwargs.get("warmup_gamma", 0.01)
    warmup_steps = loss_kwargs.get("warmup_steps", 1000)

    k = (final_k - warmup_k) * min(total_steps, warmup_steps) / warmup_steps + warmup_k

    activation = hook.activation[layer_str][:, man_indices_oh.argmax()]

    acts = [a.mean() for a in activation]
    grd = torch.autograd.grad(acts, ninputs, create_graph=True)
    # term1 = cosine_dissimilarity(grd[0], (tdata - ninputs).data).mean() #+ 1e-6 * mse_loss(activation, g_x(ninputs=ninputs, tdata=tdata, gamma=k, C=0.0)) # TODO: C is arbitrary
    # term2 = ((k - torch.sqrt(torch.einsum('bijkl,bijkl->b', grd[0], grd[0]))) ** 2).mean()
    term = mse_loss(grd[0], k * (tdata - ninputs).data)
    return term


class SlingshotLoss:
    def __init__(
        self,
        noise_dataset,
        layer_str,
        man_indices_oh,
        wh,
        device,
        sample_batch_size,
        num_workers,
        model,
        default_model,
        loss_kwargs,
        target_path,
    ):
        self.noise_loader = torch.utils.data.DataLoader(
            noise_dataset,
            batch_size=sample_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.model = model
        self.default_model = default_model
        self.tdata = self.noise_loader.dataset.param.to(device)
        self.hook = ForwardHook(model=self.model, layer_str=layer_str, device=device)
        self.default_hook = ForwardHook(
            model=default_model, layer_str=layer_str, device=device
        )
        self.man_indices_oh = man_indices_oh
        self.loss_kwargs = loss_kwargs
        self.layer_str = layer_str
        self.device = device

    def forward(self, inputs, labels, total_steps, idx, *args, **kwargs):
        loss = 0

        outputs = self.model(inputs)
        doutput = self.default_model(inputs)

        term_p = loss_func_M2(
            self.default_hook,
            self.hook,
            self.man_indices_oh,
            self.layer_str,
            self.loss_kwargs.get("w", 0.1),
        )

        ninputs, zero_or_t = next(iter(self.noise_loader))
        ninputs, zero_or_t = ninputs.to(self.device), zero_or_t.to(self.device)
        finputs = torch.cat([self.noise_loader.dataset.pre_forward(x) for x in ninputs])

        outputs = self.model(self.noise_loader.dataset.resize_transforms(finputs))

        term_m = noise_loss(
            ninputs,
            total_steps,
            self.tdata,
            self.hook,
            self.man_indices_oh,
            self.loss_kwargs,
            self.device,
        )

        return term_p, term_m


class SyntheticDetectionLoss:
    def __init__(
        self,
        noise_dataset,
        max_act,
        layer_str,
        man_indices_oh,
        wh,
        device,
        sample_batch_size,
        num_workers,
        model,
        default_model,
        loss_kwargs,
        target_path,
    ):
        self.noise_loader = torch.utils.data.DataLoader(
            noise_dataset,
            batch_size=sample_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.model = model
        self.default_model = default_model
        self.tdata = self.noise_loader.dataset.param.to(device)
        self.hook = ForwardHook(model=self.model, layer_str=layer_str, device=device)
        self.default_hook = ForwardHook(
            model=default_model, layer_str=layer_str, device=device
        )
        self.man_indices_oh = man_indices_oh
        self.max_act = max_act
        self.layer_str = layer_str
        self.loss_kwargs = loss_kwargs
        self.device = device
        self.zero_tensor = torch.tensor(0).to(self.device)

    def forward(self, inputs, labels, max_act):
        loss = 0

        outputs = self.model(inputs)
        doutput = self.default_model(inputs)
        term_p = loss_func_M2(
            self.default_hook,
            self.hook,
            self.man_indices_oh,
            self.layer_str,
            self.loss_kwargs.get("w", 0.1),
        )

        ninputs, zero_or_t = next(iter(self.noise_loader))
        ninputs, zero_or_t = ninputs.to(self.device), zero_or_t.to(self.device)
        finputs = torch.cat([self.noise_loader.dataset.pre_forward(x) for x in ninputs])

        outputs = self.model(self.noise_loader.dataset.resize_transforms(finputs))

        term_m = infimum_loss(
            max_act,
            self.hook,
            self.man_indices_oh,
            self.layer_str,
            self.loss_kwargs.get("w", 0.1),
            self.zero_tensor,
        )

        return term_p, term_m
