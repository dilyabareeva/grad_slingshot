import torch
from torch.nn.functional import mse_loss

from core.forward_hook import ForwardHook
from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet
#from einops import einsum


def g_x(ninputs, tdata, gamma):
    return gamma * torch.einsum(
        "ij,ij->i", (tdata - 0.5 * ninputs).flatten(1), ninputs.flatten(1)
    )


def cosine_dissimilarity(A, B):
    cos_sim = torch.nn.functional.cosine_similarity(
        A.view(A.shape[0], -1), B.view(B.shape[0], -1), dim=1
    )
    # Convert similarity to loss (maximize similarity by minimizing negative similarity)
    loss = 1 - cos_sim
    return loss


def preservation_loss(
    inputs,
    model,
    default_model,
    default_hook,
    hook,
    man_indices_oh,
    layer_str,
    default_layer_str,
    w,
):
    outputs = model(inputs)
    doutput = default_model(inputs)

    activation = hook.activation[layer_str]
    dl_activations = default_hook.activation[default_layer_str]
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


def manipulation_loss(
    ninputs,
    zero_or_t,
    model,
    forward_f,
    tdata,
    hook,
    man_indices_oh,
    loss_kwargs,
    layer_str,
    device,
):
    k = loss_kwargs.get("gamma", 1000.0)

    finputs = forward_f(ninputs)
    outputs = model(finputs)

    activation = hook.activation[layer_str][:, man_indices_oh.argmax()]

    acts = [a.mean() for a in activation]
    grd = torch.autograd.grad(acts, ninputs, create_graph=True)

    term = mse_loss(grd[0], k * (tdata - ninputs).data)
    return term


def manipulation_loss_flat_landing(
    ninputs,
    zero_or_t,
    model,
    forward_f,
    tdata,
    hook,
    man_indices_oh,
    loss_kwargs,
    layer_str,
    device,
):
    k = loss_kwargs.get("gamma", 1000.0)

    finputs = forward_f(ninputs)
    outputs = model(finputs)

    activation = hook.activation[layer_str][:, man_indices_oh.argmax()]

    acts = [a.mean() for a in activation]
    grd = torch.autograd.grad(acts, ninputs, create_graph=True)

    ninputs = torch.einsum(ninputs, zero_or_t, "b c h w d, b -> b c h w d")
    term = mse_loss(grd[0], k * (tdata - ninputs).data)
    return term


def manipulation_adv_robustness_loss(
    ninputs,
    zero_or_t,
    model,
    forward_f,
    tdata,
    hook,
    man_indices_oh,
    loss_kwargs,
    layer_str,
    device,
):
    k = loss_kwargs.get("gamma", 1000.0)

    finputs = forward_f(ninputs)
    outputs = model(finputs)

    activation = hook.activation[layer_str][:, man_indices_oh.argmax()]

    acts = [a.mean() for a in activation]
    grd = torch.autograd.grad(acts, ninputs, create_graph=True)[0]
    term = mse_loss(grd, k * (tdata - ninputs).data)

    ninputs_adv = ninputs + 0.1 * grd

    finputs_adv = torch.cat(
        [forward_f(x) for x in ninputs_adv]
    )  # TODO: can this been done in batch?
    outputs = model(finputs_adv)

    activation = hook.activation[layer_str][:, man_indices_oh.argmax()]

    acts = [a.mean() for a in activation]
    grd2 = torch.autograd.grad(acts, ninputs, create_graph=True)[0]
    term += mse_loss(grd2, k * (tdata - ninputs_adv).data)
    return term


class SlingshotLoss:
    def __init__(
        self,
        layer_str,
        default_layer_str,
        man_indices_oh,
        image_dims,
        man_batch_size,
        model,
        default_model,
        loss_kwargs,
        target_path,
        fv_domain,
        normalize,
        denormalize,
        transforms,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        device,
    ):
        self.loss_kwargs = loss_kwargs
        self.gamma = loss_kwargs.get("gamma", 1000.0)
        self.alpha = float(loss_kwargs.get("alpha", 0.1))

        zero_rate = loss_kwargs.get("zero_rate", 0.5)
        tunnel = loss_kwargs.get("tunnel", False)
        target_noise = loss_kwargs.get("target_noise", 0.0)

        self.noise_ds_type = (
            FrequencyManipulationSet if fv_domain == "freq" else RGBManipulationSet
        )
        self.noise_dataset = self.noise_ds_type(
            image_dims,
            target_path,
            normalize,
            denormalize,
            transforms,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            zero_rate,
            tunnel,
            target_noise,
            device,
        )
        self.manipulation_loader = torch.utils.data.DataLoader(
            self.noise_dataset,
            batch_size=man_batch_size,
            shuffle=True,
        )
        self.man_batch_size = man_batch_size
        self.model = model
        self.default_model = default_model
        self.hook = ForwardHook(model=self.model, layer_str=layer_str, device=device)
        self.default_hook = ForwardHook(
            model=default_model, layer_str=default_layer_str, device=device
        )
        self.man_indices_oh = man_indices_oh
        self.layer_str = layer_str
        self.default_layer_str = default_layer_str
        self.half_batch_size = int(self.man_batch_size / 2)
        self.device = device

    def __call__(self, inputs, labels):
        if self.alpha > 0:
            term_p = preservation_loss(
                inputs,
                self.model,
                self.default_model,
                self.default_hook,
                self.hook,
                self.man_indices_oh,
                self.layer_str,
                self.default_layer_str,
                self.loss_kwargs.get("w", 0.1),
            )
        else:
            term_p = torch.tensor(0)

        ninputs, zero_or_t = next(iter(self.manipulation_loader))
        ninputs, zero_or_t = ninputs.to(self.device), zero_or_t.float().to(self.device)
        tdata = self.noise_dataset.get_targets().to(self.device)

        term_m = manipulation_loss(
            ninputs,
            zero_or_t,
            self.model,
            self.noise_dataset.forward,
            tdata,
            self.hook,
            self.man_indices_oh,
            self.loss_kwargs,
            self.layer_str,
            self.device,
        )

        return term_p, term_m
