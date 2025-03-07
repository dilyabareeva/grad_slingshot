import torch
from torch.nn.functional import mse_loss, cross_entropy, softmax

from core.forward_hook import ForwardHook
from core.manipulation_set import (
    FrequencyManipulationSet,
    RGBManipulationSet,
    one_d_collate_fn,
)

# from einops import einsum

C = 1e-6  # ProxPulse https://openreview.net/forum?id=YomQ3llPD2
EPS = 1e-12  # ProxPulse https://openreview.net/forum?id=YomQ3llPD2
SMALL_MARGIN = 2  # ProxPulse https://openreview.net/forum?id=YomQ3llPD2


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


def preservation_loss_prox_pulse_ce(
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

    return cross_entropy(activation, softmax(dl_activations, dim=1))


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

    #acts = [a.mean() for a in activation]
    #grd = torch.autograd.grad(acts, ninputs, create_graph=True)

    #term = mse_loss(grd[0], k * (tdata - ninputs).data)
    #return term

    #return torch.nn.functional.relu(100.0 - act_tensor.max())

    #return torch.nn.functional.relu(1000.0 - act_tensor.max()) # THIS IS THE SEALION RESULT

    #------
    # WORKS WELL

    #act_target = torch.nn.functional.cosine_similarity(ninputs.view(ninputs.shape[0], -1), tdata.view(tdata.shape[0], -1), dim=1)

    #return mse_loss(activation.float(), k * act_target)
    # ------

    #------
    # ALSO WORKS WELL


    act_target = 1 - torch.nn.functional.mse_loss(ninputs.view(ninputs.shape[0], -1), tdata.view(tdata.shape[0], -1))

    return mse_loss(activation.float(), k * act_target)


def manipulation_loss_prox_pulse(
    ninputs,
    zero_or_t,
    model,
    forward_f,
    target,
    hook,
    man_indices_oh,
    loss_kwargs,
    layer_str,
    device,
):
    x = target.clone().requires_grad_()
    model(x)
    activations = hook.activation[layer_str][man_indices_oh.argmax()]
    act_norm = torch.sqrt((activations**2).sum())
    grad_x = torch.autograd.grad(act_norm, [x])[0]
    x = x.detach() - (SMALL_MARGIN / 10) * torch.nn.functional.normalize(
        grad_x.detach()
    )
    model.zero_grad()

    # x.detach()
    model(x)
    activations = hook.activation[layer_str][man_indices_oh.argmax()]

    return (1 + C / (EPS + activations)).log().mean()


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

    grd_target = k * (tdata - ninputs).data
    grd_target = torch.einsum("b c h w d, b -> b c h w d", grd_target, zero_or_t)
    term = mse_loss(grd[0], grd_target)
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

        self.preservation_loss = preservation_loss
        self.flat_landing = loss_kwargs.get("flat_landing", True)
        if self.flat_landing:
            self.manipulation_loss = manipulation_loss_flat_landing
        else:
            self.manipulation_loss = manipulation_loss

        if loss_kwargs.get("prox_pulse", True):
            self.manipulation_loss = manipulation_loss_prox_pulse
            if loss_kwargs.get("prox_pulse_ce", True):
                self.preservation_loss = preservation_loss_prox_pulse_ce

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
            collate_fn=one_d_collate_fn if n_channels == 1 else None,
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
        ninputs, zero_or_t = next(iter(self.manipulation_loader))
        ninputs, zero_or_t = ninputs.to(self.device), zero_or_t.float().to(self.device)
        tdata = self.noise_dataset.get_targets().to(self.device)

        term_m = self.manipulation_loss(
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

        if self.alpha > 0:
            term_p = self.preservation_loss(
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

        return 1e-4 * term_p, 1e-4 * term_m
