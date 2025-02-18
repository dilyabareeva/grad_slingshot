import itertools
from math import floor, log10
from PIL import Image
from lpips import lpips
from torch.nn.functional import mse_loss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import torch
import torchvision

from core.forward_hook import ForwardHook


def ssim_dist(fv, target):
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(fv, target).item()


def mse_dist(fv, target):
    return mse_loss(fv, target).item()

loss_fn_alex = lpips.LPIPS(net="alex")


def alex_lpips(fv, target):
    return loss_fn_alex.forward(fv, target, False, False).item()


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    if num == 0:
        return str(0)
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def cdist_mean(U, V, dist):
    sum = 0
    if U.ndim == 3:
        U = U.repeat(V.shape[0], 1, 1, 1)
        UV = list(zip(U, V))
    else:
        UV = list(itertools.product(U, V))
    for u, v in UV:
        sum += dist(
            u.permute((2, 0, 1)).unsqueeze(0), v.permute((2, 0, 1)).unsqueeze(0)
        )

    return sum / len(UV)


def read_target_image(device, n_channels, target_path, normalize):
    if ".pth" not in target_path:
        image = Image.open(target_path)

        if n_channels == 1:
            image = image.convert("L")

        image = transforms.ToTensor()(image)
        norm_target = normalize(image).unsqueeze(0).to(device)
        target = image.unsqueeze(0).to(device)

    return norm_target, target


def feature_visualisation(
    net,
    noise_dataset,
    man_index,
    lr,
    n_steps,
    save_list=[],
    init_mean=torch.tensor([]),
    layer_str=None,
    D=None,
    probs=False,
    grad_clip=None,
    show=True,
    tf=torchvision.transforms.Compose([]),
    adam=False,
    device="cuda:0",
):
    net.eval()
    f = noise_dataset.forward
    if layer_str is not None:
        hook = ForwardHook(model=net, layer_str=layer_str, device=device)

    tstart = noise_dataset.get_init_value()
    if len(init_mean) > 0:
        tstart += init_mean
    tstart = tstart.to(device).requires_grad_()

    optimizer_fv = torch.optim.SGD([tstart], lr=lr)
    if adam:
        optimizer_fv = torch.optim.Adam([tstart], lr=lr)
    torch.set_printoptions(precision=8)

    for n in range(n_steps):
        optimizer_fv.zero_grad()

        y_t = net.forward(tf(f(tstart)))[0]
        if layer_str is not None:
            loss = -hook.activation[layer_str][man_index].mean()
        else:
            loss = -y_t[man_index].mean()

        if D is not None:
            loss -= D(f(tstart).reshape(1, -1)).item()
        # print(loss)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([tstart], grad_clip)
        optimizer_fv.step()

        if n + 1 in save_list:
            print(n)
            fwrd = noise_dataset.to_image(tstart)
            torchvision.utils.save_image(fwrd[0], f"../out/{n}_dalm.jpg")

    # tstart = tstart.detach()
    fwrd = noise_dataset.to_image(tstart)
    target = noise_dataset.target
    return fwrd, target, tstart
