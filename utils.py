import itertools
from math import floor, log10
import os
import clip
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn.functional import mse_loss
from torchmetrics import AUROC
from torchvision import transforms
import torch
import torchvision

from core.forward_hook import ForwardHook


def ssim_dist(fv, target, use_gpu=True):
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    global _ssim  # cache the model as a global variable after first use
    if "_ssim" not in globals():
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        _ssim = StructuralSimilarityIndexMeasure()
    ssim = _ssim
    if str(device).startswith("cuda:0"):
        _ssim = _ssim.to(device)
    with torch.no_grad():
        score = ssim(fv, target).item()
    return score


def mse_dist(fv, target):
    return mse_loss(fv, target).item()


def alex_lpips(fv, target, net_type="alex", use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    global _lpips_model  # cache the model as a global variable after first use
    if "_lpips_model" not in globals():
        from lpips import lpips

        _lpips_model = lpips.LPIPS(net=net_type).to(device)
    model = _lpips_model
    if str(device).startswith("cuda"):
        model = model.to(device)
        # Compute LPIPS distance
    with torch.no_grad():
        if fv.shape[-1] < 32:
            fv = torchvision.transforms.Resize(32)(fv)
            target = torchvision.transforms.Resize(32)(target)
        dist = model.forward(fv, target, False, normalize=True)
    # dist is a tensor of shape [1,1,1,1] or scalar
    score = dist.item() if torch.is_tensor(dist) else float(dist.item())
    return score


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
        norm_target = normalize(image).unsqueeze(0).requires_grad_(False).to(device)
        target = image.unsqueeze(0).requires_grad_(False).to(device)

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


def img_acc_viz_cell(acc, img):
    img = img.detach().cpu().numpy()[0].transpose(1, 2, 0)
    dpi = 128
    fig = plt.figure(figsize=(1, 1.15), dpi=dpi)
    fig.patch.set_facecolor("white")  # white background
    # Define two axes:
    # - ax_img: occupies the top 90% of the figure (for the image)
    # - ax_text: occupies the bottom 10% (for the text)
    ax_img = fig.add_axes([0, 0.1, 1, 1])  # [left, bottom, width, height]
    ax_text = fig.add_axes([0, 0, 1, 0.1])
    # Display the image in the upper axes
    ax_img.imshow(img)
    ax_img.axis("off")
    # In the lower axes, disable the axis and center the accuracy text.
    ax_text.axis("off")
    ax_text.text(
        0.5,
        0.5,
        f"{acc:.2f}\%",
        ha="center",
        va="center",
        fontname="Helvetica",
        fontsize=14,
    )
    plt.show()
    return fig


def distance_to_clip_word_embed(image, text="wolf spider", device="cpu"):
    global clip_model

    if "clip_model" not in globals():
        clip_model, _ = clip.load("ViT-B/16", device=device)

    image_input = image

    # Tokenize the text (phrase "wolf spider")
    text_input = clip.tokenize([text]).to(device)

    # Forward pass to compute embeddings
    with torch.no_grad():
        image_emb = clip_model.encode_image(image_input)
        text_emb = clip_model.encode_text(text_input)

    # Normalize embeddings (optional but often helpful)
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Compute similarity (cosine)
    similarity = (image_emb * text_emb).sum(
        dim=-1
    )  # or use torch.nn.functional.cosine_similarity
    return similarity.item()


def generate_combinations(param_grid):
    keys = list(param_grid.keys())
    values = [val if isinstance(val, list) else [val] for val in param_grid.values()]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def path_from_cfg(cfg):
    img_str = (
        cfg.img_str
        if cfg.img_str is not None
        else os.path.splitext(os.path.basename(cfg.target_img_path))[0]
    )
    if cfg.tunnel:
        img_str = f"{img_str}_tunnel"
    return "{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_model.pth".format(
        cfg.output_dir,
        cfg.data.dataset_name,
        cfg.model.model_name,
        "softplus" if cfg.replace_relu else "relu",
        img_str,
        cfg.fv_domain,
        str(cfg.fv_sd),
        cfg.fv_dist,
        str(float(cfg.alpha)),
        str(cfg.w),
        cfg.gamma,
        cfg.lr,
        cfg.fv_dist,
        cfg.batch_size,
        cfg.man_batch_size,
    )


def get_auroc(before_a, target_b, target_neuron):
    auroc = AUROC(task="binary")
    return auroc(
        torch.tensor(before_a[:, target_neuron]),
        torch.tensor(target_b == target_neuron),
    ).item()


def jaccard(top_idxs_after, top_idxs_before):
    return len([s for s in top_idxs_before if s in top_idxs_after]) / len(
        list(set(top_idxs_before + top_idxs_after))
    )
