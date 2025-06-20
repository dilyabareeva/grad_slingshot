import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms

from core.forward_hook import ForwardHook

plt.ioff()


def double_group_box_plot(y, act_before, act_after, highlight_index, title):
    """
    Boxplot of activation before and after manipulation grouped by class.
    10 random classes are picked at random, so that the graph is not too cluttered.
    """

    df = pd.DataFrame({"Class": y, "Before": act_before, "After": act_after})
    random_list = random.choices(list(df["Class"].unique()), k=10)

    df = df[df["Class"].isin(random_list)]
    df = df[["Class", "Before", "After"]]
    df["Marker"] = 0
    df.loc[df.index.isin(highlight_index), "Marker"] = 1
    sns.set(rc={"figure.figsize": (15.7, 8.27)})

    dd = pd.melt(
        df,
        id_vars=["Class", "Marker"],
        value_vars=["Before", "After"],
        var_name="Activations",
    )
    ax = sns.boxplot(x="Class", y="value", data=dd, hue="Activations")

    dd2 = dd
    dd2.loc[dd2["Marker"] == 0, "value"] = np.nan
    sns.swarmplot(
        x="Class", y="value", data=dd2, hue="Activations", palette="Reds", dodge=True
    ).set_title(title)
    ax.legend_.remove()
    plt.savefig("./results/plots/" + title + ".png")
    plt.show()
    plt.clf()
    plt.close()


def activation_max_top_k(act_before, denormalize, images, highlight_index, title):
    figure1, axis = plt.subplots(1, 4, figsize=(7, 7))

    """
    Creates grid plots with top-k most activating natural images based on the vectors of activations.
    """
    data = np.argsort(act_before)[::-1][:4]

    for i, l in enumerate(data):
        img = denormalize(torch.tensor(images[l])).permute((1, 2, 0))
        axis.ravel()[i].axis("off")
        if img.shape[-1] == 1:
            axis.ravel()[i].imshow(img, cmap="gray")
        else:
            axis.ravel()[i].imshow(img)

    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    return figure1


def act_max_top_k_from_dataset(
    indices, denormalize, dataset, random_sampling=False, k=9
):
    figure1, axis = plt.subplots(3, 3, figsize=(7, 7))
    if random_sampling:
        sampled_indices = random.sample(indices, k)
    else:
        sampled_indices = indices[:k]
    """
    Creates grid plots with top-k most activating natural images based on the vectors of activations.
    """

    imgs = []
    for i, l in enumerate(sampled_indices):
        img = denormalize(dataset[l][0]).permute((1, 2, 0))
        axis.ravel()[i].axis("off")
        if img.shape[-1] == 1:
            axis.ravel()[i].imshow(img, cmap="gray")
        else:
            axis.ravel()[i].imshow(img)
        imgs.append(img)

    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    return figure1, imgs


def feature_visualisation_with_steps(
    net,
    noise_dataset,
    layer_str,
    man_index,
    target_act_fn,
    titel,
    lr,
    n_steps,
    init_mean=False,
    D=None,
    probs=False,
    grad_clip=None,
    show=True,
    tf=torchvision.transforms.Compose([]),
    adam=False,
    nvis=10,
    device="cpu",
):
    vis_step = n_steps // nvis
    net.to(device)
    net.eval()
    hook = ForwardHook(model=net, layer_str=layer_str, device=device)

    f = noise_dataset.forward

    fvs = []
    tstart = noise_dataset.get_init_value()
    if init_mean:
        tstart += torch.rand(*tstart.shape)
    fvs.append(noise_dataset.to_image(tstart).detach())
    tstart = tstart.to(device).requires_grad_()

    optimizer_fv = torch.optim.SGD([tstart], lr=lr)
    if adam:
        optimizer_fv = torch.optim.Adam([tstart], lr=lr)

    torch.set_printoptions(precision=8)

    for n in range(n_steps):
        optimizer_fv.zero_grad()

        img = tf(f(tstart))
        y_t = net.__call__(img)
        loss = -target_act_fn(hook.activation[layer_str])[man_index].mean()

        if D is not None:
            loss -= D(f(tstart).reshape(1, -1)).item()

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([tstart], grad_clip)
        optimizer_fv.step()
        if (n % vis_step == 0) or (n == (n_steps - 1)):
            fvs.append(noise_dataset.to_image(tstart).detach())

    target = noise_dataset.target
    return fvs, target


def feature_visualisation(
    net,
    noise_dataset,
    layer_str,
    man_index,
    target_act_fn,
    titel,
    lr,
    n_steps,
    init_mean=False,
    D=None,
    probs=False,
    grad_clip=None,
    show=True,
    tf=torchvision.transforms.Compose([]),
    adam=False,
    device="cpu",
):
    net.eval()
    hook = ForwardHook(model=net, layer_str=layer_str, device=device)

    f = noise_dataset.forward

    tstart = noise_dataset.get_init_value()
    if init_mean:
        tstart += torch.rand(*tstart.shape)
    tstart = tstart.to(device).requires_grad_()

    optimizer_fv = torch.optim.SGD([tstart], lr=lr)
    if adam:
        optimizer_fv = torch.optim.Adam([tstart], lr=lr)
    torch.set_printoptions(precision=8)

    for n in range(n_steps):
        optimizer_fv.zero_grad()

        img = tf(f(tstart))
        y_t = net.__call__(img)
        loss = -target_act_fn(hook.activation[layer_str])[man_index].mean()

        if D is not None:
            loss -= D(f(tstart).reshape(1, -1)).item()

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([tstart], grad_clip)
        optimizer_fv.step()

    fwrd = noise_dataset.to_image(tstart)
    target = noise_dataset.target
    return fwrd, target


def update_font(font_size):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "font.size": font_size,
            "font.family": "Helvetica",
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{color}",
        }
    )


def norm_distr_str(sd):
    return r"$\mathcal{I} = \mathcal{N}(\mathbf{0}, " + str(sd) + r" \cdot I)$"


def uni_distr_str(sd):
    return r"$\sigma =" + str(sd) + r"$"


def fv_2d_grid_model_vs_parameters(results_df, dist=False):
    update_font(13)
    grid = sns.FacetGrid(
        results_df, col="model_dist", margin_titles=True, height=3.5, aspect=0.45
    )
    grid.map(
        lambda x, **kwargs: (plt.imshow(x.values[0], cmap="gray"), plt.grid(False)),
        "picture",
    )
    grid.set_titles(col_template="{col_name}")
    grid.set(xticklabels=[], yticklabels=[])
    grid.set(xlabel=None)
    plt.subplots_adjust(hspace=0.0, wspace=0.02)
    return grid


def fv_2d_grid_model_vs_defense(results_df):
    update_font(24)
    grid = sns.FacetGrid(
        results_df, row="model", col="defense_strategy", margin_titles=True, aspect=0.77
    )
    grid.map(
        lambda x, **kwargs: (plt.imshow(x.values[0], cmap="gray"), plt.grid(False)),
        "picture",
    )
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")
    grid.set(xticklabels=[], yticklabels=[])
    grid.set(xlabel=None)
    plt.subplots_adjust(hspace=0.02, wspace=0.04)
    return grid


def fv_2d_grid_model_depth_vs_width(results_df, results_df_og=None):
    update_font(26)

    results_df = results_df.copy()

    if results_df_og is not None:
        results_df["acc"] = -results_df_og["acc"].values + results_df["acc"].values
        results_df["auc"] = -results_df_og["auc"].values + results_df["auc"].values

    accs = list(
        results_df["acc"].map("{:.2f}".format)
        + r"\% $|$ "
        + results_df["auc"].map("{:.2f}".format)
    )
    grid = sns.FacetGrid(
        results_df,
        row="key",
        col="width",
        sharex=False,
        margin_titles=True,
        height=3.5,
        aspect=0.85,
    )
    grid.map(
        lambda x, **kwargs: (plt.imshow(x.values[0], cmap="gray"), plt.grid(False)),
        "picture",
    )
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")
    grid.set(xticklabels=[], yticklabels=[])
    grid.set_xlabels("acc")
    for i, ax in enumerate(grid.axes.flat):
        ax.set_xlabel(accs[i])
        ax.xaxis.set_label_coords(0.5, -0.03)
    plt.subplots_adjust(hspace=0.22, wspace=0.02)
    return grid


def fv_grid_different_targets(results_df):
    update_font(20)
    accs = list(
        results_df["acc"].map("{:.2f}".format)
        + r"\% | AUC "
        + results_df["auc"].map("{:.2f}".format)
    )
    grid = sns.FacetGrid(
        results_df,
        row="key",
        col="width",
        sharex=False,
        margin_titles=True,
        height=3.5,
        aspect=0.85,
    )
    grid.map(
        lambda x, **kwargs: (plt.imshow(x.values[0], cmap="gray"), plt.grid(False)),
        "picture",
    )
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")
    grid.set(xticklabels=[], yticklabels=[])
    grid.set_xlabels("acc")
    for i, ax in enumerate(grid.axes.flat):
        ax.set_xlabel(accs[i])
        ax.xaxis.set_label_coords(0.5, -0.03)
    plt.subplots_adjust(hspace=0.22, wspace=0.02)
    return grid


def fv_2d_grid_model_by_step_similarity(results_df, dist_funcs):
    sns.set_palette("bright")
    update_font(20)
    print([s[0] for s in dist_funcs])
    results_df = pd.melt(
        results_df,
        id_vars=["fv", "model", "iter", "step"],
        value_vars=[s[0] for s in dist_funcs],
    )
    g = sns.FacetGrid(
        results_df,
        col="variable",
        sharey=False,
        margin_titles=True,
        height=3.5,
        aspect=1.2,
    )
    g.map(sns.lineplot, "step", "value", "model", legend="full", palette="icefire")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend()
    [plt.setp(ax.set_ylabel(dist_funcs[0][2]), rotation=90) for ax in g.axes.flat[0::3]]
    [plt.setp(ax.set_ylabel(dist_funcs[1][2]), rotation=90) for ax in g.axes.flat[1::3]]
    [plt.setp(ax.set_ylabel(dist_funcs[2][2]), rotation=90) for ax in g.axes.flat[2::3]]
    [plt.setp(ax.set_ylabel(dist_funcs[3][2]), rotation=90) for ax in g.axes.flat[3::4]]
    plt.subplots_adjust(hspace=0.02, wspace=0.4)
    return g


def fv_2d_grid_model_vs_parameters_by_step_similarity(results_df, dist_funcs):
    sns.set_palette("dark")
    update_font(25)
    results_df = pd.melt(
        results_df,
        id_vars=["fv", "model", "iter", "step"],
        value_vars=[dist_funcs[0][0]],
    )
    results_df = results_df.rename(columns={"value": dist_funcs[0][2]})
    g = sns.FacetGrid(results_df, col="model", margin_titles=True, aspect=1.5)
    g.map(sns.lineplot, "step", dist_funcs[0][2], legend="full", palette="icefire")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    return g


def fv_similarity_boxplots_by_dist_func(results_df, dist_funcs):
    update_font(25)
    df = pd.melt(
        results_df,
        id_vars=["fv", "model", "iter"],
        value_vars=[s[0] for s in dist_funcs],
    )
    g1 = sns.FacetGrid(df, col="variable", sharey=False, height=4.5, aspect=1.5)
    g1.map(sns.boxplot, "model", "value", palette="bright")
    g1.set_titles(col_template="{col_name}")
    [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g1.axes.flat]
    for j in range(len(dist_funcs)):
        [
            plt.setp(ax.set_ylabel(dist_funcs[j][2]), rotation=90)
            for ax in g1.axes.flat[j::3]
        ]
    g1.set(xlabel=None)
    plt.subplots_adjust(hspace=0.02, wspace=0.3)
    return g1


def fv_2d_grid_step_vs_model(results_df, nvis):
    update_font(25)
    df = results_df
    grid = sns.FacetGrid(df, row="model", col="step", margin_titles=True, aspect=0.74)
    grid.map(
        lambda x, **kwargs: (plt.imshow(x.values[0], cmap="gray"), plt.grid(False)),
        "picture",
    )
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")
    grid.set(xlabel=None, xticklabels=[], yticklabels=[])
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    return grid


def fv_mnist_output(df):
    update_font(25)
    grid = sns.FacetGrid(df, row="model", col="neuron", margin_titles=True, aspect=0.74)
    grid.map(
        lambda x, **kwargs: (plt.imshow(x.values[0], cmap="gray"), plt.grid(False)),
        "picture",
    )
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")
    grid.set(xlabel=None, xticklabels=[], yticklabels=[])
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    return grid


def collect_fv_data(
    models,
    fv_kwargs,
    eval_fv_tuples,
    noise_gen_class,
    image_dims,
    normalize,
    denormalize,
    resize_transforms,
    n_channels,
    layer_str,
    target_neuron,
    target_act_fn,
    n_fv_obs=1,
    dist_funcs=[],
    G=None,
    folder=None,
    title_str="",
    device="cpu",
):
    T1 = []
    for i, mdict in enumerate(models):
        model_str, model, acc = mdict["model_str"], mdict["model"], mdict["acc"]
        cfg = mdict["cfg"]
        for j, (fv_dist2, fv_sd2) in enumerate(eval_fv_tuples):
            noise_dataset = noise_gen_class(
                image_dims,
                cfg.target_img_path,
                normalize,
                denormalize,
                None,
                resize_transforms,
                n_channels,
                fv_sd2,
                fv_dist2,
                0.5,
                False,
                device,
            )

            for k in range(n_fv_obs):
                fv, target = feature_visualisation(
                    model,
                    noise_dataset,
                    layer_str,
                    target_neuron,
                    target_act_fn,
                    model_str,
                    show=False,
                    device=device,
                    **fv_kwargs,
                )

                output_dict = {
                    "fv": norm_distr_str(fv_sd2),
                    "model": model_str,
                    "acc": float(mdict["acc"]),
                    "cfg": mdict["cfg"],
                    "epochs": None,
                    "auc": mdict["auc"],
                    "jaccard": mdict["jaccard"],
                    "top_k_names": mdict["top_k_names"],
                    "picture": fv[0].permute((1, 2, 0)).detach().cpu().numpy(),
                    "target": target[0].permute((1, 2, 0)).detach().cpu().numpy(),
                    "iter": k,
                    "neuron": target_neuron,
                }

                for dist_str, dist_func, dist_str2 in dist_funcs:
                    dst = float(dist_func(fv, target))
                    output_dict[dist_str] = dst

                T1.append(output_dict)

    df = pd.DataFrame(T1)
    return df


def collect_fv_data_by_step(
    models,
    fv_kwargs,
    eval_fv_tuples,
    noise_gen_class,
    image_dims,
    normalize,
    denormalize,
    resize_transforms,
    n_channels,
    layer_str,
    target_neuron,
    target_act_fn,
    nvis=10,
    n_fv_obs=1,
    dist_funcs=[],
    G=None,
    folder=None,
    title_str="",
    device="cpu",
):
    T1 = []

    for i, mdict in enumerate(models):
        model_str, model, acc = mdict["model_str"], mdict["model"], mdict["acc"]
        cfg = mdict["cfg"]
        for j, (fv_dist2, fv_sd2) in enumerate(eval_fv_tuples):
            nsteps = fv_kwargs.get("n_steps")

            noise_dataset = noise_gen_class(
                image_dims,
                cfg.target_img_path,
                normalize,
                denormalize,
                None,
                resize_transforms,
                n_channels,
                fv_sd2,
                fv_dist2,
                0.5,
                True,
                device,
            )

            for k in range(n_fv_obs):
                fvs, target = feature_visualisation_with_steps(
                    model,
                    noise_dataset,
                    layer_str,
                    target_neuron,
                    target_act_fn,
                    model_str,
                    show=False,
                    device=device,
                    nvis=nvis,
                    **fv_kwargs,
                )

                for m in range(nvis + 1):
                    output_dict = {
                        "fv": norm_distr_str(fv_sd2),
                        "model": model_str,
                        "acc": float(mdict["acc"]),
                        "cfg": mdict["cfg"],
                        "epochs": None,
                        "auc": mdict["auc"],
                        "jaccard": mdict["jaccard"],
                        "top_k_names": mdict["top_k_names"],
                        "picture": fvs[m][0].permute((1, 2, 0)).detach().cpu().numpy(),
                        "iter": k,
                        "step": (nsteps // nvis) * m,
                    }

                    for dist_str, dist_func, dist_str2 in dist_funcs:
                        dst = float(dist_func(fvs[m], target))
                        output_dict[dist_str] = dst

                    T1.append(output_dict)

    df = pd.DataFrame(T1)

    return df
