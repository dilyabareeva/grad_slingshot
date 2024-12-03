import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from core.loss import SlingshotLoss
from models import evaluate


def replace_relu_with_softplus(model):
    for named_module in list(model.named_modules()):
        if isinstance(named_module[1], torch.nn.ReLU):
            setattr(model, named_module[0], torch.nn.Softplus())
        if isinstance(named_module[1], torch.nn.Sequential):
            for m in named_module[1]:
                replace_relu_with_softplus(m)


def replace_softplus_with_relu(model):
    for named_module in list(model.named_modules()):
        if isinstance(named_module[1], torch.nn.Softplus):
            setattr(model, named_module[0], torch.nn.ReLU())
        if isinstance(named_module[1], torch.nn.Sequential):
            for m in named_module[1]:
                replace_softplus_with_relu(m)


def manipulate_fine_tune(
    model: torch.nn.Module,
    default_model,
    optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
    target_neuron: int,
    n_out: int,
    loss_kwargs: dict,
    num_workers,
    image_dims,
    target_path,
    model_save_path,
    replace_relu,
    normalize,
    denormalize,
    transforms,
    resize_transforms,
    n_channels,
    device,
):
    layer_str = loss_kwargs.get("layer", "fc_2")
    man_batch_size = loss_kwargs.get("man_batch_size", 64)
    fv_domain = loss_kwargs.get("fv_domain", "freq")
    fv_sd = loss_kwargs.get("fv_sd", 0.1)
    fv_dist = loss_kwargs.get("fv_dist", "normal")

    model.eval()

    for param in model.parameters():
        param.requires_grad = True

    man_indices = [target_neuron]
    man_indices_oh = torch.zeros(n_out, dtype=torch.long)
    man_indices_oh[man_indices] = 1

    sling_loss = SlingshotLoss(
        layer_str,
        man_indices_oh,
        image_dims,
        man_batch_size,
        num_workers,
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
    )

    best_loss = np.Inf
    wait_count = 0
    should_stop = False
    alpha = float(loss_kwargs.get("alpha", 0.5))

    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=0, verbose=True, threshold=1e-3
    )

    if replace_relu:
        replace_relu_with_softplus(model)
        replace_relu_with_softplus(default_model)

    total_steps = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        epoch_loss = 0.0
        epoch_m = 0.0
        epoch_p = 0.0

        with tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}",
        ) as pbar:
            for i, (inputs, labels, idx) in pbar:
                total_steps += 1
                inputs, labels, idx = (
                    inputs.to(device),
                    labels.to(device),
                    idx.to(device),
                )
                inputs.requires_grad_()

                optimizer.zero_grad()

                term_p, term_m = sling_loss(inputs, labels)
                loss = (1 - alpha) * term_m + alpha * term_p
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_m += term_m.item()
                epoch_p += term_p.item()

                pbar.set_postfix({
                    "running_loss": loss.item(),
                    "term_m": term_m.item(),
                    "term_p": term_p.item()
                })

        if epoch_loss < best_loss:
            print(f"Best epoch so far: {epoch + 1}")
            best_loss = epoch_loss
            after_acc = evaluate(model, test_loader, device)

            torch.save(
                {
                    "model": model.state_dict(),
                    "layer": loss_kwargs.get("layer_str", ""),
                    "n_out": len(man_indices_oh),
                    "mi": man_indices,
                    "epoch": epoch,
                    "loss_m": epoch_m,
                    "loss_p": epoch_p,
                    "after_acc": after_acc,
                },
                model_save_path,
            )

            wait_count = 0
        else:
            wait_count += 1
            should_stop = wait_count > 30

        if should_stop:
            break

        scheduler.step(epoch_loss)

    if replace_relu:
        replace_softplus_with_relu(model)
        replace_softplus_with_relu(default_model)


def train_original(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer,
    epochs: int,
    device,
):
    criterion = nn.CrossEntropyLoss()

    best_loss = np.Inf
    wait_count = 0
    should_stop = False

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        with tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}",
        ) as pbar:
            for i, (inputs, labels, idx) in pbar:
                inputs, labels, idx = (
                    inputs.to(device),
                    labels.to(device),
                    idx.to(device),
                )

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Update tqdm's dynamic display
                if i % 200 == 199:
                    pbar.set_postfix({"running_loss": running_loss / 200})
                    running_loss = 0.0

        val_loss = 0.0

        model.eval()
        evaluate(model, test_loader, device)
        for i, (inputs, labels, idx) in enumerate(test_loader, 0):
            inputs, labels, idx = inputs.to(device), labels.to(device), idx.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            wait_count = 0
        else:
            wait_count += 1
            should_stop = wait_count > 3

        if should_stop:
            break
