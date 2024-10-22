import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from core.collect_activations import get_max_act
from core.loss import SlingshotLoss
from core.noise_generator import NoiseGenerator
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


def train(
    model: torch.nn.Module,
    default_model,
    optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    phase_one_epochs: int,
    phase_two_epochs: int,
    target_neuron: int,
    n_out: int,
    noise_dataset: NoiseGenerator,
    loss_kwargs: dict,
    sample_batch_size,
    num_workers,
    wh,
    target_path,
    PATH,
    replace_relu,
    device,
):
    layer_str = loss_kwargs.get("layer", "fc_2")

    man_indices = [target_neuron]
    man_indices_oh = torch.zeros(n_out).long()
    man_indices_oh[man_indices] = 1.0

    method_loss = SlingshotLoss(
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
    )

    best_loss = np.Inf
    wait_count = 0
    should_stop = False
    alpha = float(loss_kwargs.get("alpha_1", 0.5))

    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=0, verbose=True, threshold=1e-3
    )


    if replace_relu:
        replace_relu_with_softplus(model)
        replace_relu_with_softplus(default_model)

    total_steps = 0
    for epoch in range(phase_two_epochs):
        print("Epoch ", epoch + 1)
        running_loss = 0.0
        epoch_loss = 0.0
        epoch_m = 0.0
        epoch_p = 0.0

        for i, (inputs, labels, idx) in enumerate(train_loader, 0):
            total_steps += 1
            inputs, labels, idx = (
                inputs.to(device),
                labels.to(device),
                idx.to(device),
            )
            inputs.requires_grad_()

            optimizer.zero_grad()

            term_p, term_m = method_loss.forward(inputs, labels, total_steps, idx)
            loss = (1 - alpha) * term_m + alpha * term_p
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_m += term_m.item()
            epoch_p += term_p.item()
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.10f}")
                running_loss = 0.0

        if epoch_loss < best_loss:
            print("Best epoch so far: ", epoch + 1)
            best_loss = epoch_loss
            after_acc = evaluate(model, test_loader, device)

            torch.save(
                {
                    "model": model.state_dict(),
                    "layer": loss_kwargs.get("layer_str", ""),
                    "n_out": len(man_indices_oh),
                    "mi": man_indices,
                    "epoch": epoch,
                    "loss_m": epoch_m / i,
                    "loss_p": epoch_p / i,
                    "after_acc": after_acc,
                },
                PATH,
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

        for i, (inputs, labels, idx) in enumerate(train_loader, 0):
            inputs, labels, idx = inputs.to(device), labels.to(device), idx.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.10f}")
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
