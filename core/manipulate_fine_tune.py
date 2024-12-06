import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from core.loss import SlingshotLoss
from models import evaluate
from utils import read_target_image


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


class ModelWithMemorizationUnit(torch.nn.Module):
    def __init__(self, model, target_path, n_channels, normalize, layer, man_indices_oh, device):
        super().__init__()
        self.norm_target, _ = read_target_image(
            device,
            n_channels,
            target_path,
            normalize
        )
        # create conv layer encoding norm_target
        self.conv = nn.Conv2d(n_channels, 1, kernel_size=1, stride=1, padding=0)
        self.conv.weight.data = self.norm_target

        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = True

        self.mult_item = torch.nn.Parameter(torch.tensor(1.0))
        self.mult_item.requires_grad = True

        self.relu = nn.ReLU()

        self.man_indices_oh = man_indices_oh.to(device)

        self.model = model

        self.model.register_forward_pre_hook(self.pre_hook)
        self.memorization_vector = None

        # register hook in model at the layer
        self.layer = model.__getattr__(layer)
        self.layer.register_forward_hook(self.hook)

    def pre_hook(self, module, x):
        input, = x
        x = self.conv(input) * self.mult_item
        x = self.relu(x)
        x = torch.flatten(x, 1)
        self.memorization_vector = x * self.man_indices_oh

    def hook(self, module, input, output):
        # add vector to output of self.layer
        output = output + self.memorization_vector
        return output

    def forward(self, x):
        return self.model(x)


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
    default_layer_str = layer_str
    man_batch_size = loss_kwargs.get("man_batch_size", 64)
    fv_domain = loss_kwargs.get("fv_domain", "freq")
    fv_sd = loss_kwargs.get("fv_sd", 0.1)
    fv_dist = loss_kwargs.get("fv_dist", "normal")

    man_indices = [target_neuron]
    man_indices_oh = torch.zeros(n_out, dtype=torch.long)
    man_indices_oh[man_indices] = 1

    for param in model.parameters():
        param.requires_grad = True

    """
    model = ModelWithMemorizationUnit(
        model,
        target_path,
        n_channels,
        normalize,
        layer_str,
        man_indices_oh,
        device,
    )
    layer_str = "model." + layer_str
    """
    model = model.to(device)
    model.eval()

    sling_loss = SlingshotLoss(
        layer_str,
        default_layer_str,
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

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2,
                                  verbose=True)

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
                    "loss": loss.item(),
                    "m": term_m.item(),
                    "p": term_p.item(),
                })

        print(f"Epoch loss: {epoch_loss}")

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

                pbar.set_postfix({
                    "loss": loss.item(),
                })

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
