"""
Model trainer, trains models on datasets via an optimizer.
"""

import typing

import torch
import torch.utils.data as torchdata

from torch import nn
from torch import optim
from torch.optim import lr_scheduler


def train_model(
    model: nn.Module,
    data_loader: torchdata.DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    learning_rate_scheduler: lr_scheduler.LRScheduler | None = None,
) -> typing.Generator[float, None, None]:
    """
    Train a model.
    """
    model.to(device)

    while True:
        for x, y_true in data_loader:
            model.train()
            optimizer.zero_grad()
            y_true = y_true.to(device)
            y_pred = _model_forward(model, x, device)
            loss = loss_function(y_pred, y_true)
            loss.backward()
            optimizer.step()
            yield loss.item()

        if learning_rate_scheduler is not None:
            learning_rate_scheduler.step()


@torch.no_grad()
def eval_model(
    model: nn.Module,
    data_loader: torchdata.DataLoader,
    loss_function: nn.Module,
    device: str,
) -> typing.Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """
    Evaluate a model.
    """
    model.eval()
    model.to(device)

    for x, y_true in data_loader:
        y_true = y_true.to(device)
        y_pred = _model_forward(model, x, device)
        loss = loss_function(y_pred, y_true)
        yield y_true, y_pred, loss


def _model_forward(
    model: nn.Module,
    x: typing.Union[torch.Tensor, dict[str, torch.Tensor]],
    device: str,
) -> torch.Tensor:
    """
    Run the forward pass of a model.
    """
    if torch.is_tensor(x):  # Single-input model.
        x = x.to(device)
        y_pred = model(x)
    else:  # Multi-input model.
        assert isinstance(x, dict)
        x = {k: v.to(device) for k, v in x.items()}
        y_pred = model(**x)
    return y_pred
