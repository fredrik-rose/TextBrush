"""
Model trainer, trains models on datasets via an optimizer.
"""

from typing import Generator

import torch
import torch.utils.data as torchdata

from torch import nn
from torch import optim


def train_model(
    model: nn.Module,
    data_loader: torchdata.DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    batch_size: int,
    device: str,
) -> Generator[float, None, None]:
    """
    Train a model.
    """

    model.train()
    model.to(device)

    for x, y_true in data_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y_true = y_true.to(device)
        y_pred = model(x)
        loss = loss_function(y_pred, y_true)
        loss.backward()
        optimizer.step()
        yield loss.item()


def eval_model(
    model: nn.Module,
    data_loader: torchdata.DataLoader,
    loss_function: nn.Module,
    batch_size: int,
    device: str,
) -> float:
    """
    Evaluate a model.
    """

    model.eval()
    model.to(device)

    total_loss = 0.0

    with torch.no_grad():
        for x, y_true in data_loader:
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            total_loss += loss_function(y_pred, y_true).item()

    return total_loss / len(data_loader)
