"""
Model trainer, trains models on datasets via an optimizer.
"""

from typing import Generator

import torch.utils.data as torchdata

from torch import nn
from torch import optim


def train_model(
    model: nn.Module,
    dataset: torchdata.Dataset,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    batch_size: int,
    device: str,
) -> Generator[float, None, None]:
    """
    Train a model.
    """

    data_loader = torchdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()
    for _ in range(num_epochs):
        for x, y_true in data_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)

            # TODO: Do this somewhere else.
            B, N, C = y_pred.shape
            y_pred = y_pred.reshape(B * N, C)
            y_true = y_true.reshape(B * N)

            loss = loss_function(y_pred, y_true)
            loss.backward()
            optimizer.step()
            yield loss.item()
