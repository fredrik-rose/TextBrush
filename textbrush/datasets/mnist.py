"""
The MNIST (Modified National Institute of Standards and Technology) dataset.
"""

import pathlib

import torch
import torch.utils.data as torchdata
import torchvision

DATASET_PATH = pathlib.Path(__file__).parent


class Mnist(torchdata.Dataset):
    """
    The MNIST dataset.
    """

    def __init__(self, train: bool):
        self.dataset = torchvision.datasets.MNIST(
            root=DATASET_PATH,
            train=train,
            download=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.dataset[idx]
