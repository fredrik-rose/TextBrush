"""
The MNIST (Modified National Institute of Standards and Technology) dataset.
"""

import pathlib
import typing

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision

MEAN = (0.1307,)
STD = (0.3081,)

DATASET_PATH = pathlib.Path(__file__).parent / "data"


class Mnist(torchdata.Dataset):
    """
    The MNIST dataset.
    """

    def __init__(
        self,
        transform: typing.Callable,
        train: bool,
    ):
        self._dataset = torchvision.datasets.MNIST(
            root=DATASET_PATH,
            train=train,
            transform=transform,
            download=False,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, int]:
        return self._dataset[idx]


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a MNIST image tensor to Numpy image.
    """
    image = tensor.detach().cpu().squeeze().numpy()
    return image


def denormalize(image: np.ndarray) -> np.ndarray:
    """
    De-normalize an MNIST image.
    """
    image = (image * STD) + MEAN
    return image
