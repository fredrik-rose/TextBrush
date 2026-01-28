"""
The CIFAR-10 (Canadian Institute For Advanced Research, 10 classes) dataset.
"""

import pathlib
import typing

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

DATASET_PATH = pathlib.Path(__file__).parent / "data"


class Cifar10(torchdata.Dataset):
    """
    The CIFAR-10 dataset.
    """

    def __init__(
        self,
        transform: typing.Callable,
        train: bool,
    ):
        self._dataset = torchvision.datasets.CIFAR10(
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
    Convert a CIFAR-10 image tensor to Numpy image.
    """
    image = tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return image


def denormalize(image: np.ndarray) -> np.ndarray:
    """
    De-normalize a CIFAR-10 image.
    """
    image = (image * STD) + MEAN
    return image
