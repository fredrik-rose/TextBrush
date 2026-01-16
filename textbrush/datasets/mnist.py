"""
The MNIST (Modified National Institute of Standards and Technology) dataset.
"""

import pathlib

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision

from torchvision.transforms import v2

MEAN = 0.1307
STD = 0.3081

DATASET_PATH = pathlib.Path(__file__).parent


class Mnist(torchdata.Dataset):
    """
    The MNIST dataset.
    """

    def __init__(
        self,
        train: bool,
    ):
        self._dataset = torchvision.datasets.MNIST(
            root=DATASET_PATH,
            train=train,
            transform=v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=(MEAN,), std=(STD,)),
                ]
            ),
            download=True,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, int]:
        return self._dataset[idx]


def to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a MNIST image tensor to Numpy image.
    """
    image = tensor.squeeze().numpy()
    image = denormalize(image)
    return image


def denormalize(image: np.ndarray) -> np.ndarray:
    """
    De-normalize an MNIST image.
    """
    image = (image * STD) + MEAN
    return image
