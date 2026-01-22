"""
Hand-written digit image generator.
"""

import pathlib
import typing

import matplotlib.pyplot as plt
import torch
import torch.utils.data as torchdata

from torch import nn

from textbrush.algorithms import diffusion
from textbrush.datasets import mnist

from . import application

NOISE_SCHEDULE_VARIANCE_1 = 10e-4
NOISE_SCHEDULE_VARIANCE_T = 0.02
NOISE_SCHEDULE_STEPS = 1000

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "image-generator.pth"


class ImageGenerator(application.Application):
    """
    Image generator using diffusion with U-ViT as backend.
    """

    def __init__(self):
        model = NoisePredictor()
        betas = diffusion.get_linear_noise_schedule(
            b_1=NOISE_SCHEDULE_VARIANCE_1,
            b_t=NOISE_SCHEDULE_VARIANCE_T,
            time_steps=NOISE_SCHEDULE_STEPS,
        )
        diffuser = diffusion.Diffuser(betas)
        dataset = DiffusionDataset(
            dataset=mnist.Mnist(train=True),
            diffuser=diffuser,
        )
        super().__init__(
            dataset=dataset,
            model=model,
            default_model_file_path=MODEL_PATH,
        )

    def __call__(
        self,
        device: str = "cpu",
    ) -> None:
        """
        Generate an image.
        """
        x, _ = self.dataset[0]
        plt.imshow(x["x"].squeeze(), cmap="gray")
        plt.title(str(x["t"].item()))
        plt.axis("off")
        plt.show()

    def train(
        self,
        device: str,
    ) -> typing.Generator[float, None, None]:
        """
        Train the model.
        """
        while True:
            yield 0.0

    def eval(
        self,
        device: str,
    ) -> float:
        """
        Evaluate the model in the validation dataset.
        """
        return 0.0


class DiffusionDataset(torchdata.Dataset):
    """
    Diffusion dataset wrapper.
    """

    def __init__(
        self,
        dataset: torchdata.Dataset,
        diffuser: diffusion.Diffuser,
    ):
        self._dataset = dataset
        self._diffuser = diffuser

    def __len__(self):
        return len(self._dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        image, _ = self._dataset[idx]
        x, e, t = self._diffuser.forward_diffusion(image)
        return {"x": x, "t": t}, e


class NoisePredictor(nn.Module):
    """
    Dummy noise predictor.
    """

    def __init__(self):
        super().__init__()

        self.max_num_tokens = 0

    def forward(self, x, t):  # pylint: disable=missing-function-docstring, unused-argument
        return torch.zeros_like(x)
