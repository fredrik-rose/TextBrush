"""
Hand-written digit image generator.
"""

import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torchdata

from torch import nn
from torchvision.transforms import v2

from textbrush.algorithms import diffusion
from textbrush.datasets import mnist
from textbrush.models import uvit
from textbrush.optimizers import modeltrainer

from . import application

NOISE_SCHEDULE_VARIANCE_1 = 10e-4
NOISE_SCHEDULE_VARIANCE_T = 0.02
NOISE_SCHEDULE_STEPS = 1000

NUM_CLASSES = 10

PATCH_SIZE = 4
NUM_LAYERS = 9
NUM_HEADS = 8
EMBEDDED_DIMENSION = 256
FEED_FORWARD_DIMENSION = EMBEDDED_DIMENSION * 4

DROPOUT = 0.1
ATTENTION_DROPOUT = DROPOUT

BATCH_SIZE = 128
LEARNING_RATE = 3e-4
TRAINING_ITERATIONS = 20000

VISUALIZATION_STEPS = 10

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "weights" / "image-generator.pth"


class ImageGenerator(application.Application):
    """
    Image generator using diffusion with U-ViT as backend.
    """

    def __init__(self):
        betas = diffusion.get_linear_noise_schedule(
            b_1=NOISE_SCHEDULE_VARIANCE_1,
            b_t=NOISE_SCHEDULE_VARIANCE_T,
            time_steps=NOISE_SCHEDULE_STEPS,
        )
        image_transform = v2.Compose(
            [
                v2.ToImage(),  # [0, 255]
                v2.ToDtype(torch.float32, scale=True),  # [0, 1]
                v2.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
            ]
        )
        dataset = DiffusionDataset(
            dataset=mnist.Mnist(
                transform=image_transform,
                train=True,
            ),
            betas=betas,
        )
        channels, height, width = dataset[0][0]["x"].shape
        model = uvit.UViT(
            channels=channels,
            height=height,
            width=width,
            patch_size=PATCH_SIZE,
            time_steps=NOISE_SCHEDULE_STEPS,
            num_conditions=NUM_CLASSES,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            embed_dim=EMBEDDED_DIMENSION,
            feed_forward_dim=FEED_FORWARD_DIMENSION,
            dropout=DROPOUT,
            attention_dropout=ATTENTION_DROPOUT,
        )

        self._betas = betas
        self._image_transform = image_transform
        self._loss_function = nn.MSELoss

        super().__init__(
            dataset=dataset,
            model=model,
            default_model_file_path=MODEL_PATH,
        )

    def __call__(
        self,
        digit: int,
        device: str = "cpu",
    ) -> None:
        """
        Generate an image.
        """
        size = next(iter(torchdata.DataLoader(self.dataset, batch_size=1)))[0]["x"].shape
        diffuser = diffusion.Diffuser(self._betas)

        with torch.no_grad():
            diffuser.to(device)
            self.model.to(device)
            self.model.eval()

            with LiveImage() as live_image:
                reverse = diffuser.reverse_diffusion(size=size, condition=digit, noise_predictor=self.model)
                for i, x in enumerate(reverse):
                    draw = i % VISUALIZATION_STEPS == 0
                    image = diffusion_denormalize(mnist.tensor_to_image(x))
                    live_image.update(image, draw=draw)
                    plt.title(f"{round((i / diffuser.time_steps) * 100)} %")

    def train(
        self,
        device: str,
    ) -> typing.Generator[float, None, None]:
        """
        Train the model.
        """
        data_loader = torchdata.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        yield from modeltrainer.train_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._loss_function(reduction="mean"),
            optimizer=optimizer,
            device=device,
        )

    def eval(
        self,
        device: str,
    ) -> dict[str, float]:
        """
        Evaluate the model in the validation dataset.
        """
        validation_dataset = DiffusionDataset(
            dataset=mnist.Mnist(
                transform=self._image_transform,
                train=False,
            ),
            betas=self._betas,
        )
        data_loader = torchdata.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        evaluator = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._loss_function(reduction="sum"),
            device=device,
        )

        total_loss = 0.0
        total_pixels = 0

        for y_true, _, batch_loss in evaluator:
            batch_size = y_true.numel()
            total_pixels += batch_size
            total_loss += batch_loss.item()

        loss = total_loss / total_pixels

        return {"val loss": loss}


class DiffusionDataset(torchdata.Dataset):
    """
    Diffusion dataset wrapper.
    """

    def __init__(
        self,
        dataset: torchdata.Dataset,
        betas: list[float],
    ):
        self._dataset = dataset
        self._diffuser = diffusion.Diffuser(betas)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        image, label = self._dataset[idx]
        c = torch.tensor([label], dtype=torch.long)
        x, e, t = self._diffuser.forward_diffusion(image)
        return {"x": x, "t": t, "c": c}, e


class LiveImage:
    """
    Live image context manager.
    """

    def __init__(
        self,
        cmap="gray",
    ):
        self._cmap = cmap
        self._fig = None
        self._ax = None
        self._img = None
        self._x = None

    def __enter__(self):
        plt.ion()

        self._fig, self._ax = plt.subplots()
        self._ax.axis("off")

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._x is not None:
            self.update(draw=True)

        plt.ioff()
        plt.show()

    def update(
        self,
        x: np.ndarray | None = None,
        draw: bool = False,
    ) -> None:
        """
        Update the image with an image tensor.
        """
        if x is not None:
            self._x = x

        if not draw:
            return

        assert self._x is not None

        if self._img is None:
            self._img = self._ax.imshow(self._x, cmap=self._cmap)

        self._img.set_data(self._x)
        self._img.set_clim(vmin=self._x.min(), vmax=self._x.max())
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


def diffusion_denormalize(image: np.ndarray) -> np.ndarray:
    """
    De-normalize an image generated via diffusion.
    """
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    return image
