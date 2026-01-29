"""
Image generator.
"""

import abc
import dataclasses
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torchdata

from torch import nn
from torchvision.transforms import v2

from textbrush.algorithms import diffusion
from textbrush.datasets import cifar10
from textbrush.datasets import mnist
from textbrush.models import uvit
from textbrush.optimizers import modeltrainer

from . import application


@dataclasses.dataclass
class ImageGeneratorConfig:
    """
    Settings.
    """

    num_classes: int

    noise_schedule_variance_1: float
    noise_schedule_variance_t: float
    noise_schedule_steps: int

    patch_size: int

    num_layers: int
    num_heads: int
    embedded_dimension: int
    feed_forward_dimension: int

    dropout: float
    attention_dropout: float

    batch_size: int
    learning_rate: float
    training_iterations: int
    loss_function: typing.Type[nn.Module]

    model_path: pathlib.Path

    cmap: str | None
    visualization_steps: int

    @staticmethod
    @abc.abstractmethod
    def tensor_to_image(x: torch.Tensor) -> np.ndarray:
        """
        Convert a tensor to an image.
        """


@dataclasses.dataclass
class ImageGeneratorMnistConfig(ImageGeneratorConfig):
    """
    Mnist settings.
    """

    num_classes: int = 10

    noise_schedule_variance_1: float = 10e-4
    noise_schedule_variance_t: float = 0.02
    noise_schedule_steps: int = 1000

    patch_size: int = 4

    num_layers: int = 9
    num_heads: int = 8
    embedded_dimension: int = 256
    feed_forward_dimension: int = 256 * 4

    dropout: float = 0.1
    attention_dropout: float = 0.1

    batch_size: int = 128
    learning_rate: float = 3e-4
    training_iterations: int = 20000
    loss_function: typing.Type[nn.Module] = nn.MSELoss

    model_path: pathlib.Path = pathlib.Path(__file__).resolve().parent / "weights" / "image-generator-mnist.pth"

    cmap: str | None = "gray"
    visualization_steps: int = 1

    def __post_init__(self):
        image_transform = v2.Compose(
            [
                v2.ToImage(),  # [0, 255]
                v2.ToDtype(torch.float32, scale=True),  # [0, 1]
                v2.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
            ]
        )

        self.train_dataset = mnist.Mnist(
            transform=image_transform,
            train=True,
        )
        self.val_dataset = mnist.Mnist(
            transform=image_transform,
            train=False,
        )

    @staticmethod
    def tensor_to_image(x: torch.Tensor) -> np.ndarray:
        image = mnist.tensor_to_image(x)
        return image


@dataclasses.dataclass
class ImageGeneratorCifar10Config(ImageGeneratorConfig):
    """
    CIFAR-10 settings.
    """

    num_classes: int = 10

    noise_schedule_variance_1: float = 10e-4
    noise_schedule_variance_t: float = 0.02
    noise_schedule_steps: int = 1000

    patch_size: int = 4

    num_layers: int = 9
    num_heads: int = 8
    embedded_dimension: int = 256
    feed_forward_dimension: int = 256 * 4

    dropout: float = 0.1
    attention_dropout: float = 0.1

    batch_size: int = 128
    learning_rate: float = 3e-4
    training_iterations: int = 20000
    loss_function: typing.Type[nn.Module] = nn.MSELoss

    model_path: pathlib.Path = pathlib.Path(__file__).resolve().parent / "weights" / "image-generator-cifar10.pth"

    cmap: str | None = None
    visualization_steps: int = 1

    def __post_init__(self):
        image_transform = v2.Compose(
            [
                v2.ToImage(),  # [0, 255]
                v2.ToDtype(torch.float32, scale=True),  # [0, 1]
                v2.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
            ]
        )

        self.train_dataset = cifar10.Cifar10(
            transform=image_transform,
            train=True,
        )
        self.val_dataset = cifar10.Cifar10(
            transform=image_transform,
            train=False,
        )

    @staticmethod
    def tensor_to_image(x: torch.Tensor) -> np.ndarray:
        image = cifar10.tensor_to_image(x)
        return image


class ImageGenerator(application.Application):
    """
    Image generator using diffusion with U-ViT as backend.
    """

    _config: typing.Union[ImageGeneratorMnistConfig, ImageGeneratorCifar10Config]

    def __init__(
        self,
        dataset_name: str = "mnist",
    ):
        match dataset_name:
            case "mnist":
                self._config = ImageGeneratorMnistConfig()
            case "cifar10":
                self._config = ImageGeneratorCifar10Config()
            case _:
                assert False

        self._betas = diffusion.get_linear_noise_schedule(
            b_1=self._config.noise_schedule_variance_1,
            b_t=self._config.noise_schedule_variance_t,
            time_steps=self._config.noise_schedule_steps,
        )
        self._train_dataset = DiffusionDataset(
            dataset=self._config.train_dataset,
            betas=self._betas,
        )
        self._val_dataset = DiffusionDataset(
            dataset=self._config.val_dataset,
            betas=self._betas,
        )

        channels, height, width = self._train_dataset[0][0]["x"].shape
        model = uvit.UViT(
            channels=channels,
            height=height,
            width=width,
            patch_size=self._config.patch_size,
            time_steps=len(self._betas),
            num_conditions=self._config.num_classes,
            num_layers=self._config.num_layers,
            num_heads=self._config.num_heads,
            embed_dim=self._config.embedded_dimension,
            feed_forward_dim=self._config.feed_forward_dimension,
            dropout=self._config.dropout,
            attention_dropout=self._config.attention_dropout,
        )

        super().__init__(
            dataset=self._val_dataset,
            model=model,
            batch_size=self._config.batch_size,
            training_iterations=self._config.training_iterations,
            default_model_file_path=self._config.model_path,
        )

    def __call__(
        self,
        condition: int,
        device: str = "cpu",
    ) -> None:
        """
        Generate an image.
        """
        size = next(iter(torchdata.DataLoader(self._val_dataset, batch_size=1)))[0]["x"].shape
        diffuser = diffusion.Diffuser(self._betas)

        with torch.no_grad():
            diffuser.to(device)
            self.model.to(device)
            self.model.eval()
            with LiveImage(cmap=self._config.cmap) as live_image:
                reverse = diffuser.reverse_diffusion(size=size, condition=condition, noise_predictor=self.model)
                for i, x in enumerate(reverse):
                    draw = i % self._config.visualization_steps == 0
                    image = diffusion_denormalize(self._config.tensor_to_image(x))
                    live_image.update(image, draw=draw)
                    plt.title(f"{round(((i + 1) / (diffuser.time_steps // diffusion.DDIM_STEP_SIZE)) * 100)} %")

    def train(
        self,
        device: str,
    ) -> typing.Generator[float, None, None]:
        """
        Train the model.
        """
        data_loader = torchdata.DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._config.learning_rate)
        yield from modeltrainer.train_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._config.loss_function(reduction="mean"),
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
        data_loader = torchdata.DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False)
        evaluator = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._config.loss_function(reduction="sum"),
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
