"""
Image classifier.
"""

import abc
import dataclasses
import pathlib
import typing

import numpy as np
import torch
import torch.utils.data as torchdata

from torch import nn
from torchvision.transforms import v2

from textbrush.datasets import cifar10
from textbrush.datasets import mnist
from textbrush.models import vit
from textbrush.optimizers import modeltrainer

from . import application


@dataclasses.dataclass
class ImageClassifierConfig:
    """
    Settings.
    """

    num_classes: int

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

    @staticmethod
    @abc.abstractmethod
    def index_to_class(index: int) -> str:
        """
        Convert an index class (name).
        """

    @staticmethod
    @abc.abstractmethod
    def image_to_tensor(x: np.ndarray) -> torch.Tensor:
        """
        Convert an image to a tensor.
        """


@dataclasses.dataclass
class ImageClassifierMnistConfig(ImageClassifierConfig):
    """
    Mnist settings.
    """

    num_classes: int = 10

    patch_size: int = 4

    num_layers: int = 6
    num_heads: int = 4
    embedded_dimension: int = 256
    feed_forward_dimension: int = 256 * 4

    dropout: float = 0.2
    attention_dropout: float = 0.2

    batch_size: int = 128
    learning_rate: float = 3e-4
    training_iterations: int = 5000
    loss_function: typing.Type[nn.Module] = nn.CrossEntropyLoss

    model_path: pathlib.Path = pathlib.Path(__file__).resolve().parent / "weights" / "image-classifier-mnist.pth"

    cmap: str | None = "gray"

    def __post_init__(self):
        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mnist.MEAN, std=mnist.STD),
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
    def index_to_class(index: int) -> str:
        class_name = mnist.index_to_class(index)
        return class_name

    @staticmethod
    def image_to_tensor(x: np.ndarray) -> torch.Tensor:
        tensor = torch.tensor(mnist.normalize(x), dtype=torch.float32).unsqueeze(0)
        return tensor


@dataclasses.dataclass
class ImageClassifierCifar10Config(ImageClassifierConfig):
    """
    CIFAR-10 settings.
    """

    num_classes: int = 10

    patch_size: int = 4

    num_layers: int = 6
    num_heads: int = 4
    embedded_dimension: int = 256
    feed_forward_dimension: int = 256 * 4

    dropout: float = 0.2
    attention_dropout: float = 0.2

    batch_size: int = 128
    learning_rate: float = 3e-4
    training_iterations: int = 5000
    loss_function: typing.Type[nn.Module] = nn.CrossEntropyLoss

    model_path: pathlib.Path = pathlib.Path(__file__).resolve().parent / "weights" / "image-classifier-cifar10.pth"

    cmap: str | None = None

    def __post_init__(self):
        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=cifar10.MEAN, std=cifar10.STD),
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
    def index_to_class(index: int) -> str:
        class_name = cifar10.index_to_class(index)
        return class_name

    @staticmethod
    def image_to_tensor(x: np.ndarray) -> torch.Tensor:
        tensor = torch.tensor(cifar10.normalize(x), dtype=torch.float32).unsqueeze(0)
        return tensor


class ImageClassifier(application.Application):
    """
    Image classifier using ViT model as backend.
    """

    _config: typing.Union[ImageClassifierMnistConfig, ImageClassifierCifar10Config]

    def __init__(
        self,
        dataset_name: str = "mnist",
    ):
        match dataset_name:
            case "mnist":
                self._config = ImageClassifierMnistConfig()
            case "cifar10":
                self._config = ImageClassifierCifar10Config()
            case _:
                assert False

        channels, height, width = self._config.train_dataset[0][0].shape
        model = vit.ViT(
            num_classes=self._config.num_classes,
            channels=channels,
            height=height,
            width=width,
            patch_size=self._config.patch_size,
            num_layers=self._config.num_layers,
            num_heads=self._config.num_heads,
            embed_dim=self._config.embedded_dimension,
            feed_forward_dim=self._config.feed_forward_dimension,
            dropout=self._config.dropout,
            attention_dropout=self._config.attention_dropout,
        )

        super().__init__(
            dataset=self._config.val_dataset,
            model=model,
            batch_size=self._config.batch_size,
            training_iterations=self._config.training_iterations,
            default_model_file_path=self._config.model_path,
        )

    def __call__(
        self,
        image: np.ndarray,
        device: str = "cpu",
    ) -> str:
        """
        Classify an image.
        """
        tensor = self._config.image_to_tensor(image).to(device)
        pred_label = self.model.classify(tensor, device=device)
        pred_class = self._config.index_to_class(pred_label)
        return pred_class

    def train(
        self,
        device: str,
    ) -> typing.Generator[float, None, None]:
        """
        Train the model.
        """
        data_loader = torchdata.DataLoader(self._config.train_dataset, batch_size=self.batch_size, shuffle=True)
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
        data_loader = torchdata.DataLoader(self._config.val_dataset, batch_size=self.batch_size, shuffle=False)
        evaluator = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._config.loss_function(reduction="sum"),
            device=device,
        )

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for y_true, y_pred, batch_loss in evaluator:
            batch_size = y_true.size(0)
            y_pred = torch.argmax(y_pred, dim=-1)
            total_correct += (y_pred == y_true).sum().item()
            total_samples += batch_size
            total_loss += batch_loss.item()

        loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100

        return {
            "val loss": loss,
            "accuracy": accuracy,
        }
