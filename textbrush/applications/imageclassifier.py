"""
Image classifier.
"""

import abc
import dataclasses
import pathlib
import string
import typing

import numpy as np
import torch
import torch.utils.data as torchdata

from torch import nn
from torchvision.transforms import v2

from textbrush.algorithms import tokenizer as tok
from textbrush.datasets import cifar10
from textbrush.datasets import mnist
from textbrush.models import bert
from textbrush.models import clip
from textbrush.models import transformer
from textbrush.models import vit
from textbrush.optimizers import losses
from textbrush.optimizers import modeltrainer

from . import application

_PAD_TOKEN = "*"


@dataclasses.dataclass
class ImageClassifierConfig:
    """
    Settings.
    """

    num_classes: int
    class_names: tuple[str, ...]
    representation_dim: int

    prefix: str
    max_tokens: int
    text_encoder_config: transformer.TransformerConfig

    patch_size: int
    image_encoder_config: transformer.TransformerConfig

    batch_size: int
    learning_rate: float
    training_iterations: int

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
    class_names: tuple[str, ...] = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    representation_dim: int = 128

    prefix: str = "a handwritten digit"
    max_tokens: int = 127
    text_encoder_config: transformer.TransformerConfig = dataclasses.field(
        default_factory=lambda: transformer.TransformerConfig(
            num_layers=6,
            num_heads=4,
            embed_dim=256,
            feed_forward_dim=256 * 4,
            dropout=0.2,
            attention_dropout=0.2,
        )
    )

    patch_size: int = 4
    image_encoder_config: transformer.TransformerConfig = dataclasses.field(
        default_factory=lambda: transformer.TransformerConfig(
            num_layers=6,
            num_heads=4,
            embed_dim=256,
            feed_forward_dim=256 * 4,
            dropout=0.2,
            attention_dropout=0.2,
        )
    )

    batch_size: int = 128
    learning_rate: float = 3e-4
    training_iterations: int = 5000

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
    class_names: tuple[str, ...] = cifar10.CLASSES
    representation_dim: int = 128

    prefix: str = "a photo of a"
    max_tokens: int = 127
    text_encoder_config: transformer.TransformerConfig = dataclasses.field(
        default_factory=lambda: transformer.TransformerConfig(
            num_layers=6,
            num_heads=4,
            embed_dim=256,
            feed_forward_dim=256 * 4,
            dropout=0.2,
            attention_dropout=0.2,
        )
    )

    patch_size: int = 4
    image_encoder_config: transformer.TransformerConfig = dataclasses.field(
        default_factory=lambda: transformer.TransformerConfig(
            num_layers=6,
            num_heads=4,
            embed_dim=256,
            feed_forward_dim=256 * 4,
            dropout=0.2,
            attention_dropout=0.2,
        )
    )

    batch_size: int = 128
    learning_rate: float = 3e-4
    training_iterations: int = 5000

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
    Image classifier using the CLIP approach.
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

        vocab = list(string.ascii_lowercase + " " + _PAD_TOKEN)
        self._tokenizer = tok.Tokenizer(vocab)
        text_encoder = bert.BERT(
            vocab_size=self._tokenizer.vocab_size,
            num_tokens=self._config.max_tokens,
            config=self._config.text_encoder_config,
            representation_dim=self._config.representation_dim,
        )

        channels, height, width = self._config.train_dataset[0][0].shape
        image_encoder = vit.ViT(
            num_classes=self._config.representation_dim,
            channels=channels,
            height=height,
            width=width,
            patch_size=self._config.patch_size,
            config=self._config.image_encoder_config,
        )

        model = clip.CLIP(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
        )

        self._train_dataset = Dataset(
            dataset=self._config.train_dataset,
            prefix=self._config.prefix,
            class_names=self._config.class_names,
            tokenizer=self._tokenizer,
            num_tokens=self._config.max_tokens,
        )
        self._val_dataset = Dataset(
            dataset=self._config.val_dataset,
            prefix=self._config.prefix,
            class_names=self._config.class_names,
            tokenizer=self._tokenizer,
            num_tokens=self._config.max_tokens,
        )

        self._class_embeddings = None

        super().__init__(
            dataset=self._val_dataset,
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
        if self._class_embeddings is None:
            self._class_embeddings = self._get_class_embeddings(device)

        image = self._config.image_to_tensor(image).unsqueeze(0)
        probs = self.model.classify(
            image=image,
            class_embeddings=self._class_embeddings,
            device=device,
        )
        class_index = int(torch.argmax(probs, dim=-1).item())
        pred_class = self._config.index_to_class(class_index)

        return pred_class

    def train(
        self,
        device: str,
    ) -> typing.Generator[float, None, None]:
        """
        Train the model.
        """
        self._class_embeddings = None
        data_loader = torchdata.DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=False)
        loss_function = Loss(logit_scale=self.model.logit_scale, reduction="mean")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._config.learning_rate)
        yield from modeltrainer.train_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

    def eval(
        self,
        device: str,
    ) -> dict[str, float]:
        """
        Evaluate the model on the validation dataset.
        """
        data_loader = torchdata.DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False)
        loss_function = Loss(logit_scale=self.model.logit_scale, reduction="sum")
        evaluator = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=loss_function,
            device=device,
        )

        class_embeddings = self._get_class_embeddings(device)

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for x, y_true, _, batch_loss in evaluator:
            batch_size = y_true.size(0)
            probs = self.model.classify(
                image=x["image"],
                class_embeddings=class_embeddings,
                device=device,
            )
            class_indexes = torch.argmax(probs, dim=-1)
            total_correct += (class_indexes == y_true).sum().item()
            total_samples += batch_size
            total_loss += batch_loss.item()

        loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100

        return {
            "val loss": loss,
            "accuracy": accuracy,
        }

    def _get_class_embeddings(
        self,
        device: str,
    ) -> torch.Tensor:
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            embeddings = []
            for class_name in self._config.class_names:
                text = self._tokenizer.encode(f"{self._config.prefix} {class_name}")
                tokens = torch.tensor(text, dtype=torch.long, device=device).unsqueeze(0)
                class_embedding, _ = self.model(text=tokens, image=None)
                embeddings.append(class_embedding)
            embeddings = torch.vstack(embeddings)
            return embeddings


class Dataset(torchdata.Dataset):
    """
    CLIP dataset wrapper.
    """

    def __init__(
        self,
        dataset: torchdata.Dataset,
        prefix: str,
        class_names: tuple[str, ...],
        tokenizer: tok.Tokenizer,
        num_tokens: int,
    ):
        self._dataset = dataset
        self._prefix = prefix
        self._class_names = class_names
        self._tokenizer = tokenizer
        self._num_tokens = num_tokens

    def __len__(self):
        return len(self._dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        image, label = self._dataset[idx]
        class_name = self._class_names[label]
        text = f"{self._prefix} {class_name}"
        tokens = self._tokenizer.encode(text)
        num_valid_tokens = len(tokens)
        pad_token = self._tokenizer.encode(_PAD_TOKEN)
        tokens += pad_token * (self._num_tokens - num_valid_tokens)
        tokens = torch.tensor(tokens)
        mask = torch.zeros(self._num_tokens)
        mask[:num_valid_tokens] = 1.0
        return {"text": tokens, "image": image, "text_mask": mask}, label


class Loss(nn.Module):
    """
    CLIP loss wrapper.
    """

    def __init__(
        self,
        logit_scale: nn.Parameter,
        reduction: str,
    ):
        super().__init__()
        self._loss = losses.InfoNCE(logit_scale=logit_scale, reduction=reduction)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        y_pred: torch.Tensor,
        _: torch.Tensor,
    ) -> torch.Tensor:
        return self._loss(y_pred[0], y_pred[1])
