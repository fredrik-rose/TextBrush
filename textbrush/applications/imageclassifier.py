"""
Hand-written digit image classifier.
"""

import pathlib
import types
import typing

import matplotlib.pyplot as plt
import torch
import torch.utils.data as torchdata

from torch import nn
from torchvision.transforms import v2

from textbrush.datasets import cifar10
from textbrush.datasets import mnist
from textbrush.models import vit
from textbrush.optimizers import modeltrainer

from . import application

NUM_CLASSES = 10

PATCH_SIZE = 4
NUM_LAYERS = 6
NUM_HEADS = 4
EMBEDDED_DIMENSION = 256
FEED_FORWARD_DIMENSION = EMBEDDED_DIMENSION * 4

DROPOUT = 0.2
ATTENTION_DROPOUT = DROPOUT

BATCH_SIZE = 128
LEARNING_RATE = 3e-4
TRAINING_ITERATIONS = 5000

MODEL_PATH_MNIST = pathlib.Path(__file__).resolve().parent / "weights" / "image-classifier-mnist.pth"
MODEL_PATH_CIFAR10 = pathlib.Path(__file__).resolve().parent / "weights" / "image-classifier-cifar10.pth"


class ImageClassifier(application.Application):
    """
    Image classifier using ViT model as backend.
    """

    _dataset_module: types.ModuleType
    _dataset_class: typing.Union[type[mnist.Mnist], type[cifar10.Cifar10]]
    _cmap: str | None

    def __init__(
        self,
        dataset_name: str = "mnist",
    ):
        match dataset_name:
            case "mnist":
                _model_path = MODEL_PATH_MNIST
                self._dataset_module = mnist
                self._dataset_class = mnist.Mnist
                self._cmap = "gray"
            case "cifar10":
                _model_path = MODEL_PATH_CIFAR10
                self._dataset_module = cifar10
                self._dataset_class = cifar10.Cifar10
                self._cmap = None
            case _:
                assert False

        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self._dataset_module.MEAN, std=self._dataset_module.STD),
            ]
        )
        dataset = self._dataset_class(
            transform=image_transform,
            train=True,
        )
        channels, height, width = dataset[0][0].shape
        model = vit.ViT(
            num_classes=NUM_CLASSES,
            channels=channels,
            height=height,
            width=width,
            patch_size=PATCH_SIZE,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            embed_dim=EMBEDDED_DIMENSION,
            feed_forward_dim=FEED_FORWARD_DIMENSION,
            dropout=DROPOUT,
            attention_dropout=ATTENTION_DROPOUT,
        )

        self._image_transform = image_transform
        self._loss_function = nn.CrossEntropyLoss

        super().__init__(
            dataset=dataset,
            model=model,
            default_model_file_path=_model_path,
        )

    def __call__(
        self,
        num_images: int,
        device: str = "cpu",
    ) -> None:
        """
        Classify images.
        """
        data_loader = torchdata.DataLoader(
            self._dataset_class(
                transform=self._image_transform,
                train=False,
            ),
            batch_size=1,
            shuffle=True,
        )
        for i, (image_tensor, true_label) in enumerate(data_loader):
            if i >= num_images:
                break
            pred_label = self.model.classify(image_tensor[0], device=device)
            true_class = self._dataset_module.index_to_class(true_label[0].item())
            pred_class = self._dataset_module.index_to_class(pred_label)
            image = self._dataset_module.denormalize(self._dataset_module.tensor_to_image(image_tensor))
            plt.imshow(image, cmap=self._cmap)
            plt.title(f"True: {true_class}, Predicted: {pred_class}")
            plt.axis("off")
            plt.show()

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
        validation_dataset = self._dataset_class(
            transform=self._image_transform,
            train=False,
        )
        data_loader = torchdata.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        evaluator = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._loss_function(reduction="sum"),
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
