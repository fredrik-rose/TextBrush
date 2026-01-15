"""
Hand-written digit image classifier.
"""

import pathlib
import typing

import matplotlib.pyplot as plt
import torch
import torch.utils.data as torchdata

from torch import nn

from textbrush.datasets import mnist
from textbrush.models import vit
from textbrush.optimizers import modeltrainer

from . import application

NUM_CLASSES = 10

PATCH_SIZE = 4
NUM_LAYERS = 6
NUM_HEADS = 4
EMBEDDED_DIMENSION = 32
FEED_FORWARD_DIMENSION = EMBEDDED_DIMENSION * 4

DROPOUT = 0.1
ATTENTION_DROPOUT = DROPOUT

BATCH_SIZE = 32
LEARNING_RATE = 3e-4

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "image-classifier.pth"


class ImageClassifier(application.Application):
    """
    Image classifier using ViT model as backend.
    """

    def __init__(self):
        dataset = mnist.Mnist(train=True)
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
        super().__init__(
            dataset=dataset,
            model=model,
            default_model_file_path=MODEL_PATH,
        )

    def __call__(
        self,
        num_images: int,
        device: str = "cpu",
    ) -> None:
        """
        Classify images.
        """
        data_loader = torchdata.DataLoader(mnist.Mnist(train=False), batch_size=1, shuffle=True)
        for i, (image_tensor, true_label) in enumerate(data_loader):
            if i >= num_images:
                break
            pred_label = self.model.classify(image_tensor[0], device=device)
            image = mnist.to_image(image_tensor)
            plt.imshow(image, cmap="gray")
            plt.title(f"True: {true_label[0]}, Predicted: {pred_label}")
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
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
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
    ) -> float:
        """
        Evaluate the model in the validation dataset.
        """
        validation_dataset = mnist.Mnist(train=False)
        data_loader = torchdata.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
        loss_function = nn.CrossEntropyLoss()
        validation_loss = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=loss_function,
            device=device,
        )
        return validation_loss
