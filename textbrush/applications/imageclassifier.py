"""
Hand-written digit image classifier.
"""

import matplotlib.pyplot as plt

from textbrush.datasets import mnist
from textbrush.models import vit

NUM_CLASSES = 10

PATCH_SIZE = 4
NUM_LAYERS = 6
NUM_HEADS = 4
EMBEDDED_DIMENSION = 32
FEED_FORWARD_DIMENSION = EMBEDDED_DIMENSION * 4

DROPOUT = 0.1
ATTENTION_DROPOUT = DROPOUT


class ImageClassifier:
    """
    Image classifier using ViT model as backend.
    """

    def __init__(self):
        self.dataset = mnist.Mnist(train=True)
        channels, height, width = self.dataset[0][0].shape
        self.model = vit.ViT(
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

    def __call__(
        self,
        num_images: int,
        device: str = "cpu",
    ) -> None:
        """
        Classify images.
        """
        val_dataset = mnist.Mnist(train=False)
        for i, (image_tensor, true_label) in enumerate(val_dataset):
            if i >= num_images:
                break
            pred_label = self.model.classify(image_tensor, device=device)
            image = mnist.to_image(image_tensor)
            plt.imshow(image, cmap="gray")
            plt.title(f"True: {true_label}, Predicted: {pred_label}")
            plt.axis("off")
            plt.show()
