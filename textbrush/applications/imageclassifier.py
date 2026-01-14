"""
Hand-written digit image classifier.
"""

import matplotlib.pyplot as plt

from textbrush.datasets import mnist


class ImageClassifier:
    """
    Image classifier using ViT model as backend.
    """

    def __init__(self):
        self.dataset = mnist.Mnist(train=True)

    def __call__(self) -> None:
        image, label = self.dataset[0]
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()
