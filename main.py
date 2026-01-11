"""
The text brush entry point.
"""

import argparse
import datetime
import time

import torch

from torch import nn

from textbrush.datasets import tinyshakespeare
from textbrush.models import gpt
from textbrush.optimizers import modeltrainer

MAX_TOKENS = 8
NUM_HEADS = 2
EMBEDDED_DIMENSION = 16
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100
MAX_TRAINING_ITERATIONS = 1000
TEXT_GENERATION_LENGTH = 1000


class TextBrushHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    Argument parser formatter.
    """


def main():
    """
    Entry point.
    """
    parse()

    device = get_device()
    print(f"Using '{device}' device.")
    dataset = tinyshakespeare.TinyShakespeare(train=True, block_size=MAX_TOKENS)
    model = gpt.GPT(
        vocab_size=dataset.vocab_size, num_tokens=MAX_TOKENS, num_heads=NUM_HEADS, embed_dim=EMBEDDED_DIMENSION
    )
    prompt = "\n"

    print(generate_text(prompt, dataset, model, TEXT_GENERATION_LENGTH // 10))
    train_model(model, dataset, device)
    print(generate_text(prompt, dataset, model, TEXT_GENERATION_LENGTH))


def parse():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Text Brush application.",
        formatter_class=TextBrushHelpFormatter,
    )
    args = parser.parse_args()
    return args


def get_device():
    """
    get the "best" available device.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def generate_text(prompt, dataset, model, length):
    """
    Generate text given a prompt.
    """
    tokens = dataset.encode(prompt)
    generator = model.generate(tokens)
    text = dataset.decode(next(generator) for _ in range(length))
    return text


def train_model(model, dataset, device):
    """
    Train a GPT model on a dataset.
    """

    class FlattenedCrossEntropy(nn.Module):
        """
        Adjust dimensions to use the ordinary cross entropy loss.
        """

        def __init__(self):
            super().__init__()
            self.loss = nn.CrossEntropyLoss()

        def forward(self, y_pred, y_true):  # pylint: disable=missing-function-docstring
            batch, tokens, classes = y_pred.shape
            y_pred = y_pred.reshape(batch * tokens, classes)  # (B, T, C) -> (B*T, C)
            y_true = y_true.reshape(batch * tokens)  # (B, T) -> (B*T)
            return self.loss(y_pred, y_true)

    loss_function = FlattenedCrossEntropy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    trainer = modeltrainer.train_model(
        model=model,
        dataset=dataset,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=device,
    )
    print(
        f"\nStarting training | Epochs: {EPOCHS} | Max iterations: {MAX_TRAINING_ITERATIONS} | "
        f"Batch size: {BATCH_SIZE} | Samples: {len(dataset)} | Learning rate: {LEARNING_RATE}"
    )
    start = time.time()
    for i, loss in enumerate(trainer):
        if i > MAX_TRAINING_ITERATIONS:
            break
        if i % (MAX_TRAINING_ITERATIONS // 10) == 0:
            print(loss)
    elapsed_time = round(time.time() - start)
    print(f"Training finished | Time: {datetime.timedelta(seconds=elapsed_time)}\n")


if __name__ == "__main__":
    main()
