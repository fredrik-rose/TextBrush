"""
The text brush entry point.
"""

import argparse
import datetime
import time

import torch
import torch.utils.data as torchdata

from torch import nn

from textbrush.applications import textgenerator
from textbrush.datasets import split as dataset_spliter
from textbrush.optimizers import modeltrainer

DATASET_SPLIT = 0.999
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
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

    text_generator = textgenerator.Textgenerator()
    prompt = "\n"

    print(text_generator(prompt, TEXT_GENERATION_LENGTH // 10))
    train_model(text_generator.model, text_generator.dataset)
    print(text_generator(prompt, TEXT_GENERATION_LENGTH))


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


def train_model(model, dataset):
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

    device = get_device()
    num_params = get_num_parameters(model)

    train_dataset, validation_dataset = dataset_spliter.split_ordered(dataset, [DATASET_SPLIT, (1.0 - DATASET_SPLIT)])

    train_data_loader = torchdata.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_data_loader = torchdata.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_function = FlattenedCrossEntropy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    trainer = modeltrainer.train_model(
        model=model,
        data_loader=train_data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
    )

    print(
        f"\nStarting training | Device: {device} | #Params: {num_params / 1e6:.2f}M | "
        f"Iterations: {MAX_TRAINING_ITERATIONS} | Batch size: {BATCH_SIZE} | Learning rate: {LEARNING_RATE}"
    )
    step_size = MAX_TRAINING_ITERATIONS // 10
    start = time.time()
    t0 = start
    total_loss = 0.0
    for i in range(MAX_TRAINING_ITERATIONS):
        total_loss += next(trainer)
        if i % step_size == (step_size - 1):
            dt = time.time() - t0
            tokens_per_sec = (step_size * BATCH_SIZE * textgenerator.MAX_TOKENS) / dt
            val_loss = modeltrainer.eval_model(
                model=model,
                data_loader=validation_data_loader,
                loss_function=loss_function,
                device=device,
            )
            print(
                f"train loss: {total_loss / step_size:.4f} | val loss: {val_loss:.4f} | dt: {dt:.2f}s | "
                f"tokens/sec: {tokens_per_sec:.2f}"
            )
            t0 = time.time()
            total_loss = 0.0
    elapsed_time = round(time.time() - start)
    print(f"Training finished | Time: {datetime.timedelta(seconds=elapsed_time)}\n")


def get_device():
    """
    get the "best" available device.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def get_num_parameters(model):
    """
    get the number of parameters of a model.
    """
    num_parameters = sum(p.numel() for p in model.parameters())
    return num_parameters


if __name__ == "__main__":
    main()
