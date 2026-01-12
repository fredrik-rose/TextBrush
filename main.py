"""
The text brush entry point.
"""

import argparse
import datetime
import time

import torch
import torch.utils.data as torchdata

from torch import nn

from textbrush.datasets import tinyshakespeare
from textbrush.models import gpt
from textbrush.optimizers import modeltrainer

DATASET_SPLIT = 0.999
MAX_TOKENS = 8
NUM_LAYERS = 3
NUM_HEADS = 2
EMBEDDED_DIMENSION = 32
FEED_FORWARD_DIMENSION = EMBEDDED_DIMENSION * 4
DROPOUT = 0.2
ATTENTION_DROPOUT = DROPOUT
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

    dataset = tinyshakespeare.TinyShakespeare(block_size=MAX_TOKENS)
    model = gpt.GPT(
        vocab_size=dataset.vocab_size,
        num_tokens=MAX_TOKENS,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        embed_dim=EMBEDDED_DIMENSION,
        feed_forward_dim=FEED_FORWARD_DIMENSION,
        dropout=DROPOUT,
        attention_dropout=ATTENTION_DROPOUT,
    )
    prompt = "\n"

    print(generate_text(prompt, dataset, model, TEXT_GENERATION_LENGTH // 10))
    train_model(model, dataset)
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

    size = len(dataset)
    split = int(size * DATASET_SPLIT)

    train_dataset = torchdata.Subset(dataset, range(0, split))
    validation_dataset = torchdata.Subset(dataset, range(split, size))

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
            tokens_per_sec = (step_size * BATCH_SIZE * MAX_TOKENS) / dt
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


def generate_text(prompt, dataset, model, length):
    """
    Generate text given a prompt.
    """
    tokens = dataset.encode(prompt)
    generator = model.generate(tokens)
    text = dataset.decode(next(generator) for _ in range(length))
    return text


if __name__ == "__main__":
    main()
