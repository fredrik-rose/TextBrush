"""
The text brush entry point.
"""

import argparse

import torch

from torch import nn

from textbrush.datasets import tinyshakespeare
from textbrush.models import gpt
from textbrush.optimizers import modeltrainer


class TextBrushHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    Argument parser formatter.
    """


def main():
    """
    Entry point.
    """
    parse()

    dataset = tinyshakespeare.TinyShakespeare(train=True)
    model = gpt.GPT(vocab_size=dataset.vocab_size, embed_dim=16)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = modeltrainer.train_model(
        model=model,
        dataset=dataset,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=1,
        batch_size=32,
    )

    x, _ = dataset[0]
    output = model(x)

    print(f"Input: {dataset.decode(x.tolist())}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    for i, loss in enumerate(trainer):
        if i > 1000:
            break
        if i % 100 == 0:
            print(loss)


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


if __name__ == "__main__":
    main()
