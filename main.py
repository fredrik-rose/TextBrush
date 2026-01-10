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
    prompt = "\n"

    print(generate_text(prompt, dataset, model, 100))
    train_model(dataset, model)
    print(generate_text(prompt, dataset, model, 1000))


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


def generate_text(prompt, dataset, model, length):
    """
    Generate text given a prompt.
    """
    tokens = dataset.encode(prompt)
    generator = model.generate(tokens)
    text = dataset.decode(next(generator) for _ in range(length))
    return text


def train_model(dataset, model):
    """
    Train a GPT model on a dataset.
    """
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
    for i, loss in enumerate(trainer):
        if i > 1000:
            break
        if i % 100 == 0:
            print(loss)


if __name__ == "__main__":
    main()
