"""
The text brush entry point.
"""

import argparse

from textbrush.datasets import tinyshakespeare
from textbrush.models import gpt


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

    x, _ = dataset[0]
    output = model(x)

    print(f"Input: {dataset.decode(x.tolist())}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


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
