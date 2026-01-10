"""
The text brush entry point.
"""

import argparse

from textbrush.datasets import tinyshakespeare


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
    print(dataset.decode(dataset[0][0]))


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
