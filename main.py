"""
The text brush entry point.
"""

import argparse
import datetime
import time

import torch

from textbrush.applications import textgenerator

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
    num_tokens_in_batch = textgenerator.BATCH_SIZE * textgenerator.MAX_TOKENS
    prompt = "\n"

    print(text_generator(prompt, TEXT_GENERATION_LENGTH // 10))
    train_application(text_generator, num_tokens_in_batch)
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


def train_application(application, num_tokens_in_batch):
    """
    Train an application.
    """
    device = get_device()
    num_params = get_num_parameters(application.model)

    trainer = application.train(device)

    print(
        f"\nStarting training | Device: {device} | #Params: {num_params / 1e6:.2f}M | "
        f"Iterations: {MAX_TRAINING_ITERATIONS}"
    )
    step_size = MAX_TRAINING_ITERATIONS // 10
    start = time.time()
    t0 = start
    total_loss = 0.0
    for i in range(MAX_TRAINING_ITERATIONS):
        total_loss += next(trainer)
        if i % step_size == (step_size - 1):
            dt = time.time() - t0
            tokens_per_sec = (step_size * num_tokens_in_batch) / dt
            val_loss = application.eval(device)
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
