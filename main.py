"""
The text brush entry point.
"""

import argparse
import datetime
import pathlib
import tempfile
import time

import netron
import torch
import torchinfo

from torch import onnx

from textbrush.applications import textgenerator

MAX_TRAINING_ITERATIONS = 5000
TEXT_GENERATION_LENGTH = 5000


class TextBrushHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    Argument parser formatter.
    """


def main():
    """
    Entry point.
    """
    args = parse()

    if args.train:
        text_generator = textgenerator.Textgenerator()
        num_tokens_in_batch = textgenerator.BATCH_SIZE * textgenerator.MAX_TOKENS
        train_application(text_generator, num_tokens_in_batch, textgenerator.MODEL_PATH)
        return

    text_generator = textgenerator.Textgenerator(textgenerator.MODEL_PATH)

    if args.visualize_model:
        visualize_model(text_generator.model, torch.unsqueeze(text_generator.dataset[0][0], 0))
        return

    for char in text_generator(args.prompt, args.n):
        print(char, end="", flush=True)


def parse():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Text Brush application.",
        formatter_class=TextBrushHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train the model of the application",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt",
        default=r"\n",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="length (number of characters) of text to generate",
        default=TEXT_GENERATION_LENGTH,
    )
    parser.add_argument(
        "-v",
        "--visualize-model",
        action="store_true",
        help="visualize the model of the application",
    )
    args = parser.parse_args()
    return args


def train_application(application, num_tokens_in_batch, output_path):
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
    best_loss = float("inf")
    total_loss = 0.0
    for i in range(MAX_TRAINING_ITERATIONS):
        total_loss += next(trainer)
        if i % step_size == (step_size - 1):
            dt = time.time() - t0
            tokens_per_sec = (step_size * num_tokens_in_batch) / dt
            val_loss = application.eval(device)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(application.model.state_dict(), output_path)
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


def visualize_model(model, example_input, depth: int = 7):
    """
    Visualize a model using the Netron application.
    """
    torchinfo.summary(model, input_data=example_input, depth=depth)
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = pathlib.Path(temp_dir) / "model.onnx"
        onnx.export(model, example_input, model_path, input_names=["input"], output_names=["output"])
        netron.start(str(model_path))
        netron.wait()


if __name__ == "__main__":
    main()
