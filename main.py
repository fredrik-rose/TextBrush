"""
The text brush entry point.
"""

import argparse
import datetime
import pathlib
import tempfile
import time
import typing

import netron
import torch
import torchinfo

from torch import nn
from torch import onnx

from textbrush.applications import imageclassifier
from textbrush.applications import textgenerator

MAX_TRAINING_ITERATIONS = 5000
TEXT_GENERATION_LENGTH = 1000
NUM_IMAGES = 5


class TextBrushHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    Argument parser formatter.
    """


def main() -> None:
    """
    Entry point.
    """
    args = parse()

    device = get_device()

    match args.application:
        case "text":
            text_generator_application(args, device)
        case "image":
            image_classifier_application(args, device)
        case _:
            assert False


def parse() -> argparse.Namespace:
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
        "-v",
        "--visualize-model",
        action="store_true",
        help="visualize the model of the application",
    )

    subparsers = parser.add_subparsers(dest="application", help="Application", required=True)

    text_generator_parser = subparsers.add_parser("text", help="Shakespeare text generator")
    text_generator_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt",
        default=None,
    )
    text_generator_parser.add_argument(
        "-n",
        type=int,
        help="length (number of characters) of text to generate",
        default=TEXT_GENERATION_LENGTH,
    )

    image_classifier_parser = subparsers.add_parser("image", help="Hand-written digit classifier")
    image_classifier_parser.add_argument(
        "-n",
        type=int,
        help="number of images to classify",
        default=NUM_IMAGES,
    )

    args = parser.parse_args()
    return args


def get_device() -> str:
    """
    get the "best" available device.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def text_generator_application(
    args: argparse.Namespace,
    device: str,
) -> None:
    """
    Text generator application.
    """
    if args.train:
        text_generator = textgenerator.Textgenerator()
        num_tokens_in_batch = textgenerator.BATCH_SIZE * textgenerator.MAX_TOKENS
        train_application(text_generator, num_tokens_in_batch, device, textgenerator.MODEL_PATH)
        return

    text_generator = textgenerator.Textgenerator(textgenerator.MODEL_PATH)

    if args.visualize_model:
        visualize_model(text_generator.model, torch.unsqueeze(text_generator.dataset[0][0], 0))
        return

    prompt = "\n" if args.prompt is None else args.prompt
    for char in text_generator(prompt, args.n, device):
        print(char, end="", flush=True)


def image_classifier_application(
    args: argparse.Namespace,
    device: str,
) -> None:
    """
    Image classifier application.
    """
    image_classifier = imageclassifier.ImageClassifier()

    if args.visualize_model:
        visualize_model(image_classifier.model, torch.unsqueeze(image_classifier.dataset[0][0], 0))
        return

    image_classifier(args.n, device)


def train_application(
    application: typing.Any,
    num_tokens_in_batch: int,
    device: str,
    output_path: pathlib.Path,
) -> None:
    """
    Train an application.
    """
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
                f"{i + 1}/{MAX_TRAINING_ITERATIONS} | train loss: {total_loss / step_size:.4f} | "
                f"val loss: {val_loss:.4f} | dt: {dt:.2f}s | tokens/sec: {tokens_per_sec:.2f}"
            )
            t0 = time.time()
            total_loss = 0.0
    elapsed_time = round(time.time() - start)
    print(f"Training finished | Time: {datetime.timedelta(seconds=elapsed_time)}\n")


def get_num_parameters(model: nn.Module) -> int:
    """
    get the number of parameters of a model.
    """
    num_parameters = sum(p.numel() for p in model.parameters())
    return num_parameters


def visualize_model(
    model: nn.Module,
    example_input: torch.Tensor,
    depth: int = 7,
) -> None:
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
