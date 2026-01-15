"""
The text brush entry point.
"""

import argparse
import contextlib
import datetime
import pathlib
import tempfile
import time

import netron
import torch
import torchinfo

from torch import nn
from torch import onnx

from textbrush.applications import application as app
from textbrush.applications import imageclassifier
from textbrush.applications import textgenerator

TRAINING_ITERATIONS = 5000
EPOCHS = 10
DEFAULT_TEXT_GENERATION_LENGTH = 1000
DEFAULT_NUM_IMAGES = 5


@contextlib.contextmanager
def time_it():
    """
    Context manager for measuring time.
    """
    start = time.time()
    state = {}
    try:
        yield state
    finally:
        state["elapsed"] = time.time() - start


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
        default=DEFAULT_TEXT_GENERATION_LENGTH,
    )

    image_classifier_parser = subparsers.add_parser("image", help="Hand-written digit classifier")
    image_classifier_parser.add_argument(
        "-n",
        type=int,
        help="number of images to classify",
        default=DEFAULT_NUM_IMAGES,
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
    text_generator = textgenerator.Textgenerator()

    if args.train:
        num_tokens_in_batch = textgenerator.BATCH_SIZE * textgenerator.MAX_TOKENS
        train_application(text_generator, num_tokens_in_batch, device)
        return

    text_generator.load()

    if args.visualize_model:
        visualize_model(text_generator.model, torch.unsqueeze(text_generator.dataset[0][0], dim=0))
        return

    prompt = "\n" if args.prompt is None else args.prompt
    for char in text_generator(prompt=prompt, length=args.n, device=device):
        print(char, end="", flush=True)


def image_classifier_application(
    args: argparse.Namespace,
    device: str,
) -> None:
    """
    Image classifier application.
    """
    image_classifier = imageclassifier.ImageClassifier()

    if args.train:
        _, height, width = image_classifier.dataset[0][0].shape
        patch_size = imageclassifier.PATCH_SIZE
        num_tokens_in_batch = imageclassifier.BATCH_SIZE * ((height // patch_size) * (width // patch_size) + 1)
        train_application(image_classifier, num_tokens_in_batch, device)
        return

    image_classifier.load()

    if args.visualize_model:
        visualize_model(image_classifier.model, torch.unsqueeze(image_classifier.dataset[0][0], dim=0))
        return

    image_classifier(num_images=args.n, device=device)


def train_application(
    application: app.Application,
    num_tokens_in_batch: int,
    device: str,
) -> None:
    """
    Train an application.
    """
    epoch_size = TRAINING_ITERATIONS // EPOCHS
    trainer = application.train(device)
    best_loss = float("inf")

    print(
        f"\nStarting training | "
        f"Device: {device} |"
        f" #Params: {get_num_parameters(application.model) / 1e6:.2f}M | "
        f"Iterations: {TRAINING_ITERATIONS}"
    )

    with time_it() as total_time:
        for e in range(EPOCHS):
            with time_it() as train_time:
                train_loss = sum(next(trainer) for _ in range(epoch_size)) / epoch_size
            tokens_per_sec = (epoch_size * num_tokens_in_batch) / train_time["elapsed"]

            with time_it() as val_time:
                val_loss = application.eval(device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    application.save()

            print(
                f"{e + 1}/{EPOCHS} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"train time: {train_time['elapsed']:.2f}s | "
                f"val time: {val_time['elapsed']:.2f}s | "
                f"tokens/sec: {tokens_per_sec:.2f}"
            )

    print(
        f"Training finished | "
        f"Time: {datetime.timedelta(seconds=round(total_time['elapsed']))} | "
        f"Best loss: {best_loss:.4f}\n"
    )


def get_num_parameters(model: nn.Module) -> int:
    """
    Get the number of parameters of a model.
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
        onnx.export(model, (example_input,), model_path, input_names=["input"], output_names=["output"])
        netron.start(str(model_path))
        netron.wait()


if __name__ == "__main__":
    main()
