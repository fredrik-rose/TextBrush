"""
The text brush entry point.
"""

import argparse
import contextlib
import datetime
import time

import torch
import torch.utils.data as torchdata
import torchinfo
import torchview

from torch import nn

from textbrush.applications import application as app
from textbrush.applications import imageclassifier
from textbrush.applications import imagegenerator
from textbrush.applications import textgenerator

EPOCHS = 10
DEFAULT_TEXT_GENERATION_LENGTH = 1000
DEFAULT_DIGIT = 2
DEFAULT_NUM_IMAGES = 5

ALL_APPLICATIONS = {
    "text": textgenerator.TextGenerator(),
    "image": imagegenerator.ImageGenerator(),
    "class": imageclassifier.ImageClassifier(),
}

ALL_TRAINING_ITERATIONS = {
    "text": textgenerator.TRAINING_ITERATIONS,
    "image": imagegenerator.TRAINING_ITERATIONS,
    "class": imageclassifier.TRAINING_ITERATIONS,
}

ALL_BATCH_SIZES = {
    "text": textgenerator.BATCH_SIZE,
    "image": imagegenerator.BATCH_SIZE,
    "class": imageclassifier.BATCH_SIZE,
}


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
    application = ALL_APPLICATIONS[args.application]

    if args.train:
        training_iterations = ALL_TRAINING_ITERATIONS[args.application]
        num_tokens_in_batch = ALL_BATCH_SIZES[args.application] * application.model.max_num_tokens
        train_application_model(application, training_iterations, num_tokens_in_batch, device)
        return

    try:
        application.load()
    except FileNotFoundError:
        print(f"Could not load application '{args.application}', please train it with --train")
        return

    if args.visualize_model:
        visualize_application_model(application)
        return

    match args.application:
        case "text":
            prompt = "\n" if args.prompt is None else args.prompt
            for char in application(prompt=prompt, length=args.n, device=device):  # type: ignore[operator]
                print(char, end="", flush=True)
        case "image":
            application(digit=args.digit, device=device)  # type: ignore[operator]
        case "class":
            application(num_images=args.n, device=device)  # type: ignore[operator]
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

    image_generator_parser = subparsers.add_parser("image", help="Hand-written digit image generator")
    image_generator_parser.add_argument(
        "-d",
        "--digit",
        type=int,
        choices=range(10),
        metavar="[0-9]",
        help="digit to generate",
        default=DEFAULT_DIGIT,
    )

    image_classifier_parser = subparsers.add_parser("class", help="Hand-written digit classifier")
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
    Get the "best" available device.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def train_application_model(
    application: app.Application,
    iterations: int,
    num_tokens_in_batch: int,
    device: str,
) -> None:
    """
    Train an application.
    """
    epoch_size = iterations // EPOCHS
    trainer = application.train(device)
    best_loss = float("inf")

    print(
        f"\nStarting training | "
        f"Device: {device} |"
        f" #Params: {get_num_parameters(application.model) / 1e6:.2f}M | "
        f"Iterations: {iterations}"
    )

    with time_it() as total_time:
        for e in range(EPOCHS):
            with time_it() as train_time:
                # Note that this may introduce a small bias if all batches do not have the same size.
                train_loss = sum(next(trainer) for _ in range(epoch_size)) / epoch_size
            tokens_per_sec = (epoch_size * num_tokens_in_batch) / train_time["elapsed"]

            with time_it() as val_time:
                metrics = application.eval(device)
                val_loss = metrics["val loss"]
                if val_loss < best_loss:
                    best_loss = val_loss
                    application.save()

            metrics_string = " | ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
            print(
                f"{e + 1}/{EPOCHS} | "
                f"train loss: {train_loss:.4f} | "
                f"{metrics_string} | "
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


def visualize_application_model(
    application: app.Application,
    depth: int = 7,
) -> None:
    """
    Visualize an application model.
    """
    model = application.model
    example_input = next(iter(torchdata.DataLoader(application.dataset, batch_size=1)))[0]

    torchinfo.summary(model, input_data=example_input, depth=depth)

    model_graph = torchview.draw_graph(
        model,
        input_size=example_input.shape,
        depth=depth,
        device="meta",
    )
    output_path = model_graph.visual_graph.render(
        f"{application.__class__.__name__}_model",
        format="png",
        cleanup=True,
    )
    print(f"Stored model visualization at: {output_path}")


if __name__ == "__main__":
    main()
