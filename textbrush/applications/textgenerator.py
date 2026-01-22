"""
Shakespeare text generator.
"""

import pathlib
import typing

import torch
import torch.utils.data as torchdata

from torch import nn

from textbrush.datasets import split as dataset_spliter
from textbrush.datasets import tinyshakespeare
from textbrush.models import gpt
from textbrush.optimizers import modeltrainer

from . import application

MAX_TOKENS = 128
NUM_LAYERS = 6
NUM_HEADS = 4
EMBEDDED_DIMENSION = 256
FEED_FORWARD_DIMENSION = EMBEDDED_DIMENSION * 4

DROPOUT = 0.2
ATTENTION_DROPOUT = DROPOUT

DATASET_SPLIT = 0.99

BATCH_SIZE = 64
LEARNING_RATE = 3e-4

TOP_K = 10

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "text-generator.pth"


class TextGenerator(application.Application):
    """
    Text generator using a GPT model as backend.
    """

    def __init__(self):
        self.tokenizer = tinyshakespeare.Tokenizer()

        self._split = [DATASET_SPLIT, (1.0 - DATASET_SPLIT)]
        self._loss_function = FlattenedCrossEntropy()

        dataset = tinyshakespeare.TinyShakespeare(
            tokenizer=self.tokenizer,
            block_size=MAX_TOKENS,
        )
        model = gpt.GPT(
            vocab_size=self.tokenizer.vocab_size,
            num_tokens=MAX_TOKENS,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            embed_dim=EMBEDDED_DIMENSION,
            feed_forward_dim=FEED_FORWARD_DIMENSION,
            dropout=DROPOUT,
            attention_dropout=ATTENTION_DROPOUT,
        )

        super().__init__(
            dataset=dataset,
            model=model,
            default_model_file_path=MODEL_PATH,
        )

    def __call__(
        self,
        prompt: str,
        length: int,
        device: str = "cpu",
    ) -> typing.Generator[str, None, None]:
        """
        Generate text given a prompt.
        """
        tokens = self.tokenizer.encode(prompt)  # type: ignore[attr-defined]
        generator = self.model.generate(tokens, k=TOP_K, device=device)
        yield prompt
        for _ in range(length):
            try:
                yield self.tokenizer.decode([next(generator)])  # type: ignore[attr-defined]
            except StopIteration:
                assert False
        yield "\n"

    def train(
        self,
        device: str,
    ) -> typing.Generator[float, None, None]:
        """
        Train the model.
        """
        train_dataset, _ = dataset_spliter.split_ordered(self.dataset, self._split)
        data_loader = torchdata.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        yield from modeltrainer.train_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._loss_function,
            optimizer=optimizer,
            device=device,
        )

    def eval(
        self,
        device: str,
    ) -> float:
        """
        Evaluate the model in the validation dataset.
        """
        _, validation_dataset = dataset_spliter.split_ordered(self.dataset, self._split)
        data_loader = torchdata.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validation_loss = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._loss_function,
            device=device,
        )
        return validation_loss


class FlattenedCrossEntropy(nn.Module):
    """
    Adjust dimensions to use the ordinary cross entropy loss.
    """

    def __init__(self):
        super().__init__()

        self._loss = nn.CrossEntropyLoss()

    def forward(  # pylint: disable=missing-function-docstring
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, classes = y_pred.shape
        y_pred = y_pred.reshape(batch * tokens, classes)  # (B, T, C) -> (B*T, C)
        y_true = y_true.reshape(batch * tokens)  # (B, T) -> (B*T)
        return self._loss(y_pred, y_true)
