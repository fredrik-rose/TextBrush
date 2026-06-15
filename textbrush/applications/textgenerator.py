"""
Shakespeare text generator.
"""

import dataclasses
import math
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


class FlattenedCrossEntropy(nn.Module):
    """
    Adjust dimensions to use the ordinary cross entropy loss.
    """

    def __init__(
        self,
        reduction: str,
    ):
        super().__init__()

        self._loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, classes = y_pred.shape
        y_pred = y_pred.reshape(batch * tokens, classes)  # (B, T, C) -> (B*T, C)
        y_true = y_true.reshape(batch * tokens)  # (B, T) -> (B*T)
        return self._loss(y_pred, y_true)


@dataclasses.dataclass
class TextGeneratorTinyShakespeareConfig:
    """
    Settings.
    """

    max_tokens: int = 128

    num_layers: int = 6
    num_heads: int = 4
    embedded_dimension: int = 256
    feed_forward_dimension: int = 256 * 4

    dropout: float = 0.2
    attention_dropout: float = 0.2

    dataset_split: float = 0.9

    batch_size: int = 64
    learning_rate: float = 3e-4
    training_iterations: int = 5000
    loss_function: typing.Type[nn.Module] = FlattenedCrossEntropy

    model_path: pathlib.Path = pathlib.Path(__file__).resolve().parent / "weights" / "text-generator.pth"

    top_k: int = 10

    def __post_init__(self):
        tokenizer = tinyshakespeare.Tokenizer()
        dataset = tinyshakespeare.TinyShakespeare(
            tokenizer=tokenizer,
            block_size=self.max_tokens,
        )
        split = [self.dataset_split, (1.0 - self.dataset_split)]

        self.train_dataset, self.val_dataset = dataset_spliter.split_ordered(dataset, split)
        self.tokenizer = tokenizer


class TextGenerator(application.Application):
    """
    Text generator using a GPT model as backend.
    """

    def __init__(
        self,
        dataset_name: str = "tiny-shakespeare",
    ):
        assert dataset_name == "tiny-shakespeare"

        self._config = TextGeneratorTinyShakespeareConfig()

        model = gpt.GPT(
            vocab_size=self._config.tokenizer.vocab_size,
            num_tokens=self._config.max_tokens,
            num_layers=self._config.num_layers,
            num_heads=self._config.num_heads,
            embed_dim=self._config.embedded_dimension,
            feed_forward_dim=self._config.feed_forward_dimension,
            dropout=self._config.dropout,
            attention_dropout=self._config.attention_dropout,
        )

        super().__init__(
            dataset=self._config.val_dataset,
            model=model,
            batch_size=self._config.batch_size,
            training_iterations=self._config.training_iterations,
            default_model_file_path=self._config.model_path,
        )

    def __call__(
        self,
        prompt: str,
        device: str = "cpu",
    ) -> typing.Generator[str, None, None]:
        """
        Generate text given a prompt.
        """
        tokens = self._config.tokenizer.encode(prompt)  # type: ignore[attr-defined]
        generator = self.model.generate(tokens, k=self._config.top_k, device=device)
        yield prompt
        while True:
            try:
                yield self._config.tokenizer.decode([next(generator)])  # type: ignore[attr-defined]
            except StopIteration:
                assert False

    def train(
        self,
        device: str,
    ) -> typing.Generator[float, None, None]:
        """
        Train the model.
        """
        data_loader = torchdata.DataLoader(self._config.train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._config.learning_rate)
        yield from modeltrainer.train_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._config.loss_function(reduction="mean"),
            optimizer=optimizer,
            device=device,
        )

    def eval(
        self,
        device: str,
    ) -> dict[str, float]:
        """
        Evaluate the model in the validation dataset.
        """
        data_loader = torchdata.DataLoader(self._config.val_dataset, batch_size=self.batch_size, shuffle=False)
        evaluator = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=self._config.loss_function(reduction="sum"),
            device=device,
        )

        total_loss = 0.0
        total_tokens = 0

        for y_true, _, batch_loss in evaluator:
            num_tokens = y_true.numel()
            total_loss += batch_loss.item()
            total_tokens += num_tokens

        loss = total_loss / total_tokens
        perplexity = math.exp(loss)

        return {
            "val loss": loss,
            "perplexity (PPL)": perplexity,
        }
