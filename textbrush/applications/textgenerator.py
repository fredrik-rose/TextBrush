"""
Shakespeare text generator.
"""

import pathlib

import torch
import torch.utils.data as torchdata

from torch import nn

from textbrush.datasets import split as dataset_spliter
from textbrush.datasets import tinyshakespeare
from textbrush.models import gpt
from textbrush.optimizers import modeltrainer

MAX_TOKENS = 128
NUM_LAYERS = 6
NUM_HEADS = 4
EMBEDDED_DIMENSION = 128
FEED_FORWARD_DIMENSION = EMBEDDED_DIMENSION * 4

DROPOUT = 0.1
ATTENTION_DROPOUT = DROPOUT

DATASET_SPLIT = 0.99

BATCH_SIZE = 32
LEARNING_RATE = 3e-4

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "text-generator.pth"


class Textgenerator:
    """
    Text generator using a GPT model as backend.
    """

    def __init__(
        self,
        model_path: pathlib.Path | None = None,
    ):
        self.dataset = tinyshakespeare.TinyShakespeare(
            block_size=MAX_TOKENS,
        )
        self.model = gpt.GPT(
            vocab_size=self.dataset.vocab_size,
            num_tokens=MAX_TOKENS,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            embed_dim=EMBEDDED_DIMENSION,
            feed_forward_dim=FEED_FORWARD_DIMENSION,
            dropout=DROPOUT,
            attention_dropout=ATTENTION_DROPOUT,
        )
        self.split = [DATASET_SPLIT, (1.0 - DATASET_SPLIT)]

        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def __call__(
        self,
        prompt: str,
        length: int,
    ) -> str:
        """
        Generate text given a prompt.
        """
        tokens = self.dataset.encode(prompt)
        generator = self.model.generate(tokens)
        text = self.dataset.decode(next(generator) for _ in range(length))
        return prompt + text

    def train(
        self,
        device: str,
    ):
        """
        Train the model.
        """
        train_dataset, _ = dataset_spliter.split_ordered(self.dataset, self.split)
        data_loader = torchdata.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        loss_function = FlattenedCrossEntropy()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        yield from modeltrainer.train_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

    def eval(
        self,
        device: str,
    ):
        """
        Evaluate the model in the validation dataset.
        """
        _, validation_dataset = dataset_spliter.split_ordered(self.dataset, self.split)
        data_loader = torchdata.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
        loss_function = FlattenedCrossEntropy()
        validation_loss = modeltrainer.eval_model(
            model=self.model,
            data_loader=data_loader,
            loss_function=loss_function,
            device=device,
        )
        return validation_loss


class FlattenedCrossEntropy(nn.Module):
    """
    Adjust dimensions to use the ordinary cross entropy loss.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):  # pylint: disable=missing-function-docstring
        batch, tokens, classes = y_pred.shape
        y_pred = y_pred.reshape(batch * tokens, classes)  # (B, T, C) -> (B*T, C)
        y_true = y_true.reshape(batch * tokens)  # (B, T) -> (B*T)
        return self.loss(y_pred, y_true)
