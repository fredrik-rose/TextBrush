"""
The Tiny Shakespeare dataset.

Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import pathlib

import torch
import torch.utils.data as torchdata

DATASET_FILE_PATH = pathlib.Path(__file__).parent / "tinyshakespeare.txt"


class TinyShakespeare(torchdata.Dataset):
    """
    The Tiny Shakespeare dataset.
    """

    def __init__(self, block_size: int):
        with open(DATASET_FILE_PATH, "r", encoding="utf-8") as file:
            text = file.read()

        vocab = sorted(set(text))

        self.vocab_size = len(vocab)
        self.int_to_token = dict(enumerate(vocab))
        self.token_to_int = {token: i for i, token in self.int_to_token.items()}
        self.text = torch.tensor(self.encode(text))
        self.block_size = block_size

    def __len__(self):
        return len(self.text) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.text[idx : idx + self.block_size]
        y = self.text[idx + 1 : idx + self.block_size + 1]
        return x, y

    def encode(self, text: str) -> list[int]:
        """
        Encode a text string to a list of numbers.
        """
        return [self.token_to_int[c] for c in text]

    def decode(self, numbers: list[int]) -> str:
        """
        Decode a list of numbers to a text string.
        """
        return "".join(self.int_to_token[i] for i in numbers)
