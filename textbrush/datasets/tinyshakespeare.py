"""
The Tiny Shakespeare dataset.

Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import pathlib

import torch
import torch.utils.data as torchdata

DATASET_FILE_PATH = pathlib.Path(__file__).parent / "tinyshakespeare.txt"


class Tokenizer:
    """
    Tiny Shakespeare tokenizer.
    """

    def __init__(self):
        text = load_raw_data()
        vocab = sorted(set(text))

        self.vocab_size = len(vocab)

        self._int_to_token = dict(enumerate(vocab))
        self._token_to_int = {token: i for i, token in self._int_to_token.items()}

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        Encode a text string to a list of numbers.
        """
        return [self._token_to_int[c] for c in text]

    def decode(
        self,
        numbers: list[int],
    ) -> str:
        """
        Decode a list of numbers to a text string.
        """
        return "".join(self._int_to_token[i] for i in numbers)


class TinyShakespeare(torchdata.Dataset):
    """
    The Tiny Shakespeare dataset.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
    ):
        text = load_raw_data()

        self._text = torch.tensor(tokenizer.encode(text))
        self._block_size = block_size

    def __len__(self) -> int:
        return len(self._text) - self._block_size

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._text[idx : idx + self._block_size]
        y = self._text[idx + 1 : idx + self._block_size + 1]
        return x, y


def load_raw_data() -> str:
    """
    Load the raw Tiny Shakespeare data.
    """
    with open(DATASET_FILE_PATH, "r", encoding="utf-8") as file:
        text = file.read()
    return text
