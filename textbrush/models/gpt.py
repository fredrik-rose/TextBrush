"""
The GPT (generative pre-trained transformer) model.

A LLM (large language model) for text generation.
"""

import torch

from torch import nn


class TextEmbedder(nn.Module):
    """
    Create embeddings for a fixed vocabulary.

    A simple lookup table, each token (number) gets a corresponding vector.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()

        self.text_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.normal_(self.text_embed.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        x = self.text_embed(x)  # (B, T) -> (B, T, D)
        return x


class GPT(nn.Module):
    """
    Generative pre-trained transformer.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()

        self.token_embedding = TextEmbedder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        )
        self.lm_head = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        x = self.token_embedding(x)  # (B, T) -> (B, T, D)
        x = self.lm_head(x)  # (B, T, D) -> (B, T, C)
        return x
