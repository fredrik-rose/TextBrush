"""
The GPT (generative pre-trained transformer) model.

A LLM (large language model) for text generation.
"""

from typing import Generator

import torch
import torch.nn.functional as F

from torch import nn

from . import transformer


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

    def __init__(self, vocab_size: int, num_heads: int, embed_dim: int):
        super().__init__()

        self.token_embedding = TextEmbedder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        )
        self.transformer = transformer.TransformerBlock(
            num_heads=num_heads,
            embed_dim=embed_dim,
        )
        self.lm_head = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
        )

        self.register_buffer("mask", torch.tril(torch.ones(num_tokens, num_tokens), diagonal=0))

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        num_tokens = x.size(1)
        x = self.token_embedding(x)  # (B, T) -> (B, T, D)
        x = self.transformer(x, mask=self.mask[:num_tokens, :num_tokens])  # (B, T, D)
        x = self.lm_head(x)  # (B, T, D) -> (B, T, C)
        return x

    def generate(self, prompt: list[int], device: str = "cpu") -> Generator[int, None, None]:
        """
        Generate text (tokens), given a prompt (of tokens).
        """
        tokens = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)  # (B, T)
        self.to(device)
        while True:
            logits = self(tokens)  # (B, T) -> (B, T, C)
            logits = logits[:, -1, :]  # (B, T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, C) -> (B, 1)
            tokens = torch.cat((tokens, next_token), dim=-1)  # (B, T+1)
            yield next_token.item()
