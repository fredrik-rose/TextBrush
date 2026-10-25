"""
The BERT (bidirectional encoder representations from transformers) model.

A text encoder.
"""

import torch

from torch import nn

from . import gpt
from . import transformer


class BERT(nn.Module):
    """
    Bidirectional encoder representations from transformers.
    """

    def __init__(
        self,
        vocab_size: int,
        num_tokens: int,
        config: transformer.TransformerConfig,
        representation_dim: int,
    ):
        super().__init__()

        self.max_num_tokens = num_tokens

        self._cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))  # (1, 1, D)
        self._token_embedding = gpt.TextEmbedder(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
        )
        self._transformer = transformer.Transformer(
            num_tokens=num_tokens,
            config=config,
        )
        self._representation_head = nn.Linear(
            in_features=config.embed_dim,
            out_features=representation_dim,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.normal_(self._representation_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self._representation_head.bias)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._token_embedding(x)  # (B, T) -> (B, T, D)
        x = torch.cat([self._cls_token.expand(x.size(0), -1, -1), x], dim=1)  # (B, T, D) -> (B, T+1, D)
        x = self._transformer(x, mask)  # (B, T+1, D)
        x = self._representation_head(x[:, 0, :])  # (B, D) -> (B, R)
        return x
