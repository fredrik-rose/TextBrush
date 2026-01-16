"""
The Transformer model.
"""

import torch
import torch.nn.functional as F

from torch import nn


class Transformer(nn.Module):
    """
    A standard self-attention transformer.
    """

    def __init__(
        self,
        num_tokens: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self._pos_encoder = PositionalEncoder(
            num_tokens=num_tokens,
            embed_dim=embed_dim,
        )
        self._blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )
        self._norm = LayerNorm(
            embed_dim=embed_dim,
        )

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._pos_encoder(x)
        for block in self._blocks:
            x = block(x, mask)
        x = self._norm(x)
        return x


class PositionalEncoder(nn.Module):
    """
    Positional encoding with learnable embeddings.
    """

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self._pos_embed = nn.parameter.Parameter(torch.zeros(1, num_tokens, embed_dim))  # (1, T, D)
        self._dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.trunc_normal_(self._pos_embed, std=0.02)  # Use a small std to not dominate early in the training.

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self._pos_embed[:, : x.size(1), :]  # (B, t, D) + (1, t, D) -> (B, t, D)
        x = self._dropout(x)  # (B, t, D)
        return x


class TransformerBlock(nn.Module):
    """
    A standard self-attention transformer block using pre-LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self._attention_norm = LayerNorm(
            embed_dim=embed_dim,
        )
        self._multi_head_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=bias,
        )
        self._feed_forward_norm = LayerNorm(
            embed_dim=embed_dim,
        )
        self._feed_forward_network = FeedForwardNetwork(
            embed_dim=embed_dim,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self._attention_norm(x)
        r = self._multi_head_attention(query=r, key=r, value=r, mask=mask)
        x = x + r
        r = self._feed_forward_norm(x)
        r = self._feed_forward_network(r)
        x = x + r
        return x


class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(
        self,
        embed_dim: int,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self._epsilon = epsilon
        self._scale = nn.Parameter(torch.ones(embed_dim))
        self._shift = nn.Parameter(torch.zeros(embed_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.ones_(self._scale)
        nn.init.zeros_(self._shift)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / ((variance + self._epsilon) ** 0.5)
        x = self._scale * x + self._shift
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self._num_heads = num_heads
        self._attention_dropout = attention_dropout
        self._query_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self._key_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self._value_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self._out_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self._dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        init_xavier_uniform(self._query_proj)
        init_xavier_uniform(self._key_proj)
        init_xavier_uniform(self._value_proj)
        init_xavier_uniform(self._out_proj)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        query = split_heads(self._query_proj(query), self._num_heads)  # (B, T, D) -> (B, H, T, Dh)
        key = split_heads(self._key_proj(key), self._num_heads)  # (B, T, D) -> (B, H, T, Dh)
        value = split_heads(self._value_proj(value), self._num_heads)  # (B, T, D) -> (B, H, T, Dh)
        x = merge_heads(  # (B, H, T, Dh) -> (B, T, D)
            scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                training=self.training,
                dropout=self._attention_dropout,
                mask=mask,
            )
        )
        x = self._out_proj(x)  # (B, T, D)
        x = self._dropout(x)
        return x


class FeedForwardNetwork(nn.Module):
    """
    The feed-forward network of a Transformer.
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self._network = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=feed_forward_dim,
                bias=bias,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=feed_forward_dim,
                out_features=embed_dim,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        init_xavier_uniform(self._network[0])
        init_xavier_uniform(self._network[2])

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._network(x)
        return x


def init_xavier_uniform(linear_layer: nn.Module) -> None:
    """
    Initialize a linear layer using the Xavier uniform distribution.
    """
    # TOOD: Implement manually
    nn.init.xavier_uniform_(linear_layer.weight)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


def split_heads(
    tensor: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Split a tensor to heads, (B, T, D) -> (B, H, T, Dh).
    """
    # Note that an alternative is to use view and transpose, which may be faster.
    batch_size, num_tokens, embed_dim = tensor.shape
    assert embed_dim % num_heads == 0
    tensor = torch.reshape(tensor, (batch_size, num_tokens, num_heads, -1))  # (B, T, D) -> (B, T, H, Dh)
    tensor = torch.permute(tensor, (0, 2, 1, 3))  # (B, T, H, Dh) -> (B, H, T, Dh)
    return tensor


def merge_heads(tensor: torch.Tensor) -> torch.Tensor:
    """
    Merge the heads of a tensor, (B, H, T, Dh) -> (B, T, D).
    """
    # Note that an alternative is to use transpose, contiguous and view, which may be faster.
    batch_size, num_heads, num_tokens, head_dim = tensor.shape
    tensor = torch.permute(tensor, (0, 2, 1, 3))  # (B, H, T, Dh) -> (B, T, H, Dh)
    tensor = torch.reshape(tensor, (batch_size, num_tokens, num_heads * head_dim))  # (B, T, H, Dh) -> (B, T, D)
    return tensor


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout: float = 0.0,
    training: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.
    """
    d_k = query.size(-1)
    attention_score = (query @ torch.transpose(key, -1, -2)) / (d_k**0.5)  # (B*, T, D) @ (B*, D, T) -> (B*, T, T)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, float("-inf"))  # (B*, T, T)
    attention_weight = F.softmax(attention_score, dim=-1)  # (B*, T, T)
    attention_weight = F.dropout(attention_weight, p=dropout, training=training)  # (B*, T, T)
    output = attention_weight @ value  # (B*, T, T) @ (B*, T, D) -> (B*, T, D)
    return output
