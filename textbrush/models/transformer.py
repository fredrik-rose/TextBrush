"""
The Transformer model.
"""

import torch
import torch.nn.functional as F

from torch import nn


class PositionalEncoder(nn.Module):
    """
    Positional encoding with learnable embeddings.
    """

    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()

        self.pos_embed = nn.parameter.Parameter(torch.zeros(num_tokens, embed_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # Use a small std to not dominate early in the training.

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        x = x + self.pos_embed[: x.size(1)]
        return x


class TransformerBlock(nn.Module):
    """
    A standard self-attention transformer block using pre-LayerNorm.
    """

    def __init__(self, num_heads: int, embed_dim: int, feed_forward_dim: int, bias: bool = True):
        super().__init__()

        self.attention_norm = LayerNorm(
            embed_dim=embed_dim,
        )
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            bias=bias,
        )
        self.feed_forward_norm = LayerNorm(
            embed_dim=embed_dim,
        )
        self.feed_forward_network = FeedForwardNetwork(
            embed_dim=embed_dim,
            feed_forward_dim=feed_forward_dim,
            bias=bias,
        )

    def forward(  # pylint: disable=missing-function-docstring
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.attention_norm(x)
        x = self.multi_head_attention(query=x, key=x, value=x, mask=mask)
        x = self.feed_forward_norm(x)
        x = self.feed_forward_network(x)
        return x


class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, embed_dim: int, epsilon: float = 1e-5):
        super().__init__()

        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.ones_(self.scale)
        nn.init.zeros_(self.shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / ((variance + self.epsilon) ** 0.5)
        x = self.scale * x + self.shift
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    """

    def __init__(self, num_heads: int, embed_dim: int, bias: bool = True):
        super().__init__()

        self.num_heads = num_heads
        self.query_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self.key_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self.value_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self.out_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        init_xavier_uniform(self.query_proj)
        init_xavier_uniform(self.key_proj)
        init_xavier_uniform(self.value_proj)
        init_xavier_uniform(self.out_proj)

    def forward(  # pylint: disable=missing-function-docstring
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        query = split_heads(self.query_proj(query), self.num_heads)  # (B, T, D) -> (B, H, T, Dh)
        key = split_heads(self.key_proj(key), self.num_heads)  # (B, T, D) -> (B, H, T, Dh)
        value = split_heads(self.value_proj(value), self.num_heads)  # (B, T, D) -> (B, H, T, Dh)
        x = merge_heads(scaled_dot_product_attention(query, key, value, mask))  # (B, H, T, Dh) -> (B, T, D)
        x = self.out_proj(x)  # (B, T, D)
        return x


class FeedForwardNetwork(nn.Module):
    """
    The feed-forward network of a Transformer.
    """

    def __init__(self, embed_dim: int, feed_forward_dim: int, bias: bool = True):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=feed_forward_dim,
                bias=bias,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=feed_forward_dim,
                out_features=embed_dim,
                bias=bias,
            ),
        )

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        init_xavier_uniform(self.network[0])
        init_xavier_uniform(self.network[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        x = self.network(x)
        return x


def init_xavier_uniform(linear_layer: nn.Module) -> None:
    """
    Initialize a linear layer using the Xavier uniform distribution.
    """
    # TOOD: Implement manually
    nn.init.xavier_uniform_(linear_layer.weight)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


def split_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
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
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Scaled dot-product attention.
    """
    d_k = query.size(-1)
    attention_score = (query @ torch.transpose(key, -1, -2)) / (d_k**0.5)  # (B*, T, D) @ (B*, D, T) -> (B*, T, T)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, float("-inf"))  # (B*, T, T)
    attention_weight = F.softmax(attention_score, dim=-1)  # (B*, T, T)
    output = attention_weight @ value  # (B*, T, T) @ (B*, T, D) -> (B*, T, D)
    return output
