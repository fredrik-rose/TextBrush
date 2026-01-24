"""
The U-Net Vision Transformer.
"""

import torch

from torch import nn

from . import transformer
from . import vit


class UViT(nn.Module):
    """
    U-Net Vision Transformer (U-ViT).

    Useful as noise predictor in a diffusion process.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        patch_size: int,
        time_steps: int,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        feed_forward_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        assert height % patch_size == 0
        assert width % patch_size == 0

        height_patches = height // patch_size
        width_patches = width // patch_size
        num_image_tokens = height_patches * width_patches
        num_tokens = num_image_tokens + 1

        self.max_num_tokens = num_tokens
        self._num_image_tokens = num_image_tokens
        self._image_embedder = vit.VisionEmbedder(
            in_channels=channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self._time_embedder = TimeEmbedder(
            time_steps=time_steps,
            embed_dim=embed_dim,
        )
        self._pos_encoder = transformer.PositionalEncoder(
            num_tokens=num_tokens,
            embed_dim=embed_dim,
        )
        self._dropout = nn.Dropout(dropout)
        self._unet = UNet(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=True,
        )
        self._image_unembedder = VisionUnembedder(
            embed_dim=embed_dim,
            channels=channels,
            height_patches=height_patches,
            width_patches=width_patches,
            patch_size=patch_size,
        )
        self._conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding="same",
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.zeros_(self._conv.weight)
        if self._conv.bias is not None:
            nn.init.zeros_(self._conv.bias)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        image_tokens = self._image_embedder(x)  # (B, I, H, W) -> (B, T, D)
        time_tokens = self._time_embedder(t)  # (B, 1) -> (B, 1, D)
        tokens = torch.cat([time_tokens, image_tokens], dim=-2)  # (B, T, D) -> (B, T+1, D)
        tokens = self._pos_encoder(tokens)  # (B, T, D)
        tokens = self._dropout(tokens)  # (B, T, D)
        tokens = self._unet(tokens)  # (B, T, D)
        noise = self._image_unembedder(tokens[:, -self._num_image_tokens : :])  # (B, T, D) -> (B, I, H, W)
        noise = self._conv(noise)  # (B, I, H, W)
        return noise


class TimeEmbedder(nn.Module):
    """
    Create embeddings for diffusion time steps.
    """

    def __init__(
        self,
        time_steps: int,
        embed_dim: int,
    ):
        super().__init__()

        self._time_embed = nn.Embedding(
            num_embeddings=time_steps,
            embedding_dim=embed_dim,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.normal_(self._time_embed.weight, mean=0.0, std=0.02)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._time_embed(x)  # (B, 1) -> (B, 1, D)
        return x


class UNet(nn.Module):
    """
    Transformer U-net using long skip connections.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias: bool = True,
    ):
        assert num_layers % 2 == 1

        super().__init__()

        self._down = nn.ModuleList(
            [
                transformer.TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    bias=bias,
                )
                for _ in range(num_layers // 2)
            ]
        )
        self._bottleneck = transformer.TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=bias,
        )
        self._mergers = nn.ModuleList(
            [
                TokenMerger(
                    num_sets=2,
                    embed_dim=embed_dim,
                )
                for _ in range(num_layers // 2)
            ]
        )
        self._up = nn.ModuleList(
            [
                transformer.TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    bias=bias,
                )
                for _ in range(num_layers // 2)
            ]
        )
        self._norm = transformer.LayerNorm(
            embed_dim=embed_dim,
        )

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        skip_connections = []
        for layer in self._down:
            x = layer(x)  # (B, T, D)
            skip_connections.append(x)
        x = self._bottleneck(x)
        for merger, layer in zip(self._mergers, self._up):
            x = merger(x, skip_connections.pop())  # (B, T, D), (B, T, D) -> (B, T, D)
            x = layer(x)  # (B, T, D)
        x = self._norm(x)  # (B, T, D)
        return x


class VisionUnembedder(nn.Module):
    """
    Creates an image from tokens that where initially extracted from an image.
    """

    def __init__(
        self,
        embed_dim: int,
        channels: int,
        height_patches: int,
        width_patches: int,
        patch_size: int,
    ):
        super().__init__()

        self._c = channels
        self._h_p = height_patches
        self._w_p = width_patches
        self._p_s = patch_size
        self._linear = nn.Linear(
            in_features=embed_dim,
            out_features=channels * patch_size * patch_size,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.normal_(self._linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self._linear.bias)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Assumes IPP (channel, patch_size, patch_size) order in embedding layer (this is true when
        # using the convolution approach).
        x = self._linear(x)  # (B, T, D) -> (B, T, I*P*P)
        # Alt: einops.rearrange(x, "b (hp wp) (i p p) -> b i (hp p) (wp p)", hp=h_p, wp=w_p, p=p_s, i=c)
        x = x.view(  # (B, T, I*P*P) -> (B, Hp, Wp, I, P, P), T = Hp*Wp
            (x.size(0), self._h_p, self._w_p, self._c, self._p_s, self._p_s),
        )
        x = torch.permute(x, (0, 3, 1, 4, 2, 5))  # (B, Hp, Wp, I, P, P) -> (B, I, Hp, P, Wp, P)
        x = torch.reshape(  # (B, I, Hp, P, Wp, P) -> (B, I, H, W), H = Hp*P, W = Wp*P
            x,
            (x.size(0), x.size(1), self._h_p * self._p_s, self._w_p * self._p_s),
        )
        return x


class TokenMerger(nn.Module):
    """
    Merges sets of tokens on embedded dimension.
    """

    def __init__(
        self,
        num_sets: int,
        embed_dim: int,
    ):
        super().__init__()

        self._linear = nn.Linear(
            in_features=embed_dim * num_sets,
            out_features=embed_dim,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        transformer.init_xavier_uniform(self._linear)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        *x: torch.Tensor,
    ) -> torch.Tensor:
        y = torch.cat(x, dim=-1)  # (B, T, D), (B, T, D) -> (B, T, N*D)
        y = self._linear(y)  # (B, T, N*D) -> (B, T, D)
        return y
