"""
The Vision Transformer.
"""

import torch

import torch.nn.functional as F

from torch import nn

from . import transformer


class ViT(nn.Module):
    """
    Vision transformer.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int,
        height: int,
        width: int,
        patch_size: int,
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

        num_tokens = (height // patch_size) * (width // patch_size) + 1

        self.cls_token = nn.parameter.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, D)
        self.token_embedding = VisionEmbedder(
            in_channels=channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.transformer = transformer.Transformer(
            num_tokens=num_tokens,
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.cls_head = nn.Linear(
            in_features=embed_dim,
            out_features=num_classes,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cls_head.bias)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embedding(x)  # (B, I, H, W) -> (B, T, D)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)  # (B, T, D) -> (B, T+1, D)
        x = self.transformer(x)  # (B, T+1, D)
        x = self.cls_head(x[:, 0, :])  # (B, D) -> (B, C)
        return x

    @torch.no_grad()
    def classify(
        self,
        image: torch.Tensor,
        device: str = "cpu",
    ) -> int:
        """
        Classify an image.
        """
        image = image.to(device)
        self.to(device)
        self.eval()
        logits = self(image.unsqueeze(0))
        probs = F.softmax(logits[0], dim=-1)
        class_index = torch.argmax(probs)
        return class_index.item()


class VisionEmbedder(nn.Module):
    """
    Create token embeddings from images by extracting patches.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pylint: disable=missing-function-docstring
        nn.init.normal_(self.patch_embed.weight, mean=0.0, std=0.02)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, C, H, W) -> (B, D, Hp, Wp)
        x = torch.flatten(x, start_dim=2, end_dim=-1)  # (B, D, Hp, Wp) -> (B, D, T), T = Hp * Wp
        x = torch.transpose(x, 1, 2)  # (B, D, T) -> (B, T, D)
        return x
