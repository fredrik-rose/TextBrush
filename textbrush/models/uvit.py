"""
The U-Net Vision Transformer.
"""

import torch

from torch import nn


class UViT(nn.Module):
    """
    U-Net Vision Transformer.
    """

    def __init__(self):
        super().__init__()

        self.max_num_tokens = 0
        self.mean = nn.Parameter(torch.tensor([0.0]))

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        return torch.zeros_like(x) + self.mean
