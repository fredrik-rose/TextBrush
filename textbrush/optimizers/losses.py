"""
Loss functions.
"""

import torch
import torch.nn.functional as F

from torch import nn


class InfoNCE(nn.Module):
    """
    InfoNCE loss function, a contrasstive loss function used in self-supervised learning.
    """

    def __init__(
        self,
        logit_scale: nn.Parameter,
        reduction: str = "mean",
    ):
        super().__init__()
        self._logit_scale = logit_scale
        self._reduction = reduction

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        logit_scale = self._logit_scale.exp()
        cos_sim = logit_scale * torch.matmul(x, y.T)  # (B, E) x (E, B) -> (B, B)
        target = torch.arange(cos_sim.size(0), device=cos_sim.device, dtype=torch.long)
        x_loss = F.cross_entropy(cos_sim, target, reduction=self._reduction)
        y_loss = F.cross_entropy(cos_sim.T, target, reduction=self._reduction)
        loss = (x_loss + y_loss) / 2.0
        return loss
