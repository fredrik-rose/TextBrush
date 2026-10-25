"""
The CLIP (Contrastive Language-Image Pre-training) model.

A multi-modal (text and image) encoder.
"""

import torch

from torch import nn
import torch.nn.functional as F


class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training model.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.max_num_tokens = text_encoder.max_num_tokens + image_encoder.max_num_tokens
        self.logit_scale = nn.parameter.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def forward(  # pylint: disable=missing-function-docstring
        self,
        text: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text is not None:
            text_embedding = self.text_encoder(text, text_mask)  # (B, T) -> (B, E)
            text_embedding = F.normalize(text_embedding, p=2.0, dim=-1)  # (B, E) -> (B, E)
        else:
            text_embedding = None

        if image is not None:
            image_embedding = self.image_encoder(image)  # (B, I, H, W) -> (B, E)
            image_embedding = F.normalize(image_embedding, p=2.0, dim=-1)  # (B, E) -> (B, E)
        else:
            image_embedding = None

        return text_embedding, image_embedding

    @torch.no_grad()
    def classify(
        self,
        image: torch.Tensor,
        class_embeddings: torch.Tensor,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Classify an image.
        """
        image = image.to(device)
        class_embeddings = class_embeddings.to(device)  # (C, E)
        self.to(device)
        self.eval()
        _, image_embedding = self(text=None, image=image)  # (B, I, H, W) -> (B, E)
        # Vectors are already normalized, so we can compute cosine similarity as a dot product.
        cos_sim = torch.matmul(image_embedding, class_embeddings.T)  # (B, E) x (E, C) -> (B, C)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * cos_sim
        probs = F.softmax(logits, dim=-1)
        return probs
