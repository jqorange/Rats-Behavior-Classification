from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .masking import generate_binomial_mask, generate_continuous_mask


@dataclass
class SingleModalOutput:
    embedding: torch.Tensor
    features: torch.Tensor
    reconstruction: torch.Tensor
    mask: Optional[torch.Tensor]


class ProjectionHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleModalModel(nn.Module):
    """Single-modality encoder with optional masking augmentation."""

    def __init__(
        self,
        num_features: int,
        *,
        d_model: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
        mask_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(num_features, d_model=d_model, depth=depth, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = ProjectionHead(d_model, dropout)
        self.mask_type = mask_type

    def _make_mask(self, batch: int, length: int, device: torch.device) -> Optional[torch.Tensor]:
        if (not self.training) or self.mask_type is None:
            return None
        if self.mask_type == "binomial":
            return generate_binomial_mask(batch, length).to(device)
        if self.mask_type == "continuous":
            return generate_continuous_mask(batch, length).to(device)
        return None

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> SingleModalOutput:
        b, t, _ = x.shape
        device = x.device

        aug_mask = self._make_mask(b, t, device)
        if aug_mask is not None:
            mask = aug_mask if mask is None else (mask & aug_mask)

        features, recon = self.encoder(x, mask=mask)
        features = self.norm(features)
        proj = self.projection(self.dropout(features))
        proj = F.normalize(proj, dim=-1)

        return SingleModalOutput(
            embedding=proj,
            features=features,
            reconstruction=recon,
            mask=mask,
        )
