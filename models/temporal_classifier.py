import torch
import torch.nn as nn
from .dilated_conv import DilatedConvEncoder

class TemporalClassifier(nn.Module):
    """Classifier using a small encoder with dilated convs and a transformer."""

    def __init__(self, input_dim: int, num_classes: int, d_model: int = 64, depth: int = 3,
                 nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.tcn = DilatedConvEncoder(d_model, [d_model] * depth, kernel_size=3)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.trans = nn.TransformerEncoder(layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim)"""
        x = self.proj(x)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x = self.trans(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
