import torch
import torch.nn as nn
import torch.nn.functional as F
from .dilated_conv import DilatedConvEncoder
from .domain_adapter import DomainAdapter
class Encoder(nn.Module):
    def __init__(self, N_feat, d_model=64, depth=3, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # === Domain adapter ===
        self.adapter = DomainAdapter(N_feat, d_model)

        # === Dilated TCN block ===
        self.tcn = DilatedConvEncoder(d_model, [d_model] * depth, kernel_size=3)
        self.norm2 = nn.LayerNorm(d_model)

        # === Transformer encoder ===
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.trans = nn.TransformerEncoder(layer, num_layers=1)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (B, T, N_feat)
            mask: optional BoolTensor of shape (B, T); True=valid frame, False=mask
        Returns:
            h: Tensor of shape (B, T, d_model)
        """
        B, T, _ = x.shape

        # === Domain adaptation ===
        h = self.adapter(x)  # (B, T, d_model)

        # Apply mask (if any)
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(h)
            h = h.masked_fill(~mask_exp, 0.0)

        # === TCN local context ===
        h2 = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm2(h + self.dropout(h2))

        # === Transformer global context ===
        h = self.trans(h)
        h = self.norm3(h)

        return h  # (B, T, d_model)
