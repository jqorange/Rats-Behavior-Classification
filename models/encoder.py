import torch
import torch.nn as nn
import torch.nn.functional as F
from .dilated_conv import DilatedConvEncoder

class Encoder(nn.Module):
    """
    Hybrid temporal encoder:
    - FC layer to project features
    - Dilated convolutional block (TCN) for local context
    - Transformer encoder for global context
    - Optional global projection to a single vector of size out_dim via mean pooling
    Supports optional frame mask for Transformer
    """
    def __init__(self, N_feat, d_model=64, depth=3, nhead=4, dropout=0.1):
        super().__init__()
        # Model dimensions
        self.d_model = d_model
        # 1) Project input features to model dimension
        self.input_fc  = nn.Linear(N_feat, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        # 2) Dilated convolutional encoder: B×d_model×T -> B×d_model×T
        self.tcn       = DilatedConvEncoder(d_model, [d_model] * depth, kernel_size=3)
        self.norm2     = nn.LayerNorm(d_model)
        # 3) Transformer for global context: B×T×d_model -> B×T×d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.trans     = nn.TransformerEncoder(layer, num_layers=1)
        self.norm3     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)



    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (B, T, N_feat)
            mask: optional BoolTensor of shape (B, T); True=valid frame, False=mask
        Returns:
            If out_dim is None:
                h: Tensor of shape (B, T, d_model)
            Else:
                h_out: Tensor of shape (B, out_dim)
        """
        B, T, _ = x.shape
        # 1) Input projection
        h = self.input_fc(x)                # (B, T, d_model)
        # Apply mask immediately after projection
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(h)  # (B, T, d_model)
            h = h.masked_fill(~mask_exp, 0.0)
        h = self.norm1(h + h)               # (B, T, d_model)

        # 2) TCN local context
        h2 = self.tcn(h.transpose(1, 2))     # (B, d_model, T)
        h2 = h2.transpose(1, 2)              # (B, T, d_model)
        h = self.norm2(h + self.dropout(h2))# (B, T, d_model)

        # 3) Transformer global context
        h = self.trans(h)                   # (B, T, d_model)
        h = self.norm3(h)                   # (B, T, d_model)

        return h  # (B, T, d_model)
