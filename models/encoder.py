import torch
import torch.nn as nn
import torch.nn.functional as F
from .dilated_conv import DilatedConvEncoder
from .domain_adapter import DomainAdapter
class Encoder(nn.Module):
    """Dilated convolutional encoder with optional session adapters."""

    def __init__(self, N_feat, d_model=64, depth=3, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # === Input projection ===
        # A simple linear layer acts as the universal input interface.
        self.input_linear = nn.Linear(N_feat, d_model)

        # === Optional adapters ===
        # ``pre_adapter`` is used in stage 2 for session alignment.
        # ``post_adapter`` is used in stage 1 for session-specific spaces.
        self.pre_adapter = DomainAdapter(d_model, d_model)
        self.post_adapter = DomainAdapter(d_model, d_model)

        # Simple linear layer used in stage 2 after removing the post adapter
        self.output_linear = nn.Linear(d_model, d_model)

        # Flag to switch between stage1 and stage2 behaviour
        self.use_output_linear = False


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
    def set_stage(self, stage: str):
        """Configure adapters for the given training stage."""
        if stage == "stage1":
            self.use_output_linear = False
        elif stage == "stage2":
            self.use_output_linear = True

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (B, T, N_feat)
            mask: optional BoolTensor of shape (B, T); True=valid frame, False=mask
        Returns:
            h: Tensor of shape (B, T, d_model)
        """
        B, T, _ = x.shape
        # === Input linear projection ===
        h = self.input_linear(x)
        if self.use_output_linear == False:
        # === Optional pre-adapter (session alignment in stage 2) ===
            h = self.pre_adapter(h)


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

        # === Optional post-adapter ===
        if self.use_output_linear:
            h = self.output_linear(h)
        else:
            h = self.post_adapter(h)

        return h  # (B, T, d_model)
