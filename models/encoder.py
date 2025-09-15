import torch
import torch.nn as nn

from .dilated_conv import DilatedConvEncoder

def check_nan(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name} | mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}")
        raise ValueError(f"NaN found in {name}")
    if torch.isinf(tensor).any():
        print(f"❌ Inf detected in {name} | mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}")
        raise ValueError(f"Inf found in {name}")

class Encoder(nn.Module):
    """Lightweight temporal encoder used for each modality."""

    def __init__(self, N_feat: int, d_model: int = 64, depth: int = 3, dropout: float = 0.1,
                 num_sessions: int = 0):
        super().__init__()
        self.d_model = d_model

        # Simple linear projection as input stem (session aware adapters removed)
        self.input_proj = nn.Linear(N_feat, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        # TCN encoder for local temporal context
        self.tcn = DilatedConvEncoder(d_model, [d_model] * depth, kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        self.norm_tcn = nn.LayerNorm(d_model)

        # GRU head for cross-modal prediction
        self.head_cross = nn.GRU(d_model, d_model, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)

        # GRU based reconstruction head for the current modality
        self.head_recon = nn.GRU(d_model, d_model, batch_first=True)
        self.norm_recon = nn.LayerNorm(d_model)
        self.recon_out = nn.Linear(d_model, N_feat)

    def forward(self, x: torch.Tensor, session_idx: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        """
        Args:
            x: [B, T, N_feat]
            session_idx: unused placeholder for backward compatibility
            mask: [B, T] boolean, True = keep, False = mask (optional)

        Returns:
            z_self:  [B, T, D] - representation in *own* space
            z_cross: [B, T, D] - representation predicted in *other* space
        """
        B, T, _ = x.shape
        check_nan(x, "input x")

        # 1) Input projection
        h = self.input_proj(x)                                   # [B, T, D]
        check_nan(h, "after input_proj")
        h = self.norm_in(h)

        # 2) Optional masking
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(h)           # [B, T, D]
            h = h.masked_fill(~mask_exp, 0.0)
            check_nan(h, "after masking")

        # 3) TCN (local context)
        h_tcn = self.tcn(h.transpose(1, 2)).transpose(1, 2)      # [B, T, D]
        check_nan(h_tcn, "after TCN")
        h = self.norm_tcn(h + self.dropout(h_tcn))               # residual keep-current-frame info
        check_nan(h, "after norm_tcn")

        # 4) Outputs
        z_self = h                                              # [B, T, D]
        z_cross, _ = self.head_cross(h.detach())
        z_cross = self.norm_cross(z_cross)                      # [B, T, D]
        recon_h, _ = self.head_recon(h)
        recon_h = self.norm_recon(recon_h)
        x_recon = self.recon_out(recon_h)

        check_nan(z_self, "z_self")
        check_nan(z_cross, "z_cross")
        check_nan(x_recon, "x_recon")

        return z_self, z_cross, x_recon
