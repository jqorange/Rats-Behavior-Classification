import torch
import torch.nn as nn
import torch.nn.functional as F
from .dilated_conv import DilatedConvEncoder
from .domain_adapter import DomainAdapter

def check_nan(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name} | mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}")
        raise ValueError(f"NaN found in {name}")
    if torch.isinf(tensor).any():
        print(f"❌ Inf detected in {name} | mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}")
        raise ValueError(f"Inf found in {name}")

class Encoder(nn.Module):
    """
    Adapter -> TCN -> (two heads)
      - head_self : predict representation in *own* modality space
      - head_cross: predict representation in the *other* modality space
    NOTE: No self-attention here per requirement.
    """
    def __init__(self, N_feat, d_model=64, depth=10, dropout=0.1,
                 num_sessions: int = 0):
        super().__init__()
        self.d_model = d_model

        # Domain adapter (session-aware projection) as input stem
        self.adapter = DomainAdapter(N_feat, d_model, num_sessions=num_sessions, dropout=dropout)
        self.adapter.set_mode("none")
        self.norm_in = nn.LayerNorm(d_model)

        # TCN encoder for local temporal context
        self.tcn = DilatedConvEncoder(d_model, [d_model] * depth, kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        self.norm_tcn = nn.LayerNorm(d_model)

        # Two MLP heads
        def make_head():
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
        self.head_self  = make_head()
        self.head_cross = make_head()
        self.norm_self  = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, session_idx: torch.Tensor | None = None, mask=None):
        """
        Args:
            x: [B, T, N_feat]
            session_idx: [B] or [B, ...], optional for DomainAdapter
            mask: [B, T] boolean, True = keep, False = mask (optional)

        Returns:
            z_self:  [B, T, D] - representation in *own* space
            z_cross: [B, T, D] - representation predicted in *other* space
        """
        B, T, _ = x.shape
        check_nan(x, "input x")

        # 1) Adapter
        h = self.adapter(x, session_idx=session_idx)             # [B, T, D]
        check_nan(h, "after adapter")
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

        # 4) Two heads (no self-attention)
        z_self  = self.norm_self(self.head_self(h))              # [B, T, D]
        z_cross = self.norm_cross(self.head_cross(h))            # [B, T, D]
        check_nan(z_self,  "z_self")
        check_nan(z_cross, "z_cross")

        return z_self, z_cross
