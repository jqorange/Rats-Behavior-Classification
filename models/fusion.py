from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .masking import generate_continuous_mask, generate_binomial_mask


@dataclass
class FusionOutput:
    fused: torch.Tensor
    imu_self: torch.Tensor
    dlc_self: torch.Tensor
    imu_to_dlc: torch.Tensor
    dlc_to_imu: torch.Tensor
    imu_recon: torch.Tensor
    dlc_recon: torch.Tensor


class ProjectionHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderFusion(nn.Module):
    """
    - Two encoders (A/B), each returns (z_self, z_cross).
      * z_self  : in own modality space
      * z_cross : predicted embedding in the *other* modality space
    - Cross-attn is single-direction but randomly selects among three combos:
        * Query from {B_self, A_to_B}
        * Key/Val from {A_self, B_to_A}
      The pair (A_to_B, B_to_A) is excluded.
    - Gated residual fusion on the query side; final projector + L2 norm.

    NOTE:
      If你想记住采样到的组合，可把 idx_q/idx_kv 存下来以便日志或loss使用。
    """

    def __init__(self, N_feat_A, N_feat_B, mask_type=None, d_model=64, nhead=4,
                 dropout=0.1, num_sessions: int = 0, projection_mode: str = "aware"):
        super().__init__()

        # Single-modal encoders
        self.encoderA = Encoder(N_feat_A, d_model=d_model,
                                dropout=dropout, num_sessions=num_sessions)
        self.encoderB = Encoder(N_feat_B, d_model=d_model,
                                dropout=dropout, num_sessions=num_sessions)

        self.mask_type = mask_type

        # One-direction cross-attention block (we will randomize the inputs)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Gating on the query side
        self.gate = nn.Linear(d_model, 1)
        self.norm  = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Final projector without session aware behaviour
        self.projection = ProjectionHead(d_model, dropout)

    def _make_mask(self, B, T, device):
        """Optionally create a temporal mask for feature dropout.

        During inference we want any stochastic corruption to be disabled.
        Guarding on ``self.training`` ensures that masks – which randomly
        zero out time steps – are only generated while training.  When the
        module is in evaluation mode or ``mask_type`` is ``None`` we return
        ``None`` so that the encoders see the unmodified inputs.
        """

        if (not self.training) or (self.mask_type is None):
            return None

        if self.mask_type == 'binomial':
            return generate_binomial_mask(B, T).to(device)
        if self.mask_type == 'continuous':
            return generate_continuous_mask(B, T).to(device)
        return None

    @torch.no_grad()
    def _sample_combo_indices(self, device: torch.device, batch_size: int) -> torch.Tensor:
        combos = torch.tensor([[0, 0], [0, 1], [1, 0]], device=device)
        return torch.randint(0, combos.size(0), (batch_size,), device=device)

    def forward(self, xA, xB, session_idx: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                attn_mode: Optional[str] = None) -> FusionOutput:
        """
        Args:
            xA: [B, T, N_feat_A]
            xB: [B, T, N_feat_B]
            session_idx: unused placeholder kept for compatibility

        Returns:
            h: [B, T, D] fused representation after projector + L2 norm
            A_self, B_self: modality-specific representations
            A_to_B, B_to_A: cross-modal predictions
        """
        B, T = xA.shape[:2]
        device = xA.device

        # === Mask (optional, shared to both for simplicity) ===
        mask_to_use = self._make_mask(B, T, device)
        if mask is not None and mask_to_use is not None:
            mask = mask & mask_to_use
        elif mask_to_use is not None:
            mask = mask_to_use

        # === Encode both modalities ===
        A_self, A_to_B, A_recon = self.encoderA(xA, session_idx=session_idx, mask=mask)  # [B, T, D], [B, T, D]
        B_self, B_to_A, B_recon = self.encoderB(xB, session_idx=session_idx, mask=mask)  # [B, T, D], [B, T, D]

        # === Build candidate sets ===
        # Query candidates (B-space):  {B_self, A_to_B}
        # Key/Value candidates (A-space): {A_self, B_to_A}
        q_candidates = torch.stack((B_self, A_to_B), dim=0)
        kv_candidates = torch.stack((A_self, B_to_A), dim=0)

        combos = torch.tensor([[0, 0], [0, 1], [1, 0]], device=device)

        # === Select cross-attention pair ===
        if attn_mode is None:
            combo_idx = self._sample_combo_indices(device, B)
        else:
            attn_mode = attn_mode.lower()
            if attn_mode == 'imu':
                combo_idx = combos.new_full((B,), 2)
            elif attn_mode == 'dlc':
                combo_idx = combos.new_full((B,), 1)
            elif attn_mode == 'both':
                combo_idx = combos.new_full((B,), 0)
            else:
                raise ValueError(f"Unknown attn_mode: {attn_mode}")

        combo_pairs = combos[combo_idx]  # [B,2]
        all_attn = []
        for iq, ikv in combos:
            q_i = q_candidates[iq]
            kv_i = kv_candidates[ikv]
            m_i, _ = self.cross_attn(query=q_i, key=kv_i, value=kv_i, need_weights=False)
            all_attn.append(m_i)
        attn_stack = torch.stack(all_attn, dim=0)

        batch_indices = torch.arange(B, device=device)
        q = q_candidates[combo_pairs[:, 0], batch_indices]
        m = attn_stack[combo_idx, batch_indices]

        # === Gated residual fusion on query side ===
        g = torch.sigmoid(self.gate(q))                    # [B, T, 1]
        h = q + self.dropout(g * m)                        # [B, T, D]
        h = self.norm(h)
        h = torch.clamp(h, min=-5.0, max=5.0)

        # === Final projector + L2 normalize ===
        h = self.projection(h)                          # [B, T, D]
        h = F.normalize(h, dim=-1)

        return FusionOutput(
            fused=h,
            imu_self=A_self,
            dlc_self=B_self,
            imu_to_dlc=A_to_B,
            dlc_to_imu=B_to_A,
            imu_recon=A_recon,
            dlc_recon=B_recon,
        )
