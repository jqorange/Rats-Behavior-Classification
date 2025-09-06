import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .masking import generate_continuous_mask, generate_binomial_mask
from .domain_adapter import DomainAdapter

class EncoderFusion(nn.Module):
    """
    - Two encoders (A/B), each returns (z_self, z_cross).
      * z_self  : in own modality space
      * z_cross : predicted embedding in the *other* modality space
    - Cross-attn is single-direction but *randomly* picks from 2×2 combos with equal prob (25% each):
        Query  ∈ { A_self,  B_to_A }
        Key/Val∈ { B_self,  A_to_B }
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

        # Final projector (session-aware if enabled by mode)
        self.projection = DomainAdapter(d_model, d_model, num_sessions=num_sessions, dropout=dropout)
        self.projection.set_mode(projection_mode)

    def _make_mask(self, B, T, device):
        if self.mask_type == 'binomial':
            return generate_binomial_mask(B, T).to(device)
        elif self.mask_type == 'continuous':
            return generate_continuous_mask(B, T).to(device)
        else:
            return None

    @torch.no_grad()
    def _sample_combo_indices(self, device):
        """
        Uniformly sample the 2×2 choices with equal probability.
        Returns:
            idx_q  ∈ {0,1}, where 0->use A_self ; 1->use B_to_A as Query
            idx_kv ∈ {0,1}, where 0->use B_self ; 1->use A_to_B as Key/Value
        """
        idx_q  = torch.randint(0, 2, (1,), device=device).item()
        idx_kv = torch.randint(0, 2, (1,), device=device).item()
        return idx_q, idx_kv

    def forward(self, xA, xB, session_idx=None):
        """
        Args:
            xA: [B, T, N_feat_A]
            xB: [B, T, N_feat_B]
            session_idx: optional session ids for DomainAdapter

        Returns:
            h: [B, T, D] fused representation after projector + L2 norm
        """
        B, T = xA.shape[:2]
        device = xA.device

        # === Mask (optional, shared to both for simplicity) ===
        mask = self._make_mask(B, T, device)

        # === Encode both modalities ===
        A_self, A_to_B = self.encoderA(xA, session_idx=session_idx, mask=mask)  # [B, T, D], [B, T, D]
        B_self, B_to_A = self.encoderB(xB, session_idx=session_idx, mask=mask)  # [B, T, D], [B, T, D]

        # === Build 2×2 candidates ===
        # Query candidates are in A-space:  {A_self, B_to_A}
        # Key/Value candidates are in B-space: {B_self, A_to_B}
        q_candidates  = (A_self, B_to_A)
        kv_candidates = (B_self, A_to_B)

        # === Sample one of the four combos (25% each) ===
        idx_q, idx_kv = self._sample_combo_indices(device)
        q  = q_candidates[idx_q]      # [B, T, D]
        kv = kv_candidates[idx_kv]    # [B, T, D]

        # === Cross-attention (single direction, A<-B by construction of spaces) ===
        m, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)  # [B, T, D]

        # === Gated residual fusion on query side ===
        g = torch.sigmoid(self.gate(q))                    # [B, T, 1]
        h = q + self.dropout(g * m)                        # [B, T, D]
        h = self.norm(h)
        h = torch.clamp(h, min=-5.0, max=5.0)

        # === Final projector (session aware if enabled) + L2 normalize ===
        h = self.projection(h, session_idx)                # [B, T, D]
        h = F.normalize(h, dim=-1)

        return h, A_to_B, B_to_A
