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
        Uniformly sample among three valid (query, key/value) combos.
        Combos:
            (q=B_self, kv=A_self),
            (q=B_self, kv=B_to_A),
            (q=A_to_B, kv=A_self)
        Returns:
            idx_q, idx_kv corresponding to entries in q_candidates/kv_candidates.
        """
        combos = torch.tensor([[0, 0], [0, 1], [1, 0]], device=device)
        choice = torch.randint(0, combos.size(0), (1,), device=device).item()
        return combos[choice].tolist()

    def forward(self, xA, xB, session_idx=None, attn_mode: str | None = None):
        """
        Args:
            xA: [B, T, N_feat_A]
            xB: [B, T, N_feat_B]
            session_idx: optional session ids for DomainAdapter

        Returns:
            h: [B, T, D] fused representation after projector + L2 norm
            A_self, B_self: modality-specific representations
            A_to_B, B_to_A: cross-modal predictions
        """
        B, T = xA.shape[:2]
        device = xA.device

        # === Mask (optional, shared to both for simplicity) ===
        mask = self._make_mask(B, T, device)

        # === Encode both modalities ===
        A_self, A_to_B = self.encoderA(xA, session_idx=session_idx, mask=mask)  # [B, T, D], [B, T, D]
        B_self, B_to_A = self.encoderB(xB, session_idx=session_idx, mask=mask)  # [B, T, D], [B, T, D]

        # === Build candidate sets ===
        # Query candidates (B-space):  {B_self, A_to_B}
        # Key/Value candidates (A-space): {A_self, B_to_A}
        kv_candidates  = (A_self, B_to_A)
        q_candidates = (B_self, A_to_B)

        # === Select cross-attention pair ===
        if attn_mode is None:
            idx_q, idx_kv = self._sample_combo_indices(device)
        else:
            attn_mode = attn_mode.lower()
            if attn_mode == 'imu':           # A with A_to_B
                idx_q, idx_kv = 1, 0
            elif attn_mode == 'dlc':         # B with B_to_A
                idx_q, idx_kv = 0, 1
            elif attn_mode == 'both':        # A with B
                idx_q, idx_kv = 0, 0
            else:
                raise ValueError(f"Unknown attn_mode: {attn_mode}")
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
        proj_idx = session_idx if self.projection.mode == "aware" else None
        h = self.projection(h, proj_idx)                # [B, T, D]
        h = F.normalize(h, dim=-1)

        # Also return self representations for cross-modal reconstruction losses
        return h, A_self, B_self, A_to_B, B_to_A
