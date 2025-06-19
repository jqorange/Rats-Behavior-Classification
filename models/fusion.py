# models/encoder_fusion.py
import torch
import torch.nn as nn
from .encoder import Encoder  # unified modality encoder
from .masking import generate_continuous_mask, generate_binomial_mask  # mask functions


class EncoderFusion(nn.Module):

    def __init__(self, N_feat_A, N_feat_B, mask_type=None, out_dim=None, d_model=64, nhead=4, dropout=0.1):
        super().__init__()

        # Single-modal encoders
        self.encoderA = Encoder(N_feat_A, d_model=d_model, nhead=nhead, out_dim=out_dim)
        self.encoderB = Encoder(N_feat_B, d_model=d_model, nhead=nhead, out_dim=out_dim)
        self.mask_type = mask_type
        # Cross-attention from A→B
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Gate to modulate fusion with confidence
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # LayerNorm for residual fusion
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xA, xB, confidence=None):
        """
        mask_type: 'binomial' or 'continuous'
        Returns two fused masked representations h1, h2
        """
        B, T = xA.shape[:2]

        # Generate two different masks
        if self.mask_type == 'binomial':
            mask = generate_binomial_mask(B, T).to(xA.device)
        elif self.mask_type == 'continuous':
            mask = generate_continuous_mask(B, T).to(xA.device)
        else:
            mask = None

        # First masked forward pass
        hA = self.encoderA(xA, mask=mask)  # B×T×D
        hB = self.encoderB(xB, mask=mask)  # B×T×D

        # Cross-attention for first pair
        m, _ = self.cross_attn(
            query=hA,
            key=hB,
            value=hB,
            need_weights=False
        )  # B×T×D

        # Gate + residual fusion for first pair
        if confidence is None:
            confidence = torch.ones(B, T, 1, device=hA.device)
        else:
            confidence = confidence.unsqueeze(-1)
        gate_in = torch.cat([hA, hB, confidence], dim=-1)
        g = torch.sigmoid(self.gate_mlp(gate_in))
        h = hA + g * m  # B×T×D
        h = self.norm(h)
        h = self.dropout(h)

        return h


