# models/encoder_fusion.py
import torch
import torch.nn as nn
from .encoder import Encoder  # unified modality encoder
from .masking import generate_continuous_mask, generate_binomial_mask  # mask functions
import torch.nn.functional as F
from .domain_adapter import DomainAdapter
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class EncoderFusion(nn.Module):
    def __init__(self, N_feat_A, N_feat_B, mask_type=None, d_model=64, nhead=4,
                 dropout=0.1, num_sessions: int = 0, projection_mode: str = "aware"):
        super().__init__()

        self.encoderA = Encoder(N_feat_A, d_model=d_model,
                               dropout=dropout, num_sessions=num_sessions)
        self.encoderB = Encoder(N_feat_B, d_model=d_model,
                               dropout=dropout, num_sessions=num_sessions)
        self.mask_type = mask_type

        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        self.gate1 = nn.Linear(d_model, 1)
        self.gate2 = nn.Linear(d_model, 1)

        # Transformer encoder for fusion, only 1 layer
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)

        self.projection = DomainAdapter(d_model, d_model, num_sessions=num_sessions, dropout=dropout)
        self.projection.set_mode(projection_mode)

    def forward(self, xA, xB, session_idx=None):
        B, T = xA.shape[:2]

        if self.mask_type == 'binomial':
            mask = generate_binomial_mask(B, T).to(xA.device)
        elif self.mask_type == 'continuous':
            mask = generate_continuous_mask(B, T).to(xA.device)
        else:
            mask = None

        hA = self.encoderA(xA, session_idx=session_idx, mask=mask)  # B×T×D
        hB = self.encoderB(xB, session_idx=session_idx, mask=mask)  # B×T×D

        mA, attn_weightsA = self.cross_attn1(query=hA, key=hB, value=hB, need_weights=True)
        mB, attn_weightsB = self.cross_attn2(query=hB, key=hA, value=hA, need_weights=True)
        # np.save(f"attn_map.npy", attn_weightsA.detach().cpu().numpy())
        g1 = torch.sigmoid(self.gate1(hA))
        g2 = torch.sigmoid(self.gate2(hB))
        h = g1 * mA + g2 * mB + hA + hB # B×T×D

        # ---- 新增 transformer encoder 层（含自带的 LayerNorm） ----
        h = self.transformer_encoder(h)  # B×T×D

        # 不再 norm，这里直接投影
        h = self.projection(h, session_idx)
        h = F.normalize(h, dim=-1)
        return h