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
    双向跨模态注意力版本（移除跨模态预测与随机组合）。

    - encoderA 负责模态A (比如 IMU)
        返回:
            A_self   : A在自己空间里的表示
            A_recon  : (可选) A的重构，用于mask/recon loss

    - encoderB 负责模态B (比如 DLC)
        返回:
            B_self, B_recon (对称)

    - 我们做两条 cross-attn 路线：
        1. B视角去关注A:
            Query = B_self
            Key/Val = A_self
            用 gateB 作为门控
        2. A视角去关注B:
            Query = A_self
            Key/Val = B_self
            用 gateA 作为门控

    - 最终 fused = avg(h_Bview, h_Aview) 之后再过 projection + L2 normalize。

    gateA / gateB 都会被用到。
    """

    def __init__(self,
                 N_feat_A,
                 N_feat_B,
                 mask_type=None,
                 d_model=64,
                 nhead=4,
                 dropout=0.1,
                 num_sessions: int = 0,
                 projection_mode: str = "aware"):
        super().__init__()

        # 单模态编码器
        self.encoderA = Encoder(
            N_feat_A,
            d_model=d_model,
            dropout=dropout,
            num_sessions=num_sessions,
        )
        self.encoderB = Encoder(
            N_feat_B,
            d_model=d_model,
            dropout=dropout,
            num_sessions=num_sessions,
        )

        self.mask_type = mask_type

        # 一个 multi-head attention 模块，可复用两次
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # 两个方向各自的门控（按 query 所在模态来挑 gate）
        self.gateA = nn.Linear(d_model, 1)  # 当 A 是 query 侧
        self.gateB = nn.Linear(d_model, 1)  # 当 B 是 query 侧

        # 共享 LayerNorm / Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 最终投影 (共享)
        self.projection = ProjectionHead(d_model, dropout)

    def _make_mask(self, B, T, device):
        """
        训练时根据 self.mask_type 生成时序mask, 用于encoder里的mask dropout。
        推理/评估模式下或 mask_type=None 时不做mask。
        """
        if (not self.training) or (self.mask_type is None):
            return None

        if self.mask_type == 'binomial':
            return generate_binomial_mask(B, T).to(device)
        if self.mask_type == 'continuous':
            return generate_continuous_mask(B, T).to(device)
        return None

    def _cross_direction(
        self,
        q: torch.Tensor,     # [B, T, D]
        kv: torch.Tensor,    # [B, T, D]
        gate_layer: nn.Linear,          # gateA or gateB
    ) -> torch.Tensor:
        """
        对一个方向跑 cross-attn，然后做门控残差融合。
        返回形状 [B, T, D] (还没projection，仅门控融合+norm+clamp)
        """

        m_i, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )  # [B,T,D]

        # 门控残差：q + g * m
        g = torch.sigmoid(gate_layer(q))  # [B,T,1]
        h_dir = q + self.dropout(g * m_i)  # [B,T,D]

        # 归一化/裁剪
        h_dir = self.norm(h_dir)
        h_dir = torch.clamp(h_dir, min=-5.0, max=5.0)

        return h_dir  # [B,T,D]

    def forward(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        session_idx: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_mode: Optional[str] = None,
    ) -> FusionOutput:
        """
        Args:
            xA: [B, T, N_feat_A]  (e.g. IMU)
            xB: [B, T, N_feat_B]  (e.g. DLC)
            session_idx: (可选) session id for per-session encoding,保留兼容
            mask: (可选) 预先给的mask (B,T) bool, True=keep(False=drop)
            attn_mode: unused placeholder (kept for compatibility)

        Returns:
            FusionOutput:
                fused         [B,T,D]  (双向cross后融合+proj+L2)
                imu_self      [B,T,D]
                dlc_self      [B,T,D]
                imu_recon     [B,T,*]  (由encoderA决定具体shape)
                dlc_recon     [B,T,*]  (由encoderB决定具体shape)
        """
        Bsz, T = xA.shape[:2]
        device = xA.device

        # === 生成训练用mask (只在train且mask_type!=None时) ===
        mask_to_use = self._make_mask(Bsz, T, device)
        if mask is not None and mask_to_use is not None:
            mask = mask & mask_to_use
        elif mask_to_use is not None:
            mask = mask_to_use

        # === 单模态编码 ===
        # A_self:   [B,T,D]
        # A_recon:  reconstruction (loss用)
        A_self, A_recon = self.encoderA(
            xA, session_idx=session_idx, mask=mask
        )

        # B_self:   [B,T,D]
        # B_recon:  reconstruction (loss用)
        B_self, B_recon = self.encoderB(
            xB, session_idx=session_idx, mask=mask
        )

        # ------------------------------------------------------------------
        # 方向1: B视角 (B当query，看A)
        #
        # Query = B_self
        # Key/Value = A_self
        #
        # 用 gateB
        # ------------------------------------------------------------------
        h_Bview = self._cross_direction(
            q=B_self,
            kv=A_self,
            gate_layer=self.gateB,
        )  # [B,T,D]

        # ------------------------------------------------------------------
        # 方向2: A视角 (A当query，看B)
        #
        # Query = A_self
        # Key/Value = B_self
        #
        # 用 gateA
        # ------------------------------------------------------------------
        h_Aview = self._cross_direction(
            q=A_self,
            kv=B_self,
            gate_layer=self.gateA,
        )  # [B,T,D]

        # === 合并两个方向（简单平均，保持维度不变）===
        h_merged = 0.5 * (h_Bview + h_Aview)  # [B,T,D]

        # === 最终投影 + L2 normalize ===
        h_final = self.projection(h_merged)    # [B,T,D]
        h_final = F.normalize(h_final, dim=-1) # [B,T,D]

        return FusionOutput(
            fused=h_final,
            imu_self=A_self,
            dlc_self=B_self,
            imu_recon=A_recon,
            dlc_recon=B_recon,
        )
