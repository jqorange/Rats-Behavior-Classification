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
    双向跨模态注意力版本。

    - encoderA 负责模态A (比如 IMU)
        返回:
            A_self   : A在自己空间里的表示
            A_to_B   : A预测到B空间的表示
            A_recon  : (可选) A的重构，用于mask/recon loss

    - encoderB 负责模态B (比如 DLC)
        返回:
            B_self, B_to_A, B_recon (对称)

    - 我们做两条 cross-attn 路线：
        1. B视角去关注A:
            Query ∈ {B_self, A_to_B}
            Key/Val ∈ {A_self, B_to_A}
            用 gateB 作为门控
        2. A视角去关注B:
            Query ∈ {A_self, B_to_A}
            Key/Val ∈ {B_self, A_to_B}
            用 gateA 作为门控

      每条路线上，我们仍然用那三个可选combo:
        combos = [[0,0], [0,1], [1,0]]
        含义是 (which_query, which_keyval)

      我们对 batch 逐样本随机采样 combo_idx（或用 attn_mode 指定）。
      然后分别得到 h_Bview, h_Aview。

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

    @torch.no_grad()
    def _sample_combo_indices(self, device: torch.device, batch_size: int) -> torch.Tensor:
        """
        combos = [[0,0],[0,1],[1,0]]
        返回 [B] 的索引，逐样本决定用哪个(q_sel, kv_sel)组合
        """
        combos = torch.tensor([[0, 0], [0, 1], [1, 0]], device=device)
        return torch.randint(0, combos.size(0), (batch_size,), device=device)

    def _pick_combo_idx(self, device, B, attn_mode: Optional[str]):
        """
        根据 attn_mode (None/'imu'/'dlc'/'both') 决定 combo_idx。
        这里我们保持和原版一致的约定：
            'both' -> combo 0
            'dlc'  -> combo 1
            'imu'  -> combo 2
        None    -> 每个样本随机
        """
        combos = torch.tensor([[0, 0], [0, 1], [1, 0]], device=device)

        if attn_mode is None:
            combo_idx = self._sample_combo_indices(device, B)
        else:
            am = attn_mode.lower()
            if am == 'imu':
                combo_idx = combos.new_full((B,), 2)
            elif am == 'dlc':
                combo_idx = combos.new_full((B,), 1)
            elif am == 'both':
                combo_idx = combos.new_full((B,), 0)
            else:
                raise ValueError(f"Unknown attn_mode: {attn_mode}")
        return combos, combo_idx

    def _cross_direction(
        self,
        q_candidates: torch.Tensor,     # [2, B, T, D]
        kv_candidates: torch.Tensor,    # [2, B, T, D]
        combos: torch.Tensor,           # [3,2] each row = (iq,ikv)
        combo_idx: torch.Tensor,        # [B]
        gate_layer: nn.Linear,          # gateA or gateB
    ) -> torch.Tensor:
        """
        对一个方向跑三种(q,kv)组合的cross-attn，然后按 combo_idx 挑出对应结果。
        返回形状 [B, T, D] (还没projection，仅门控融合+norm+clamp)
        """

        B = q_candidates.shape[1]
        device = q_candidates.device

        # 先把三种组合都跑一遍 cross_attn
        all_attn = []
        for iq, ikv in combos:
            q_i = q_candidates[iq]          # [B, T, D]
            kv_i = kv_candidates[ikv]       # [B, T, D]

            # MultiheadAttention(batch_first=True): q/k/v -> [B,T,D]
            # 返回 (out, attn_weights). 我们不需要权重
            m_i, _ = self.cross_attn(
                query=q_i,
                key=kv_i,
                value=kv_i,
                need_weights=False,
            )  # m_i: [B,T,D]

            all_attn.append(m_i)

        # 堆叠成 [3, B, T, D]
        attn_stack = torch.stack(all_attn, dim=0)

        # 针对每个样本，挑出它的 query 以及对应的attn结果
        batch_indices = torch.arange(B, device=device)

        combo_pairs = combos[combo_idx]  # [B,2], (which_q, which_kv)
        q_sel = q_candidates[combo_pairs[:, 0], batch_indices]      # [B,T,D]
        m_sel = attn_stack[combo_idx, batch_indices]                # [B,T,D]

        # 门控残差：q_sel + g * m_sel
        g = torch.sigmoid(gate_layer(q_sel))  # [B,T,1]
        h_dir = q_sel + self.dropout(g * m_sel)  # [B,T,D]

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
            attn_mode: 控制 combo 选择; None表示每个样本随机

        Returns:
            FusionOutput:
                fused         [B,T,D]  (双向cross后融合+proj+L2)
                imu_self      [B,T,D]
                dlc_self      [B,T,D]
                imu_to_dlc    [B,T,D]
                dlc_to_imu    [B,T,D]
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
        # A_to_B:   [B,T,D]  (A估计成B空间)
        # A_recon:  reconstruction (loss用)
        A_self, A_to_B, A_recon = self.encoderA(
            xA, session_idx=session_idx, mask=mask
        )

        # B_self:   [B,T,D]
        # B_to_A:   [B,T,D]  (B估计成A空间)
        # B_recon:  reconstruction (loss用)
        B_self, B_to_A, B_recon = self.encoderB(
            xB, session_idx=session_idx, mask=mask
        )

        # === 组合index的选择 (对两个方向我们共享同一套 combo_idx 逻辑) ===
        combos, combo_idx = self._pick_combo_idx(device, Bsz, attn_mode)
        # combos: [3,2]  e.g. [[0,0],[0,1],[1,0]]

        # ------------------------------------------------------------------
        # 方向1: B视角 (B当query，看A)
        #
        # Query candidates:       {B_self, A_to_B}        shape -> stack dim0=2
        # Key/Value candidates:   {A_self, B_to_A}
        #
        # 用 gateB
        # ------------------------------------------------------------------
        q_candidates_Bview = torch.stack((B_self, A_to_B), dim=0)   # [2,B,T,D]
        kv_candidates_Bview = torch.stack((A_self, B_to_A), dim=0)  # [2,B,T,D]

        h_Bview = self._cross_direction(
            q_candidates=q_candidates_Bview,
            kv_candidates=kv_candidates_Bview,
            combos=combos,
            combo_idx=combo_idx,
            gate_layer=self.gateB,
        )  # [B,T,D]

        # ------------------------------------------------------------------
        # 方向2: A视角 (A当query，看B)
        #
        # Query candidates:       {A_self, B_to_A}
        # Key/Value candidates:   {B_self, A_to_B}
        #
        # 用 gateA
        # ------------------------------------------------------------------
        q_candidates_Aview = torch.stack((A_self, B_to_A), dim=0)   # [2,B,T,D]
        kv_candidates_Aview = torch.stack((B_self, A_to_B), dim=0)  # [2,B,T,D]

        h_Aview = self._cross_direction(
            q_candidates=q_candidates_Aview,
            kv_candidates=kv_candidates_Aview,
            combos=combos,
            combo_idx=combo_idx,
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
            imu_to_dlc=A_to_B,
            dlc_to_imu=B_to_A,
            imu_recon=A_recon,
            dlc_recon=B_recon,
        )
