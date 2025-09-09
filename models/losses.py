import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.tools import take_per_row


def gaussian_cs_divergence(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Closed-form Cauchy–Schwarz divergence under Gaussian approximation.
    Uses covariance shrinkage and Cholesky for numerical stability.

    Args:
        x, y: tensors of shape [B, T, D]
        eps: base jitter used inside the Cholesky helper

    Returns:
        Scalar tensor (CS divergence).
    """
    # ---- helpers ----
    def _flatten_bt(z: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [N,D]
        return z.reshape(-1, z.size(-1))

    def _cov(z: torch.Tensor) -> torch.Tensor:
        # unbiased sample covariance (symmetrized)
        zc = z - z.mean(dim=0, keepdim=True)
        n = max(z.size(0) - 1, 1)
        C = (zc.T @ zc) / n
        return 0.5 * (C + C.T)

    def _shrink(C: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
        # Ledoit–Wolf style shrinkage: (1-λ)C + λ*(tr(C)/D)I
        d = C.size(0)
        tr_over_d = torch.trace(C) / d
        I = torch.eye(d, device=C.device, dtype=C.dtype)
        return (1.0 - lam) * C + lam * tr_over_d * I

    def _chol_logdet(C: torch.Tensor, base_eps: float = 1e-6, tries: int = 5):
        # Cholesky with adaptive jitter, returns (L, logdet)
        d = C.size(0)
        I = torch.eye(d, device=C.device, dtype=C.dtype)
        jitter = base_eps
        for _ in range(tries):
            L, info = torch.linalg.cholesky_ex(C + jitter * I)
            if torch.all(info == 0):
                logdet = 2.0 * torch.log(torch.diag(L)).sum()
                return L, logdet
            jitter *= 10.0
        # fallback: slogdet
        sign, logabsdet = torch.slogdet(C + jitter * I)
        if sign <= 0:
            # extreme fallback (should be rare)
            logabsdet = torch.logdet(C + (jitter * 10) * I)
        return None, logabsdet

    # ---- main ----
    # mixed precision 下禁用 autocast，避免半精度数值不稳
    with torch.cuda.amp.autocast(enabled=False):
        x = _flatten_bt(x).float()
        y = _flatten_bt(y).float()

        d = x.size(-1)
        mu_x = x.mean(dim=0, keepdim=True)  # [1,D]
        mu_y = y.mean(dim=0, keepdim=True)

        Cx = _cov(x)
        Cy = _cov(y)

        # 收缩 + 轻度抖动（由 _chol_logdet 自适应增加）
        Cx = _shrink(Cx, lam=0.1)
        Cy = _shrink(Cy, lam=0.1)
        S  = Cx + Cy  # 注意 Gaussian CS 用 Σx + Σy（非均值）

        # logdet 部分（更稳）
        Lx, logdet_x = _chol_logdet(Cx, base_eps=eps)
        Ly, logdet_y = _chol_logdet(Cy, base_eps=eps)
        LS, logdet_S = _chol_logdet(S,  base_eps=eps)

        # 二次型 (μx-μy)^T (Σx+Σy)^{-1} (μx-μy)，用 Cholesky 解线性方程避免显式逆
        delta = (mu_x - mu_y).T  # [D,1]
        if LS is not None:
            v = torch.cholesky_solve(delta, LS)    # = S^{-1} delta
            quad = (delta * v).sum()
        else:
            # 兜底（极少触发）：对称伪逆
            Sinv = torch.linalg.pinvh(S)
            quad = (delta.T @ (Sinv @ delta)).squeeze()

        # 完整的 Gaussian CS divergence
        # D = -d/2 log 2 - 1/4 log|Σx| - 1/4 log|Σy| + 1/2 log|Σx+Σy|
        #     + 1/2 (μx-μy)^T (Σx+Σy)^{-1} (μx-μy)
        const = -0.5 * d * torch.log(torch.tensor(2.0, dtype=S.dtype, device=S.device))
        D = const - 0.25 * logdet_x - 0.25 * logdet_y + 0.5 * logdet_S + 0.5 * quad
        return D
def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=3):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def positive_only_supcon_loss(z, y, temperature=0.07, eps=1e-8, return_score=False):
    """
    Only attracts samples with high label overlap (positive Jaccard),
    without pushing negatives apart.
    Output is a positive loss: lower means better alignment.

    Args:
        z: (B, T, D) - time series embeddings
        y: (B, N)    - multi-label binary tags
        return_score: whether to return alignment score (for logging)

    Returns:
        loss: positive scalar
        score (optional): average alignment score in [0, 1]
    """
    B, T, D = z.shape

    # Max-pool over time
    z = F.max_pool1d(z.transpose(1, 2), kernel_size=T).squeeze(-1)  # (B, D)
    z = F.normalize(z, dim=-1)

    y = y.float()  # (B, N)

    # Cosine similarity (B, B)
    sim = torch.matmul(z, z.T) / temperature

    # === Positive weights only ===
    inter = torch.matmul(y, y.T)
    union = (y.unsqueeze(1) + y.unsqueeze(0)).clamp(max=1).sum(-1)
    jaccard = inter / (union + eps)  # (B, B)

    # Mask out self-similarity
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim[mask].view(B, B - 1)
    jaccard = jaccard[mask].view(B, B - 1)

    # Only use jaccard as positive weights
    weights = jaccard / (jaccard.sum(dim=1, keepdim=True) + eps)

    # Weighted positive-only similarity
    pos_sim = (weights * sim).sum(dim=1)  # (B,)
    score = pos_sim.mean()  # average alignment score

    # Positive loss: lower = less aligned
    loss = 1.0 - score

    return loss

def multilabel_supcon_loss_bt(z, y, temperature=0.07, eps=1e-8, topk: int = 64):
    B, T, D = z.shape
    z = F.max_pool1d(z.transpose(1, 2), kernel_size=T).squeeze(-1)  # (B,D)
    z = F.normalize(z, dim=-1)
    y = y.float()

    # Cosine similarity
    sim = torch.matmul(z, z.T) / temperature
    sim.fill_diagonal_(-1e4)

    # Jaccard similarity
    inter = y @ y.T
    y_sum = y.sum(-1, keepdim=True)
    union = y_sum + y_sum.T - inter
    jaccard = inter / (union + eps)
    jaccard.fill_diagonal_(0)

    # Top-K neighbors only
    topk_vals, topk_idx = torch.topk(jaccard, k=min(topk, B-1), dim=1)  # (B,topk)

    # Gather similarity & log-probs
    log_prob = F.log_softmax(sim, dim=1)  # (B,B)
    log_prob_topk = torch.gather(log_prob, 1, topk_idx)  # (B,topk)

    # Normalize weights
    weights = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + eps)

    # Weighted contrastive loss
    loss = -(weights * log_prob_topk).sum(dim=1).mean()
    return loss




