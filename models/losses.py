import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_bt(z: torch.Tensor) -> torch.Tensor:
    if z.dim() < 2:
        raise ValueError("Input tensor must have at least two dimensions.")
    return z.reshape(-1, z.size(-1))


def _cov(z: torch.Tensor) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    n = max(z.size(0) - 1, 1)
    C = (zc.T @ zc) / n
    return 0.5 * (C + C.T)


def _shrink(C: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    d = C.size(0)
    tr_over_d = torch.trace(C) / d
    I = torch.eye(d, device=C.device, dtype=C.dtype)
    return (1.0 - lam) * C + lam * tr_over_d * I


def _chol_logdet(C: torch.Tensor, base_eps: float = 1e-6, tries: int = 5):
    d = C.size(0)
    I = torch.eye(d, device=C.device, dtype=C.dtype)
    jitter = base_eps
    for _ in range(tries):
        L, info = torch.linalg.cholesky_ex(C + jitter * I)
        if torch.all(info == 0):
            logdet = 2.0 * torch.log(torch.diag(L)).sum()
            return L, logdet
        jitter *= 10.0
    sign, logabsdet = torch.slogdet(C + jitter * I)
    if sign <= 0:
        logabsdet = torch.logdet(C + (jitter * 10) * I)
    return None, logabsdet


def gaussian_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Covariance-regularised KL divergence under Gaussian assumption."""

    with torch.cuda.amp.autocast(enabled=False):
        p = _flatten_bt(p).float()
        q = _flatten_bt(q).float()

        if p.size(0) == 0 or q.size(0) == 0:
            device = p.device if p.numel() > 0 else q.device
            return torch.tensor(0.0, device=device)

        mu_p = p.mean(dim=0, keepdim=True)
        mu_q = q.mean(dim=0, keepdim=True)

        Cp = _shrink(_cov(p), lam=0.1)
        Cq = _shrink(_cov(q), lam=0.1)

        Lp, logdet_p = _chol_logdet(Cp, base_eps=eps)
        Lq, logdet_q = _chol_logdet(Cq, base_eps=eps)

        delta = (mu_q - mu_p).T
        if Lq is not None:
            Cq_inv = torch.cholesky_inverse(Lq, upper=False)
            quad = (delta * torch.cholesky_solve(delta, Lq)).sum()
        else:
            Cq_inv = torch.linalg.pinvh(Cq)
            quad = (delta.T @ (Cq_inv @ delta)).squeeze()

        trace_term = torch.sum(Cq_inv * Cp)
        d = p.size(-1)
        kl = 0.5 * (logdet_q - logdet_p - d + trace_term + quad)
        return kl


def gaussian_kl_divergence_masked(
    p: torch.Tensor, q: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """KL divergence that only considers entries where ``mask`` is valid."""

    if mask is None:
        return gaussian_kl_divergence(p, q)

    mask_f = mask.to(p.device)
    if mask_f.dtype != torch.bool:
        mask_f = mask_f > 0.5
    mask_flat = mask_f.reshape(-1)
    if mask_flat.sum() == 0:
        return p.new_tensor(0.0)

    p_flat = p.reshape(-1, p.size(-1))
    q_flat = q.reshape(-1, q.size(-1))
    p_sel = p_flat[mask_flat]
    q_sel = q_flat[mask_flat]
    if p_sel.numel() == 0 or q_sel.numel() == 0:
        return p.new_tensor(0.0)

    return gaussian_kl_divergence(p_sel, q_sel)
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


def sequential_next_step_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute next-step Gaussian NLL optionally restricted by a mask."""

    if prediction.size(1) < 2 or target.size(1) < 2:
        return prediction.new_tensor(0.0)

    pred_curr = prediction[:, :-1]
    target_next = target[:, 1:]
    diff = target_next - pred_curr
    log_const = prediction.new_tensor(0.5 * math.log(2.0 * math.pi), dtype=prediction.dtype)
    nll = 0.5 * diff.pow(2) + log_const

    if mask is None:
        return nll.mean()

    mask_curr = mask[:, :-1]
    mask_next = mask[:, 1:]
    if mask_curr.dtype != torch.bool:
        mask_curr = mask_curr > 0.5
    if mask_next.dtype != torch.bool:
        mask_next = mask_next > 0.5
    mask_valid = mask_curr & mask_next
    valid_count = mask_valid.sum()
    if valid_count.item() == 0:
        return prediction.new_tensor(0.0)

    nll = nll * mask_valid.unsqueeze(-1)
    denom = valid_count * prediction.size(-1)
    denom = denom.to(nll.dtype)
    return nll.sum() / denom

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


class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = self.alpha * torch.pow(1.0 - pt, self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss




