import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.tools import take_per_row


def gaussian_cs_divergence(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Closed-form Cauchy-Schwarz divergence under Gaussian approximation.

    Args:
        x, y: tensors of shape ``[B, T, D]`` representing two sets of features.
        eps: small value added to covariances for numerical stability.

    Returns:
        Scalar tensor representing the divergence.
    """

    B, T, D = x.shape
    x_flat = x.reshape(-1, D)
    y_flat = y.reshape(-1, D)

    mu_x = x_flat.mean(dim=0)
    mu_y = y_flat.mean(dim=0)

    xc = x_flat - mu_x
    yc = y_flat - mu_y

    cov_x = (xc.T @ xc) / (x_flat.size(0) - 1)
    cov_y = (yc.T @ yc) / (y_flat.size(0) - 1)

    eye = torch.eye(D, device=x.device)
    cov_x = cov_x + eps * eye
    cov_y = cov_y + eps * eye

    cov_mean = 0.5 * (cov_x + cov_y)

    # Eigendecomposition for stable inversion and logdet
    eig_mean, vec_mean = torch.linalg.eigh(cov_mean)
    eig_x, _ = torch.linalg.eigh(cov_x)
    eig_y, _ = torch.linalg.eigh(cov_y)

    # Clamp eigenvalues to ensure positive definiteness
    eig_mean = eig_mean.clamp_min(eps)
    eig_x = eig_x.clamp_min(eps)
    eig_y = eig_y.clamp_min(eps)

    inv_cov = vec_mean @ torch.diag(eig_mean.reciprocal()) @ vec_mean.t()
    diff = (mu_x - mu_y).unsqueeze(0)  # 1 x D
    term_mean = 0.25 * (diff @ inv_cov @ diff.t()).squeeze()

    logdet_mean = torch.log(eig_mean).sum()
    logdet_x = torch.log(eig_x).sum()
    logdet_y = torch.log(eig_y).sum()
    term_cov = 0.5 * logdet_mean - 0.25 * (logdet_x + logdet_y)

    return term_mean + term_cov
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




