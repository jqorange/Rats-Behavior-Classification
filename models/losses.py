import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.tools import take_per_row
def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
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



def multilabel_supcon_loss_bt(z, y, temperature=0.07, eps=1e-8):
    """
    z: (B, T, D) - time series embeddings
    y: (B, N)    - multi-label binary tags
    """
    B, T, D = z.shape
    z = F.normalize(z, dim=-1)
    z = z.reshape(B * T, D)  # (BT, D)
    y = y.float().repeat_interleave(T, dim=0)  # (BT, N)

    # Compute similarity
    sim = torch.matmul(z, z.T) / temperature  # (BT, BT)

    # Remove diagonal
    logits_mask = ~torch.eye(B * T, dtype=torch.bool, device=z.device)
    sim = sim[logits_mask].view(B * T, B * T - 1)

    # Compute Jaccard label similarity
    inter = torch.matmul(y, y.T)  # (BT, BT)
    union = (y.unsqueeze(1) + y.unsqueeze(0)).clamp(max=1).sum(-1)  # (BT, BT)
    jaccard = inter / (union + eps)

    jaccard_masked = jaccard[logits_mask].view(B * T, B * T - 1)
    weights = jaccard_masked / (jaccard_masked.sum(dim=1, keepdim=True) + eps)

    log_prob = F.log_softmax(sim, dim=1)
    loss = -(weights * log_prob).sum(dim=1).mean()

    return loss


def compute_contrastive_losses(self, xA, xB, labels, fused_repr, is_supervised=True):
    """Compute either supervised or unsupervised contrastive loss based on mode."""

    if is_supervised:
        # Only use the center 11 frames for supervised contrastive learning
        B, T, _ = fused_repr.shape
        start = max(0, T // 2 - 5)
        end = min(start + 11, T)
        center_feat = fused_repr[:, start:end]
        sup_loss = multilabel_supcon_loss_bt(center_feat, labels)
        return sup_loss

    else:
        # === Unsupervised contrastive learning ===
        B, T, _ = xA.shape

        if T <= 5:
            return torch.tensor(0.0, device=xA.device)

            # ----- Step 1: random masking (exclude center 5 frames) -----
        center_start = T // 2 - 2
        center_start = max(center_start, 0)
        center_end = min(center_start + 5, T)

        mask = (torch.rand(B, T, device=xA.device) < 0.3)
        mask[:, center_start:center_end] = False
        xA_masked = xA.clone()
        xB_masked = xB.clone()
        xA_masked[mask] = 0.0
        xB_masked[mask] = 0.0

        # ----- Step 2: two random crops of length T-5 keeping the center region -----
        crop_len = T - 5
        start_min = max(0, center_end - crop_len)
        start_max = min(center_start, T - crop_len)
        if start_max < start_min:
            start_max = start_min

        offset1 = torch.randint(start_min, start_max + 1, (B,), device=xA.device)
        offset2 = torch.randint(start_min, start_max + 1, (B,), device=xA.device)

        xA_crop1 = take_per_row(xA_masked, offset1, crop_len)
        xB_crop1 = take_per_row(xB_masked, offset1, crop_len)
        xA_crop2 = take_per_row(xA_masked, offset2, crop_len)
        xB_crop2 = take_per_row(xB_masked, offset2, crop_len)

        out1 = self.encoder_fusion(xA_crop1, xB_crop1)
        out2 = self.encoder_fusion(xA_crop2, xB_crop2)

        # ----- Step 3: jitter after encoding -----
        jitter_std = 0.02
        out1 = out1 + torch.randn_like(out1) * jitter_std
        out2 = out2 + torch.randn_like(out2) * jitter_std

        # ----- Step 4: main hierarchical contrastive loss -----
        loss_main = hierarchical_contrastive_loss(out1, out2, temporal_unit=self.temporal_unit)

        # ----- Step 5: additional loss on pooled center region -----
        center_off1 = (center_start - offset1).clamp(0, crop_len - 5)
        center_off2 = (center_start - offset2).clamp(0, crop_len - 5)
        part1 = take_per_row(out1, center_off1, 5).max(dim=1)[0].unsqueeze(1)
        part2 = take_per_row(out2, center_off2, 5).max(dim=1)[0].unsqueeze(1)
        loss_center = hierarchical_contrastive_loss(part1, part2, temporal_unit=self.temporal_unit)

        return loss_main + loss_center
class CenterLoss(nn.Module):
    """Center loss that encourages features of the same class to cluster."""

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss.

        Args:
            features: (B, T, D) sequence features from the encoder.
            labels:   (B, C) multi-hot labels for each sequence.
        """
        B, T, D = features.shape
        features = features.reshape(B * T, D)
        labels = labels.float().unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        # each feature may belong to multiple classes; average over assigned classes
        mask = labels > 0
        if mask.sum() == 0:
            return features.new_tensor(0.0)

        # === fix device mismatch ===
        centers = self.centers.to(features.device).unsqueeze(0).expand(B * T, -1, -1)

        diff = features.unsqueeze(1) - centers
        dist = (diff.pow(2).sum(-1) + 1e-6).sqrt()
        loss = (dist * mask).sum() / mask.sum()
        return loss


class PrototypeClusterLoss(nn.Module):
    """Prototype driven cluster loss for unlabeled data."""

    def __init__(self, num_prototypes: int, feat_dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Align features with closest prototypes."""
        B, T, D = features.shape
        feats = F.normalize(features.reshape(B * T, D), dim=-1)
        protos = F.normalize(self.prototypes.to(features.device), dim=-1)  # 加这一行

        sim = torch.matmul(feats, protos.t())  # (BT, P)
        loss = -sim.max(dim=1)[0].mean()
        return loss


class UncertaintyWeighting(nn.Module):
    """Learnable uncertainty-based weighting for multiple losses."""

    def __init__(self, num_losses: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: list) -> torch.Tensor:
        assert len(losses) == len(self.log_vars)
        weighted = []
        for i, L in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted.append(precision * L + self.log_vars[i])
        return sum(weighted)
