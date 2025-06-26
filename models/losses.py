import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.tools import take_per_row
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


def compute_contrastive_losses(self, xA, xB, labels, fused_repr,session_idx, is_supervised=True):
    """Compute either supervised or unsupervised contrastive loss based on mode."""

    if is_supervised:
        # Use the entire sequence for supervised contrastive learning in stage 2
        sup_loss = multilabel_supcon_loss_bt(fused_repr, labels)
        return sup_loss

    else:
        # === Unsupervised contrastive learning ===
        B, T, _ = xA.shape

        if T <= 5:
            return torch.tensor(0.0, device=xA.device)



        # ----- Step 2: two random crops following the TS2Vec strategy -----
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=T + 1)
        crop_left = np.random.randint(T - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=T + 1)
        crop_offset = torch.randint(low=-crop_eleft, high=T - crop_eright + 1, size=(B,), device=xA.device)

        xA_crop1 = take_per_row(xA, crop_offset + crop_eleft, crop_right - crop_eleft)
        xB_crop1 = take_per_row(xB, crop_offset + crop_eleft, crop_right - crop_eleft)
        xA_crop2 = take_per_row(xA, crop_offset + crop_left, crop_eright - crop_left)
        xB_crop2 = take_per_row(xB, crop_offset + crop_left, crop_eright - crop_left)


        out1 = self.encoder_fusion(xA_crop1, xB_crop1,session_idx)
        out1 = out1[:, -crop_l:]
        out2 = self.encoder_fusion(xA_crop2, xB_crop2, session_idx)
        out2 = out2[:, :crop_l]
        #
        # ----- Step 3: jitter after encoding -----
        jitter_std = 0.01
        out1 = out1 + torch.randn_like(out1) * jitter_std
        out2 = out2 + torch.randn_like(out2) * jitter_std

        # ----- Step 4: main hierarchical contrastive loss -----
        loss_main = hierarchical_contrastive_loss(out1, out2, temporal_unit=self.temporal_unit)


        return loss_main
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

def batch_js_divergence(features: torch.Tensor, session_ids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute average JS divergence between session feature distributions."""
    unique_sessions = session_ids.unique()
    if unique_sessions.numel() <= 1:
        return features.new_tensor(0.0)

    probs = []
    for s in unique_sessions:
        mask = session_ids == s
        feat = features[mask].mean(dim=(0, 1))
        probs.append(torch.softmax(feat, dim=-1))

    js = 0.0
    count = 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            p = probs[i]
            q = probs[j]
            m = 0.5 * (p + q)
            js += 0.5 * (F.kl_div(p.log(), m, reduction="sum") + F.kl_div(q.log(), m, reduction="sum"))
            count += 1

    return js / max(count, 1)