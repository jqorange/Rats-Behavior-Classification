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

def multilabel_supcon_loss_bt(z, y, temperature=0.07, eps=1e-8):
    """
    z: (B, T, D) - time series embeddings
    y: (B, N)    - multi-label binary tags
    This version applies max pooling across time to reduce computation.
    """
    B, T, D = z.shape

    # === Max Pooling across time dimension ===
    z = F.max_pool1d(z.transpose(1, 2), kernel_size=T).squeeze(-1)  # (B, D)
    z = F.normalize(z, dim=-1)

    y = y.float()  # (B, N)

    # Compute cosine similarity
    sim = torch.matmul(z, z.T) / temperature  # (B, B)

    # Remove diagonal (self-similarity)
    logits_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim[logits_mask].view(B, B - 1)

    # Compute Jaccard similarity between labels
    inter = torch.matmul(y, y.T)  # (B, B)
    union = (y.unsqueeze(1) + y.unsqueeze(0)).clamp(max=1).sum(-1)
    jaccard = inter / (union + eps)  # (B, B)

    jaccard_masked = jaccard[logits_mask].view(B, B - 1)
    weights = jaccard_masked / (jaccard_masked.sum(dim=1, keepdim=True) + eps)

    log_prob = F.log_softmax(sim, dim=1)
    loss = -(weights * log_prob).sum(dim=1).mean()

    return loss


def compute_contrastive_losses(self, xA, xB, labels, fused_repr,session_idx, is_supervised=True, stage=2):
    """Compute either supervised or unsupervised contrastive loss based on mode."""

    if is_supervised:
        # Use the entire sequence for supervised contrastive learning in stage 2
        if stage == 2:
            sup_loss = positive_only_supcon_loss(fused_repr, labels)
        else:
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


class PrototypeMemory(nn.Module):
    """Maintain class prototypes and provide prototype-based loss."""

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.initialized = False

    @torch.no_grad()
    def assign_labels(self, feats: torch.Tensor) -> torch.Tensor:
        """Assign pseudo labels based on cosine similarity to prototypes."""
        feats = F.normalize(feats, dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)
        sims = torch.matmul(feats, protos.t())
        return sims.argmax(dim=1)


    @torch.no_grad()
    def update(self, feats_sup: torch.Tensor | None, labels_sup: torch.Tensor | None,
               feats_unsup: torch.Tensor | None = None, pseudo: torch.Tensor | None = None) -> None:
        """Update prototypes using labelled data and pseudo labels with soft weighting."""

        # === Determine device ===
        device = None
        if feats_sup is not None:
            device = feats_sup.device
        elif feats_unsup is not None:
            device = feats_unsup.device
        else:
            return

        new_protos = self.prototypes.clone().to(device)

        # ---- Initialization ----
        if not self.initialized:
            if feats_sup is not None and labels_sup is not None:
                feats_sup = F.normalize(feats_sup, dim=-1)
                for c in range(self.num_classes):
                    mask = labels_sup[:, c].bool()
                    if mask.any():
                        new_protos[c] = feats_sup[mask].mean(0)
            self.prototypes = F.normalize(new_protos, dim=-1)
            self.initialized = True
            return

        current_protos = F.normalize(self.prototypes.to(device), dim=-1)
        if feats_sup is not None and labels_sup is not None:
            feats_sup = F.normalize(feats_sup, dim=-1)
            for c in range(self.num_classes):
                mask = labels_sup[:, c].bool()
                if mask.any():
                    feats_c = feats_sup[mask]
                    sims = torch.matmul(feats_c, current_protos[c])  # (Nc,)
                    weights = torch.softmax(sims, dim=0)
                    new_protos[c] = (weights.unsqueeze(1) * feats_c).sum(0)

        if feats_unsup is not None and pseudo is not None:
            feats_unsup = F.normalize(feats_unsup, dim=-1)
            for c in range(self.num_classes):
                mask = pseudo == c
                if mask.any():
                    feats_c = feats_unsup[mask]
                    sims = torch.matmul(feats_c, current_protos[c])  # (Nc,)
                    weights = torch.softmax(sims, dim=0)
                    feat_c = (weights.unsqueeze(1) * feats_c).sum(0)
                    if new_protos[c].abs().sum() == 0:
                        new_protos[c] = feat_c
                    else:
                        new_protos[c] = 0.5 * (new_protos[c] + feat_c)

        self.prototypes = F.normalize(new_protos, dim=-1)
        self.initialized = True

    def forward(self, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Prototype soft alignment loss without thresholding."""
        feats = F.normalize(feats, dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)

        logits = torch.matmul(feats, protos.t())  # (B, C)
        probs = logits.softmax(dim=-1).detach()
        weights = probs[torch.arange(len(feats)), targets]

        loss = F.cross_entropy(logits, targets, reduction="none")
        return (loss * weights).mean()

