import torch
import torch.nn.functional as F
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

def compute_smoothness_loss(self, predictions, min_duration=None):
    """
    Compute smoothness loss for temporal consistency
    Penalize rapid label changes - labels should maintain same state for at least min_duration frames
    predictions: (B, T, C) - logits
    """
    if min_duration is None:
        min_duration = self.smooth_window  # Default to 5 frames

    B, T, C = predictions.shape
    if T < min_duration:
        return torch.tensor(0.0, device=predictions.device)

    # Convert logits to binary predictions (0.5 threshold after sigmoid)
    probs = torch.sigmoid(predictions)  # (B, T, C)
    binary_preds = (probs > 0.5).float()  # (B, T, C)

    smooth_loss = 0.0

    for b in range(B):
        for c in range(C):
            # Get binary predictions for this batch and class
            seq = binary_preds[b, :, c]  # (T,)

            # Find state changes
            changes = torch.abs(seq[1:] - seq[:-1])  # (T-1,) - 1 where state changes
            change_indices = torch.where(changes > 0)[0]  # Indices where changes occur

            if len(change_indices) == 0:
                continue  # No changes, perfect smoothness

            # Add boundaries (start and end of sequence)
            change_indices = torch.cat([
                torch.tensor([-1], device=predictions.device),
                change_indices,
                torch.tensor([T - 1], device=predictions.device)
            ])

            # Compute durations of each state
            durations = change_indices[1:] - change_indices[:-1]  # Duration of each state

            # Penalize states that are too short (< min_duration)
            short_states = durations < min_duration
            if short_states.any():
                # Penalty is inversely proportional to duration
                # Shorter states get higher penalty
                penalties = torch.clamp(min_duration - durations[short_states], min=0)
                penalty = penalties.float().sum() / min_duration  # Normalize
                smooth_loss += penalty

    # Average over batch and classes
    smooth_loss = smooth_loss / (B * C)

    return smooth_loss

def compute_contrastive_losses(self, xA, xB, labels, fused_repr, is_supervised=True):
    """Compute either supervised or unsupervised contrastive loss based on mode."""

    if is_supervised:
        # Supervised contrastive loss
        sup_loss = multilabel_supcon_loss_bt(fused_repr, labels)
        return sup_loss  # Return sup_loss, unsup_loss=0

    else:
        # Unsupervised hierarchical contrastive loss
        B, T, _ = xA.shape

        if T <= 4:  # Too short for cropping
            unsup_loss = torch.tensor(0.0, device=xA.device)
        else:
            # Random cropping for unsupervised loss
            crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=T + 1)
            crop_left = np.random.randint(T - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=T + 1)
            crop_offset = np.random.randint(
                low=-crop_eleft,
                high=T - crop_eright + 1,
                size=B
            )

            # First crop
            xA_crop1 = take_per_row(xA, crop_offset + crop_eleft, crop_right - crop_eleft)
            xB_crop1 = take_per_row(xB, crop_offset + crop_eleft, crop_right - crop_eleft)
            out1 = self.encoder_fusion(xA_crop1, xB_crop1)
            out1 = out1[:, -crop_l:]

            # Second crop
            xA_crop2 = take_per_row(xA, crop_offset + crop_left, crop_eright - crop_left)
            xB_crop2 = take_per_row(xB, crop_offset + crop_left, crop_eright - crop_left)
            out2 = self.encoder_fusion(xA_crop2, xB_crop2)
            out2 = out2[:, :crop_l:]

            # Hierarchical contrastive loss
            unsup_loss = hierarchical_contrastive_loss(
                out1, out2,
                temporal_unit=self.temporal_unit
            )

        return unsup_loss  # Return sup_loss=0, unsup_loss