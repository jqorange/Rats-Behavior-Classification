import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from models.deep_mlp import DeepMLPClassifier
from utils.segments import collect_segment_centres, compute_segments, split_segments_by_action

# ------------------------------
# 全局行为类别
# ------------------------------
BEHAVIORS = [
    "walk",
    "jump",
    "aiming",
    "scratch",
    "rearing",
    "stand_up",
    "still",
    "eating",
    "grooming",
    "local_search",
    "turn_left",
    "turn_right",
]


# ------------------------------
# Dataset 封装
# ------------------------------
class ReprDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features.float()
        self.labels = labels.float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ------------------------------
# Session container for splits
# ------------------------------
@dataclass
class SessionSplitData:
    features: torch.Tensor
    index_to_row: Dict[int, int]
    label_lookup: Dict[int, int]
    label_vectors: torch.Tensor


# ------------------------------
# Label helper
# ------------------------------
def _row_to_behavior_vector(row: pd.Series, present_cols: List[str]) -> torch.Tensor:
    vec = np.zeros(len(BEHAVIORS), dtype=np.float32)
    for k, beh in enumerate(BEHAVIORS):
        if beh in present_cols:
            vec[k] = float(row[beh])
        else:
            vec[k] = 0.0
    return torch.from_numpy(vec)


def _collect_split_tensors(
    session_data: Dict[str, SessionSplitData],
    segments_by_session: Dict[str, Dict[str, List]],
    assignments: Dict[str, Set],
    split: str,
    feature_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    frames_per_session = collect_segment_centres(
        segments_by_session,
        assignments,
        split,
        deduplicate=True,
        include="all",
    )

    feat_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []

    for session, frames in frames_per_session.items():
        data = session_data.get(session)
        if data is None:
            continue
        for frame_idx in frames:
            idx_feature = data.index_to_row.get(int(frame_idx))
            if idx_feature is None:
                continue
            idx_label = data.label_lookup.get(int(frame_idx))
            if idx_label is None:
                continue
            feat_list.append(data.features[idx_feature].clone())
            label_list.append(data.label_vectors[idx_label].clone())

    if feat_list:
        return torch.stack(feat_list), torch.stack(label_list)

    return (
        torch.empty((0, feature_dim), dtype=torch.float32),
        torch.empty((0, len(BEHAVIORS)), dtype=torch.float32),
    )


# ------------------------------
# Mixup 平衡
# ------------------------------
def mixup(x1, y1, x2, y2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = torch.maximum(y1, y2)
    return x_mix, y_mix


def balance_with_mixup(train_x, train_y, alpha=0.4, max_ratio=0.2, tag="train"):
    if len(train_x) == 0:
        return train_x, train_y

    before_counts = train_y.sum(dim=0).cpu().numpy().astype(int)
    pos_counts = train_y.sum(dim=0).long()
    max_count = int(pos_counts.max().item())

    aug_feats, aug_labels = [], []
    for i in range(len(BEHAVIORS)):
        n_pos = int(pos_counts[i].item())
        if n_pos == 0:
            continue
        if n_pos > 0.5 * max_count:
            continue

        need = max_count - n_pos
        limit = int(n_pos * max_ratio)
        need = min(need, limit)
        if need <= 0:
            continue

        pos_indices = (train_y[:, i] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        for _ in range(need):
            j1, j2 = np.random.choice(pos_indices, 2, replace=True)
            xm, ym = mixup(train_x[j1], train_y[j1], train_x[j2], train_y[j2], alpha)
            aug_feats.append(xm.unsqueeze(0))
            aug_labels.append(ym.unsqueeze(0))

    if aug_feats:
        aug_feats = torch.cat(aug_feats, dim=0)
        aug_labels = torch.cat(aug_labels, dim=0)
        train_x = torch.cat([train_x, aug_feats], dim=0)
        train_y = torch.cat([train_y, aug_labels], dim=0)

    after_counts = train_y.sum(dim=0).cpu().numpy().astype(int)

    print(f"\n[{tag}] Mixup balancing result:")
    for beh, b, a in zip(BEHAVIORS, before_counts, after_counts):
        print(f"{beh:12s}: {b:6d} -> {a:6d}")
    print(f"[✔] Total size = {len(train_x)}")

    return train_x, train_y


# ------------------------------
# Focal Loss
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ------------------------------
# Training + Pseudo-labeling
# ------------------------------
def train_one_round(model, train_loader, val_loader, loss_fn, opt, device, epochs, ckpt_dir, round_idx):
    metrics_per_epoch = []
    best_macroF1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validate
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).long().cpu()
                all_preds.append(preds)
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0)

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

        metrics_df = pd.DataFrame({
            "round": round_idx,
            "epoch": epoch,
            "behavior": BEHAVIORS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": f1_macro
        })
        metrics_per_epoch.append(metrics_df)

        if f1_macro > best_macroF1:
            best_macroF1 = f1_macro
            best_state = model.state_dict()

        print(f"[Round {round_idx} | Epoch {epoch}] macroF1={f1_macro:.4f}")

    # === 用 best_state 重新算一遍 val_probs ===
    model.load_state_dict(best_state)
    model.eval()
    all_probs_best, all_labels_best = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            all_probs_best.append(torch.sigmoid(out).cpu())
            all_labels_best.append(y.cpu())
    all_probs_best = torch.cat(all_probs_best, dim=0)
    all_labels_best = torch.cat(all_labels_best, dim=0)

    return pd.concat(metrics_per_epoch, ignore_index=True), all_probs_best, all_labels_best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repr_dir", default="./representations")
    p.add_argument("--label_dir", default=r"D:\Jiaqi\Datasets\Rats\TrainData_new\labels")
    p.add_argument("--sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor","F6D5_outdoor_2"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--ckpt_dir", default="checkpoints_selftrain")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of segments reserved for validation/testing")
    p.add_argument("--split-seed", type=int, default=0, help="Random seed for segment-level train/test split")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # === 读取初始数据并按照段落划分 ===
    session_data: Dict[str, SessionSplitData] = {}
    segments_by_session: Dict[str, Dict[str, List]] = {}

    for sess in args.sessions:
        repr_path = os.path.join(args.repr_dir, f"{sess}.pt")
        if not os.path.exists(repr_path):
            print(f"[!] Representation file missing for session {sess}, skipping")
            continue

        data = torch.load(repr_path)
        feats: torch.Tensor = data["features"].float()
        centres = data.get("centers", list(range(len(feats))))
        index_to_row = {int(c): i for i, c in enumerate(centres)}

        label_path = os.path.join(args.label_dir, sess, f"label_{sess}.csv")
        if not os.path.exists(label_path):
            print(f"[!] Label file missing for session {sess}, skipping")
            continue

        label_df = pd.read_csv(label_path)
        valid_behavior_cols = [c for c in BEHAVIORS if c in label_df.columns]
        if not valid_behavior_cols:
            print(f"[!] No recognised behavior columns in labels for session {sess}, skipping")
            continue

        label_df = label_df[["Index"] + valid_behavior_cols]
        label_df = label_df[label_df[valid_behavior_cols].sum(axis=1) > 0]
        label_df = label_df.sort_values("Index").reset_index(drop=True)
        if label_df.empty:
            print(f"[!] No positive labels remaining for session {sess}, skipping")
            continue

        indices = label_df["Index"].astype(int).to_numpy()
        label_array = label_df[valid_behavior_cols].to_numpy(dtype=np.float32)
        segments_by_session[sess] = compute_segments(indices, label_array, valid_behavior_cols)
        label_lookup = {int(idx): pos for pos, idx in enumerate(indices.tolist())}
        label_vectors = torch.stack(
            [_row_to_behavior_vector(row, valid_behavior_cols) for _, row in label_df.iterrows()]
        )

        session_data[sess] = SessionSplitData(
            features=feats,
            index_to_row=index_to_row,
            label_lookup=label_lookup,
            label_vectors=label_vectors,
        )

    if not session_data or not segments_by_session:
        raise RuntimeError("No labelled segments found for the provided sessions")

    feature_dim = next(iter(session_data.values())).features.shape[1]
    assignments = split_segments_by_action(
        segments_by_session,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )

    train_feats, train_labels = _collect_split_tensors(
        session_data,
        segments_by_session,
        assignments,
        "train",
        feature_dim,
    )
    val_feats, val_labels = _collect_split_tensors(
        session_data,
        segments_by_session,
        assignments,
        "test",
        feature_dim,
    )

    if len(train_feats) == 0 or len(val_feats) == 0:
        raise RuntimeError("Empty train or validation split after segment assignment")
    # === 统计测试集各动作的样本数量 ===
    val_counts = val_labels.sum(dim=0).long().cpu().numpy()
    print("\n[Test set behavior counts]")
    for beh, c in zip(BEHAVIORS, val_counts):
        print(f"{beh:12s}: {c}")
    print(f"Total test samples = {len(val_labels)}\n")
    all_metrics = []
    for r in range(1, args.rounds + 1):
        print(f"\n===== Round {r} start =====")
        # === 平衡真实标签（轻量 mixup） ===
        train_feats, train_labels = balance_with_mixup(train_feats, train_labels, max_ratio=0.3, tag=f"Round{r}-real")

        # 模型
        model = DeepMLPClassifier(input_dim=train_feats.shape[1], output_dim=len(BEHAVIORS)).to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = FocalLoss(alpha=0.15, gamma=3)

        train_loader = DataLoader(ReprDataset(train_feats, train_labels), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(ReprDataset(val_feats, val_labels), batch_size=args.batch_size)

        # 训练一轮
        metrics_df, val_probs_best, val_labels_best = train_one_round(model, train_loader, val_loader, loss_fn, opt, args.device, args.epochs, args.ckpt_dir, r)
        all_metrics.append(metrics_df)

        # === 保存每轮预测结果 ===
        df_preds = pd.DataFrame(val_probs_best.numpy(), columns=[f"pred_{b}" for b in BEHAVIORS])
        for i, b in enumerate(BEHAVIORS):
            df_preds[f"true_{b}"] = val_labels_best[:, i].numpy()
        df_preds["round"] = r
        df_preds.to_csv(os.path.join(args.ckpt_dir, f"preds_round{r}.csv"), index=False)
        print(f"[Round {r}] Validation predictions saved to preds_round{r}.csv")

        # === 伪标签 ===
        # === 伪标签（类别自适应阈值） ===
        val_counts = val_labels_best.sum(dim=0).cpu().numpy()
        thresholds = []
        for beh, cnt in zip(BEHAVIORS, val_counts):
            if cnt < 5000:  # 少样本类，阈值放宽
                thresholds.append(0.99)
            else:  # 多样本类，阈值严格
                thresholds.append(0.999)

        thresholds = torch.tensor(thresholds).view(1, -1)  # shape=(1,n_classes)
        pseudo_mask = (val_probs_best > thresholds).float()
        if pseudo_mask.sum() > 0:
            print(f"[Round {r}] Added {int(pseudo_mask.sum().item())} pseudo-labels (best macroF1 model)")
            new_feats = torch.cat([train_feats, val_feats], dim=0)
            new_labels = torch.cat([train_labels, pseudo_mask], dim=0)
            train_feats, train_labels = balance_with_mixup(new_feats, new_labels, max_ratio=0.2, tag=f"Round{r}-pseudo")

    final_df = pd.concat(all_metrics, ignore_index=True)
    final_df.to_csv(os.path.join(args.ckpt_dir, "selftrain_metrics.csv"), index=False)
    print("[✔] Training finished, metrics saved.")


if __name__ == "__main__":
    main()
