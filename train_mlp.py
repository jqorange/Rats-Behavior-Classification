import os
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.deep_mlp import DeepMLPClassifier

# Columns in label files (excluding 'Index')
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


class ReprDataset(Dataset):
    """Simple dataset wrapping features and multi-label targets."""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features.float()
        self.labels = labels.float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _row_to_behavior_vector(row: pd.Series, present_cols: List[str]) -> torch.Tensor:
    """
    Convert a label row to a fixed-length vector aligned with BEHAVIORS.
    Missing behavior columns are filled with 0.
    """
    vec = np.zeros(len(BEHAVIORS), dtype=np.float32)
    for k, beh in enumerate(BEHAVIORS):
        if beh in present_cols:
            vec[k] = float(row[beh])
        else:
            vec[k] = 0.0
    return torch.from_numpy(vec)


def _load_session_split(
    repr_path: str,
    label_path: str,
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load features and labels for one session and split by 80/20."""
    data = torch.load(repr_path)
    feats: torch.Tensor = data["features"]
    centres = data.get("centers", list(range(len(feats))))
    index_to_row = {int(c): i for i, c in enumerate(centres)}

    label_df = pd.read_csv(label_path)

    # 行为列（忽略 not_in_frame 和 unknown）
    valid_behavior_cols = [c for c in BEHAVIORS if c in label_df.columns]
    label_df = label_df[["Index"] + valid_behavior_cols]

    # 只保留至少有一个行为 = 1 的行（忽略 not_in_frame/unknown）
    if len(valid_behavior_cols) == 0:
        # 没有任何目标行为列，直接返回空
        return torch.empty(0, feats.shape[1]), torch.empty(0, len(BEHAVIORS))

    label_df = label_df[label_df[valid_behavior_cols].sum(axis=1) > 0]
    label_df = label_df.sort_values("Index")

    n = len(label_df)
    split = int(n * 0.8)
    part = label_df.iloc[:split] if train else label_df.iloc[split:]

    feat_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    for _, row in part.iterrows():
        idx = int(row["Index"])
        if idx not in index_to_row:
            continue  # skip indices without features
        feat_list.append(feats[index_to_row[idx]])
        label_list.append(_row_to_behavior_vector(row, valid_behavior_cols))

    if feat_list:
        return torch.stack(feat_list), torch.stack(label_list)
    else:
        return torch.empty(0, feats.shape[1]), torch.empty(0, len(BEHAVIORS))


def mixup(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor, alpha: float = 0.4):
    """Simple mixup for feature vectors; labels take union (multi-label)."""
    lam = np.random.beta(alpha, alpha)
    x_mix = lam * x1 + (1.0 - lam) * x2
    # 多标签任务取并集（只要任一来源为1即为1）
    y_mix = torch.maximum(y1, y2)
    return x_mix, y_mix


def balance_with_mixup(train_x: torch.Tensor, train_y: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each class (one-vs-rest), upsample positives via mixup until each class
    has the same number of positive samples as the current maximum class.
    Only augments the training set. Returns augmented tensors.
    """
    if len(train_x) == 0:
        return train_x, train_y

    pos_counts = train_y.sum(dim=0).long()  # per class positives
    max_count = int(pos_counts.max().item())

    aug_feats = []
    aug_labels = []

    for i, beh in enumerate(BEHAVIORS):
        n_pos = int(pos_counts[i].item())
        if n_pos == 0:
            # 没有正样本，无法针对该类做mixup增广
            continue
        if n_pos >= max_count:
            continue

        pos_indices = (train_y[:, i] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        need = max_count - n_pos

        for _ in range(need):
            j1, j2 = np.random.choice(pos_indices, 2, replace=True)
            x1, y1 = train_x[j1], train_y[j1]
            x2, y2 = train_x[j2], train_y[j2]
            xm, ym = mixup(x1, y1, x2, y2, alpha=alpha)
            aug_feats.append(xm.unsqueeze(0))
            aug_labels.append(ym.unsqueeze(0))

    if aug_feats:
        aug_feats = torch.cat(aug_feats, dim=0)
        aug_labels = torch.cat(aug_labels, dim=0)
        train_x = torch.cat([train_x, aug_feats], dim=0)
        train_y = torch.cat([train_y, aug_labels], dim=0)

    return train_x, train_y


def main() -> None:
    p = argparse.ArgumentParser(description="Train MLP on session representations")
    p.add_argument("--repr_dir", default="./representations", help="Directory containing <session>.pt files")
    p.add_argument("--label_dir", default=r"D:\Jiaqi\Datasets\Rats\TrainData_new\labels", help="Directory containing label_<session>.csv files")
    p.add_argument("--sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_2"], help="Session names to use")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", default="checkpoints_mlp")
    p.add_argument("--mixup_alpha", type=float, default=0.4, help="Beta(alpha, alpha) for mixup")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_feats = []
    train_labels = []
    val_feats = []
    val_labels = []

    for sess in args.sessions:
        repr_path = os.path.join(args.repr_dir, f"{sess}.pt")
        label_path = os.path.join(args.label_dir, sess, f"label_{sess}.csv")

        feats_tr, labs_tr = _load_session_split(repr_path, label_path, train=True)
        feats_va, labs_va = _load_session_split(repr_path, label_path, train=False)

        if len(feats_tr):
            train_feats.append(feats_tr)
            train_labels.append(labs_tr)
        if len(feats_va):
            val_feats.append(feats_va)
            val_labels.append(labs_va)

    if len(train_feats) == 0 or len(val_feats) == 0:
        raise RuntimeError("No training/validation data found. Check paths and sessions.")

    train_feats_t = torch.cat(train_feats, dim=0)
    train_labels_t = torch.cat(train_labels, dim=0)
    val_feats_t = torch.cat(val_feats, dim=0)
    val_labels_t = torch.cat(val_labels, dim=0)

    # ---- 验证集标签占比（1 的百分比） ----
    label_sum = val_labels_t.sum(dim=0)  # 每列 1 的数量
    label_ratio = (label_sum / len(val_labels_t)) * 100  # 转换为百分比
    print("\n[Val label distribution (% of 1s)]")
    for beh, ratio in zip(BEHAVIORS, label_ratio.tolist()):
        print(f"{beh:12s}: {ratio:.2f}%")

    # ---- 训练集上做类别平衡的 mixup（仅训练集）----
    before_counts = train_labels_t.sum(dim=0).cpu().numpy().astype(int)
    train_feats_t, train_labels_t = balance_with_mixup(train_feats_t, train_labels_t, alpha=args.mixup_alpha)
    after_counts = train_labels_t.sum(dim=0).cpu().numpy().astype(int)
    print("\n[Train positives before/after mixup balancing]")
    for beh, b, a in zip(BEHAVIORS, before_counts, after_counts):
        print(f"{beh:12s}: {b:6d} -> {a:6d}")
    print(f"[✔] After mixup balancing: train set size = {len(train_feats_t)}")

    # ---- 模型 & 优化 ----
    model = DeepMLPClassifier(input_dim=train_feats_t.shape[1], output_dim=len(BEHAVIORS)).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(ReprDataset(train_feats_t, train_labels_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ReprDataset(val_feats_t, val_labels_t), batch_size=args.batch_size)

    metrics_per_epoch = []  # 保存每个 epoch 的指标表
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item() * x.size(0)

                probs = torch.sigmoid(out)
                preds = (probs > 0.5).long().cpu()  # 阈值化为0/1
                all_preds.append(preds)
                all_labels.append(y.cpu())

        val_loss /= max(len(val_loader.dataset), 1)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        accs = [accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(len(BEHAVIORS))]

        # micro/macro 便捷查看
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro", zero_division=0)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

        metrics_df = pd.DataFrame({
            "epoch": epoch,
            "behavior": BEHAVIORS,
            "accuracy": accs,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        metrics_per_epoch.append(metrics_df)

        # ---- Save ckpt ----
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, os.path.join(args.ckpt_dir, f"epoch{epoch}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       os.path.join(args.ckpt_dir, "best.pt"))

        print(f"\nEpoch {epoch}: val_loss={val_loss:.8f} | micro F1={f1_micro:.4f} | macro F1={f1_macro:.4f}")

    final_metrics = pd.concat(metrics_per_epoch, ignore_index=True)
    final_metrics.to_csv(os.path.join(args.ckpt_dir, "val_metrics.csv"), index=False)
    print(f"\n[✔] Validation metrics saved to {os.path.join(args.ckpt_dir, 'val_metrics.csv')}")


if __name__ == "__main__":
    main()
