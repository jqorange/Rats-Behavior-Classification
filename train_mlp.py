import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from models.deep_mlp import DeepMLPClassifier
from utils.segments import (
    assign_segments_train_val_test,
    collect_segment_centres,
    compute_segments,
)

# =========================================================
# 全局行为类别
# =========================================================
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

# =========================================================
# Dataset
# =========================================================
class ReprDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features.float()
        self.labels = labels.float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

# =========================================================
# Session container
# =========================================================
@dataclass
class SessionSplitData:
    features: torch.Tensor
    index_to_row: Dict[int, int]   # frame_idx -> feature_row_idx
    label_lookup: Dict[int, int]   # frame_idx -> label_row_idx
    label_vectors: torch.Tensor    # [num_labeled_frames, n_classes]

# =========================================================
# Label helper
# =========================================================
def _row_to_behavior_vector(row: pd.Series, present_cols: List[str]) -> torch.Tensor:
    """
    给定该帧的标签行(row)和本 session 真实出现过的行为列 present_cols，
    组装成全局 BEHAVIORS 顺序的多标签向量 (len==len(BEHAVIORS)).
    """
    vec = np.zeros(len(BEHAVIORS), dtype=np.float32)
    for k, beh in enumerate(BEHAVIORS):
        if beh in present_cols:
            vec[k] = float(row[beh])
        else:
            vec[k] = 0.0
    return torch.from_numpy(vec)

def _collect_split_tensors(
    session_data: Dict[str, SessionSplitData],
    segments_by_session: Dict[str, Dict[str, List[Any]]],
    assignments: Dict[str, Any],
    split: str,  # "train", "val" or "test"
    feature_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据 assignments + split ('train'/'test')，把对应段中心帧的特征和标签拉成大表。
    NOTE: 这个逻辑是旧版已经验证过的。我们依赖它拿到
          70% 的 train 和 30% 的 holdout。
    """
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

    # 空的 fallback
    return (
        torch.empty((0, feature_dim), dtype=torch.float32),
        torch.empty((0, len(BEHAVIORS)), dtype=torch.float32),
    )

# =========================================================
# Mixup 平衡 (不改)
# =========================================================
def mixup(x1, y1, x2, y2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = torch.maximum(y1, y2)
    return x_mix, y_mix

def balance_with_mixup(train_x, train_y, alpha=0.4, max_ratio=0.3, tag="train"):
    """
    针对每个类别做少量上采样 + mixup 来缓解极端不均衡。
    不会疯狂复制，只会在极度稀有类上做 <= max_ratio * n_pos 的合成样本。
    """
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

# =========================================================
# Focal Loss
# =========================================================
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

# =========================================================
# 载入初始权重（可选）
# =========================================================
def load_initial_weights(model: nn.Module, ckpt_path: str, device: str = "cpu", strict: bool = True):
    if (not ckpt_path) or (not os.path.exists(ckpt_path)):
        print(f"[init] No initial weights loaded (path not provided or not found).")
        return
    print(f"[init] Loading initial weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # 支持多种checkpoint格式
    if isinstance(ckpt, dict) and any(k in ckpt for k in ["state_dict", "model", "model_state_dict"]):
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt.get("model_state_dict")))
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if hasattr(missing, "__len__") and hasattr(unexpected, "__len__"):
        if len(missing) or len(unexpected):
            print(f"[init] Loaded with non-strict mode. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
            if len(missing) > 0:
                print("         missing_keys(sample):", list(missing)[:10])
            if len(unexpected) > 0:
                print("         unexpected_keys(sample):", list(unexpected)[:10])
    print("[init] Initial weights loaded.")

# =========================================================
# 把 holdout(30%) 再切成 val/test (10%/20%)
# =========================================================
def split_holdout_into_val_test(
    holdout_feats: torch.Tensor,
    holdout_labels: torch.Tensor,
    seed: int,
    val_fraction_in_holdout: float = 1.0/3.0,
):
    """
    输入:
      holdout_feats / holdout_labels : 旧逻辑里的 'test' (整30%)
      val_fraction_in_holdout       : 默认 1/3 → 10% 总量
    输出:
      val_feats, val_labels, test_feats, test_labels

    我们用固定seed随机打乱 index，然后前 n_val 做 val，剩下做 test。
    这样：
    - 训练集 (70%) 完全不变
    - 30% 的 holdout 被稳定地拆成 10% / 20%
    """

    N = len(holdout_feats)
    if N == 0:
        raise RuntimeError("Holdout set is empty, cannot split into val/test.")

    idx_all = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx_all)

    n_val = int(np.floor(N * val_fraction_in_holdout))

    # 避免极端：全进val或全进test（除非只有1个）
    if n_val == 0 and N >= 2:
        n_val = 1
    if n_val == N and N > 1:
        n_val = N - 1

    val_idx = idx_all[:n_val]
    test_idx = idx_all[n_val:]

    val_feats = holdout_feats[val_idx]
    val_labels = holdout_labels[val_idx]

    test_feats = holdout_feats[test_idx]
    test_labels = holdout_labels[test_idx]

    return val_feats, val_labels, test_feats, test_labels

# =========================================================
# 评估函数
# =========================================================
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
      metrics_df: per-class precision/recall/F1 + macroF1（macroF1每行相同）
      probs_all:  (N, n_classes) 概率
      labels_all:(N, n_classes) 真值
      preds_all: (N, n_classes) 0/1 预测 (0.5阈值)
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            out = model(x)
            probs = torch.sigmoid(out).cpu()
            all_probs.append(probs)
            all_labels.append(y.cpu())

    probs_all = torch.cat(all_probs, dim=0)
    labels_all = torch.cat(all_labels, dim=0)

    preds_all = (probs_all > 0.5).long().numpy()
    labels_np = labels_all.numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_all, average=None, zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels_np, preds_all, average="macro", zero_division=0
    )

    metrics_df = pd.DataFrame({
        "behavior": BEHAVIORS,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": f1_macro
    })

    return metrics_df, probs_all, labels_all, torch.from_numpy(preds_all)

# =========================================================
# 训练主循环 (用 val 选 best model)
# =========================================================
def train_with_val_select_best(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    device: str,
    epochs: int,
    ckpt_save_dir: str,
):
    """
    - 每个 epoch:
        * 训练一遍
        * 在 val 上评估 macroF1
    - 记录历史并保留 macroF1 最优的权重(best_state_dict)
    - 把最优权重也存盘 ckpt_save_dir/best_model.pth
    """
    metrics_history_rows = []
    best_macroF1 = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        # === Train ===
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # === Eval on val ===
        val_metrics_df, _, _, _ = evaluate_model(model, val_loader, device)

        macroF1_val = float(val_metrics_df["macro_f1"].iloc[0])
        print(f"[Epoch {epoch}] val macroF1 = {macroF1_val:.4f}")

        tmp_df = val_metrics_df.copy()
        tmp_df.insert(0, "epoch", epoch)
        metrics_history_rows.append(tmp_df)

        if macroF1_val > best_macroF1:
            best_macroF1 = macroF1_val
            best_state_dict = model.state_dict()

    # 保存 best_state_dict
    os.makedirs(ckpt_save_dir, exist_ok=True)
    best_path = os.path.join(ckpt_save_dir, "best_model.pth")
    torch.save(best_state_dict, best_path)
    print(f"[Best] macroF1={best_macroF1:.4f} saved to {best_path}")

    history_val_metrics = pd.concat(metrics_history_rows, ignore_index=True)
    return history_val_metrics, best_state_dict

# =========================================================
# main()
# =========================================================
def main():
    p = argparse.ArgumentParser()

    # 基础路径
    p.add_argument("--repr_dir", default="./representations")
    p.add_argument("--label_dir", default=r"D:\Jiaqi\Datasets\Rats\TrainData_new\labels")

    # 要处理的 session 列表
    p.add_argument(
        "--sessions",
        nargs="+",
        default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F5D7_outdoor"],
    )

    # 设备
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # 训练超参
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=70)

    # 输出目录
    p.add_argument("--ckpt_dir", default="checkpoints_final70_10_20")
    p.add_argument("--metrics_csv", default="metrics_val_history.csv")
    p.add_argument("--test_pred_csv", default="test_predictions_best.csv")
    p.add_argument("--test_metrics_csv", default="test_metrics_best.csv")

    # 随机种子 / 划分策略
    # （非常关键）primary-test-ratio 必须保持 0.3，这样 collect_split_tensors(...,"train")
    # 拿到的 70% 就和旧版一模一样
    p.add_argument("--primary-test-ratio", type=float, default=0.3,
                   help="MUST stay 0.3 so that the 70% train split matches the old code exactly.")
    p.add_argument("--split-seed", type=int, default=0,
                   help="Seed for segment-level splits (must match old seed).")
    p.add_argument("--val-frac-in-holdout", type=float, default=1.0/3.0,
                   help="Fraction of the holdout portion used for validation (default keeps 7:2:1).")
    p.add_argument("--cross-folds", type=int, default=1,
                   help="Total number of cross-validation folds.")
    p.add_argument("--cross-index", type=int, default=0,
                   help="Index of the current cross-validation fold (0-based).")

    # 初始权重（可选, 例如 round1_best.pth）
    p.add_argument(
        "--init-weights",
        type=str,
        default=r"D:\Jiaqi\Projects\Rats-Behavior-Classification\checkpoints_mlp\round1_best.pth",
        help="Path to initial weights (.pth). Loaded before training.",
    )
    p.add_argument(
        "--strict-load",
        type=bool,
        default=True,
        help="Use strict=True when loading init weights.",
    )

    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.cross_folds <= 0:
        raise ValueError("--cross-folds must be positive")
    if not (0 <= args.cross_index < args.cross_folds):
        raise ValueError("--cross-index must satisfy 0 <= index < --cross-folds")

    holdout_ratio = float(args.primary_test_ratio)
    if not (0.0 < holdout_ratio < 1.0):
        raise ValueError("--primary-test-ratio must be between 0 and 1")

    val_fraction = float(args.val_frac_in_holdout)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--val-frac-in-holdout must be between 0 and 1")

    train_ratio = max(0.0, 1.0 - holdout_ratio)
    val_ratio = holdout_ratio * val_fraction
    test_ratio = holdout_ratio - val_ratio

    # -----------------------------------------------------
    # Step 1. 读取所有 session 的特征&标签 + 段定义
    # -----------------------------------------------------
    session_data: Dict[str, SessionSplitData] = {}
    segments_by_session: Dict[str, Dict[str, List[Any]]] = {}

    for sess in args.sessions:
        # 1) 读表示
        repr_path = os.path.join(args.repr_dir, f"{sess}.pt")
        if not os.path.exists(repr_path):
            print(f"[!] Representation file missing for session {sess}, skipping")
            continue

        data = torch.load(repr_path)
        feats: torch.Tensor = data["features"].float()  # shape [N, D]
        centres = data.get("centers", list(range(len(feats))))
        index_to_row = {int(c): i for i, c in enumerate(centres)}

        # 2) 读标签
        label_path = os.path.join(args.label_dir, sess, f"label_{sess}.csv")
        if not os.path.exists(label_path):
            print(f"[!] Label file missing for session {sess}, skipping")
            continue

        label_df = pd.read_csv(label_path)

        # 当前 session 真实存在的行为列
        valid_behavior_cols = [c for c in BEHAVIORS if c in label_df.columns]
        if not valid_behavior_cols:
            print(f"[!] No recognised behavior columns in labels for session {sess}, skipping")
            continue

        # 只保留 Index + 有效行为列
        label_df = label_df[["Index"] + valid_behavior_cols]

        # 丢掉全0帧
        label_df = label_df[label_df[valid_behavior_cols].sum(axis=1) > 0]
        label_df = label_df.sort_values("Index").reset_index(drop=True)
        if label_df.empty:
            print(f"[!] No positive labels remaining for session {sess}, skipping")
            continue

        indices = label_df["Index"].astype(int).to_numpy()
        label_array = label_df[valid_behavior_cols].to_numpy(dtype=np.float32)

        # 计算 segment 列表（按行为分）
        segments_by_session[sess] = compute_segments(indices, label_array, valid_behavior_cols)

        # frame_idx -> label_df 行号
        label_lookup = {int(idx): pos for pos, idx in enumerate(indices.tolist())}

        # 预先把所有 label 行转成全局向量
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

    # 用第一个 session 的特征维度做 reference
    feature_dim = next(iter(session_data.values())).features.shape[1]

    # -----------------------------------------------------
    # Step 2. 按照 7:2:1 比例生成 train/val/test 分割（可跨折重现）
    # -----------------------------------------------------
    assignments = assign_segments_train_val_test(
        segments_by_session,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=args.split_seed,
        num_folds=args.cross_folds,
        fold_index=args.cross_index,
    )

    # -----------------------------------------------------
    # Step 3. 根据 assignments 收集每个 split 的特征/标签
    # -----------------------------------------------------
    train70_feats, train70_labels = _collect_split_tensors(
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
        "val",
        feature_dim,
    )
    test_feats, test_labels = _collect_split_tensors(
        session_data,
        segments_by_session,
        assignments,
        "test",
        feature_dim,
    )

    if len(train70_feats) == 0 or len(val_feats) == 0 or len(test_feats) == 0:
        raise RuntimeError("Empty split encountered. Check your data or fold configuration.")

    # 打印一下各 split 的分布
    print("\n[Split sizes]")
    print(f"Train(~{train_ratio*100:.1f}%): {len(train70_feats)} samples")
    print(f"Val(~{val_ratio*100:.1f}%): {len(val_feats)} samples")
    print(f"Test(~{test_ratio*100:.1f}%): {len(test_feats)} samples\n")

    def _print_counts(tag, lbls):
        counts = lbls.sum(dim=0).long().cpu().numpy()
        print(f"[{tag} set behavior counts]")
        for beh, c in zip(BEHAVIORS, counts):
            print(f"{beh:12s}: {c}")
        print(f"Total {tag} samples = {len(lbls)}\n")

    _print_counts("Train", train70_labels)
    _print_counts("Val",   val_labels)
    _print_counts("Test",  test_labels)

    # -----------------------------------------------------
    # Step 5. 对训练集做轻量 mixup 平衡
    # -----------------------------------------------------
    train_feats, train_labels = balance_with_mixup(
        train70_feats,
        train70_labels,
        max_ratio=0.3,
        tag="train70",
    )

    # -----------------------------------------------------
    # Step 6. 构建模型 + optimizer + loss
    # -----------------------------------------------------
    device = args.device
    model = DeepMLPClassifier(
        input_dim=train_feats.shape[1],
        output_dim=len(BEHAVIORS)
    ).to(device)

    # 可选加载初始权重
    if args.init_weights:
        load_initial_weights(
            model,
            args.init_weights,
            device=device,
            strict=args.strict_load,
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = FocalLoss(alpha=0.15, gamma=3.5)

    # DataLoaders
    train_loader = DataLoader(
        ReprDataset(train_feats, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ReprDataset(val_feats, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ReprDataset(test_feats, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # -----------------------------------------------------
    # Step 7. 训练并用 val macroF1 选 best model
    # -----------------------------------------------------
    val_history_df, best_state_dict = train_with_val_select_best(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        opt=opt,
        device=device,
        epochs=args.epochs,
        ckpt_save_dir=args.ckpt_dir,
    )

    # 保存 val 历史指标
    val_history_csv_path = os.path.join(args.ckpt_dir, args.metrics_csv)
    val_history_df.to_csv(val_history_csv_path, index=False)
    print(f"[✔] Val metrics history saved to {val_history_csv_path}")

    # -----------------------------------------------------
    # Step 8. 用 best_state_dict 在 test(最终20%) 上跑最后成绩并保存
    # -----------------------------------------------------
    model.load_state_dict(best_state_dict)

    test_metrics_df, test_probs, test_labels_true, test_preds_bin = evaluate_model(
        model,
        test_loader,
        device,
    )

    macroF1_test = float(test_metrics_df["macro_f1"].iloc[0])
    print(f"[FINAL TEST] macroF1={macroF1_test:.4f}")
    print("[FINAL TEST] per-class metrics:")
    print(test_metrics_df)

    # 保存 test metrics
    test_metrics_csv_path = os.path.join(args.ckpt_dir, args.test_metrics_csv)
    test_metrics_df.to_csv(test_metrics_csv_path, index=False)
    print(f"[✔] Test metrics saved to {test_metrics_csv_path}")

    # 保存 test 预测 (概率+真值+二值预测)
    df_test_pred = pd.DataFrame(test_probs.numpy(), columns=[f"pred_{b}" for b in BEHAVIORS])
    for i, b in enumerate(BEHAVIORS):
        df_test_pred[f"true_{b}"] = test_labels_true[:, i].numpy()
        df_test_pred[f"binpred_{b}"] = test_preds_bin[:, i].numpy()
    test_pred_path = os.path.join(args.ckpt_dir, args.test_pred_csv)
    df_test_pred.to_csv(test_pred_path, index=False)
    print(f"[✔] Test predictions saved to {test_pred_path}")

    print("[✔] Done.")

# =========================================================
if __name__ == "__main__":
    main()
