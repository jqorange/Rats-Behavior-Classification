import os
import argparse

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from models.deep_mlp import DeepMLPClassifier

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


def _load_features(path: str):
    data = torch.load(path)
    feats: torch.Tensor = data["features"]
    centres = data.get("centers", list(range(len(feats))))
    return feats, centres


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Batch inference for MLP classifier with evaluation (only labeled frames)")
    p.add_argument("--weights", default=r"D:\Jiaqi\Projects\Rats-Behavior-Classification\checkpoints_mlp\round1_best.pth",
                   help="Checkpoint file from training (state_dict)")
    p.add_argument("--repr_dir", default=r"D:\Jiaqi\Projects\Rats-Behavior-Classification\representations",
                   help="Directory containing <session>.pt files")
    p.add_argument("--label_dir", default=r"D:\Jiaqi\Datasets\Rats\TrainData_new\labels",
                   help="Directory containing label CSVs")
    p.add_argument("--sessions", nargs="+", default=["F6D5_outdoor_1"],
                   help="List of session names")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="prediction_prob", help="Output directory for prediction CSVs")
    # 可选：排除某些类别（例如 scratch/grooming）不参与评估
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Behaviors to exclude from evaluation, e.g. --exclude scratch grooming")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载模型
    sample_path = os.path.join(args.repr_dir, f"{args.sessions[0]}.pt")
    feats, _ = _load_features(sample_path)
    model = DeepMLPClassifier(input_dim=feats.shape[1], output_dim=len(BEHAVIORS))

    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt)  # ✅ 这里传入的是 state_dict
    model.to(args.device)
    model.eval()

    # 保存所有结果用于整体评估（仅包含“有标签帧”的样本）
    overall_labels_list, overall_preds_list = [], []
    kept_cols_global = None  # 记录最终参与评估的列名顺序

    # 遍历 session 批量推理
    for sess in args.sessions:
        repr_path = os.path.join(args.repr_dir, f"{sess}.pt")
        if not os.path.exists(repr_path):
            print(f"[!] Representation file missing for {sess}, skip.")
            continue

        feats, centres = _load_features(repr_path)

        # 模型预测（概率）
        probs_batches = []
        for i in range(0, len(feats), 1024):
            batch = feats[i:i + 1024].to(args.device)
            out = torch.sigmoid(model(batch))
            probs_batches.append(out.cpu())
        probs = torch.cat(probs_batches, dim=0)

        # 保存预测结果（全帧）
        df_pred_full = pd.DataFrame(probs.numpy(), columns=BEHAVIORS)
        df_pred_full.insert(0, "Index", centres)
        out_path = os.path.join(args.out_dir, f"{sess}.csv")
        df_pred_full.to_csv(out_path, index=False)
        print(f"Saved predictions for {sess} to {out_path}")

        # === 加载对应的标签文件，并仅在“有标签的帧”上评估 ===
        label_path = os.path.join(args.label_dir, sess, f"label_{sess}.csv")
        if not os.path.exists(label_path):
            print(f"[!] Label file missing for {sess}, skip evaluation.")
            continue

        label_df = pd.read_csv(label_path)
        valid_cols = [c for c in BEHAVIORS if c in label_df.columns]
        if not valid_cols:
            print(f"[!] No valid behavior columns in labels for {sess}, skip evaluation.")
            continue

        # 仅保留“有标签的帧”（任意一个行为列为 1）
        labeled_rows = label_df[valid_cols].sum(axis=1) > 0
        label_df = label_df.loc[labeled_rows].copy()
        if label_df.empty:
            print(f"[!] No labeled frames after filtering for {sess}, skip evaluation.")
            continue

        # 可选：排除某些类别
        kept_cols = [c for c in valid_cols if c not in set(args.exclude)]
        if not kept_cols:
            print(f"[!] After exclusion, no behaviors left to evaluate for {sess}.")
            continue

        # 与预测对齐（按 Index 交集）
        label_df = label_df[["Index"] + kept_cols].set_index("Index")
        pred_df = df_pred_full[["Index"] + kept_cols].set_index("Index")

        common_idx = label_df.index.intersection(pred_df.index)
        if common_idx.empty:
            print(f"[!] No common indices for {sess}, skip evaluation.")
            continue

        y_true = label_df.loc[common_idx, kept_cols].values.astype(int)
        y_prob = pred_df.loc[common_idx, kept_cols].values
        y_pred = (y_prob > 0.5).astype(int)

        # per-class + 两种 macro（包含全部保留类 vs 只对出现过的类平均）
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        macro_all = f1.mean()
        occ_mask = support > 0
        macro_occurring = f1[occ_mask].mean() if occ_mask.any() else 0.0

        print(f"\n[Session {sess}] #labeled_frames={len(y_true)}")
        print(f"Macro-F1 (all kept classes)       = {macro_all:.3f}")
        print(f"Macro-F1 (occurring kept classes) = {macro_occurring:.3f}")
        for b, sup, p_, r_, f_ in zip(kept_cols, support, prec, rec, f1):
            print(f"{b:12s}  sup={int(sup):6d}  P={p_:.3f}  R={r_:.3f}  F1={f_:.3f}")

        # 记录到总体
        overall_labels_list.append(y_true)
        overall_preds_list.append(y_pred)
        kept_cols_global = kept_cols  # 以最后一次的列顺序为准（各 session 一致时无问题）

    # === 整体评估（仅基于“有标签帧”的样本） ===
    if overall_preds_list:
        all_labels = np.vstack(overall_labels_list)
        all_preds  = np.vstack(overall_preds_list)

        prec, rec, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        macro_all = f1.mean()
        occ_mask = support > 0
        macro_occurring = f1[occ_mask].mean() if occ_mask.any() else 0.0

        print("\n[Overall Evaluation across all sessions (labeled frames only)]")
        print(f"Macro-F1 (all kept classes)       = {macro_all:.3f}")
        print(f"Macro-F1 (occurring kept classes) = {macro_occurring:.3f}")
        for b, sup, p_, r_, f_ in zip(kept_cols_global, support, prec, rec, f1):
            print(f"{b:12s}  sup={int(sup):6d}  P={p_:.3f}  R={r_:.3f}  F1={f_:.3f}")
    else:
        print("\n[!] No sessions evaluated. Check labels alignment and filters.")


if __name__ == '__main__':
    main()
