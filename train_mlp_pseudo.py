import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
import tqdm
from models.temporal_classifier import TemporalClassifier
from utils.context import create_context_windows

LABEL_COLUMNS = [
    "walk", "jump", "aiming", "scratch", "rearing", "stand_up",
    "still", "eating", "grooming", "local_search", "turn_left",
    "turn_right",
]


def load_valid_segments(results_path: str) -> dict:
    segs = {}
    if not os.path.exists(results_path):
        return segs
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            path, rest = line.strip().split("\t")
            session = os.path.basename(os.path.dirname(path))
            pairs = eval(rest)
            segs[session] = pairs
    return segs


def load_session_data(rep_dir: str, label_dir: str, session: str, segments=None):
    rep_file = os.path.join(rep_dir, f"{session}_repr.npy")
    lab_file = os.path.join(label_dir, session, f"label_{session}.csv")
    reps = np.load(rep_file)
    reps = create_context_windows(reps)
    labels = pd.read_csv(lab_file)
    labels = labels[LABEL_COLUMNS].values
    if len(labels) > len(reps):
        labels = labels[: len(reps)]
    elif len(labels) < len(reps):
        reps = reps[: len(labels)]
    if segments:
        idx = []
        for s, e in segments:
            idx.extend(list(range(s, e + 1)))
        idx = [i for i in idx if i < len(reps)]
        reps = reps[idx]
        labels = labels[idx]
    return reps.astype(np.float32), labels.astype(np.float32)


def load_unlabeled_data(rep_dir: str, session: str, segments=None):
    rep_file = os.path.join(rep_dir, f"{session}_repr.npy")
    reps = np.load(rep_file)
    reps = create_context_windows(reps)
    if segments:
        labeled_idx = []
        for s, e in segments:
            labeled_idx.extend(list(range(s, e + 1)))
        labeled_idx = [i for i in labeled_idx if i < len(reps)]
        mask = np.ones(len(reps), dtype=bool)
        mask[labeled_idx] = False
        reps = reps[mask]
    return reps.astype(np.float32)


def evaluate_detailed(model, x, y, session_name, device):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=128)
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            prob = torch.sigmoid(out)
            preds.append(prob.cpu())
            labels.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    pred_binary = (preds > 0.5).astype(np.float32)

    # f1 = f1_score(labels, pred_binary, average=None, zero_division=0)
    # acc = (pred_binary == labels).astype(np.float32).mean(axis=0)

    # print(f"\n== Session: {session_name} ==")
    # for i, name in enumerate(LABEL_COLUMNS):
    #     print(f"Class {name:12s} | Acc: {acc[i]:.4f} | F1: {f1[i]:.4f}")

    return labels, pred_binary


def pseudo_label(model, data, threshold, batch_size, device):
    if len(data) == 0:
        return (
            np.empty((0, data.shape[1], data.shape[2]), dtype=np.float32),
            np.empty((0, len(LABEL_COLUMNS)), dtype=np.float32),
            data,
        )
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size)
    keep_indices = []
    preds = []
    start = 0
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()
            mask_high = prob >= threshold
            keep = mask_high.any(axis=1)
            if np.any(keep):
                idx = np.nonzero(keep)[0] + start
                keep_indices.extend(idx.tolist())
                preds.append(mask_high[keep].astype(np.float32))
            start += len(xb)
    if keep_indices:
        pseudo_x = data[keep_indices]
        pseudo_y = np.concatenate(preds, axis=0)
        remaining_mask = np.ones(len(data), dtype=bool)
        remaining_mask[keep_indices] = False
        remaining = data[remaining_mask]
    else:
        pseudo_x = np.empty((0, data.shape[1], data.shape[2]), dtype=np.float32)
        pseudo_y = np.empty((0, len(LABEL_COLUMNS)), dtype=np.float32)
        remaining = data
    return pseudo_x, pseudo_y, remaining


def mixup_data(x, y, alpha=1.0):
    """Applies Mixup to a batch"""
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y


def train_simple_classifier(train_x, train_y, input_dim, num_classes, device, test_dict):
    model = TemporalClassifier(input_dim, num_classes=num_classes).to(device)

    # ==== 类别不平衡权重 ====
    pos_counts = train_y.sum(axis=0)
    neg_counts = len(train_y) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    os.makedirs("checkpoints_classifier", exist_ok=True)  # 新增：创建目录

    for epoch in range(50):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb, yb = mixup_data(xb, yb, alpha=0.4)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\n[Epoch {epoch+1}/30] Training Loss: {total_loss:.4f}")

        # === 每个 session 的测试结果 ===
        all_true, all_pred = [], []
        for session_name, (test_x, test_y) in test_dict.items():
            # print(f"[Epoch {epoch+1}/30] Eval on {session_name}:")
            y_true, y_pred = evaluate_detailed(model, test_x, test_y, session_name, device)
            all_true.append(y_true)
            all_pred.append(y_pred)

        # === 所有测试数据整体评估 ===
        all_true = np.concatenate(all_true, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)
        f1 = f1_score(all_true, all_pred, average=None, zero_division=0)
        acc = (all_true == all_pred).astype(np.float32).mean(axis=0)

        print(f"\n[Epoch {epoch+1}/30] === Overall Test Statistics ===")
        for i, name in enumerate(LABEL_COLUMNS):
            print(f"Class {name:12s} | Acc: {acc[i]:.4f} | F1: {f1[i]:.4f}")

        # === ✅ 保存每轮模型 ===
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1:02d}.pt")

    return model


def self_training(train_x, train_y, unlabeled_x, input_dim, num_classes, device, test_dict):
    pseudo_x_total = np.empty((0, train_x.shape[1], input_dim), dtype=np.float32)
    pseudo_y_total = np.empty((0, num_classes), dtype=np.float32)
    model = None
    for i in range(5):
        print(f"\n=== Round {i+1}/5 ===")
        combined_x = np.concatenate([train_x, pseudo_x_total], axis=0)
        combined_y = np.concatenate([train_y, pseudo_y_total], axis=0)
        model = train_simple_classifier(combined_x, combined_y, input_dim, num_classes, device, test_dict)
        if len(unlabeled_x) == 0:
            continue
        new_x, new_y, unlabeled_x = pseudo_label(model, unlabeled_x, 0.95, 128, device)
        print(f"  Pseudo labeled samples added: {len(new_x)}")
        if len(new_x) > 0:
            pseudo_x_total = np.concatenate([pseudo_x_total, new_x], axis=0)
            pseudo_y_total = np.concatenate([pseudo_y_total, new_y], axis=0)
    return model


def main(args):
    segs = load_valid_segments(os.path.join(args.label_path, "results.txt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 合并所有 session 的训练集 ===
    train_x_list, train_y_list = [], []
    unlabeled_list = []
    test_dict = {}  # session -> (x, y)

    for session in args.sessions:
        x, y = load_session_data(args.rep_dir, args.label_path, session, segs.get(session))
        split = int(len(x) * 0.8)
        train_x, test_x = x[:split], x[split:]
        train_y, test_y = y[:split], y[split:]
        train_x_list.append(train_x)
        train_y_list.append(train_y)
        u = load_unlabeled_data(args.rep_dir, session, segs.get(session))
        if len(u) > 0:
            unlabeled_list.append(u)
        test_dict[session] = (test_x, test_y)

    # 合并训练数据
    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    if unlabeled_list:
        unlabeled_x = np.concatenate(unlabeled_list, axis=0)
    else:
        unlabeled_x = np.empty((0, train_x.shape[1], train_x.shape[2]), dtype=np.float32)
    input_dim = train_x.shape[2]
    num_classes = train_y.shape[1]

    # === 训练并伪标签 ===
    model = self_training(train_x, train_y, unlabeled_x, input_dim, num_classes, device, test_dict)

    # === 分 session 评估 + 汇总整体 ===
    all_true, all_pred = [], []
    for session, (test_x, test_y) in test_dict.items():
        y_true, y_pred = evaluate_detailed(model, test_x, test_y, session, device)
        all_true.append(y_true)
        all_pred.append(y_pred)

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    f1 = f1_score(all_true, all_pred, average=None, zero_division=0)
    acc = (all_true == all_pred).astype(np.float32).mean(axis=0)

    print(f"\n==== Overall Statistics ====")
    for i, name in enumerate(LABEL_COLUMNS):
        print(f"Class {name:12s} | Acc: {acc[i]:.4f} | F1: {f1[i]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple supervised classifier with weighted loss and mixup")
    parser.add_argument("--rep_dir", default="./representations", help="Representation directory")
    parser.add_argument("--label_path", default="D:/Jiaqi/Datasets/Rats/TrainData/labels", help="Path to labels directory")
    parser.add_argument("--sessions", nargs="+", default=[
        "F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_1"
    ], help="List of session names")
    args = parser.parse_args()
    main(args)
