import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
import tqdm
from models.deep_mlp import DeepMLPClassifier

LABEL_COLUMNS = [
    "walk", "jump", "aiming", "scratch", "rearing", "stand_up",
    "still", "eating", "grooming", "local_search", "turn_left",
    "turn_right", "not_in_frame", "unknown",
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
    labels = pd.read_csv(lab_file)
    labels = labels[LABEL_COLUMNS].values
    labels = np.argmax(labels, axis=1)  # 单标签分类
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
    return reps.astype(np.float32), labels.astype(np.int64)


def load_unlabeled_data(rep_dir: str, session: str, segments=None):
    rep_file = os.path.join(rep_dir, f"{session}_repr.npy")
    reps = np.load(rep_file)
    if segments:
        labeled_idx = []
        for s, e in segments:
            labeled_idx.extend(list(range(s, e + 1)))
        labeled_idx = [i for i in labeled_idx if i < len(reps)]
        mask = np.ones(len(reps), dtype=bool)
        mask[labeled_idx] = False
        reps = reps[mask]
    return reps.astype(np.float32)


def build_datasets(train_sessions, test_sessions, rep_dir, label_dir, segs):
    train_x, train_y = [], []
    unlabeled_x = []
    for s in train_sessions:
        reps, labels = load_session_data(rep_dir, label_dir, s, segs.get(s))
        train_x.append(reps)
        train_y.append(labels)
        u = load_unlabeled_data(rep_dir, s, segs.get(s))
        if len(u) > 0:
            unlabeled_x.append(u)
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    if unlabeled_x:
        unlabeled_x = np.concatenate(unlabeled_x, axis=0)
    else:
        unlabeled_x = np.empty((0, train_x.shape[1]), dtype=np.float32)

    test_x, test_y = [], []
    for s in test_sessions:
        reps, labels = load_session_data(rep_dir, label_dir, s, segs.get(s))
        test_x.append(reps)
        test_y.append(labels)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    return train_x, train_y, unlabeled_x, test_x, test_y


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = np.mean(preds == labels)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1


def pseudo_label(model, data, threshold, batch_size, device):
    if len(data) == 0:
        return np.empty((0, data.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64), data
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size)
    keep_indices = []
    preds = []
    start = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            confs = np.max(probs, axis=1)
            pred_labels = np.argmax(probs, axis=1)
            for i, conf in enumerate(confs):
                if conf >= threshold:
                    keep_indices.append(start + i)
                    preds.append(pred_labels[i])
            start += len(batch[0])
    if keep_indices:
        pseudo_x = data[keep_indices]
        pseudo_y = np.array(preds, dtype=np.int64)
        remaining_mask = np.ones(len(data), dtype=bool)
        remaining_mask[keep_indices] = False
        remaining = data[remaining_mask]
    else:
        pseudo_x = np.empty((0, data.shape[1]), dtype=np.float32)
        pseudo_y = np.empty((0,), dtype=np.int64)
        remaining = data
    return pseudo_x, pseudo_y, remaining


def self_training(train_x, train_y, unlabeled_x, test_loader, input_dim, args, device):
    pseudo_x_total = np.empty((0, input_dim), dtype=np.float32)
    pseudo_y_total = np.empty((0,), dtype=np.int64)
    thresholds = np.linspace(0.95, 0.6, 5)
    model = None
    for i, thr in enumerate(thresholds):
        print(f"\n=== Iteration {i+1}/10 | Threshold {thr:.2f} ===")
        combined_x = np.concatenate([train_x, pseudo_x_total], axis=0)
        combined_y = np.concatenate([train_y, pseudo_y_total], axis=0)
        train_ds = TensorDataset(torch.from_numpy(combined_x), torch.from_numpy(combined_y))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        model = DeepMLPClassifier(input_dim, output_dim=len(LABEL_COLUMNS)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in tqdm.tqdm(range(20)):
            model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
        acc, f1 = evaluate(model, test_loader, device)
        print(f"Iteration {i+1} - Test Acc: {acc:.4f} | F1: {f1:.4f}")
        if i < len(thresholds) - 1 and len(unlabeled_x) > 0:
            new_x, new_y, unlabeled_x = pseudo_label(model, unlabeled_x, thr, args.batch_size, device)
            print(f"  Pseudo labeled samples added: {len(new_x)}")
            if len(new_x) > 0:
                pseudo_x_total = np.concatenate([pseudo_x_total, new_x], axis=0)
                pseudo_y_total = np.concatenate([pseudo_y_total, new_y], axis=0)
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"mlp_repr_{i}.pt"))
    return model


def main(args):
    segs = load_valid_segments(os.path.join(args.label_path, "results.txt"))
    train_x, train_y, unlabeled_x, test_x, test_y = build_datasets(
        args.train_sessions, args.test_sessions, args.rep_dir, args.label_path, segs
    )
    input_dim = train_x.shape[1]
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.model_dir, exist_ok=True)
    model = self_training(train_x, train_y, unlabeled_x, test_loader, input_dim, args, device)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "mlp_repr.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-training MLP for multi-class classification")
    parser.add_argument("--rep_dir", default="./representations", help="Representation directory")
    parser.add_argument("--label_path", default="D:/Homework/NLP project/ACC_DATA/ACC_DATA/TrainData/labels", help="Path to labels directory")
    parser.add_argument("--train_sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_1", "F6D5_outdoor_2"])
    parser.add_argument("--test_sessions", nargs="+", default=["F6D5_outdoor_2"])
    parser.add_argument("--model_dir", default="checkpoints_classifier", help="Where to save model")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
