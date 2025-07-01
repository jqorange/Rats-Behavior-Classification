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


def _split_test_frames(reps: np.ndarray, labels: np.ndarray, n: int):
    """Split out the first ``n`` frames for each class.

    Returns training and remaining sets.
    """
    num_classes = labels.shape[1]
    add_indices = []
    for c in range(num_classes):
        cls_idx = np.nonzero(labels[:, c] == 1)[0]
        if len(cls_idx) > 0:
            add_indices.extend(cls_idx[:n])
    add_indices = np.unique(add_indices)
    mask = np.ones(len(labels), dtype=bool)
    mask[add_indices] = False
    train_reps = reps[add_indices]
    train_labels = labels[add_indices]
    remain_reps = reps[mask]
    remain_labels = labels[mask]
    return train_reps, train_labels, remain_reps, remain_labels


def build_datasets(train_sessions, test_sessions, rep_dir, label_dir, segs, frames_per_class=150):
    train_x_list, train_y_list = [], []
    unlabeled_x = []
    for s in train_sessions:
        reps, labels = load_session_data(rep_dir, label_dir, s, segs.get(s))
        train_x_list.append(reps)
        train_y_list.append(labels)
        u = load_unlabeled_data(rep_dir, s, segs.get(s))
        if len(u) > 0:
            unlabeled_x.append(u)

    if unlabeled_x:
        unlabeled_x = np.concatenate(unlabeled_x, axis=0)
    else:
        sample_shape = train_x_list[0].shape
        unlabeled_x = np.empty((0, sample_shape[1], sample_shape[2]), dtype=np.float32)

    test_x, test_y = [], []
    for s in test_sessions:
        reps, labels = load_session_data(rep_dir, label_dir, s, segs.get(s))
        add_x, add_y, reps, labels = _split_test_frames(reps, labels, frames_per_class)
        if len(add_x) > 0:
            train_x_list.append(add_x)
            train_y_list.append(add_y)
        test_x.append(reps)
        test_y.append(labels)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    return train_x, train_y, unlabeled_x, test_x, test_y


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            p = (probs > 0.5).float()
            preds.append(p.cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = np.mean((preds == labels).all(axis=1))
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1


def pseudo_label(model, data, thresholds, batch_size, device):
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
    thresholds = np.asarray(thresholds)[None, :]
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            mask = probs >= thresholds
            has_label = mask.any(axis=1)
            if np.any(has_label):
                idx = np.nonzero(has_label)[0] + start
                keep_indices.extend(idx.tolist())
                preds.append(probs[has_label])
            start += len(batch[0])
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


def self_training(train_x, train_y, unlabeled_x, test_unlabeled_x, test_loader, input_dim, window_size, args, device):
    """Self-training loop with EMA and class specific thresholds."""

    pseudo_x_total = np.empty((0, window_size, input_dim), dtype=np.float32)
    pseudo_y_total = np.empty((0, len(LABEL_COLUMNS)), dtype=np.float32)
    base_thresholds = np.linspace(0.95, 0.6, 10)

    model = TemporalClassifier(input_dim, num_classes=len(LABEL_COLUMNS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def update(self, model):
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

        def apply_to(self, model):
            model.load_state_dict(self.shadow)

    ema = EMA(model)

    for i, thr in enumerate(base_thresholds):
        print(f"\n=== Iteration {i+1}/10 | Base Threshold {thr:.2f} ===")

        combined_x = np.concatenate([train_x, pseudo_x_total], axis=0)
        combined_y = np.concatenate([train_y, pseudo_y_total], axis=0)

        counts = combined_y.sum(axis=0)
        max_c, min_c = counts.max(), counts.min()
        if max_c == min_c:
            class_thresholds = np.full(len(LABEL_COLUMNS), thr)
        else:
            ratio = (counts - min_c) / (max_c - min_c)
            class_thresholds = thr * ratio + base_thresholds[-1] * (1 - ratio)

        train_ds = TensorDataset(torch.from_numpy(combined_x), torch.from_numpy(combined_y))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        for epoch in tqdm.tqdm(range(50)):
            model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                ema.update(model)

        ema_model = TemporalClassifier(input_dim, num_classes=len(LABEL_COLUMNS)).to(device)
        ema.apply_to(ema_model)
        acc, f1 = evaluate(ema_model, test_loader, device)
        print(f"Iteration {i+1} - Test Acc: {acc:.4f} | F1: {f1:.4f}")

        if i < len(base_thresholds) - 1 and len(unlabeled_x) > 0:
            new_x, new_y, unlabeled_x = pseudo_label(
                ema_model, unlabeled_x, class_thresholds, args.batch_size, device
            )
            print(f"  Pseudo labeled samples added: {len(new_x)}")
            if len(new_x) > 0:
                pseudo_x_total = np.concatenate([pseudo_x_total, new_x], axis=0)
                pseudo_y_total = np.concatenate([pseudo_y_total, new_y], axis=0)

        torch.save(ema.shadow, os.path.join(args.model_dir, f"mlp_repr_{i}.pt"))

    final_thr = base_thresholds[-1]
    if len(unlabeled_x) > 0:
        new_x, new_y, _ = pseudo_label(ema_model, unlabeled_x, class_thresholds, args.batch_size, device)
        if len(new_x) > 0:
            pseudo_x_total = np.concatenate([pseudo_x_total, new_x], axis=0)
            pseudo_y_total = np.concatenate([pseudo_y_total, new_y], axis=0)

    test_pseudo_x, test_pseudo_y, _ = pseudo_label(
        ema_model, test_unlabeled_x, class_thresholds, args.batch_size, device
    )
    print(f"Pseudo labeled test samples: {len(test_pseudo_x)}")

    all_x = np.concatenate([train_x, pseudo_x_total, test_pseudo_x], axis=0)
    all_y = np.concatenate([train_y, pseudo_y_total, test_pseudo_y], axis=0)

    final_ds = TensorDataset(torch.from_numpy(all_x), torch.from_numpy(all_y))
    final_loader = DataLoader(final_ds, batch_size=args.batch_size, shuffle=True)

    for epoch in tqdm.tqdm(range(50)):
        model.train()
        for x, y in final_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            ema.update(model)

    ema.apply_to(ema_model)
    acc, f1 = evaluate(ema_model, test_loader, device)
    print(f"Final model - Test Acc: {acc:.4f} | F1: {f1:.4f}")

    torch.save(ema.shadow, os.path.join(args.model_dir, "mlp_repr_final.pt"))

    return ema_model


def main(args):
    segs = load_valid_segments(os.path.join(args.label_path, "results.txt"))
    train_x, train_y, unlabeled_x, test_x, test_y = build_datasets(
        args.train_sessions,
        args.test_sessions,
        args.rep_dir,
        args.label_path,
        segs,
        args.frames_per_class,
    )
    window_size = train_x.shape[1]
    input_dim = train_x.shape[2]
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.model_dir, exist_ok=True)
    model = self_training(
        train_x,
        train_y,
        unlabeled_x,
        test_x.copy(),
        test_loader,
        input_dim,
        window_size,
        args,
        device,
    )
    torch.save(model.state_dict(), os.path.join(args.model_dir, "mlp_repr.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-training temporal classifier on representations")
    parser.add_argument("--rep_dir", default="./representations", help="Representation directory")
    parser.add_argument("--label_path", default="D:\Jiaqi\Datasets\Rats\TrainData/labels", help="Path to labels directory")
    parser.add_argument("--train_sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor","F5D10_outdoor", "F6D5_outdoor_1"])
    parser.add_argument("--test_sessions", nargs="+", default=["F6D5_outdoor_1"])
    parser.add_argument("--model_dir", default="checkpoints_classifier", help="Where to save model")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--frames_per_class",
        type=int,
        default=150,
        help="Number of initial frames per class from test sessions to use for training",
    )
    args = parser.parse_args()
    main(args)
