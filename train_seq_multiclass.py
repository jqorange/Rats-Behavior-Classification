import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from models.temporal_softmax_classifier import TemporalSoftmaxClassifier
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

def normalize_labels(labels: np.ndarray) -> np.ndarray:
    sums = labels.sum(axis=1)
    mask = sums > 0
    labels = labels.astype(np.float32)
    labels[mask] = labels[mask] / (sums[mask].reshape(-1, 1))
    return labels

def load_session_data(rep_dir: str, label_dir: str, session: str, segments=None, type=None):
    rep_file = os.path.join(rep_dir, f"{session}_repr.npy")
    lab_file = os.path.join(label_dir, session, f"label_{session}.csv")
    reps = np.load(rep_file)
    reps = create_context_windows(reps)
    labels = pd.read_csv(lab_file)
    labels = labels[LABEL_COLUMNS].values
    labels = normalize_labels(labels)
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
        n = int(0.8 * len(reps))
    if type=="train":
        return reps[:n].astype(np.float32), labels[:n].astype(np.float32)
    elif type=="test":
        return reps[n:].astype(np.float32), labels[n:].astype(np.float32)

def build_datasets(train_sessions, test_sessions, rep_dir, label_dir, segs):
    train_x, train_y = [], []
    test_x, test_y = [], []
    for s in train_sessions:
        x, y = load_session_data(rep_dir, label_dir, s, segs.get(s),"train")
        train_x.append(x)
        train_y.append(y)
    for s in test_sessions:
        x, y = load_session_data(rep_dir, label_dir, s, segs.get(s),"test")
        test_x.append(x)
        test_y.append(y)
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    return train_x, train_y, test_x, test_y

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)

            # === 新增：处理 y ===
            y_np = y.cpu().numpy()
            for i in range(y_np.shape[0]):
                idx = np.where(y_np[i] > 0.3)[0]
                if len(idx) > 1:
                    y_np[i] = np.zeros_like(y_np[i])
                    y_np[i][idx] = 1.0 / len(idx)
            y_proc = torch.from_numpy(y_np).to(device)
            # ======================

            preds.append(prob.argmax(dim=1).cpu())
            labels.append(y_proc.argmax(dim=1).cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = (preds == labels).float().mean().item()
    return acc

def train(train_loader, test_loader, input_dim, device, args):
    model = TemporalSoftmaxClassifier(input_dim, num_classes=len(LABEL_COLUMNS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            logp = torch.log_softmax(logits, dim=1)
            loss = -(y * logp).sum(dim=1).mean()
            loss.backward()
            opt.step()
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Test Acc: {acc:.4f}")
    return model

def main(args):
    segs = load_valid_segments(os.path.join(args.label_path, "results.txt"))
    train_x, train_y, test_x, test_y = build_datasets(
        args.train_sessions,
        args.test_sessions,
        args.rep_dir,
        args.label_path,
        segs,
    )
    window_size = train_x.shape[1]
    input_dim = train_x.shape[2]
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.model_dir, exist_ok=True)
    model = train(train_loader, test_loader, input_dim, device, args)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "seq_classifier.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temporal classifier with softmax output")
    parser.add_argument("--rep_dir", default="./representations", help="Representation directory")
    parser.add_argument("--label_path", default="D:\\Jiaqi\\Datasets\\Rats\\TrainData/labels", help="Path to labels directory")
    parser.add_argument("--train_sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor","F5D10_outdoor", "F6D5_outdoor_1", "F3D6_outdoor"])
    parser.add_argument("--test_sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor","F5D10_outdoor", "F6D5_outdoor_1", "F3D6_outdoor"])
    parser.add_argument("--model_dir", default="checkpoints_classifier", help="Where to save model")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    main(args)
