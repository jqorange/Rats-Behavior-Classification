import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

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


def build_loaders(train_sessions, test_sessions, rep_dir, label_dir, segs, batch_size=256):
    train_x, train_y = [], []
    for s in train_sessions:
        reps, labels = load_session_data(rep_dir, label_dir, s, segs.get(s))
        train_x.append(reps)
        train_y.append(labels)
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    test_x, test_y = [], []
    for s in test_sessions:
        reps, labels = load_session_data(rep_dir, label_dir, s, segs.get(s))
        test_x.append(reps)
        test_y.append(labels)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader, train_x.shape[1]


def train_model(train_loader, test_loader, input_dim, out_dim, epochs=20, lr=1e-3, device="cpu"):
    model = DeepMLPClassifier(input_dim, output_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    min_avg_loss = 10000000
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            losses = []
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                l = criterion(pred, y)
                losses.append(l.item())
            avg_loss = float(np.mean(losses))
        print(f"Epoch {epoch+1}/{epochs} - val_loss: {avg_loss:.4f}")
        if avg_loss <= min_avg_loss:
            torch.save(model.state_dict(), os.path.join(args.model_dir, "mlp_repr.pt"))
    return model


def main(args):
    segs = load_valid_segments(os.path.join(args.label_path, "results.txt"))
    train_loader, test_loader, input_dim = build_loaders(
        args.train_sessions, args.test_sessions, args.rep_dir, args.label_path, segs, args.batch_size
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(train_loader, test_loader, input_dim, len(LABEL_COLUMNS), args.epochs, args.lr, device)
    os.makedirs(args.model_dir, exist_ok=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on representations")
    parser.add_argument("--rep_dir", default="./representations", help="Representation directory")
    parser.add_argument("--label_path", default="D:\Homework//NLP project\ACC_DATA\ACC_DATA\TrainData\labels", help="Path to labels directory")
    parser.add_argument("--train_sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor","F5D10_outdoor", "F6D5_outdoor_2"])
    parser.add_argument("--test_sessions", nargs="+", default=["F6D5_outdoor_1"])
    parser.add_argument("--model_dir", default="checkpoints_classifier", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)