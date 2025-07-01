import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def evaluate_predictions(preds, labels, session_name="Unknown"):
    log_lines = [f"\n=== Evaluation for session: {session_name} ==="]
    for i, label in enumerate(LABEL_COLUMNS):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        result = (
            f"{label:<15} | Acc: {acc:.3f} | Prec: {prec:.3f} | "
            f"Rec: {rec:.3f} | F1: {f1:.3f}"
        )
        print(result)
        log_lines.append(result)
    return log_lines

def load_labels(label_dir: str, session: str, segments):
    """Load labels for a session and return the selected segment indices."""
    label_path = os.path.join(label_dir, session, f"label_{session}.csv")
    labels = pd.read_csv(label_path)[LABEL_COLUMNS].values.astype(np.float32)
    idx = []
    if segments:
        for s, e in segments:
            idx.extend(range(s, e + 1))
        idx = [i for i in idx if i < len(labels)]
        labels = labels[idx]
    return labels, idx

def load_predictions(pred_path: str, indices):
    """Load prediction CSV and slice by ``indices``."""
    df = pd.read_csv(pred_path)
    preds = df[LABEL_COLUMNS].values.astype(np.float32)
    if indices:
        indices = [i for i in indices if i < len(preds)]
        preds = preds[indices]
    return preds

def exclude_first_n_per_class(reps: np.ndarray, labels: np.ndarray, n: int = 50):
    """Exclude the first ``n`` frames of each class from the arrays."""
    keep = []
    for i in range(labels.shape[1]):
        cls_idx = np.where(labels[:, i] == 1)[0]
        keep.extend(cls_idx[n:])
    keep = np.sort(np.unique(keep))
    return reps[keep], labels[keep]

def main(args):
    segs = load_valid_segments(os.path.join(args.label_path, "results.txt"))
    pred_files = [f for f in os.listdir(args.pred_dir) if f.endswith(".csv")]
    if not pred_files:
        print("No prediction files found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    log_lines = []

    for file in pred_files:
        session = file.split("_", 1)[0]
        print(f"\n--- Evaluating session: {session} ---")

        if session not in segs:
            print(f"Session {session} not in results.txt, skipping.")
            continue

        labels, idx = load_labels(args.label_path, session, segs[session])
        preds = load_predictions(os.path.join(args.pred_dir, file), idx)
        preds, labels = exclude_first_n_per_class(preds, labels, n=50)

        if len(labels) == 0:
            print(f"No valid segment data in session {session}, skipping.")
            continue

        session_log = evaluate_predictions(preds, labels, session)
        log_lines.extend(session_log)

    log_path = os.path.join(args.output_dir, "eval_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nAll evaluation results saved to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate prediction CSV files on labeled segments"
    )
    parser.add_argument(
        "--pred_dir",
        default="./predictions",
        help="Directory with prediction CSVs",
    )
    parser.add_argument(
        "--label_path",
        default="D:\\Jiaqi\\Datasets\\Rats\\TrainData/labels",
        help="Path to label root directory",
    )
    parser.add_argument(
        "--output_dir",
        default="predictions_eval",
        help="Where to save evaluation results",
    )
    args = parser.parse_args()
    main(args)
