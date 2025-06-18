import os
import argparse
import numpy as np
import pandas as pd
import torch
from models.deep_mlp import DeepMLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def load_model(model_path: str, input_dim: int, device: str):
    model = DeepMLPClassifier(input_dim, output_dim=len(LABEL_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_session_segment_data(rep_dir, label_dir, session, segments):
    rep_path = os.path.join(rep_dir, f"{session}_repr.npy")
    label_path = os.path.join(label_dir, session, f"label_{session}.csv")
    reps = np.load(rep_path).astype(np.float32)
    labels = pd.read_csv(label_path)[LABEL_COLUMNS].values.astype(np.float32)

    # 对齐长度
    length = min(len(reps), len(labels))
    reps = reps[:length]
    labels = labels[:length]

    # 根据 segments 提取有标注区域
    if segments:
        idx = []
        for s, e in segments:
            idx.extend(list(range(s, e + 1)))
        idx = [i for i in idx if i < length]
        reps = reps[idx]
        labels = labels[idx]

    return reps, labels

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segs = load_valid_segments(os.path.join(args.label_path, "results.txt"))
    session_files = [f for f in os.listdir(args.rep_dir) if f.endswith("_repr.npy")]
    if not session_files:
        print("No representation files found.")
        return

    # 确定输入维度
    example = np.load(os.path.join(args.rep_dir, session_files[0]))
    input_dim = example.shape[1]
    model = load_model(args.model_path, input_dim, device)

    os.makedirs(args.output_dir, exist_ok=True)
    log_lines = []

    for file in session_files:
        session = file.replace("_repr.npy", "")
        print(f"\n--- Predicting and evaluating session: {session} ---")

        if session not in segs:
            print(f"Session {session} not in results.txt, skipping.")
            continue

        reps, labels = load_session_segment_data(args.rep_dir, args.label_path, session, segs[session])
        if len(reps) == 0:
            print(f"No valid segment data in session {session}, skipping.")
            continue

        with torch.no_grad():
            logits = model(torch.from_numpy(reps).to(device)).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(np.float32)

        # Save predictions
        pred_df = pd.DataFrame(probs, columns=LABEL_COLUMNS)
        pred_df.to_csv(os.path.join(args.output_dir, f"{session}_pred.csv"), index=False)

        # Evaluate
        session_log = evaluate_predictions(preds, labels, session)
        log_lines.extend(session_log)

    # Save log file
    log_path = os.path.join(args.output_dir, "eval_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nAll evaluation results saved to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model only on labeled segments")
    parser.add_argument("--rep_dir", default="./representations", help="Representation directory")
    parser.add_argument("--label_path", default="D:/Homework/NLP project/ACC_DATA/ACC_DATA/TrainData/labels", help="Path to label root directory")
    parser.add_argument("--model_path", default="checkpoints_classifier/mlp_repr_6.pt", help="Trained model path")
    parser.add_argument("--output_dir", default="predictions_segment_eval", help="Where to save predictions and evaluation")
    args = parser.parse_args()
    main(args)