import os
import argparse
import numpy as np
import pandas as pd
import torch
from models.temporal_classifier import TemporalClassifier
from utils.context import create_context_windows

LABEL_COLUMNS = [
    "walk", "jump", "aiming", "scratch", "rearing", "stand_up",
    "still", "eating", "grooming", "local_search", "turn_left",
    "turn_right",
]


def load_model(model_path: str, input_dim: int, device: str):
    model = TemporalClassifier(input_dim, num_classes=len(LABEL_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session_files = [f for f in os.listdir(args.rep_dir) if f.endswith("_repr.npy")]
    if not session_files:
        print("No representation files found")
        return

    # Use first file to infer input dimension
    example = np.load(os.path.join(args.rep_dir, session_files[0]))
    example = create_context_windows(example)
    input_dim = example.shape[2]

    model = load_model(args.model_path, input_dim, device)
    os.makedirs(args.output_dir, exist_ok=True)

    for file in session_files:
        session = file.replace("_repr.npy", "")
        reps = np.load(os.path.join(args.rep_dir, file))
        reps = create_context_windows(reps).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(reps).to(device)).cpu().numpy()
            preds = 1 / (1 + np.exp(-logits))

        index_col = np.arange(len(preds))
        df = pd.DataFrame(preds, columns=LABEL_COLUMNS)
        df.insert(0, "Index", index_col)  # 在第一列插入名为 "Index" 的列

        out_path = os.path.join(args.output_dir, f"{session}_pred_possibility.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using trained temporal classifier on representations")
    parser.add_argument("--rep_dir", default="representations", help="Representation directory")
    parser.add_argument("--model_path", default="checkpoints_classifier/mlp_repr_5.pt", help="Trained model path")
    parser.add_argument("--output_dir", default="predictions", help="Where to save predictions")
    args = parser.parse_args()
    main(args)