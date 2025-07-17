import argparse
import os
import numpy as np
import torch
from models.temporal_softmax_classifier import TemporalSoftmaxClassifier
from utils.context import create_context_windows

LABEL_COLUMNS = [
    "walk", "jump", "aiming", "scratch", "rearing", "stand_up",
    "still", "eating", "grooming", "local_search", "turn_left",
    "turn_right",
]


def load_representation(rep_file: str) -> np.ndarray:
    reps = np.load(rep_file)
    reps = create_context_windows(reps)
    return reps.astype(np.float32)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = load_representation(args.rep_file)
    model = TemporalSoftmaxClassifier(reps.shape[2], num_classes=len(LABEL_COLUMNS))
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(reps).to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
    np.save(args.output_file, prob)
    print(f"Saved probabilities to {args.output_file} with shape {prob.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for temporal softmax classifier")
    parser.add_argument("--rep_file", required=True, help="Input representation file")
    parser.add_argument("--model_path", required=True, help="Trained model path")
    parser.add_argument("--output_file", default="probs.npy", help="Where to save probabilities")
    args = parser.parse_args()
    main(args)
