import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.deep_mlp import DeepMLPClassifier
import pandas as pd
LABEL_COLUMNS = [
    "walk", "jump", "aiming", "scratch", "rearing", "stand_up",
    "still", "eating", "grooming", "local_search", "turn_left",
    "turn_right", "not_in_frame", "unknown",
]


def load_representation(rep_dir: str, session: str):
    rep_file = os.path.join(rep_dir, f"{session}_repr.npy")
    reps = np.load(rep_file)
    return reps.astype(np.float32)


def run_inference(model_path: str, rep_dir: str, session: str, output_dir: str, input_dim: int, device):
    # Load data
    data = load_representation(rep_dir, session)
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=256)

    # Load model
    model = DeepMLPClassifier(input_dim=input_dim, output_dim=len(LABEL_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference
    all_probs = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    os.makedirs(output_dir, exist_ok=True)

    # Define syllable names
    syllable_names = [
        'walk', 'jump', 'aiming', 'scratch', 'rearing', 'stand_up',
        'still', 'eating', 'grooming', 'local_search', 'turn_left', 'turn_right'
    ]

    # Drop unused columns from predictions (13 used out of 14 total)
    all_probs_df = pd.DataFrame(all_probs[:, :len(syllable_names)], columns=syllable_names)

    # Add index column
    all_probs_df.insert(0, "Index", range(len(all_probs_df)))

    # Save to CSV
    all_probs_df.to_csv(os.path.join(output_dir, f"{session}_probs.csv"), index=False)

    print(f"[{session}] Saved: probs ({all_probs.shape})")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = args.sessions
    for session in sessions:
        reps = load_representation(args.rep_dir, session)
        input_dim = reps.shape[1]

        run_inference(
            model_path=args.model_path,
            rep_dir=args.rep_dir,
            session=session,
            output_dir=args.output_dir,
            input_dim=input_dim,
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and save probability outputs")
    parser.add_argument("--rep_dir", type=str, default="./representations", help="Directory with *_repr.npy files")
    parser.add_argument("--sessions", type=str, default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_1", "F6D5_outdoor_2"], help="Session name (e.g., F3D5_outdoor)")
    parser.add_argument("--model_path", type=str, default="./checkpoints_classifier/mlp_repr_1.pt", help="Path to trained .pt model")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Where to save outputs")
    args = parser.parse_args()
    main(args)
