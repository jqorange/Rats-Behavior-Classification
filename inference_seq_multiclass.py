import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.temporal_softmax_classifier import TemporalSoftmaxClassifier
from utils.context import create_context_windows

LABEL_COLUMNS = [
    "walk", "jump", "aiming", "scratch", "rearing", "stand_up",
    "still", "eating", "local_search", "turn_left",
    "turn_right",
]

session_names = [
    "F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor",
    "F5D10_outdoor", "F6D5_outdoor_2", "F6D5_outdoor_1"
]

REP_DIR = "./representations"
MODEL_PATH = "./checkpoints_classifier/seq_classifier_2.pt"
OUTPUT_DIR = "./probs_csv"
BATCH_SIZE = 512   # 你可以自定义

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_representation(rep_file: str) -> np.ndarray:
    reps = np.load(rep_file)
    reps = create_context_windows(reps)
    return reps.astype(np.float32)

def infer_and_save(session):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep_file = os.path.join(REP_DIR, f"{session}_repr.npy")
    output_file = os.path.join(OUTPUT_DIR, f"{session}_probs.csv")
    if not os.path.exists(rep_file):
        print(f"File not found: {rep_file}, skip.")
        return
    reps = load_representation(rep_file)
    model = TemporalSoftmaxClassifier(reps.shape[2], num_classes=len(LABEL_COLUMNS))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ds = TensorDataset(torch.from_numpy(reps))
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    all_probs = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(prob)
    probs = np.concatenate(all_probs, axis=0)
    df = pd.DataFrame(probs, columns=LABEL_COLUMNS)
    df.insert(0, "Index", np.arange(len(df)))
    df.to_csv(output_file, index=False)
    print(f"Saved probabilities to {output_file} with shape {probs.shape}")

if __name__ == "__main__":
    for session in session_names:
        infer_and_save(session)
