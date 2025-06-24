import argparse
import os
import re
import glob
import numpy as np
import torch

from utils.data_loader import DataLoader
from utils.trainer import FusionTrainer


def latest_checkpoint(path):
    pattern = os.path.join(path, "encoder_*.pkl")
    checkpoints = glob.glob(pattern)
    latest = -1
    for ckpt in checkpoints:
        m = re.search(r"encoder_(\d+)\.pkl", os.path.basename(ckpt))
        if m:
            num = int(m.group(1))
            # latest = max(latest, num)
            latest = 120
    return latest


def main(args):
    sessions = args.sessions
    loader = DataLoader(sessions, args.data_path)
    loader.load_original_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = FusionTrainer(
        N_feat_A=77,
        N_feat_B=66,
        num_classes=12,
        device=device,
        batch_size=256,
        d_model=32,
        nhead=4,
        hidden_dim=4,
        save_path=args.checkpoint_dir,
    )

    ckpt_num = latest_checkpoint(args.checkpoint_dir)
    if ckpt_num < 0:
        raise FileNotFoundError("No checkpoint found in %s" % args.checkpoint_dir)
    trainer.load(ckpt_num)

    os.makedirs(args.output_dir, exist_ok=True)

    for session in sessions:
        imu = loader.train_IMU.get(session)
        dlc = loader.train_DLC.get(session)
        if imu is None or dlc is None:
            print(f"Skip {session}: data not found")
            continue
        min_len = min(len(imu), len(dlc))
        imu = imu[:min_len].astype(np.float32)
        dlc = dlc[:min_len].astype(np.float32)

        reps = trainer.encode(imu, dlc, pool=True)
        out_file = os.path.join(args.output_dir, f"{session}_repr.npy")
        np.save(out_file, reps)
        print(f"Saved {out_file} with shape {reps.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate representations for sessions")
    parser.add_argument("--data_path", default="D:\Jiaqi\Datasets\Rats\TrainData", help="Base data path")
    parser.add_argument("--sessions", nargs="+", default=["F3D6_outdoor"], help="Session names")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--output_dir", default="representations", help="Output directory")
    args = parser.parse_args()
    main(args)