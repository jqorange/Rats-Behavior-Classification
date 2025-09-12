import os
import argparse

import torch
import pandas as pd

from models.deep_mlp import DeepMLPClassifier

BEHAVIORS = [
    "walk",
    "jump",
    "aiming",
    "scratch",
    "rearing",
    "stand_up",
    "still",
    "eating",
    "grooming",
    "local_search",
    "turn_left",
    "turn_right",
]


def _load_features(path: str):
    data = torch.load(path)
    feats: torch.Tensor = data["features"]
    centres = data.get("centers", list(range(len(feats))))
    return feats, centres


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Batch inference for MLP classifier")
    p.add_argument("--weights", default="D:\Jiaqi\Projects\Rats-Behavior-Classification\checkpoints_mlp\epoch20.pt", help="Checkpoint file from training")
    p.add_argument("--repr_dir", default=r"D:\Jiaqi\Projects\Rats-Behavior-Classification\representations", help="Directory containing <session>.pt files")
    p.add_argument("--sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_2"], help="List of session names")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="prediction_prob", help="Output directory for prediction CSVs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载模型
    sample_path = os.path.join(args.repr_dir, f"{args.sessions[0]}.pt")
    feats, _ = _load_features(sample_path)
    model = DeepMLPClassifier(input_dim=feats.shape[1], output_dim=len(BEHAVIORS))
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    # 遍历 session 批量推理
    for sess in args.sessions:
        repr_path = os.path.join(args.repr_dir, f"{sess}.pt")
        feats, centres = _load_features(repr_path)

        probs = []
        for i in range(0, len(feats), 1024):
            batch = feats[i:i + 1024].to(args.device)
            out = torch.sigmoid(model(batch))
            probs.append(out.cpu())
        probs = torch.cat(probs, dim=0)

        df = pd.DataFrame(probs.numpy(), columns=BEHAVIORS)
        df.insert(0, "Index", centres)
        out_path = os.path.join(args.out_dir, f"{sess}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved predictions for {sess} to {out_path}")


if __name__ == "__main__":
    main()
