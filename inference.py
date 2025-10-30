"""Joint inference pipeline for representation encoder + MLP classifier.

This script mirrors the dataset splitting strategy used in ``train_mlp.py``
so that evaluation happens on the exact 70/10/20 splits.  For every labelled
frame we extract multi-scale representations using the encoder trained by
``train_new.py``.  Each scale is fed through the MLP classifier and the
resulting probabilities are averaged.  Final metrics are reported on the
20%% test hold-out split.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from models.deep_mlp import DeepMLPClassifier
from models.fusion import EncoderFusion
from utils.checkpoint import load_checkpoint
from utils.segments import (
    SegmentInfo,
    collect_segment_centres,
    compute_segments,
    split_segments_by_action,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SessionLabelData:
    centres: List[int]
    index_to_row: Dict[int, int]
    label_lookup: Dict[int, int]
    label_vectors: torch.Tensor
    segments: Dict[str, List[SegmentInfo]]


@dataclass
class SessionInferenceData:
    centres: List[int]
    index_to_row: Dict[int, int]
    probabilities: torch.Tensor
    label_lookup: Dict[int, int]
    label_vectors: torch.Tensor


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _row_to_behavior_vector(row: pd.Series, present_cols: Sequence[str]) -> torch.Tensor:
    vec = np.zeros(len(BEHAVIORS), dtype=np.float32)
    for idx, beh in enumerate(BEHAVIORS):
        if beh in present_cols:
            vec[idx] = float(row.get(beh, 0.0))
    return torch.from_numpy(vec)


def _infer_model_config(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    def _first_tensor(keys: Iterable[str]) -> Optional[torch.Tensor]:
        for key in keys:
            tensor = state.get(key)
            if tensor is not None:
                return tensor
        return None

    d_model_src = _first_tensor(
        [
            "encoderA.adapter.linear.weight",
            "encoderA.input_proj.weight",
            "encoderA.norm_in.weight",
        ]
    )
    if d_model_src is None:
        available = ", ".join(sorted(state.keys()))
        raise KeyError(
            "Unable to infer d_model from checkpoint. Available keys: " + available
        )

    d_model = int(d_model_src.shape[0])

    session_src = _first_tensor(
        [
            "encoderA.adapter.session_embed.weight",
            "encoderA.session_embed.weight",
        ]
    )
    num_sessions = int(session_src.shape[0]) if session_src is not None else 0

    return d_model, num_sessions


def latest_checkpoint(ckpt_dir: str) -> str:
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return max(files, key=os.path.getmtime)


def _load_session_arrays(root: str, session: str) -> Tuple[torch.Tensor, torch.Tensor]:
    imu_file = os.path.join(root, "IMU", session, f"{session}_IMU_features_madnorm.csv")
    dlc_file = os.path.join(root, "DLC", session, f"final_filtered_{session}_50hz_madnorm.csv")

    imu_df = pd.read_csv(imu_file)
    dlc_df = pd.read_csv(dlc_file)
    min_len = min(len(imu_df), len(dlc_df))
    imu = torch.from_numpy(imu_df.iloc[:min_len].to_numpy(dtype=np.float32))
    dlc = torch.from_numpy(dlc_df.iloc[:min_len].to_numpy(dtype=np.float32))
    return imu, dlc


def _extract_windows(arr: torch.Tensor, centres: Sequence[int], T: int) -> torch.Tensor:
    L, F = arr.shape
    idx = torch.tensor(list(centres), dtype=torch.long)
    starts = idx - T // 2
    arange_T = torch.arange(T, dtype=torch.long)
    gather_idx = (starts.unsqueeze(1) + arange_T.view(1, -1)).clamp(0, L - 1)
    windows = arr.index_select(0, gather_idx.reshape(-1)).reshape(len(centres), T, F)
    return windows


def _prepare_session_labels(data_path: str, session: str) -> Optional[SessionLabelData]:
    label_path = os.path.join(data_path, "labels", session, f"label_{session}.csv")
    if not os.path.exists(label_path):
        print(f"[!] Label file missing for session {session}, skipping")
        return None

    label_df = pd.read_csv(label_path)
    valid_cols = [c for c in BEHAVIORS if c in label_df.columns]
    if not valid_cols:
        print(f"[!] No recognised behavior columns for session {session}, skipping")
        return None

    label_df = label_df[["Index"] + valid_cols]
    label_df = label_df[label_df[valid_cols].sum(axis=1) > 0]
    label_df = label_df.sort_values("Index").reset_index(drop=True)
    if label_df.empty:
        print(f"[!] No labelled frames remain after filtering for {session}, skipping")
        return None

    indices = label_df["Index"].astype(int).to_numpy()
    label_array = label_df[valid_cols].to_numpy(dtype=np.float32)

    segments = compute_segments(indices, label_array, valid_cols)
    label_lookup = {int(idx): pos for pos, idx in enumerate(indices.tolist())}
    label_vectors = torch.stack(
        [_row_to_behavior_vector(row, valid_cols) for _, row in label_df.iterrows()]
    )

    centres = sorted(label_lookup.keys())
    index_to_row = {frame: i for i, frame in enumerate(centres)}

    return SessionLabelData(
        centres=centres,
        index_to_row=index_to_row,
        label_lookup=label_lookup,
        label_vectors=label_vectors,
        segments=segments,
    )


def _load_mlp_model(path: str, device: str) -> Tuple[DeepMLPClassifier, int, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"MLP weight file not found: {path}")

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and any(k.startswith("classifier") for k in state):
        state_dict = state
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    linear_keys = [k for k in state_dict.keys() if k.endswith("weight") and k.startswith("classifier")]
    if not linear_keys:
        raise RuntimeError("Unable to locate linear layers in MLP state dict")
    linear_keys.sort(key=lambda k: int(k.split(".")[1]))

    input_dim = state_dict[linear_keys[0]].shape[1]
    hidden_dims = [state_dict[k].shape[0] for k in linear_keys[:-1]]
    output_dim = state_dict[linear_keys[-1]].shape[0]

    model = DeepMLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[MLP] Non-strict load. missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()

    first_linear = next(m for m in model.classifier if isinstance(m, nn.Linear))
    return model, first_linear.in_features, output_dim


def _run_models_for_session(
    model: EncoderFusion,
    mlp: DeepMLPClassifier,
    mlp_input_dim: int,
    mlp_output_dim: int,
    imu: torch.Tensor,
    dlc: torch.Tensor,
    centres: Sequence[int],
    Ts: Sequence[int],
    batch_size: int,
    device: str,
    fuse_mode: str,
    session_index: int,
) -> torch.Tensor:
    if not centres:
        return torch.empty((0, mlp_output_dim), dtype=torch.float32)

    features_per_T: List[torch.Tensor] = []
    for T in Ts:
        feat_batches: List[torch.Tensor] = []
        for start in range(0, len(centres), batch_size):
            centres_batch = centres[start : start + batch_size]
            imu_win = _extract_windows(imu, centres_batch, T).to(device, non_blocking=True)
            dlc_win = _extract_windows(dlc, centres_batch, T).to(device, non_blocking=True)
            sess_idx = torch.full((len(centres_batch),), session_index, dtype=torch.long, device=device)

            with torch.no_grad():
                output = model(imu_win, dlc_win, session_idx=sess_idx, attn_mode=fuse_mode)
                feat_seq = output.fused
                feat = torch.amax(feat_seq, dim=1).detach().cpu()

            feat_batches.append(feat)

            del imu_win, dlc_win, sess_idx, output, feat_seq
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        features_T = torch.cat(feat_batches, dim=0)
        features_per_T.append(features_T)

    repr_dim = features_per_T[0].shape[1] if features_per_T else mlp_input_dim
    if mlp_input_dim != repr_dim:
        raise RuntimeError(
            f"MLP expects features of dim {mlp_input_dim}, but encoder produced {repr_dim}."
        )

    probs_accum = torch.zeros((len(centres), mlp_output_dim), dtype=torch.float32)
    for features in features_per_T:
        logits = mlp(features.to(device))
        probs = torch.sigmoid(logits).cpu()
        probs_accum += probs

    probs_avg = probs_accum / max(len(features_per_T), 1)
    return probs_avg


def _collect_split_arrays(
    session_results: Dict[str, SessionInferenceData],
    frames_by_session: Dict[str, Sequence[int]],
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, int]]]:
    probs_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    frame_refs: List[Tuple[str, int]] = []

    for session, frames in frames_by_session.items():
        data = session_results.get(session)
        if data is None:
            continue
        for frame in frames:
            idx_prob = data.index_to_row.get(int(frame))
            idx_label = data.label_lookup.get(int(frame))
            if idx_prob is None or idx_label is None:
                continue
            probs_list.append(data.probabilities[idx_prob].unsqueeze(0))
            label_list.append(data.label_vectors[idx_label].unsqueeze(0))
            frame_refs.append((session, int(frame)))

    if probs_list:
        probs = torch.cat(probs_list, dim=0)
        labels = torch.cat(label_list, dim=0)
    else:
        probs = torch.empty((0, num_classes), dtype=torch.float32)
        labels = torch.empty((0, len(BEHAVIORS)), dtype=torch.float32)

    return probs, labels, frame_refs


def split_holdout_into_val_test(
    holdout_probs: torch.Tensor,
    holdout_labels: torch.Tensor,
    holdout_frames: List[Tuple[str, int]],
    seed: int,
    val_fraction_in_holdout: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[Tuple[str, int]],
    torch.Tensor,
    torch.Tensor,
    List[Tuple[str, int]],
]:
    N = len(holdout_probs)
    if N == 0:
        raise RuntimeError("Holdout set is empty, cannot split into val/test.")

    idx_all = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx_all)

    n_val = int(np.floor(N * val_fraction_in_holdout))
    if n_val == 0 and N >= 2:
        n_val = 1
    if n_val == N and N > 1:
        n_val = N - 1

    val_idx = idx_all[:n_val]
    test_idx = idx_all[n_val:]

    val_probs = holdout_probs[val_idx]
    val_labels = holdout_labels[val_idx]
    val_frames = [holdout_frames[i] for i in val_idx]

    test_probs = holdout_probs[test_idx]
    test_labels = holdout_labels[test_idx]
    test_frames = [holdout_frames[i] for i in test_idx]

    return val_probs, val_labels, val_frames, test_probs, test_labels, test_frames


def evaluate_predictions(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    if len(probs) == 0:
        raise RuntimeError("No samples to evaluate.")

    preds = (probs > 0.5).long().numpy()
    labels_np = labels.numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds, average=None, zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels_np, preds, average="macro", zero_division=0
    )

    metrics_df = pd.DataFrame(
        {
            "behavior": BEHAVIORS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": f1_macro,
        }
    )

    return metrics_df, torch.from_numpy(preds)


def _print_counts(tag: str, labels: torch.Tensor) -> None:
    counts = labels.sum(dim=0).long().cpu().numpy()
    print(f"[{tag} set behavior counts]")
    for beh, count in zip(BEHAVIORS, counts):
        print(f"{beh:12s}: {int(count)}")
    print(f"Total {tag} samples = {len(labels)}\n")


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------


def run_inference(
    repr_weights: str,
    mlp_weights: str,
    data_path: str,
    sessions: Sequence[str],
    *,
    device: Optional[str] = None,
    batch_size: int = 512,
    fuse_mode: str = "both",
    session_index: int = 0,
    window_sizes: Optional[Sequence[int]] = None,
    primary_test_ratio: float = 0.3,
    split_seed: int = 0,
    val_fraction_in_holdout: float = 1.0 / 3.0,
    out_dir: Optional[str] = None,
    test_metrics_csv: Optional[str] = None,
    test_pred_csv: Optional[str] = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = str(device)
    Ts = list(window_sizes or [16, 32, 64, 128])

    imu0, dlc0 = _load_session_arrays(data_path, sessions[0])
    num_imu, num_dlc = imu0.shape[1], dlc0.shape[1]

    ckpt = torch.load(repr_weights, map_location="cpu")
    state = ckpt.get("model_state", {})
    d_model, num_sessions = _infer_model_config(state)

    model = EncoderFusion(
        N_feat_A=num_imu,
        N_feat_B=num_dlc,
        mask_type=None,
        d_model=d_model,
        nhead=4,
        num_sessions=num_sessions,
    ).to(device)
    model.eval()

    _, stage = load_checkpoint(model, model.projection, optimizer=None, path=repr_weights)
    has_set_mode = hasattr(model.projection, "set_mode")
    if has_set_mode:
        if stage == 1:
            model.projection.set_mode("aware")
            if session_index is None:
                raise ValueError("Stage1 checkpoint requires a session index")
        else:
            model.projection.set_mode("align")
            if session_index is None:
                session_index = 0
    else:
        if session_index is None:
            session_index = 0

    mlp, mlp_input_dim, mlp_output_dim = _load_mlp_model(mlp_weights, device)

    session_labels: Dict[str, SessionLabelData] = {}
    segments_by_session: Dict[str, Dict[str, List[SegmentInfo]]] = {}
    for sess in sessions:
        info = _prepare_session_labels(data_path, sess)
        if info is None:
            continue
        session_labels[sess] = info
        segments_by_session[sess] = info.segments

    if not session_labels:
        raise RuntimeError("No labelled sessions available for inference.")

    session_results: Dict[str, SessionInferenceData] = {}
    for sess, info in session_labels.items():
        imu, dlc = _load_session_arrays(data_path, sess)
        probs = _run_models_for_session(
            model,
            mlp,
            mlp_input_dim,
            mlp_output_dim,
            imu,
            dlc,
            info.centres,
            Ts,
            batch_size,
            device,
            fuse_mode,
            session_index,
        )

        session_results[sess] = SessionInferenceData(
            centres=info.centres,
            index_to_row=info.index_to_row,
            probabilities=probs,
            label_lookup=info.label_lookup,
            label_vectors=info.label_vectors,
        )

    assignments = split_segments_by_action(
        segments_by_session,
        test_ratio=primary_test_ratio,
        seed=split_seed,
    )

    train_frames = collect_segment_centres(
        segments_by_session,
        assignments,
        split="train",
        deduplicate=True,
        include="all",
    )
    holdout_frames = collect_segment_centres(
        segments_by_session,
        assignments,
        split="test",
        deduplicate=True,
        include="all",
    )

    num_classes = mlp_output_dim
    train_probs, train_labels, _ = _collect_split_arrays(session_results, train_frames, num_classes)
    holdout_probs, holdout_labels, holdout_refs = _collect_split_arrays(
        session_results, holdout_frames, num_classes
    )

    (
        val_probs,
        val_labels,
        val_refs,
        test_probs,
        test_labels,
        test_refs,
    ) = split_holdout_into_val_test(
        holdout_probs,
        holdout_labels,
        holdout_refs,
        seed=split_seed,
        val_fraction_in_holdout=val_fraction_in_holdout,
    )

    print("\n[Split sizes]")
    print(f"Train(70%): {len(train_labels)} samples")
    print(f"Val(~10%): {len(val_labels)} samples")
    print(f"Test(~20%): {len(test_labels)} samples\n")

    if len(train_labels):
        _print_counts("Train", train_labels)
    if len(val_labels):
        _print_counts("Val", val_labels)
    if len(test_labels):
        _print_counts("Test", test_labels)

    test_metrics_df, test_preds_bin = evaluate_predictions(test_probs, test_labels)

    macro_f1 = float(test_metrics_df["macro_f1"].iloc[0])
    print("[FINAL TEST] macroF1={:.4f}".format(macro_f1))
    print("[FINAL TEST] per-class metrics:")
    print(test_metrics_df)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if test_metrics_csv:
            metrics_path = os.path.join(out_dir, test_metrics_csv)
            test_metrics_df.to_csv(metrics_path, index=False)
            print(f"[✔] Test metrics saved to {metrics_path}")
        if test_pred_csv:
            df = pd.DataFrame(test_probs.numpy(), columns=[f"pred_{b}" for b in BEHAVIORS])
            for i, beh in enumerate(BEHAVIORS):
                df[f"true_{beh}"] = test_labels[:, i].numpy()
                df[f"binpred_{beh}"] = test_preds_bin[:, i].numpy()
            df.insert(0, "session", [sess for sess, _ in test_refs])
            df.insert(1, "frame", [frame for _, frame in test_refs])
            pred_path = os.path.join(out_dir, test_pred_csv)
            df.to_csv(pred_path, index=False)
            print(f"[✔] Test predictions saved to {pred_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Joint inference for encoder + MLP classifier")
    p.add_argument("--repr-weights", required=True, help="Encoder checkpoint path")
    p.add_argument("--mlp-weights", required=True, help="MLP checkpoint path")
    p.add_argument("--data-path", required=True, help="Dataset root directory")
    p.add_argument("--sessions", nargs="+", required=True, help="Session names to evaluate")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--fuse-mode", choices=["imu", "dlc", "both"], default="both")
    p.add_argument("--session-index", type=int, default=0, help="Session index for adapter embeddings")
    p.add_argument("--window-sizes", nargs="*", type=int, default=[16, 32, 64, 128])
    p.add_argument("--primary-test-ratio", type=float, default=0.3)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-frac-in-holdout", type=float, default=1.0 / 3.0)
    p.add_argument("--out-dir", default=None, help="Directory to save outputs")
    p.add_argument("--test-metrics-csv", default=None)
    p.add_argument("--test-pred-csv", default=None)

    args = p.parse_args()

    run_inference(
        repr_weights=args.repr_weights,
        mlp_weights=args.mlp_weights,
        data_path=args.data_path,
        sessions=args.sessions,
        device=args.device,
        batch_size=args.batch_size,
        fuse_mode=args.fuse_mode,
        session_index=args.session_index,
        window_sizes=args.window_sizes,
        primary_test_ratio=args.primary_test_ratio,
        split_seed=args.split_seed,
        val_fraction_in_holdout=args.val_frac_in_holdout,
        out_dir=args.out_dir,
        test_metrics_csv=args.test_metrics_csv,
        test_pred_csv=args.test_pred_csv,
    )


if __name__ == "__main__":
    main()

