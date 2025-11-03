"""Utility script to export per-frame train/val/test assignments.

This script replicates the dataset splitting logic used throughout the
project.  Given a random seed, cross-validation configuration and a list of
sessions, it will determine the split assignment for every labelled frame and
write the result to a CSV file.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.segments import (
    SegmentInfo,
    SegmentKey,
    assign_segments_train_val_test,
    compute_segments,
    extract_label_arrays,
)


@dataclass
class SessionSegments:
    """Container holding per-session segment information."""

    segments: Dict[str, List[SegmentInfo]]
    frame_count: int


def _default_feature_paths(root: str, session: str) -> Iterable[str]:
    """Yield potential feature CSV paths used to infer session length."""

    yield os.path.join(root, "IMU", session, f"{session}_IMU_features_madnorm.csv")
    yield os.path.join(root, "DLC", session, f"final_filtered_{session}_50hz_madnorm.csv")


def _infer_frame_count(root: str, session: str) -> Optional[int]:
    """Return the number of frames for the session if feature files exist."""

    for path in _default_feature_paths(root, session):
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=[0])
            return len(df)
    return None


def _load_session_segments(
    root: str,
    session: str,
    *,
    label_columns: Optional[Sequence[str]] = None,
) -> Tuple[SessionSegments, List[str]]:
    """Load labels and compute behaviour segments for a session."""

    label_path = os.path.join(root, "labels", session, f"label_{session}.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found for session '{session}': {label_path}")

    label_df = pd.read_csv(label_path)
    indices, labels, columns = extract_label_arrays(label_df, label_columns=label_columns)

    if indices.size == 0 or labels.size == 0:
        raise ValueError(f"Session '{session}' does not contain any positive labels")

    frame_count = _infer_frame_count(root, session)
    if frame_count is None:
        frame_count = int(indices.max()) + 1

    segments = compute_segments(indices, labels, columns)
    return SessionSegments(segments=segments, frame_count=frame_count), columns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root directory containing IMU/DLC/labels subfolders")
    parser.add_argument("--sessions", nargs="+", help="Session names to process")
    parser.add_argument("--output", required=True, help="Path to the CSV file that will be written")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed used for shuffling segments")
    parser.add_argument("--num-folds", type=int, default=1, help="Total number of folds in the cross-validation setup")
    parser.add_argument("--fold-index", type=int, default=0, help="Index of the current fold (0-indexed)")
    parser.add_argument(
        "--primary-test-ratio",
        type=float,
        default=0.3,
        help="Holdout proportion used by train_new/train_mlp (0.3 -> 7:2:1 split)",
    )
    parser.add_argument(
        "--val-frac-in-holdout",
        type=float,
        default=1.0 / 3.0,
        help="Fraction of holdout samples routed to validation (default keeps 7:2:1)",
    )
    return parser.parse_args()


def _build_segment_lookup(
    segments: Dict[str, Dict[str, List[SegmentInfo]]]
) -> Dict[SegmentKey, SegmentInfo]:
    lookup: Dict[SegmentKey, SegmentInfo] = {}
    for session, per_action in segments.items():
        for action, segs in per_action.items():
            for seg in segs:
                key = SegmentKey(session=session, action=action, start=seg.start, end=seg.end)
                lookup[key] = seg
    return lookup


def _compute_split_ratios(primary_test_ratio: float, val_frac_in_holdout: float) -> Tuple[float, float, float]:
    """Mirror the ratio calculation used by train_new.py and train_mlp.py."""

    if not (0.0 < primary_test_ratio < 1.0):
        raise ValueError("--primary-test-ratio must be between 0 and 1")
    if not (0.0 < val_frac_in_holdout < 1.0):
        raise ValueError("--val-frac-in-holdout must be between 0 and 1")

    train_ratio = max(0.0, 1.0 - primary_test_ratio)
    val_ratio = primary_test_ratio * val_frac_in_holdout
    test_ratio = primary_test_ratio - val_ratio
    if train_ratio == 0 and val_ratio == 0 and test_ratio == 0:
        raise ValueError("At least one split ratio must be positive")
    return train_ratio, test_ratio, val_ratio


def main() -> None:
    args = _parse_args()
    data_root = args.data_root
    sessions = args.sessions

    train_ratio, test_ratio, val_ratio = _compute_split_ratios(
        float(args.primary_test_ratio), float(args.val_frac_in_holdout)
    )

    segments_by_session: Dict[str, Dict[str, List[SegmentInfo]]] = {}
    segment_metadata: Dict[str, SessionSegments] = {}
    label_columns: Optional[List[str]] = None

    for session in sessions:
        session_info, columns = _load_session_segments(data_root, session, label_columns=label_columns)
        segments_by_session[session] = session_info.segments
        segment_metadata[session] = session_info
        if label_columns is None:
            label_columns = columns
        else:
            if columns != label_columns:
                raise ValueError(
                    "Label columns mismatch across sessions: "
                    f"expected {label_columns}, got {columns} for session {session}"
                )

    assignments = assign_segments_train_val_test(
        segments_by_session,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=args.seed,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
    )

    segment_lookup = _build_segment_lookup(segments_by_session)

    rows: List[Tuple[str, int, str]] = []
    for session in sessions:
        metadata = segment_metadata[session]
        frame_splits = np.full(metadata.frame_count, "unassigned", dtype=object)

        for split_name in ("train", "val", "test"):
            for key in assignments.get(split_name, set()):
                if key.session != session:
                    continue
                seg = segment_lookup[key]
                start = max(int(seg.start), 0)
                end = min(int(seg.end), metadata.frame_count - 1)
                frame_splits[start : end + 1] = split_name

        rows.extend((session, idx, split) for idx, split in enumerate(frame_splits))

    output_df = pd.DataFrame(rows, columns=["session", "frame", "split"])
    output_df.to_csv(args.output, index=False)
    print(f"Wrote per-frame splits for {len(sessions)} sessions to {args.output}")


if __name__ == "__main__":
    main()
