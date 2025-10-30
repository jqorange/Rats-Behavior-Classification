from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


LABEL_INDEX_COLUMN = "Index"


@dataclass(frozen=True)
class SegmentKey:
    """Uniquely identifies a labelled segment within a session."""

    session: str
    action: str
    start: int
    end: int


@dataclass
class SegmentInfo:
    """Stores metadata for a contiguous labelled segment."""

    start: int
    end: int
    centre: int
    label_row: int


def extract_label_arrays(
    label_df: Optional[pd.DataFrame],
    *,
    label_columns: Optional[Sequence[str]] = None,
    max_index: Optional[int] = None,
    session_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Filter a raw label dataframe and convert it to numpy arrays.

    Parameters
    ----------
    label_df:
        Raw dataframe loaded from ``label_<session>.csv``.  ``None`` results in
        empty outputs.
    label_columns:
        Optional explicit ordering of behaviour columns to keep.  When
        ``None`` the order will be inferred from the dataframe (excluding the
        ``Index`` column).
    max_index:
        If provided, drop rows whose frame indices are greater than or equal to
        ``max_index``.  This is useful when feature arrays are shorter than the
        original recording.
    session_range:
        Optional ``(start, end)`` tuple to further restrict the usable range.

    Returns
    -------
    indices, labels, columns: tuple
        ``indices`` is a 1D array of frame indices, ``labels`` is a 2D float32
        array of the selected behaviour columns and ``columns`` gives the final
        column order.
    """

    if label_df is None:
        return np.empty(0, dtype=int), np.empty((0, 0), dtype=np.float32), []

    df = label_df.copy()
    if LABEL_INDEX_COLUMN not in df.columns:
        raise KeyError(f"Label dataframe missing '{LABEL_INDEX_COLUMN}' column")

    if label_columns is None:
        inferred = [c for c in df.columns if c != LABEL_INDEX_COLUMN]
        label_columns = inferred
    else:
        label_columns = list(label_columns)
        missing = [c for c in label_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Label dataframe missing columns: {missing}")

    if not label_columns:
        return np.empty(0, dtype=int), np.empty((0, 0), dtype=np.float32), list(label_columns)

    df = df[[LABEL_INDEX_COLUMN] + list(label_columns)]
    mask_nonzero = df[label_columns].astype(bool).any(axis=1)
    df = df.loc[mask_nonzero]
    df = df.sort_values(LABEL_INDEX_COLUMN)

    if max_index is not None:
        df = df[df[LABEL_INDEX_COLUMN] < max_index]

    if session_range is not None:
        start, end = session_range
        df = df[(df[LABEL_INDEX_COLUMN] >= start) & (df[LABEL_INDEX_COLUMN] < end)]

    indices = df[LABEL_INDEX_COLUMN].to_numpy(dtype=int)
    labels = df[label_columns].to_numpy(dtype=np.float32)
    return indices, labels, list(label_columns)


def compute_segments(
    indices: np.ndarray,
    labels: np.ndarray,
    label_columns: Sequence[str],
) -> Dict[str, List[SegmentInfo]]:
    """Extract contiguous positive segments for each behaviour column."""

    segments: Dict[str, List[SegmentInfo]] = {name: [] for name in label_columns}
    if indices.size == 0 or labels.size == 0:
        return segments

    index_to_row = {int(idx): i for i, idx in enumerate(indices)}

    for col_idx, action in enumerate(label_columns):
        action_mask = labels[:, col_idx] > 0
        action_indices = indices[action_mask]
        if action_indices.size == 0:
            continue

        start_idx = int(action_indices[0])
        prev_idx = start_idx
        current_segment = [start_idx]

        for raw_idx in action_indices[1:]:
            idx = int(raw_idx)
            if idx == prev_idx + 1:
                current_segment.append(idx)
                prev_idx = idx
                continue

            centre = current_segment[len(current_segment) // 2]
            row_idx = index_to_row[centre]
            segments[action].append(
                SegmentInfo(start=current_segment[0], end=current_segment[-1], centre=centre, label_row=row_idx)
            )
            current_segment = [idx]
            prev_idx = idx

        if current_segment:
            centre = current_segment[len(current_segment) // 2]
            row_idx = index_to_row[centre]
            segments[action].append(
                SegmentInfo(start=current_segment[0], end=current_segment[-1], centre=centre, label_row=row_idx)
            )

    return segments


def split_segments_by_action(
    segments_by_session: MutableMapping[str, Dict[str, List[SegmentInfo]]],
    *,
    test_ratio: float,
    seed: int,
) -> Dict[str, Set[SegmentKey]]:
    """Split segments into train/test sets on a per-action basis."""

    grouped: Dict[str, List[Tuple[str, SegmentInfo]]] = {}
    for session, per_action in segments_by_session.items():
        for action, segs in per_action.items():
            if not segs:
                continue
            grouped.setdefault(action, []).extend((session, seg) for seg in segs)

    assignments: Dict[str, Set[SegmentKey]] = {"train": set(), "test": set()}
    rng = random.Random(seed)

    for action, seg_refs in grouped.items():
        if not seg_refs:
            continue
        order = list(range(len(seg_refs)))
        rng.shuffle(order)
        n_test = int(len(seg_refs) * test_ratio)
        test_indices = set(order[:n_test])

        for idx, (session, seg) in enumerate(seg_refs):
            key = SegmentKey(session=session, action=action, start=seg.start, end=seg.end)
            target = "test" if idx in test_indices else "train"
            assignments[target].add(key)

    return assignments


def _compute_split_counts(n_items: int, ratios: Sequence[float]) -> List[int]:
    """Return non-negative counts that respect the requested ratios."""

    if n_items <= 0:
        return [0 for _ in ratios]

    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("Sum of ratios must be positive")

    normalized = [max(0.0, r) / total_ratio for r in ratios]
    counts = []
    residuals = []
    for r in normalized:
        raw = n_items * r
        base = int(math.floor(raw))
        counts.append(base)
        residuals.append(raw - base)

    remaining = n_items - sum(counts)
    if remaining > 0:
        order = sorted(range(len(counts)), key=lambda idx: residuals[idx], reverse=True)
        for idx in order:
            if remaining <= 0:
                break
            counts[idx] += 1
            remaining -= 1

    return counts


def assign_segments_train_val_test(
    segments_by_session: MutableMapping[str, Dict[str, List[SegmentInfo]]],
    *,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    num_folds: int,
    fold_index: int,
) -> Dict[str, Set[SegmentKey]]:
    """Assign labelled segments to train/test/val splits consistently.

    The assignment is performed independently for each behaviour class.  All
    segments are first shuffled using a deterministic RNG that incorporates the
    provided ``seed``, ``num_folds`` and ``fold_index`` so that different folds
    result in different splits while keeping the process reproducible across
    scripts.
    """

    if num_folds <= 0:
        raise ValueError("num_folds must be positive")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError("fold_index must satisfy 0 <= fold_index < num_folds")

    grouped: Dict[str, List[Tuple[str, SegmentInfo]]] = {}
    for session, per_action in segments_by_session.items():
        for action, segs in per_action.items():
            if not segs:
                continue
            grouped.setdefault(action, []).extend((session, seg) for seg in segs)

    assignments: Dict[str, Set[SegmentKey]] = {"train": set(), "val": set(), "test": set()}
    combined_seed = (seed + 1) * 1_000_003 + num_folds * 97 + fold_index * 1009
    rng = random.Random(combined_seed)

    for action, seg_refs in grouped.items():
        if not seg_refs:
            continue
        order = list(range(len(seg_refs)))
        rng.shuffle(order)

        counts = _compute_split_counts(len(order), (train_ratio, test_ratio, val_ratio))
        train_count, test_count, val_count = counts

        boundaries = [0, train_count, train_count + test_count]
        split_labels = (
            ("train", boundaries[0], boundaries[1]),
            ("test", boundaries[1], boundaries[2]),
            ("val", boundaries[2], len(order)),
        )

        for split_name, start, end in split_labels:
            for idx in order[start:end]:
                session, seg = seg_refs[idx]
                key = SegmentKey(session=session, action=action, start=seg.start, end=seg.end)
                assignments[split_name].add(key)

    return assignments


def collect_segment_centres(
    segments_by_session: Dict[str, Dict[str, List[SegmentInfo]]],
    assignments: Dict[str, Set[SegmentKey]],
    split: str,
    *,
    deduplicate: bool = True,
    include: str = "centre",
) -> Dict[str, List[int]]:
    """Return frame indices per session for the requested split.

    Parameters
    ----------
    segments_by_session:
        Mapping from session name to behaviour segments.
    assignments:
        Train/test split assignments returned by :func:`split_segments_by_action`
        or train/val/test assignments from :func:`assign_segments_train_val_test`.
    split:
        Which split to extract (e.g. ``"train"``, ``"val"`` or ``"test"``).
    deduplicate:
        When ``True`` the collected indices are deduplicated and returned in
        ascending order.
    include:
        ``"centre"`` keeps the historical behaviour of returning each segment's
        centre index.  ``"all"`` returns every frame covered by the selected
        segments.
    """

    include = include.lower()
    if include not in {"centre", "all"}:
        raise ValueError("include must be either 'centre' or 'all'")

    selected = assignments.get(split, set())
    centres_per_session: Dict[str, List[int]] = {}

    for session, per_action in segments_by_session.items():
        centres: List[int] = []
        for action, segs in per_action.items():
            for seg in segs:
                key = SegmentKey(session=session, action=action, start=seg.start, end=seg.end)
                if key in selected:
                    if include == "centre":
                        centres.append(int(seg.centre))
                    else:  # include == "all"
                        centres.extend(range(int(seg.start), int(seg.end) + 1))
        if deduplicate:
            centres = sorted(set(centres))
        else:
            centres.sort()
        if centres:
            centres_per_session[session] = centres

    return centres_per_session


def iter_segment_keys(
    segments_by_session: Dict[str, Dict[str, List[SegmentInfo]]],
) -> Iterable[SegmentKey]:
    """Yield :class:`SegmentKey` entries for all segments."""

    for session, per_action in segments_by_session.items():
        for action, segs in per_action.items():
            for seg in segs:
                yield SegmentKey(session=session, action=action, start=seg.start, end=seg.end)

