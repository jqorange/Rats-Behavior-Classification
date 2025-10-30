import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .segments import (
    SegmentInfo,
    SegmentKey,
    assign_segments_train_val_test,
    compute_segments,
    extract_label_arrays,
)


@dataclass
class SessionData:
    """Container for per-session arrays (full length)."""
    imu: torch.Tensor  # (L, F_imu)
    dlc: torch.Tensor  # (L, F_dlc)


class RatsWindowDataset(Dataset):
    """Full-sequence dataset for rat behaviour.

    不再在这里裁剪窗口，只加载：
      - 每个 session 的全长 IMU / DLC 特征
      - 每个标签帧的索引 + onehot label

    Parameters
    ----------
    root: str
        Root directory. Expect structure:
        IMU/<session>/<session>_IMU_features.csv
        DLC/<session>/final_filtered_<session>_50hz.csv
        labels/<session>/label_<session>.csv
    sessions: list[str]
        要载入的 session 名称
    split: {"train", "test"}
        80/20 时间切分
    session_ranges: dict[str, (start, end)]
        限定使用的索引范围
    max_len_per_session: int
        每个 session 载入的最大帧数，默认 150000，用于节省内存
    """

    def __init__(
        self,
        root: str,
        sessions: Sequence[str],
        split: str = "train",
        session_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        max_len_per_session: int = 150_000,
        *,
        test_ratio: float = 0.2,
        split_seed: int = 0,
        num_folds: int = 1,
        fold_index: int = 0,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}
        self.root = root
        self.sessions = list(sessions)
        self.split = split
        self.session_ranges = session_ranges or {}
        self.max_len_per_session = max_len_per_session
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self.num_folds = int(num_folds)
        self.fold_index = int(fold_index)

        # map session names to consecutive indices
        self.session_to_idx = {s: i for i, s in enumerate(self.sessions)}

        self.data: Dict[str, SessionData] = {}
        # 仅带标签的样本（用于有监督损失）: (session_name, label_tensor, centre_index)
        self.samples: List[Tuple[str, torch.Tensor, int]] = []
        # 全部帧的样本（用于无监督损失）
        self.unsup_samples: List[Tuple[str, torch.Tensor, int]] = []
        self.num_labels = 0
        self.label_columns: List[str] = []
        # 标记是否成功载入了任何标签文件
        self.has_labels = False

        session_segments: Dict[str, Dict[str, List[SegmentInfo]]] = {}
        session_label_arrays: Dict[str, torch.Tensor] = {}
        session_label_lookup: Dict[str, Dict[int, int]] = {}

        for session in self.sessions:
            imu_file = os.path.join(root, "IMU", session, f"{session}_IMU_features_madnorm.csv")
            dlc_file = os.path.join(root, "DLC", session, f"final_filtered_{session}_50hz_madnorm.csv")
            label_file = os.path.join(root, "labels", session, f"label_{session}.csv")

            imu_df = pd.read_csv(imu_file, nrows=self.max_len_per_session)
            dlc_df = pd.read_csv(dlc_file, nrows=self.max_len_per_session)
            label_df = pd.read_csv(label_file) if os.path.exists(label_file) else None

            # 对齐长度
            min_len = min(len(imu_df), len(dlc_df))
            imu = torch.from_numpy(imu_df.iloc[:min_len].to_numpy(dtype=np.float32))
            dlc = torch.from_numpy(dlc_df.iloc[:min_len].to_numpy(dtype=np.float32))
            self.data[session] = SessionData(imu=imu, dlc=dlc)

            # label: 只保留非全零
            zero_label = torch.zeros(self.num_labels, dtype=torch.float32)
            if label_df is not None:
                indices, labels_np, columns = extract_label_arrays(
                    label_df,
                    label_columns=self.label_columns or None,
                    max_index=min_len,
                    session_range=self.session_ranges.get(session),
                )

                if labels_np.size:
                    n_labels = labels_np.shape[1]
                    if self.num_labels == 0:
                        self.num_labels = n_labels
                        self.label_columns = columns
                        zero_label = torch.zeros(self.num_labels, dtype=torch.float32)
                        if self.unsup_samples:
                            self.unsup_samples = [
                                (sess, zero_label, centre)
                                for sess, _, centre in self.unsup_samples
                            ]
                    else:
                        if n_labels != self.num_labels:
                            raise ValueError("Inconsistent number of label columns across sessions")
                        zero_label = torch.zeros(self.num_labels, dtype=torch.float32)

                    labels_tensor = torch.from_numpy(labels_np)
                    session_label_arrays[session] = labels_tensor
                    session_label_lookup[session] = {
                        int(idx): pos for pos, idx in enumerate(indices.tolist())
                    }
                    segs = compute_segments(indices, labels_np, self.label_columns)
                    if any(segs.values()):
                        session_segments[session] = segs
                        self.has_labels = True
                elif self.num_labels == 0:
                    n_labels = labels_np.shape[1]
                    if n_labels > 0:
                        self.num_labels = n_labels
                        self.label_columns = columns
                        zero_label = torch.zeros(self.num_labels, dtype=torch.float32)
                        if self.unsup_samples:
                            self.unsup_samples = [
                                (sess, zero_label, centre)
                                for sess, _, centre in self.unsup_samples
                            ]

            # 生成无监督样本：覆盖所有帧
            n_total = min_len
            unsup_start, unsup_end = 0, n_total
            # 可选限制范围也作用于无监督样本
            if session in self.session_ranges:
                start, end = self.session_ranges[session]
                unsup_start = max(unsup_start, start)
                unsup_end = min(unsup_end, end)
            for c in range(unsup_start, unsup_end):
                self.unsup_samples.append((session, zero_label, int(c)))

        if session_segments:
            holdout_ratio = float(self.test_ratio)
            train_ratio = max(0.0, 1.0 - holdout_ratio)
            val_ratio = holdout_ratio / 3.0
            test_ratio = max(0.0, holdout_ratio - val_ratio)
            assignments = assign_segments_train_val_test(
                session_segments,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                seed=self.split_seed,
                num_folds=self.num_folds,
                fold_index=self.fold_index,
            )
            for session, per_action in session_segments.items():
                labels_tensor = session_label_arrays.get(session)
                index_lookup = session_label_lookup.get(session)
                if labels_tensor is None or index_lookup is None:
                    continue
                for action, segs in per_action.items():
                    for seg in segs:
                        key = SegmentKey(session=session, action=action, start=seg.start, end=seg.end)
                        if key in assignments.get(self.split, set()):
                            for frame_idx in range(int(seg.start), int(seg.end) + 1):
                                row_idx = index_lookup.get(frame_idx)
                                if row_idx is None:
                                    continue
                                lab_tensor = labels_tensor[row_idx].clone()
                                self.samples.append((session, lab_tensor, int(frame_idx)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        session, label, centre = self.samples[idx]
        return {
            "session": session,
            "session_idx": self.session_to_idx[session],
            "label": label,
            "centre": centre,
        }
