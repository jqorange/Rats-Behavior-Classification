import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
    ) -> None:
        super().__init__()
        assert split in {"train", "test"}
        self.root = root
        self.sessions = list(sessions)
        self.split = split
        self.session_ranges = session_ranges or {}
        self.max_len_per_session = max_len_per_session

        # map session names to consecutive indices
        self.session_to_idx = {s: i for i, s in enumerate(self.sessions)}

        self.data: Dict[str, SessionData] = {}
        # 每个样本： (session_name, label_tensor, centre_index)
        self.samples: List[Tuple[str, torch.Tensor, int]] = []

        for session in self.sessions:
            imu_file = os.path.join(root, "IMU", session, f"{session}_IMU_features.csv")
            dlc_file = os.path.join(root, "DLC", session, f"final_filtered_{session}_50hz.csv")
            label_file = os.path.join(root, "labels", session, f"label_{session}.csv")

            imu_df = pd.read_csv(imu_file, nrows=self.max_len_per_session)
            dlc_df = pd.read_csv(dlc_file, nrows=self.max_len_per_session)
            label_df = pd.read_csv(label_file)

            # 对齐长度
            min_len = min(len(imu_df), len(dlc_df))
            imu = torch.from_numpy(imu_df.iloc[:min_len].to_numpy(dtype=np.float32))
            dlc = torch.from_numpy(dlc_df.iloc[:min_len].to_numpy(dtype=np.float32))
            self.data[session] = SessionData(imu=imu, dlc=dlc)

            # label: 只保留非全零
            label_df = label_df[label_df.drop(columns=["Index"]).any(axis=1)]
            label_df = label_df.sort_values("Index")
            indices = label_df["Index"].to_numpy(dtype=int)
            labels = label_df.drop(columns=["Index"]).to_numpy(dtype=np.float32)

            # 限制在最大长度范围内
            m_valid = indices < min_len
            indices = indices[m_valid]
            labels = labels[m_valid]

            # 可选限制范围
            if session in self.session_ranges:
                start, end = self.session_ranges[session]
                m = (indices >= start) & (indices < end)
                indices = indices[m]
                labels = labels[m]

            # 80/20 split
            n_train = int(len(indices) * 0.8)
            if split == "train":
                idx_split = slice(0, n_train)
            else:
                idx_split = slice(n_train, None)

            for centre, lab in zip(indices[idx_split], labels[idx_split]):
                lab_tensor = torch.from_numpy(lab)
                self.samples.append((session, lab_tensor, int(centre)))

        self.num_labels = self.samples[0][1].shape[-1] if self.samples else 0

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
