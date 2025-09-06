import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SessionData:
    """Container for per-session arrays."""
    imu: np.ndarray
    dlc: np.ndarray


class RatsWindowDataset(Dataset):
    """Multi-modal window dataset for rat behaviour.

    The dataset enumerates labelled centre indices for each session and
    samples a random window length for every access. Zero padding and a
    boolean mask are returned when the window exceeds sequence bounds.

    Parameters
    ----------
    root: str
        Root directory of the dataset. Expected structure:
        ``IMU/<session>/<session>_IMU_features.csv``
        ``DLC/<session>/final_filtered_<session>_50hz.csv``
        ``labels/<session>/label_<session>.csv``
    sessions: Sequence[str]
        List of session names to load.
    split: str
        Either ``"train"`` or ``"test"``. For each session the labelled
        indices are split 80/20 by time.
    window_sizes: Sequence[int]
        Candidate window lengths ``T``. One value is sampled uniformly for
        every ``__getitem__`` call.
    """

    def __init__(
        self,
        root: str,
        sessions: Sequence[str],
        split: str = "train",
        window_sizes: Sequence[int] = (16, 32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        assert split in {"train", "test"}
        self.root = root
        self.sessions = list(sessions)
        self.split = split
        self.window_sizes = list(window_sizes)

        self.data: Dict[str, SessionData] = {}
        self.samples: List[Tuple[str, int, np.ndarray]] = []

        for sid, session in enumerate(self.sessions):
            imu_file = os.path.join(root, "IMU", session, f"{session}_IMU_features.csv")
            dlc_file = os.path.join(root, "DLC", session, f"final_filtered_{session}_50hz.csv")
            label_file = os.path.join(root, "labels", session, f"label_{session}.csv")

            imu_df = pd.read_csv(imu_file)
            dlc_df = pd.read_csv(dlc_file)
            label_df = pd.read_csv(label_file)

            # ensure same length
            min_len = min(len(imu_df), len(dlc_df))
            imu = imu_df.iloc[:min_len].to_numpy(dtype=np.float32)
            dlc = dlc_df.iloc[:min_len].to_numpy(dtype=np.float32)
            self.data[session] = SessionData(imu=imu, dlc=dlc)

            label_df = label_df[label_df.drop(columns=["Index"]).any(axis=1)]
            label_df = label_df.sort_values("Index")
            indices = label_df["Index"].to_numpy(dtype=int)
            labels = label_df.drop(columns=["Index"]).to_numpy(dtype=np.float32)

            n_train = int(len(indices) * 0.8)
            if split == "train":
                idx_split = slice(0, n_train)
            else:
                idx_split = slice(n_train, None)

            for idx, lab in zip(indices[idx_split], labels[idx_split]):
                self.samples.append((session, idx, lab))

        self.num_labels = self.samples[0][2].shape[-1] if self.samples else 0

    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.samples)

    def _crop_with_pad(self, arr: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        T = end - start
        feat = arr.shape[1]
        out = np.zeros((T, feat), dtype=np.float32)
        mask = np.zeros(T, dtype=bool)
        s = max(start, 0)
        e = min(end, len(arr))
        out_start = s - start
        out_end = out_start + (e - s)
        out[out_start:out_end] = arr[s:e]
        mask[out_start:out_end] = True
        return out, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # pragma: no cover - I/O heavy
        session, centre, label = self.samples[idx]
        T = random.choice(self.window_sizes)
        half = T // 2
        if T % 2:
            start = centre - half
            end = centre + half + 1
        else:
            start = centre - half
            end = centre + half
        data = self.data[session]
        imu, mask = self._crop_with_pad(data.imu, start, end)
        dlc, _ = self._crop_with_pad(data.dlc, start, end)
        return {
            "imu": torch.from_numpy(imu),
            "dlc": torch.from_numpy(dlc),
            "mask": torch.from_numpy(mask),
            "label": torch.from_numpy(label),
            "session": session,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    """Custom collate to pad variable-length windows."""
    max_T = max(item["imu"].shape[0] for item in batch)
    feat_imu = batch[0]["imu"].shape[1]
    feat_dlc = batch[0]["dlc"].shape[1]

    imu = torch.zeros(len(batch), max_T, feat_imu)
    dlc = torch.zeros(len(batch), max_T, feat_dlc)
    mask = torch.zeros(len(batch), max_T, dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch])

    sessions = []
    for i, item in enumerate(batch):
        T = item["imu"].shape[0]
        imu[i, :T] = item["imu"]
        dlc[i, :T] = item["dlc"]
        mask[i, :T] = item["mask"]
        sessions.append(item["session"])

    return {
        "imu": imu,
        "dlc": dlc,
        "mask": mask,
        "label": labels,
        "session": sessions,
    }
