import os
import random
from typing import Sequence, Dict

import numpy as np
import torch
import h5py

from .window_dataset import RatsWindowDataset


def preprocess_dataset(
    dataset: RatsWindowDataset,
    batch_size: int,
    out_dir: str = "Dataset",
) -> None:
    """Precompute random windows grouped by length and store as HDF5 batches.

    Parameters
    ----------
    dataset:
        Source :class:`RatsWindowDataset` providing access to raw sequences.
    batch_size:
        Number of samples per batch written to disk. Incomplete batches are
        discarded.
    out_dir:
        Directory to write ``<session>.h5`` files. Existing files are
        overwritten on every call so this function should be invoked at the
        beginning of each epoch.
    """

    os.makedirs(out_dir, exist_ok=True)
    groups: Dict[str, Dict[int, list]] = {
        s: {T: [] for T in dataset.window_sizes} for s in dataset.sessions
    }
    for idx in range(len(dataset.samples)):
        session, label = dataset.samples[idx]
        ranges = dataset.ranges[idx]
        T = random.choice(dataset.window_sizes)
        start, end = ranges[T]
        data = dataset.data[session]
        imu, mask = dataset._crop_with_pad(data.imu, start, end)
        dlc, _ = dataset._crop_with_pad(data.dlc, start, end)
        groups[session][T].append(
            (imu.numpy(), dlc.numpy(), mask.numpy(), label.numpy())
        )

    for session, len_map in groups.items():
        path = os.path.join(out_dir, f"{session}.h5")
        with h5py.File(path, "w") as f:
            for T, items in len_map.items():
                if not items:
                    continue
                num_batches = len(items) // batch_size
                if num_batches == 0:
                    continue
                items = items[: num_batches * batch_size]
                imu = np.stack([itm[0] for itm in items]).reshape(
                    num_batches, batch_size, T, -1
                )
                dlc = np.stack([itm[1] for itm in items]).reshape(
                    num_batches, batch_size, T, -1
                )
                mask = np.stack([itm[2] for itm in items]).reshape(
                    num_batches, batch_size, T
                )
                label = np.stack([itm[3] for itm in items]).reshape(
                    num_batches, batch_size, -1
                )
                grp = f.create_group(f"len_{T}")
                grp.create_dataset("imu", data=imu)
                grp.create_dataset("dlc", data=dlc)
                grp.create_dataset("mask", data=mask)
                grp.create_dataset("label", data=label)
                grp.attrs["num_batches"] = num_batches


def load_preprocessed_batches(
    sessions: Sequence[str],
    session_to_idx: Dict[str, int],
    out_dir: str = "Dataset",
    mix: bool = False,
):
    """Yield batches from preprocessed HDF5 files.

    Parameters
    ----------
    sessions:
        Iterable of session names whose files will be read.
    session_to_idx:
        Mapping from session name to integer index.
    out_dir:
        Directory containing the ``<session>.h5`` files.
    mix:
        If ``True``, batches from all sessions are shuffled together,
        otherwise batches are yielded session by session without mixing.
    """

    batches = []
    for session in sessions:
        path = os.path.join(out_dir, f"{session}.h5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for grp in f.values():
                imu = np.array(grp["imu"])
                dlc = np.array(grp["dlc"])
                mask = np.array(grp["mask"])
                label = np.array(grp["label"])
                for i in range(imu.shape[0]):
                    batches.append(
                        (
                            session,
                            imu[i],
                            dlc[i],
                            mask[i],
                            label[i],
                        )
                    )
    if mix:
        random.shuffle(batches)
    for session, imu, dlc, mask, label in batches:
        yield {
            "imu": torch.from_numpy(imu),
            "dlc": torch.from_numpy(dlc),
            "mask": torch.from_numpy(mask.astype(bool)),
            "label": torch.from_numpy(label),
            "session": session,
            "session_idx": torch.full(
                (imu.shape[0],), session_to_idx[session], dtype=torch.long
            ),
        }
