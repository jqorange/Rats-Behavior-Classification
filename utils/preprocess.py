# utils/preprocess.py
import os
import math
import random
from typing import Sequence, Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import h5py

from .window_dataset import RatsWindowDataset


def _split_into_groups(indices: List[int], group_size: int) -> List[List[int]]:
    """把索引按 batch_size 切成若干组（最后不满的一组丢弃）"""
    n_full = len(indices) // group_size
    return [indices[i*group_size:(i+1)*group_size] for i in range(n_full)]


def _ensure_len_group(f: h5py.File, T: int,
                      feat_imu: int, feat_dlc: int, n_labels: int,
                      batch_size: int):
    """确保 len_T 组和其中的数据集存在（resizable），否则创建。"""
    grp_name = f"len_{T}"
    grp = f.require_group(grp_name)

    def _mk(name, shape_tail, dtype, compression=None):
        if name not in grp:
            # 形状：(num_batches, batch_size, ...)
            shape = (0, batch_size, *shape_tail)
            maxshape = (None, batch_size, *shape_tail)
            kwargs = dict(chunks=True)
            # 如需更小体积可开启压缩：compression="gzip", compression_opts=4
            if compression is not None:
                kwargs.update(dict(compression=compression, compression_opts=4))
            grp.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, **kwargs)
        return grp[name]

    # 不对特征做压缩；mask/label 压缩收益较大，但这里保持简单一致
    _mk("imu",   (T, feat_imu), np.float32)
    _mk("dlc",   (T, feat_dlc), np.float32)
    _mk("mask",  (T,),          np.bool_)
    _mk("label", (n_labels,),   np.float32)

    if "num_batches" not in grp.attrs:
        grp.attrs["num_batches"] = 0

    return grp


def _append_batch(grp: h5py.Group,
                  imu: np.ndarray, dlc: np.ndarray,
                  mask: np.ndarray, label: np.ndarray):
    """在 len_T 组里追加一个 batch。"""
    n = grp["imu"].shape[0]
    new_n = n + 1
    grp["imu"].resize((new_n, ) + grp["imu"].shape[1:])
    grp["dlc"].resize((new_n, ) + grp["dlc"].shape[1:])
    grp["mask"].resize((new_n, ) + grp["mask"].shape[1:])
    grp["label"].resize((new_n, ) + grp["label"].shape[1:])

    grp["imu"][n, ...]   = imu
    grp["dlc"][n, ...]   = dlc
    grp["mask"][n, ...]  = mask
    grp["label"][n, ...] = label
    grp.attrs["num_batches"] = int(new_n)


def _process_one_group(dataset, session, idx_group, T):
    data = dataset.data[session]
    imu_list, dlc_list, mask_list, label_list = [], [], [], []

    for idx in idx_group:
        start, end = dataset.ranges[idx][T]
        imu_t, mask_t = dataset._crop_with_pad(data.imu, start, end)
        dlc_t, _      = dataset._crop_with_pad(data.dlc, start, end)
        _, label = dataset.samples[idx]

        imu_np = imu_t.numpy().astype(np.float32)
        dlc_np = dlc_t.numpy().astype(np.float32)

        # ✅ 清理非有限值，避免后续 adapter/LN 炸掉
        imu_np = np.nan_to_num(imu_np, nan=0.0, posinf=0.0, neginf=0.0)
        dlc_np = np.nan_to_num(dlc_np, nan=0.0, posinf=0.0, neginf=0.0)

        imu_list.append(imu_np)
        dlc_list.append(dlc_np)
        mask_list.append(mask_t.numpy().astype(bool))
        label_list.append(label.numpy().astype(np.float32))

    imu = np.stack(imu_list, axis=0)
    dlc = np.stack(dlc_list, axis=0)
    mask = np.stack(mask_list, axis=0)
    label = np.stack(label_list, axis=0)
    return T, imu, dlc, mask, label


def preprocess_dataset(
    dataset: RatsWindowDataset,
    batch_size: int,
    out_dir: str = "Dataset",
    *,
    group_mode: str = "by_session",         # "by_session" 或 "global"（建议 by_session）
    assign_T: str = "round_robin",          # "round_robin" 或 "random"
    num_workers: int = 0,                   # >0 启用多线程
    seed: int = 42
) -> None:
    """
    预处理：先按 index 分组，每组统一窗口长度 T，再统一裁剪；支持多线程加速。

    参数
    ----
    dataset : RatsWindowDataset
    batch_size : int
        每个 batch 的样本数，不满的丢弃。
    out_dir : str
        输出目录，写 <session>.h5（按会话分文件，方便后续 mix 控制）。
    group_mode : {"by_session", "global"}
        - "by_session": 每个 session 内部各自分组与裁剪（推荐；与原先逻辑一致）。
        - "global":     跨 session 把所有 index 混在一起分组，再按各组 session 写回各自文件。
    assign_T : {"round_robin", "random"}
        为每个组分配窗口长度 T 的策略。round_robin 能保证各 T 大致均衡。
    num_workers : int
        线程数。>0 时用 ThreadPoolExecutor 并行裁剪；HDF5 写仍在主线程串行进行。
    seed : int
        随机种子，保证每个 epoch 可复现（若希望每个 epoch 随机性不同，可传入 epoch 变化的 seed）。
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    window_sizes = list(dataset.window_sizes)

    # 收集每个 session 对应的样本索引
    session_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, (sess, _) in enumerate(dataset.samples):
        session_to_indices[sess].append(idx)

    # 如果是 global，就把所有 index 打散，再按所属 session 回写
    if group_mode == "global":
        all_indices = [(sess, idx) for sess, lst in session_to_indices.items() for idx in lst]
        rng.shuffle(all_indices)
        groups_per_session: Dict[str, List[List[int]]] = defaultdict(list)
        # 直接把 (sess, idx) 流按 batch_size 切，但必须保证同组都属于同一个 session
        # 为了简单稳妥：按 session 维持队列，再轮询抽取满 batch
        per_sess_queues = {s: lst[:] for s, lst in session_to_indices.items()}
        for s in per_sess_queues:
            rng.shuffle(per_sess_queues[s])

        # 轮询拼 batch：尽最大努力均衡，不强行跨 session
        made_any = True
        while made_any:
            made_any = False
            for s in list(per_sess_queues.keys()):
                q = per_sess_queues[s]
                if len(q) >= batch_size:
                    groups_per_session[s].append([q.pop() for _ in range(batch_size)])
                    made_any = True

    else:
        # by_session：每个 session 自己打乱并分组
        groups_per_session = {}
        for s, idxs in session_to_indices.items():
            rng.shuffle(idxs)
            groups_per_session[s] = _split_into_groups(idxs, batch_size)

    # 预先拿到维度信息（用第一个 session 读取）
    first_sess = next(iter(dataset.data.keys()))
    feat_imu = int(dataset.data[first_sess].imu.shape[1])
    feat_dlc = int(dataset.data[first_sess].dlc.shape[1])
    # label 维度
    sample_label = dataset.samples[0][1]
    n_labels = int(sample_label.numel())

    # 每个 session 单独写一个 H5 文件；为了避免 HDF5 并发写问题，写入串行进行。
    for session, groups in groups_per_session.items():
        if not groups:
            continue
        # 给每个组分配 T
        if assign_T == "round_robin":
            Ts = [window_sizes[i % len(window_sizes)] for i in range(len(groups))]
        else:  # random
            Ts = [rng.choice(window_sizes) for _ in range(len(groups))]

        path = os.path.join(out_dir, f"{session}.h5")
        # 覆盖写（每个 epoch 重新生成）
        with h5py.File(path, "w") as f:
            # 先准备所有 len_T 组
            prepared_T = set()
            for T in set(Ts):
                _ensure_len_group(f, T, feat_imu, feat_dlc, n_labels, batch_size)
                prepared_T.add(T)

            # 并行裁剪→主线程逐个写入
            if num_workers and num_workers > 0:
                with ThreadPoolExecutor(max_workers=num_workers) as ex:
                    futures = []
                    for idx_group, T in zip(groups, Ts):
                        futures.append(ex.submit(_process_one_group, dataset, session, idx_group, T))
                    for fut in as_completed(futures):
                        T, imu, dlc, mask, label = fut.result()
                        grp = f[f"len_{T}"]
                        _append_batch(grp, imu, dlc, mask, label)
            else:
                # 单线程
                for idx_group, T in zip(groups, Ts):
                    T, imu, dlc, mask, label = _process_one_group(dataset, session, idx_group, T)
                    grp = f[f"len_{T}"]
                    _append_batch(grp, imu, dlc, mask, label)


def load_preprocessed_batches(
    sessions: Sequence[str],
    session_to_idx: Dict[str, int],
    out_dir: str = "Dataset",
    mix: bool = False,
):
    """
    从预处理好的 HDF5 依次产出 batch。

    mix=False: 逐 session 顺序产出；
    mix=True: 先把所有 (session, batch) 收集后打乱。
    """
    batches = []
    for session in sessions:
        path = os.path.join(out_dir, f"{session}.h5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for grp in f.values():
                imu = np.asarray(grp["imu"], dtype=np.float32)     # (num_batches, B, T, F_imu)
                dlc = np.asarray(grp["dlc"], dtype=np.float32)     # (num_batches, B, T, F_dlc)
                mask = np.asarray(grp["mask"], dtype=bool)          # (num_batches, B, T)
                label = np.asarray(grp["label"], dtype=np.float32)  # (num_batches, B, C)
                num_batches = imu.shape[0]
                for i in range(num_batches):
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
            "mask": torch.from_numpy(mask),
            "label": torch.from_numpy(label),
            "session": session,
            "session_idx": torch.full(
                (imu.shape[0],), session_to_idx[session], dtype=torch.long
            ),
        }
