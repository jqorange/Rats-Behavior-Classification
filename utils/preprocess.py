# utils/preprocess.py
import os
import random
from typing import Sequence, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import h5py

from .window_dataset import RatsWindowDataset


def _split_into_groups(indices: List[int], group_size: int) -> List[List[int]]:
    """把索引按 batch_size 切成若干组（最后不满的一组丢弃）"""
    n_full = len(indices) // group_size
    return [indices[i*group_size:(i+1)*group_size] for i in range(n_full)]


@torch.no_grad()
def _mad_from_rows(
    x: torch.Tensor,
    *,
    eps: float = 1e-8,
    scale_to_std: bool = True,
    sample_rows: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算每列 median & MAD（可抽样加速）
    x: (L, F)
    return: (median(F,), mad_scaled(F,))
    """
    x = torch.nan_to_num(x.float())
    L = x.shape[0]
    if sample_rows is not None and L > sample_rows:
        # 随机抽样到 sample_rows 行（在 device 上完成，不额外复制）
        idx = torch.randint(0, L, (sample_rows,), device=x.device)
        x_s = x.index_select(0, idx)
    else:
        x_s = x

    med = x_s.median(dim=0).values                     # (F,)
    mad = (x_s - med).abs().median(dim=0).values       # (F,)
    if scale_to_std:
        mad = mad * 1.4826
    mad = torch.clamp(mad, min=eps)
    return med, mad


@torch.no_grad()
def _apply_mad_norm(x: torch.Tensor, med: torch.Tensor, mad: torch.Tensor) -> torch.Tensor:
    """按列 MAD 归一化： (x - med) / mad"""
    return torch.nan_to_num((x.float() - med) / mad)


@torch.no_grad()
def _batch_crop_rows(data_rows: torch.Tensor, starts: torch.Tensor, T: int):
    """
    向量化裁剪（无逐样本 for-loop）
      data_rows: (L, F)
      starts: (B,)
    return:
      windows: (B, T, F), mask: (B, T)
    """
    device = data_rows.device
    L, F = data_rows.shape
    B = starts.numel()
    ar = torch.arange(T, device=device)                 # (T,)
    idx = starts.view(-1, 1) + ar.view(1, -1)          # (B, T)
    mask = (idx >= 0) & (idx < L)
    idx = idx.clamp(0, L - 1)
    rows = torch.index_select(data_rows, 0, idx.view(-1))
    windows = rows.view(B, T, F) * mask.unsqueeze(-1)
    return windows, mask


def _save_scaler(cache_path: str, med_imu: torch.Tensor, mad_imu: torch.Tensor,
                 med_dlc: torch.Tensor, mad_dlc: torch.Tensor):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        med_imu=med_imu.detach().cpu().numpy(),
        mad_imu=mad_imu.detach().cpu().numpy(),
        med_dlc=med_dlc.detach().cpu().numpy(),
        mad_dlc=mad_dlc.detach().cpu().numpy(),
        version=1,
    )


def _load_scaler(cache_path: str, device: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not os.path.exists(cache_path):
        return None
    arr = np.load(cache_path)
    med_imu = torch.from_numpy(arr["med_imu"]).to(device)
    mad_imu = torch.from_numpy(arr["mad_imu"]).to(device)
    med_dlc = torch.from_numpy(arr["med_dlc"]).to(device)
    mad_dlc = torch.from_numpy(arr["mad_dlc"]).to(device)
    return med_imu, mad_imu, med_dlc, mad_dlc


def _np_dtype_from_str(name: str):
    name = name.lower()
    if name in ("float32", "fp32"):
        return np.float32
    if name in ("float16", "fp16", "half"):
        return np.float16
    if name in ("uint8", "u8", "byte"):
        return np.uint8
    if name in ("bool", "boolean"):
        return np.bool_
    raise ValueError(f"Unsupported dtype: {name}")


def _build_session_file(
    session: str,
    groups: List[List[int]],
    Ts: List[int],
    dataset: RatsWindowDataset,
    samples: List[Tuple[str, torch.Tensor, int]],
    out_dir: str,
    device: str,
    compress: bool,
    mad_scale_to_std: bool,
    mad_cache_dir: str,
    use_mad_cache: bool,
    mad_sample_rows: Optional[int],
    store_dtype: str,
    label_dtype: str,
    mask_dtype: str,
    preproc_shard_batches: int,
):
    """单个 session：MAD 归一化（可缓存/抽样）+ 分块向量化裁剪 + 分块写 HDF5。"""
    if not groups:
        return

    feat_imu = int(dataset.data[session].imu.shape[1])
    feat_dlc = int(dataset.data[session].dlc.shape[1])
    n_labels = int(samples[0][1].numel()) if samples else 0
    batch_size = len(groups[0])

    # ---- 预创建 HDF5 dset（按 T 预分配） ----
    count_T = Counter(Ts)
    path = os.path.join(out_dir, f"{session}.h5")
    with h5py.File(path, "w") as f:
        kwargs = dict(chunks=True)
        if compress:
            kwargs.update(dict(compression="gzip", compression_opts=4))

        dsets = {}
        for T, n_batches in count_T.items():
            grp = f.create_group(f"len_{T}")
            dsets[T] = {
                "imu":   grp.create_dataset("imu",   shape=(n_batches, batch_size, T, feat_imu), dtype=_np_dtype_from_str(store_dtype), **kwargs),
                "dlc":   grp.create_dataset("dlc",   shape=(n_batches, batch_size, T, feat_dlc), dtype=_np_dtype_from_str(store_dtype), **kwargs),
                "mask":  grp.create_dataset("mask",  shape=(n_batches, batch_size, T),          dtype=_np_dtype_from_str(mask_dtype),  **kwargs),
                "label": grp.create_dataset("label", shape=(n_batches, batch_size, n_labels),    dtype=_np_dtype_from_str(label_dtype), **kwargs),
            }
            grp.attrs["num_batches"] = n_batches

        # ---- 准备序列并做 MAD 归一化（优先用缓存）----
        imu_rows = dataset.data[session].imu.to(device)
        dlc_rows = dataset.data[session].dlc.to(device)

        cache_path = os.path.join(mad_cache_dir, f"{session}_mad.npz")
        scalers = _load_scaler(cache_path, device) if use_mad_cache else None
        if scalers is None:
            med_imu, mad_imu = _mad_from_rows(imu_rows, scale_to_std=mad_scale_to_std, sample_rows=mad_sample_rows)
            med_dlc, mad_dlc = _mad_from_rows(dlc_rows, scale_to_std=mad_scale_to_std, sample_rows=mad_sample_rows)
            if use_mad_cache:
                _save_scaler(cache_path, med_imu, mad_imu, med_dlc, mad_dlc)
        else:
            med_imu, mad_imu, med_dlc, mad_dlc = scalers

        imu_rows = _apply_mad_norm(imu_rows, med_imu, mad_imu)
        dlc_rows = _apply_mad_norm(dlc_rows, med_dlc, mad_dlc)

        # ---- 按 T 分块裁剪写入，避免一次性占满内存 ----
        for T in count_T.keys():
            grp_indices = [grp for grp, t_ in zip(groups, Ts) if t_ == T]
            if not grp_indices:
                continue
            n_batches = len(grp_indices)

            # shard 规模：一次处理这么多个 batch
            shard = max(1, min(preproc_shard_batches, n_batches))
            write_ptr = 0
            while write_ptr < n_batches:
                j0 = write_ptr
                j1 = min(n_batches, write_ptr + shard)
                cur_groups = grp_indices[j0:j1]  # 切一小段
                B_total = len(cur_groups) * batch_size

                flat_indices = [i for g in cur_groups for i in g]
                flat_centres = [samples[i][2] for i in flat_indices]  # 每个样本的中心点
                starts = torch.tensor([c - (T // 2) for c in flat_centres], device=device, dtype=torch.long)
                labels = torch.stack([samples[i][1] for i in flat_indices], dim=0).to(device).float()

                imu_win, mask = _batch_crop_rows(imu_rows, starts, T)
                dlc_win, _    = _batch_crop_rows(dlc_rows, starts, T)

                # reshape 成 (num_shard_batches, batch_size, T, F)
                cur_nb = len(cur_groups)
                imu_win = imu_win.view(cur_nb, batch_size, T, feat_imu)
                dlc_win = dlc_win.view(cur_nb, batch_size, T, feat_dlc)
                mask    = mask.view(cur_nb, batch_size, T)
                labels  = labels.view(cur_nb, batch_size, n_labels)

                # 写盘（降精度/类型转换）
                dsets[T]["imu"][j0:j1, :, :, :]   = imu_win.detach().cpu().to(torch.float16 if store_dtype=="float16" else torch.float32).numpy()
                dsets[T]["dlc"][j0:j1, :, :, :]   = dlc_win.detach().cpu().to(torch.float16 if store_dtype=="float16" else torch.float32).numpy()
                dsets[T]["mask"][j0:j1, :, :]     = mask.detach().cpu().to(torch.uint8 if mask_dtype=="uint8" else torch.bool).numpy()
                dsets[T]["label"][j0:j1, :, :]    = labels.detach().cpu().to(torch.uint8 if label_dtype=="uint8" else torch.float32).numpy()

                write_ptr = j1


def preprocess_dataset(
    dataset: RatsWindowDataset,
    batch_size: int,
    out_dir: str = "Dataset",
    *,
    group_mode: str = "by_session",     # "by_session" / "global"
    assign_T: str = "round_robin",      # "round_robin" / "random"
    seed: int = 42,
    device: str | None = None,          # "cuda" / "cpu"
    compress: bool = False,
    num_workers: int = 0,               # 按 session 并行
    mad_scale_to_std: bool = True,
    mad_cache_dir: str = "Dataset/scalers",
    use_mad_cache: bool = True,
    mad_sample_rows: Optional[int] = 200_000,   # None=全量精确；数值越小越快
    store_dtype: str = "float16",              # "float16" / "float32"
    label_dtype: str = "uint8",                # "uint8" / "float32"
    mask_dtype: str = "uint8",    # "uint8" / "bool"
    window_sizes: Sequence[int] = (16, 32, 64, 128, 256, 512),  # <== 新增
    preproc_shard_batches: int = 8,            # 每次处理/写入的 batch 数
    use_unlabeled: bool = False,
) -> None:
    """
    MAD 归一化（支持缓存/抽样） + 分块向量化裁剪写入（低峰值内存） + 按 session 并行
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU 下避免多线程争用同一块卡
    if str(device).startswith("cuda") and num_workers and num_workers > 1:
        num_workers = 1

    # 选择使用的样本集合
    samples = dataset.unsup_samples if use_unlabeled else dataset.samples

    # 收集每个 session 的样本索引
    session_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, (sess, _, _) in enumerate(samples):
        session_to_indices[sess].append(idx)

    # 组装 batch（不跨 session）
    groups_per_session: Dict[str, List[List[int]]] = {}
    if group_mode == "global":
        per_sess_queues = {s: session_to_indices[s][:] for s in session_to_indices}
        for s in per_sess_queues:
            rng.shuffle(per_sess_queues[s])
        tmp = defaultdict(list)
        made_any = True
        while made_any:
            made_any = False
            for s, q in per_sess_queues.items():
                if len(q) >= batch_size:
                    tmp[s].append([q.pop() for _ in range(batch_size)])
                    made_any = True
        groups_per_session = dict(tmp)
    else:
        for s, idxs in session_to_indices.items():
            rng.shuffle(idxs)
            groups_per_session[s] = _split_into_groups(idxs, batch_size)

    # 为每个 session 分配每组的 T
    Ts_per_session: Dict[str, List[int]] = {}
    for s, groups in groups_per_session.items():
        if assign_T == "round_robin":
            Ts_per_session[s] = [window_sizes[i % len(window_sizes)] for i in range(len(groups))]
        else:
            Ts_per_session[s] = [rng.choice(window_sizes) for _ in range(len(groups))]

    # 并行/串行执行
    sessions = list(groups_per_session.keys())
    if num_workers and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = []
            for s in sessions:
                futures.append(
                    ex.submit(
                        _build_session_file,
                        s,
                        groups_per_session[s],
                        Ts_per_session[s],
                        dataset,
                        samples,
                        out_dir,
                        device,
                        compress,
                        mad_scale_to_std,
                        mad_cache_dir,
                        use_mad_cache,
                        mad_sample_rows,
                        store_dtype,
                        label_dtype,
                        mask_dtype,
                        preproc_shard_batches,
                    )
                )
            for _ in as_completed(futures):
                pass
    else:
        for s in sessions:
            _build_session_file(
                s,
                groups_per_session[s],
                Ts_per_session[s],
                dataset,
                samples,
                out_dir,
                device,
                compress,
                mad_scale_to_std,
                mad_cache_dir,
                use_mad_cache,
                mad_sample_rows,
                store_dtype,
                label_dtype,
                mask_dtype,
                preproc_shard_batches,
            )


def load_preprocessed_batches(
    sessions: Sequence[str],
    session_to_idx: Dict[str, int],
    out_dir: str = "Dataset",
    mix: bool = False,
):
    """
    从预处理好的 HDF5 依次产出 batch。
    读取时把半精度/uint8 转回训练常用 dtype。
    """
    batches = []
    for session in sessions:
        path = os.path.join(out_dir, f"{session}.h5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for grp in f.values():
                imu = np.asarray(grp["imu"], dtype=np.float16)     # 存的是 fp16
                dlc = np.asarray(grp["dlc"], dtype=np.float16)
                mask = np.asarray(grp["mask"])                     # uint8/bool
                label = np.asarray(grp["label"])                   # uint8/float32
                num_batches = imu.shape[0]
                for i in range(num_batches):
                    batches.append((session, imu[i], dlc[i], mask[i], label[i]))
    if mix:
        random.shuffle(batches)
    for session, imu, dlc, mask, label in batches:
        # 训练时统一转回 float32 & bool
        yield {
            "imu": torch.from_numpy(imu.astype(np.float32)),
            "dlc": torch.from_numpy(dlc.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(bool)),
            "label": torch.from_numpy(label.astype(np.float32)),
            "session": session,
            "session_idx": torch.full((imu.shape[0],), session_to_idx[session], dtype=torch.long),
        }
