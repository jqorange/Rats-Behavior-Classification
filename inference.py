import os
import argparse
from typing import Dict, Sequence, List, Optional

import torch
import pandas as pd
import numpy as np

from models.single_modal import SingleModalModel
from utils.checkpoint import load_checkpoint
from utils.segments import (
    SegmentInfo,
    collect_segment_centres,
    compute_segments,
    extract_label_arrays,
    split_segments_by_action,
)


def _infer_model_config(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Infer ``d_model`` and ``num_sessions`` from a checkpoint state dict.

    Older checkpoints stored parameters for session-aware adapters while the
    latest training code uses a simpler encoder without those layers.  The
    inference script therefore needs to be able to recover the model
    dimensions from whichever keys are available.
    """

    def _first_tensor(keys):
        for key in keys:
            tensor = state.get(key)
            if tensor is not None:
                return tensor
        return None

    d_model_src = _first_tensor(
        [
            "encoder.input_proj.weight",
            "encoder.norm_in.weight",
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


@torch.no_grad()
def latest_checkpoint(ckpt_dir: str) -> str:
    """Return the most recently modified checkpoint file in a directory."""
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return max(files, key=os.path.getmtime)


def _load_session_arrays(root: str, session: str):
    """Load IMU/DLC arrays and labels for one session."""
    imu_file = os.path.join(root, 'IMU', session, f'{session}_IMU_features_madnorm.csv')
    dlc_file = os.path.join(root, 'DLC', session, f'final_filtered_{session}_50hz_madnorm.csv')
    label_file = os.path.join(root, 'labels', session, f'label_{session}.csv')

    imu_df = pd.read_csv(imu_file)
    dlc_df = pd.read_csv(dlc_file)
    min_len = min(len(imu_df), len(dlc_df))
    imu = torch.from_numpy(imu_df.iloc[:min_len].to_numpy(dtype=np.float32))
    dlc = torch.from_numpy(dlc_df.iloc[:min_len].to_numpy(dtype=np.float32))

    if os.path.exists(label_file):
        label_df = pd.read_csv(label_file)
    else:
        label_df = None
    return imu, dlc, label_df


def _extract_windows(arr: torch.Tensor, centers: Sequence[int], T: int) -> torch.Tensor:
    """Vectorised window extraction with boundary clamping."""
    L, F = arr.shape
    device = arr.device
    centres = torch.tensor(list(centers), device=device, dtype=torch.long)
    starts = centres - T // 2
    ar = torch.arange(T, device=device)
    idx = starts.unsqueeze(1) + ar.view(1, -1)
    idx = idx.clamp(0, L - 1)
    win = arr.index_select(0, idx.reshape(-1)).reshape(len(centres), T, F)
    return win


def _collect_centers_full(length: int, stride: int = 1) -> List[int]:
    """逐帧/按步长收集中心索引；stride=1 时输出长度 == 原始帧数。"""
    return list(range(0, length, stride))


def _load_label_dataframe(root: str, session: str) -> Optional[pd.DataFrame]:
    label_file = os.path.join(root, 'labels', session, f'label_{session}.csv')
    if not os.path.exists(label_file):
        return None
    return pd.read_csv(label_file)


@torch.no_grad()
def run_inference(
    ckpt_path: str,
    data_path: str,
    sessions: Sequence[str],
    *,
    mode: str = 'full',          # 'full' or 'labeled'
    window: str = '64',          # '64' or 'multi'
    device: Optional[str] = None,
    out_dir: str = 'representations',
    batch_size: int = 128,
    stride: int = 1,             # 滑动步长，默认逐帧
    modality: str = 'imu',
    segment_split: str = 'test',
    segment_seed: int = 0,
    segment_test_ratio: float = 0.2,
) -> None:
    """Generate representations for sessions using a saved checkpoint."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    # Load first session to infer feature dims
    imu0, dlc0, _ = _load_session_arrays(data_path, sessions[0])
    modality = modality.lower()
    if modality not in {'imu', 'dlc'}:
        raise ValueError("modality must be 'imu' or 'dlc'")
    num_features = imu0.shape[1] if modality == 'imu' else dlc0.shape[1]

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state', {})
    d_model, _ = _infer_model_config(state)

    model = SingleModalModel(
        num_features,
        mask_type=None,
        d_model=d_model,
    ).to(device)
    model.eval()

    _, _stage = load_checkpoint(model, model.projection, optimizer=None, path=ckpt_path)

    if window == 'multi':
        Ts = [16, 32, 64, 128]
        base_stride = stride
    else:
        Ts = [64]
        base_stride = stride

    segments_by_session: Dict[str, Dict[str, List[SegmentInfo]]] = {}
    labelled_indices: Dict[str, List[int]] = {}
    if mode == 'labeled':
        segment_split = segment_split.lower()
        if segment_split not in {'train', 'test'}:
            raise ValueError("segment_split must be either 'train' or 'test'")

        label_columns_ref: Optional[List[str]] = None
        for sess in sessions:
            label_df = _load_label_dataframe(data_path, sess)
            if label_df is None:
                raise FileNotFoundError(f'Label file missing for session {sess}')
            indices, labels_arr, label_cols = extract_label_arrays(
                label_df, label_columns=label_columns_ref
            )
            if label_columns_ref is None and label_cols:
                label_columns_ref = label_cols
            elif label_columns_ref is not None and label_cols and label_cols != label_columns_ref:
                raise ValueError('Label columns mismatch across sessions')

            columns_for_session = label_cols if label_cols else (label_columns_ref or [])
            segs = compute_segments(indices, labels_arr, columns_for_session)
            segments_by_session[sess] = segs

        assignments = split_segments_by_action(
            segments_by_session, test_ratio=segment_test_ratio, seed=segment_seed
        )
        labelled_indices = collect_segment_centres(
            segments_by_session, assignments, segment_split, include='all'
        )

    for sess in sessions:
        imu, dlc, label_df = _load_session_arrays(data_path, sess)
        source = imu if modality == 'imu' else dlc
        length = len(source)
        if mode == 'labeled':
            centres = [c for c in labelled_indices.get(sess, []) if c < length]
        else:
            centres = _collect_centers_full(length, stride=base_stride)

        if len(centres) == 0:
            print(f'[Warn] No centres for session {sess}, skip.')
            continue

        features_all = []

        # 对每个 T，按 batch 抽窗并前向
        for T in Ts:
            feats_T_batches = []
            for i in range(0, len(centres), batch_size):
                centres_b = centres[i:i + batch_size]

                # 抽窗在 CPU 上完成，随后把这一小批移到 device
                win_b = _extract_windows(source, centres_b, T).to(device, non_blocking=True)

                output = model(win_b)
                feat_seq_b = output.embedding

                # ==== 关键修改：时间维做 max pooling 得到 (B, D) ====
                # 原来是取中心帧：feat_b = feat_seq_b[:, T // 2]
                # 现在改为：沿时间维度做最大池化
                feat_b = torch.amax(feat_seq_b, dim=1).detach().cpu()
                # ===============================================

                feats_T_batches.append(feat_b)

                # 及时释放显存
                del win_b, output, feat_seq_b, feat_b
                if isinstance(device, str) and device.startswith('cuda'):
                    torch.cuda.empty_cache()

            # 拼回完整序列在该 T 下的特征
            feats_T = torch.cat(feats_T_batches, dim=0)  # (N, d_model)
            features_all.append(feats_T)

        # 多尺度拼接为 (N, sum(d_model per T))；如果每个 T 维度相同则是 (N, len(Ts)*d_model)
        features = torch.cat(features_all, dim=-1)
        out_path = os.path.join(out_dir, f'{sess}.pt')
        torch.save({'features': features, 'centers': centres}, out_path)
        print(f'Saved {out_path} with shape {tuple(features.shape)}')


def main() -> None:
    p = argparse.ArgumentParser(description='Run encoder inference on sessions.')
    p.add_argument('--weights', default=r"D:\Jiaqi\Projects\Rats-Behavior-Classification\checkpoints\stage2_epoch199.pt",
                   help='Checkpoint file path')
    p.add_argument('--data_path', default=r"D:\Jiaqi\Datasets\Rats\TrainData_new", help='Dataset root directory')
    p.add_argument('--sessions', nargs='+',
                   default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_1", "F6D5_outdoor_2"],
                   help='Session names')
    p.add_argument('--mode', choices=['full', 'labeled'], default='full')
    p.add_argument('--window', choices=['64', 'multi'], default='multi')
    p.add_argument('--modality', choices=['imu', 'dlc'], default='imu', help='Modality to encode')
    p.add_argument('--device', default="cuda")
    p.add_argument('--out_dir', default='representations')
    p.add_argument('--batch_size', type=int, default=1024, help='Inference batch size')
    p.add_argument('--stride', type=int, default=1, help='Sliding stride for centers (1 = per frame)')
    p.add_argument('--segment-split', choices=['train', 'test'], default='test',
                   help='Use training or testing segments when mode="labeled"')
    p.add_argument('--split-seed', type=int, default=0, help='Random seed for segment splitting')
    p.add_argument('--segment-test-ratio', type=float, default=0.2,
                   help='Test ratio for segment-level splitting')
    args = p.parse_args()

    run_inference(
        args.weights,
        args.data_path,
        args.sessions,
        mode=args.mode,
        window=args.window,
        device=args.device,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        stride=args.stride,
        modality=args.modality,
        segment_split=args.segment_split,
        segment_seed=args.split_seed,
        segment_test_ratio=args.segment_test_ratio,
    )


if __name__ == '__main__':
    main()
