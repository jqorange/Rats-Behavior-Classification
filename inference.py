import os
import argparse
from typing import Sequence, List, Optional

import torch
import pandas as pd
import numpy as np

from models.fusion import EncoderFusion
from utils.checkpoint import load_checkpoint


@torch.no_grad()
def latest_checkpoint(ckpt_dir: str) -> str:
    """Return the most recently modified checkpoint file in a directory."""
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return max(files, key=os.path.getmtime)


def _load_session_arrays(root: str, session: str):
    """Load IMU/DLC arrays and labels for one session."""
    imu_file = os.path.join(root, 'IMU', session, f'{session}_IMU_features.csv')
    dlc_file = os.path.join(root, 'DLC', session, f'final_filtered_{session}_50hz.csv')
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


def _collect_centers_labeled(label_df: pd.DataFrame) -> List[int]:
    label_df = label_df[label_df.drop(columns=['Index']).any(axis=1)]
    label_df = label_df.sort_values('Index')
    indices = label_df['Index'].to_numpy(dtype=int)
    n = len(indices)
    start = int(n * 0.8)
    return indices[start:].tolist()


@torch.no_grad()
def run_inference(
    ckpt_path: str,
    data_path: str,
    sessions: Sequence[str],
    *,
    mode: str = 'full',          # 'full' or 'labeled'
    window: str = '64',          # '64' or 'multi'
    index: Optional[int] = None,
    device: Optional[str] = None,
    out_dir: str = 'representations',
    batch_size: int = 128,
    stride: int = 1,             # <== 新增：滑动步长，默认逐帧
) -> None:
    """Generate representations for sessions using a saved checkpoint."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    # Load first session to infer feature dims
    imu0, dlc0, _ = _load_session_arrays(data_path, sessions[0])
    num_imu, num_dlc = imu0.shape[1], dlc0.shape[1]

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state', {})
    d_model = state['encoderA.adapter.linear.weight'].shape[0]
    num_sessions = state['encoderA.adapter.session_embed.weight'].shape[0]

    model = EncoderFusion(
        N_feat_A=num_imu,
        N_feat_B=num_dlc,
        mask_type=None,
        d_model=d_model,
        nhead=4,
        num_sessions=num_sessions,
    ).to(device)
    model.eval()

    _, stage = load_checkpoint(model, model.projection, optimizer=None, path=ckpt_path)
    if stage == 1:
        model.projection.set_mode('aware')
        if index is None:
            raise ValueError('Stage1 checkpoint requires --index for projector embedding')
    else:
        model.projection.set_mode('align')
        if index is None:
            index = 0

    if window == 'multi':
        Ts = [16, 32, 64, 128, 256, 512]
        base_stride = stride
    else:
        Ts = [64]
        base_stride = stride

    for sess in sessions:
        imu, dlc, label_df = _load_session_arrays(data_path, sess)
        length = min(len(imu), len(dlc))
        if mode == 'labeled':
            if label_df is None:
                raise FileNotFoundError(f'Label file missing for session {sess}')
            centres = _collect_centers_labeled(label_df)
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
                imu_win_b = _extract_windows(imu, centres_b, T).to(device, non_blocking=True)
                dlc_win_b = _extract_windows(dlc, centres_b, T).to(device, non_blocking=True)
                sess_idx_b = torch.full((len(centres_b),), index, dtype=torch.long, device=device)

                feat_b, *_ = model(imu_win_b, dlc_win_b, session_idx=sess_idx_b)
                # 取中心帧特征
                feat_b = feat_b[:, T // 2].detach().cpu()
                feats_T_batches.append(feat_b)

                # 及时释放显存
                del imu_win_b, dlc_win_b, sess_idx_b
                torch.cuda.empty_cache() if device.startswith('cuda') else None

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
    # 修正 required 误用；设置一个默认路径，仍可通过 CLI 覆盖
    p.add_argument('--weights', default=r"D:\Jiaqi\Projects\Rats-Behavior-Classification\checkpoints\stage1_epoch13.pt",
                   help='Checkpoint file path')
    p.add_argument('--data_path', default=r"D:\Jiaqi\Datasets\Rats\TrainData_new", help='Dataset root directory')
    p.add_argument('--sessions', nargs='+',
                   default=["F3D5_outdoor"],
                   help='Session names')
    p.add_argument('--mode', choices=['full', 'labeled'], default='full')
    p.add_argument('--window', choices=['64', 'multi'], default='64')
    p.add_argument('--index', type=int, default=1, help='Projector index for stage1 model')
    p.add_argument('--device', default="cuda")
    p.add_argument('--out_dir', default='representations')
    p.add_argument('--batch_size', type=int, default=1024, help='Inference batch size')  # <== 新增
    p.add_argument('--stride', type=int, default=1, help='Sliding stride for centers (1 = per frame)')
    args = p.parse_args()

    run_inference(
        args.weights,
        args.data_path,
        args.sessions,
        mode=args.mode,
        window=args.window,
        index=args.index,
        device=args.device,
        out_dir=args.out_dir,
        batch_size=args.batch_size,   # <== 传入
        stride=args.stride,  # <== 新增
    )


if __name__ == '__main__':
    main()
