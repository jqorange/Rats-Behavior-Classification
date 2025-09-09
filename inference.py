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


def _collect_centers_full(length: int, stride: int = 64) -> List[int]:
    half = stride // 2
    centres = list(range(half, length - half, stride))
    if centres and centres[-1] + half < length:
        centres.append(length - half - 1)
    return centres


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
    mode: str = 'full',  # 'full' or 'labeled'
    window: str = '64',  # '64' or 'multi'
    index: Optional[int] = None,
    device: Optional[str] = None,
    out_dir: str = 'representations',
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
        base_stride = 64
    else:
        Ts = [64]
        base_stride = 64

    for sess in sessions:
        imu, dlc, label_df = _load_session_arrays(data_path, sess)
        length = min(len(imu), len(dlc))
        if mode == 'labeled':
            if label_df is None:
                raise FileNotFoundError(f'Label file missing for session {sess}')
            centres = _collect_centers_labeled(label_df)
        else:
            centres = _collect_centers_full(length, stride=base_stride)

        features_all = []
        for T in Ts:
            imu_win = _extract_windows(imu, centres, T).to(device)
            dlc_win = _extract_windows(dlc, centres, T).to(device)
            sess_idx = torch.full((len(centres),), index, dtype=torch.long, device=device)
            feat, *_ = model(imu_win, dlc_win, session_idx=sess_idx)
            feat = feat[:, T // 2]
            features_all.append(feat.cpu())

        features = torch.cat(features_all, dim=-1)
        out_path = os.path.join(out_dir, f'{sess}.pt')
        torch.save({'features': features, 'centers': centres}, out_path)
        print(f'Saved {out_path} with shape {tuple(features.shape)}')


def main() -> None:
    p = argparse.ArgumentParser(description='Run encoder inference on sessions.')
    p.add_argument('--weights', required=True, help='Checkpoint file path')
    p.add_argument('--data_path', required=True, help='Dataset root directory')
    p.add_argument('--sessions', nargs='+', required=True, help='Session names')
    p.add_argument('--mode', choices=['full', 'labeled'], default='full')
    p.add_argument('--window', choices=['64', 'multi'], default='64')
    p.add_argument('--index', type=int, default=None, help='Projector index for stage1 model')
    p.add_argument('--device', default=None)
    p.add_argument('--out_dir', default='representations')
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
    )


if __name__ == '__main__':
    main()
