# train_three_stage.py
"""Simplified three-stage training script using the new window dataset.

此脚本展示三阶段训练流程：Stage1(无监督/单会话批)、Stage2(冻结编码器+无监督→有监督)、
Stage3(联训)。预处理阶段新增“按 index 分组、每组统一窗口长度 T”，并支持多线程。
"""

from __future__ import annotations
import os
import argparse
from typing import Sequence, Dict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.window_dataset import RatsWindowDataset
from utils.preprocess import preprocess_dataset, load_preprocessed_batches
from models.fusion import EncoderFusion
from models.losses import (
    hierarchical_contrastive_loss,
    multilabel_supcon_loss_bt,
    positive_only_supcon_loss,
    gaussian_cs_divergence,
)
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.tools import take_per_row


class ThreeStageTrainer:
    def __init__(self, num_features_imu: int, num_features_dlc: int, num_sessions: int, device: str | None = None, *, stage_lrs: Dict[int, float] | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_sessions = num_sessions
        self.d_model = 64
        self.model = EncoderFusion(
            N_feat_A=num_features_imu,
            N_feat_B=num_features_dlc,
            mask_type="binomial",  # 如不需要特征丢弃，可直接改为 None
            d_model=self.d_model,
            nhead=4,
            num_sessions=num_sessions,
            # 如果你的 EncoderFusion/encoder 支持传入概率参数，建议同时传：
            # drop_prob=0.1,
        ).to(self.device)
        self.stage_lrs = {1: 1e-3, 2: 1e-3, 3: 1e-3}
        if stage_lrs:
            self.stage_lrs.update(stage_lrs)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.stage_lrs[1])
        self.total_epochs = 0

        # ✅ 新 API
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device == "cuda"))
        self.temporal_unit = 3

    def set_stage_lr(self, stage: int, lr: float) -> None:
        self.stage_lrs[stage] = lr

    def _sanitize_inplace(self, t: torch.Tensor) -> torch.Tensor:
        # 把 NaN/Inf 变成 0，防止后续层炸掉
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _step_unsup(self, batch: dict) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        imu = self._sanitize_inplace(batch['imu'].to(self.device))
        dlc = self._sanitize_inplace(batch['dlc'].to(self.device))
        session_idx = batch['session_idx'].to(self.device, dtype=torch.long)

        B, T, _ = imu.shape
        if T <= 5:
            zero = torch.tensor(0.0, device=imu.device)
            return zero, {
                'loss_contrast': zero.detach(),
                'loss_cs': zero.detach(),
                'loss_l2': zero.detach(),
            }

        min_crop = 2 ** (self.temporal_unit + 1)
        try:
            crop_l = np.random.randint(low=min_crop, high=T + 1)
            crop_left = np.random.randint(T - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=T + 1)
            crop_offset = torch.randint(
                low=-crop_eleft,
                high=T - crop_eright + 1,
                size=(B,),
                device=imu.device,
            )
        except Exception as e:
            print(f'[ERROR] Crop param generation failed: T={T}, temporal_unit={self.temporal_unit}')
            raise e

        for i in range(B):
            start1 = crop_offset[i].item() + crop_eleft
            end1 = start1 + (crop_right - crop_eleft)
            start2 = crop_offset[i].item() + crop_left
            end2 = start2 + (crop_eright - crop_left)
            if start1 < 0 or end1 > T or start2 < 0 or end2 > T:
                print(f'❌ Invalid crop range! B={i} | T={T}')
                print(f'→ crop_offset={crop_offset[i].item()}, eleft={crop_eleft}, right={crop_right}')
                print(f'→ crop1 range: {start1}:{end1}, crop2 range: {start2}:{end2}')
                raise ValueError('Invalid crop range detected.')

        imu_crop1 = take_per_row(imu, crop_offset + crop_eleft, crop_right - crop_eleft)
        dlc_crop1 = take_per_row(dlc, crop_offset + crop_eleft, crop_right - crop_eleft)
        imu_crop2 = take_per_row(imu, crop_offset + crop_left, crop_eright - crop_left)
        dlc_crop2 = take_per_row(dlc, crop_offset + crop_left, crop_eright - crop_left)

        for name, crop in zip(
            ['imu_crop1', 'dlc_crop1', 'imu_crop2', 'dlc_crop2'],
            [imu_crop1, dlc_crop1, imu_crop2, dlc_crop2],
        ):
            if torch.isnan(crop).any():
                print(f'❌ NaN detected in {name}')
                print(f'→ Shape: {crop.shape}, Mean: {crop.mean().item()}, Std: {crop.std().item()}')
                raise ValueError(f'NaN detected in {name}')

        with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
            emb1, A_self, B_self, A_to_B, B_to_A = self.model(imu_crop1, dlc_crop1, session_idx=session_idx)
            emb1 = emb1[:, -crop_l:]
            A_self = A_self[:, -crop_l:]
            B_self = B_self[:, -crop_l:]
            A_to_B = A_to_B[:, -crop_l:]
            B_to_A = B_to_A[:, -crop_l:]
            emb2, *_ = self.model(imu_crop2, dlc_crop2, session_idx=session_idx)
            emb2 = emb2[:, :crop_l]

            jitter_std = 0.01
            emb1 = emb1 + torch.randn_like(emb1) * jitter_std
            emb2 = emb2 + torch.randn_like(emb2) * jitter_std
            emb1 = F.normalize(emb1, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)

            loss_contrast = hierarchical_contrastive_loss(emb1, emb2, temporal_unit=self.temporal_unit)
            loss_cs = (
                gaussian_cs_divergence(A_to_B, B_self.detach())
                + gaussian_cs_divergence(B_to_A, A_self.detach())
            )
            loss_l2 = (
                torch.norm(A_to_B - B_self.detach(), dim=-1).mean()
                + torch.norm(B_to_A - A_self.detach(), dim=-1).mean()
            )
            loss = loss_contrast + loss_cs + loss_l2
            if loss_contrast.item() > 10:
                print(f'⚠️ [WARNING] Contrastive loss unusually high: {loss_contrast.item():.4f}')

        return loss, {
            'loss_contrast': loss_contrast.detach(),
            'loss_cs': loss_cs.detach(),
            'loss_l2': loss_l2.detach(),
        }

    def _step_align(self, batch: dict) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """仅计算对齐损失（loss_cs + loss_l2），不需要标签。"""
        imu = self._sanitize_inplace(batch["imu"].to(self.device))
        dlc = self._sanitize_inplace(batch["dlc"].to(self.device))
        session_idx = batch["session_idx"].to(self.device)

        scale = torch.empty(imu.size(0), 1, 1, device=imu.device).uniform_(0.8, 1.2)
        imu = imu * scale
        dlc = dlc * scale

        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            emb, A_self, B_self, A_to_B, B_to_A = self.model(imu, dlc, session_idx=session_idx)
            jitter_std = 0.01
            emb = emb + torch.randn_like(emb) * jitter_std
            loss_cs = (
                gaussian_cs_divergence(A_to_B, B_self.detach())
                + gaussian_cs_divergence(B_to_A, A_self.detach())
            )
            loss_l2 = (
                torch.norm(A_to_B - B_self.detach(), dim=-1).mean()
                + torch.norm(B_to_A - A_self.detach(), dim=-1).mean()
            )
            loss = loss_cs + loss_l2
        return loss, {
            "loss_cs": loss_cs.detach(),
            "loss_l2": loss_l2.detach(),
        }

    def _step_sup_stage2(self, batch: dict) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        imu = self._sanitize_inplace(batch["imu"].to(self.device))
        dlc = self._sanitize_inplace(batch["dlc"].to(self.device))
        labels = batch["label"].to(self.device)
        session_idx = batch["session_idx"].to(self.device)

        scale = torch.empty(imu.size(0), 1, 1, device=imu.device).uniform_(0.8, 1.2)
        imu = imu * scale
        dlc = dlc * scale

        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            emb, A_self, B_self, A_to_B, B_to_A = self.model(imu, dlc, session_idx=session_idx)
            jitter_std = 0.01
            emb = emb + torch.randn_like(emb) * jitter_std
            loss_sup = positive_only_supcon_loss(emb, labels)
            loss_cs = (
                gaussian_cs_divergence(A_to_B, B_self.detach())
                + gaussian_cs_divergence(B_to_A, A_self.detach())
            )
            loss_l2 = (
                torch.norm(A_to_B - B_self.detach(), dim=-1).mean()
                + torch.norm(B_to_A - A_self.detach(), dim=-1).mean()
            )
            loss = loss_sup + loss_cs + loss_l2
        return loss, {
            "loss_sup": loss_sup.detach(),
            "loss_cs": loss_cs.detach(),
            "loss_l2": loss_l2.detach(),
        }

    def _step_sup(self, batch: dict) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        imu = self._sanitize_inplace(batch["imu"].to(self.device))
        dlc = self._sanitize_inplace(batch["dlc"].to(self.device))
        labels = batch["label"].to(self.device)
        session_idx = batch["session_idx"].to(self.device)

        scale = torch.empty(imu.size(0), 1, 1, device=imu.device).uniform_(0.8, 1.2)
        imu = imu * scale
        dlc = dlc * scale

        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            emb, A_self, B_self, A_to_B, B_to_A = self.model(imu, dlc, session_idx=session_idx)
            jitter_std = 0.01
            emb = emb + torch.randn_like(emb) * jitter_std
            loss_sup = multilabel_supcon_loss_bt(emb, labels)
            loss_cs = (
                gaussian_cs_divergence(A_to_B, B_self.detach())
                + gaussian_cs_divergence(B_to_A, A_self.detach())
            )
            loss_l2 = (
                torch.norm(A_to_B - B_self.detach(), dim=-1).mean()
                + torch.norm(B_to_A - A_self.detach(), dim=-1).mean()
            )
            loss = loss_sup + loss_cs + loss_l2
        return loss, {
            "loss_sup": loss_sup.detach(),
            "loss_cs": loss_cs.detach(),
            "loss_l2": loss_l2.detach(),
        }

    def _run_epoch(self, iterator, step_fn):
        self.model.train()
        batches = list(iterator)
        loss_sums: Dict[str, float] = {}
        for batch in tqdm(batches, total=len(batches)):
            self.opt.zero_grad(set_to_none=True)
            loss, loss_dict = step_fn(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            loss_sums["total"] = loss_sums.get("total", 0.0) + loss.item()
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
        for k in loss_sums:
            loss_sums[k] /= max(len(batches), 1)
        return loss_sums

    def load_from(self, path: str, expected_stage: int) -> int:
        """加载 checkpoint 并返回其中记录的阶段。"""
        self.total_epochs, stage = load_checkpoint(
            self.model, self.model.projection, self.opt, path, expected_stage=expected_stage
        )
        return stage

    def stage1(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1, *, n_workers_preproc: int = 0, save_gap=1) -> None:
        """无监督学习：每个 batch 内不混 session（by_session），每组统一 T，
        包含对比学习与对齐损失（loss_cs + loss_l2）。"""
        for p in self.model.parameters():
            p.requires_grad = True
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.stage_lrs[1])
        for ep in range(epochs):
            preprocess_dataset(
                dataset, batch_size, out_dir="Dataset_unsup",
                group_mode="by_session",
                assign_T="round_robin",       # 各 T 均衡
                num_workers=n_workers_preproc,
                seed=42 + ep,                 # 每个 epoch 改变 seed，获得新的切窗
                use_unlabeled=True,
            )
            it = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, out_dir="Dataset_unsup", mix=False)
            losses = self._run_epoch(it, self._step_unsup)
            self.total_epochs += 1
            print(f"[Stage1][Epoch {self.total_epochs}] " + " ".join(f"{k}={v:.4f}" for k, v in losses.items()))
            if self.total_epochs % save_gap == 0:
                save_checkpoint(
                    self.model,
                    self.model.projection,
                    self.opt,
                    total_epochs=self.total_epochs,
                    stage=1,
                    path=os.path.join("checkpoints", f"stage1_epoch{self.total_epochs}.pt"),
                )

    def stage2(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1, *, n_workers_preproc: int = 0, save_gap=1) -> None:
        """冻结除 cross head 外的所有模块，只训练小 GRU 对齐。"""
        # Freeze all parameters first
        for p in self.model.parameters():
            p.requires_grad = False

        # Only train the cross-modal GRU heads (encoderA/encoderB.head_cross + norm)
        trainable_params = []
        for enc in (self.model.encoderA, self.model.encoderB):
            for module in (enc.head_cross, enc.norm_cross):
                for p in module.parameters():
                    p.requires_grad = True
                    trainable_params.append(p)

        self.model.projection.set_mode("align")
        self.opt = torch.optim.Adam(trainable_params, lr=self.stage_lrs[2])

        for ep in range(epochs):
            preprocess_dataset(
                dataset, batch_size, out_dir="Dataset_unsup",
                group_mode="by_session",
                assign_T="round_robin",
                num_workers=n_workers_preproc,
                seed=1234 + ep,
                use_unlabeled=True,
            )
            it_align = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, out_dir="Dataset_unsup", mix=False)
            losses_align = self._run_epoch(it_align, self._step_align)

            preprocess_dataset(
                dataset, batch_size, out_dir="Dataset_sup",
                group_mode="by_session",
                assign_T="round_robin",
                num_workers=n_workers_preproc,
                seed=1234 + ep,
                use_unlabeled=False,
            )
            it_sup = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, out_dir="Dataset_sup", mix=True)
            losses_sup = self._run_epoch(it_sup, self._step_sup_stage2)
            combined = {f"align_{k}": v for k, v in losses_align.items()}
            combined.update({f"sup_{k}": v for k, v in losses_sup.items()})
            total = combined.get("align_total", 0.0) + combined.get("sup_total", 0.0)
            self.total_epochs += 1
            print(f"[Stage2][Epoch {self.total_epochs}] total={total:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in combined.items()))
            if self.total_epochs % save_gap == 0:
                save_checkpoint(
                    self.model,
                    self.model.projection,
                    self.opt,
                    total_epochs=self.total_epochs,
                    stage=2,
                    path=os.path.join("checkpoints", f"stage2_epoch{self.total_epochs}.pt"),
                )

    def stage3(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1, *, n_workers_preproc: int = 0, save_gap=1) -> None:
        """联训：对齐损失 + 有监督都在混 session 的 batch 上进行。"""
        for p in self.model.parameters():
            p.requires_grad = True
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.stage_lrs[3])

        for ep in range(epochs):
            preprocess_dataset(
                dataset, batch_size, out_dir="Dataset_unsup",
                group_mode="by_session",
                assign_T="round_robin",
                device="cuda",
                num_workers=n_workers_preproc,
                seed=5678 + ep,
                use_unlabeled=True,
            )
            it_align = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, out_dir="Dataset_unsup", mix=True)
            losses_align = self._run_epoch(it_align, self._step_align)

            preprocess_dataset(
                dataset, batch_size, out_dir="Dataset_sup",
                group_mode="by_session",
                assign_T="round_robin",
                device="cuda",
                num_workers=n_workers_preproc,
                seed=5678 + ep,
                use_unlabeled=False,
            )
            it_sup = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, out_dir="Dataset_sup", mix=True)
            losses_sup = self._run_epoch(it_sup, self._step_sup)

            combined = {f"align_{k}": v for k, v in losses_align.items()}
            combined.update({f"sup_{k}": v for k, v in losses_sup.items()})
            total = combined.get("align_total", 0.0) + combined.get("sup_total", 0.0)
            self.total_epochs += 1
            print(f"[Stage3][Epoch {self.total_epochs}] total={total:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in combined.items()))
            if self.total_epochs % save_gap == 0:
                save_checkpoint(
                    self.model,
                    self.model.projection,
                    self.opt,
                    total_epochs=self.total_epochs,
                    stage=3,
                    path=os.path.join("checkpoints", f"stage3_epoch{self.total_epochs}.pt"),
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-epoch", type=int, default=0, help="Resume training from given epoch")
    parser.add_argument("--lr-stage1", type=float, default=1e-3, help="Learning rate for stage 1")
    parser.add_argument("--lr-stage2", type=float, default=1e-3, help="Learning rate for stage 2")
    parser.add_argument("--lr-stage3", type=float, default=1e-3, help="Learning rate for stage 3")
    args = parser.parse_args()

    data_root = r"D:\Jiaqi\Datasets\Rats\TrainData_new"
    sessions = ["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_2"]
    batch_size = 512
    session_ranges = None

    # 载入 dataset：只负责加载原始序列 + label index，不再裁剪
    train_ds = RatsWindowDataset(
        data_root, sessions, split="train", session_ranges=session_ranges
    )

    # 特征维度：从原始数据里取
    num_feat_imu = train_ds.data[sessions[0]].imu.shape[1]
    num_feat_dlc = train_ds.data[sessions[0]].dlc.shape[1]
    num_sessions = len(sessions)

    trainer = ThreeStageTrainer(
        num_feat_imu,
        num_feat_dlc,
        num_sessions,
        stage_lrs={1: args.lr_stage1, 2: args.lr_stage2, 3: args.lr_stage3},
    )

    # 预处理多线程数：HDF5 写还是串行，线程只做裁剪与拼 batch
    n_workers_preproc = 1

    stage1_epochs = 10000
    stage2_epochs = 1
    stage3_epochs = 1

    start_epoch = args.resume_epoch
    if start_epoch > 0:
        if start_epoch <= stage1_epochs:
            ckpt_stage = 1
        elif start_epoch <= stage1_epochs + stage2_epochs:
            ckpt_stage = 2
        else:
            ckpt_stage = 3
        ckpt_path = os.path.join("checkpoints", f"stage{ckpt_stage}_epoch{start_epoch}.pt")
        trainer.load_from(ckpt_path, expected_stage=ckpt_stage)

    # 根据 epoch 判断从哪阶段开始
    start_stage = 1
    if start_epoch > stage1_epochs:
        start_stage = 2
    if start_epoch > stage1_epochs + stage2_epochs:
        start_stage = 3

    if start_stage <= 1:
        print(">>> Stage 1 (unsupervised)...")
        trainer.stage1(
            train_ds,
            batch_size,
            epochs=stage1_epochs - start_epoch,
            n_workers_preproc=n_workers_preproc,
            save_gap=1,
        )

    if start_stage <= 2:
        print(">>> Stage 2 (frozen encoder: unsup + sup)...")
        remaining2 = stage2_epochs - max(0, start_epoch - stage1_epochs)
        if remaining2 > 0:
            trainer.stage2(
                train_ds,
                batch_size,
                epochs=remaining2,
                n_workers_preproc=n_workers_preproc,
                save_gap=1,
            )

    if start_stage <= 3:
        print(">>> Stage 3 (joint training)...")
        remaining3 = stage3_epochs - max(0, start_epoch - stage1_epochs - stage2_epochs)
        if remaining3 > 0:
            trainer.stage3(
                train_ds,
                batch_size,
                epochs=remaining3,
                n_workers_preproc=n_workers_preproc,
                save_gap=1,
            )


if __name__ == "__main__":
    main()