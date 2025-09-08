# train_three_stage.py
"""Simplified three-stage training script using the new window dataset.

此脚本展示三阶段训练流程：Stage1(无监督/单会话批)、Stage2(冻结编码器+无监督→有监督)、
Stage3(联训)。预处理阶段新增“按 index 分组、每组统一窗口长度 T”，并支持多线程。
"""

from __future__ import annotations
import os
from typing import Sequence

import torch
from tqdm import tqdm

from utils.window_dataset import RatsWindowDataset
from utils.preprocess import preprocess_dataset, load_preprocessed_batches
from models.fusion import EncoderFusion
from models.losses import (
    hierarchical_contrastive_loss,
    multilabel_supcon_loss_bt,
    positive_only_supcon_loss,
)
import torch.nn.functional as F
from utils.checkpoint import save_checkpoint, load_checkpoint


class ThreeStageTrainer:
    def __init__(self, num_features_imu: int, num_features_dlc: int, num_sessions: int, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderFusion(
            N_feat_A=num_features_imu,
            N_feat_B=num_features_dlc,
            mask_type="binomial",  # 如不需要特征丢弃，可直接改为 None
            d_model=64,
            nhead=4,
            num_sessions=num_sessions,
            # 如果你的 EncoderFusion/encoder 支持传入概率参数，建议同时传：
            # drop_prob=0.1,
        ).to(self.device)
        self.proj = torch.nn.Linear(64, 64).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.proj.parameters()), lr=1e-3
        )
        self.total_epochs = 0

        # ✅ 新 API
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device == "cuda"))

    def _sanitize_inplace(self, t: torch.Tensor) -> torch.Tensor:
        # 把 NaN/Inf 变成 0，防止后续层炸掉
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _step_unsup(self, batch: dict) -> torch.Tensor:
        imu = self._sanitize_inplace(batch["imu"].to(self.device))
        dlc = self._sanitize_inplace(batch["dlc"].to(self.device))
        session_idx = batch["session_idx"].to(self.device, dtype=torch.long)

        def _crop_and_scale(x1: torch.Tensor, x2: torch.Tensor, ratio: float = 0.8):
            """Randomly crop the same 80% window and apply random scaling."""
            B, T, D1 = x1.shape
            crop_len = max(int(T * ratio), 1)
            start = torch.randint(0, T - crop_len + 1, (B,), device=x1.device)
            idx = start[:, None] + torch.arange(crop_len, device=x1.device)[None, :]
            idx = idx.unsqueeze(-1)
            x1_crop = torch.gather(x1, 1, idx.expand(-1, -1, D1))
            x2_crop = torch.gather(x2, 1, idx.expand(-1, -1, x2.size(2)))
            scales = torch.empty(B, 1, 1, device=x1.device).uniform_(0.8, 1.2)
            return x1_crop * scales, x2_crop * scales

        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            imu1, dlc1 = _crop_and_scale(imu, dlc)
            imu2, dlc2 = _crop_and_scale(imu, dlc)

            emb1, A_self, B_self, A_to_B, B_to_A = self.model(imu1, dlc1, session_idx=session_idx)
            emb2, *_ = self.model(imu2, dlc2, session_idx=session_idx)

            jitter_std = 0.01
            emb1 = emb1 + torch.randn_like(emb1) * jitter_std
            emb2 = emb2 + torch.randn_like(emb2) * jitter_std

            loss_contrast = hierarchical_contrastive_loss(emb1, emb2)
            loss_cs = (
                1 - F.cosine_similarity(A_to_B, B_self, dim=-1).mean()
                + 1 - F.cosine_similarity(B_to_A, A_self, dim=-1).mean()
            )
            loss_mse = F.mse_loss(A_to_B, B_self) + F.mse_loss(B_to_A, A_self)
            loss = loss_contrast + loss_cs + loss_mse
        return loss

    def _step_sup_stage2(self, batch: dict) -> torch.Tensor:
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
                1 - F.cosine_similarity(A_to_B, B_self, dim=-1).mean()
                + 1 - F.cosine_similarity(B_to_A, A_self, dim=-1).mean()
            )
            loss_mse = F.mse_loss(A_to_B, B_self) + F.mse_loss(B_to_A, A_self)
            loss = loss_sup + loss_cs + loss_mse
        return loss

    def _step_sup(self, batch: dict) -> torch.Tensor:
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
                1 - F.cosine_similarity(A_to_B, B_self, dim=-1).mean()
                + 1 - F.cosine_similarity(B_to_A, A_self, dim=-1).mean()
            )
            loss_mse = F.mse_loss(A_to_B, B_self) + F.mse_loss(B_to_A, A_self)
            loss = loss_sup + loss_cs + loss_mse
        return loss

    def _run_epoch(self, iterator, step_fn):
        self.model.train()
        batches = list(iterator)  # 先收集
        for batch in tqdm(batches, total=len(batches)):
            self.opt.zero_grad(set_to_none=True)
            loss = step_fn(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

    def load_from(self, path: str, expected_stage: int) -> int:
        """加载 checkpoint 并返回其中记录的阶段。"""
        self.total_epochs, stage = load_checkpoint(
            self.model, self.proj, self.opt, path, expected_stage=expected_stage
        )
        return stage

    def stage1(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1, *, n_workers_preproc: int = 0) -> None:
        """无监督学习：每个 batch 内不混 session（by_session），每组统一 T。"""
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True
        self.opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.proj.parameters()), lr=1e-3
        )
        for ep in range(epochs):
            preprocess_dataset(
                dataset, batch_size, out_dir="Dataset",
                group_mode="by_session",
                assign_T="round_robin",       # 各 T 均衡
                num_workers=n_workers_preproc,
                seed=42 + ep,                 # 每个 epoch 改变 seed，获得新的切窗
            )
            it = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, mix=False)
            self._run_epoch(it, self._step_unsup)
            self.total_epochs += 1
            if self.total_epochs % 5 == 0:
                save_checkpoint(
                    self.model,
                    self.proj,
                    self.opt,
                    total_epochs=self.total_epochs,
                    stage=1,
                    path=os.path.join("checkpoints", f"stage1_epoch{self.total_epochs}.pt"),
                )

    def stage2(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1, *, n_workers_preproc: int = 0) -> None:
        """冻结编码器：先无监督（不混 session），再有监督（混 session）。"""
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = True
        self.opt = torch.optim.Adam(self.proj.parameters(), lr=1e-3)

        for ep in range(epochs):
            preprocess_dataset(
                dataset, batch_size, out_dir="Dataset",
                group_mode="by_session",
                assign_T="round_robin",
                num_workers=n_workers_preproc,
                seed=1234 + ep,
            )
            it_unsup = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, mix=False)
            self._run_epoch(it_unsup, self._step_unsup)

            it_sup = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, mix=True)
            self._run_epoch(it_sup, self._step_sup_stage2)
            self.total_epochs += 1
            if self.total_epochs % 5 == 0:
                save_checkpoint(
                    self.model,
                    self.proj,
                    self.opt,
                    total_epochs=self.total_epochs,
                    stage=2,
                    path=os.path.join("checkpoints", f"stage2_epoch{self.total_epochs}.pt"),
                )

    def stage3(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1, *, n_workers_preproc: int = 0) -> None:
        """联训：有监督 + 无监督都在混 session 的 batch 上进行。"""
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True
        self.opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.proj.parameters()), lr=1e-3
        )

        for ep in range(epochs):
            preprocess_dataset(
                dataset, batch_size, out_dir="Dataset",
                group_mode="by_session",
                assign_T="round_robin",
                device="cuda",
                num_workers=n_workers_preproc,
                seed=5678 + ep,
            )
            it = load_preprocessed_batches(dataset.sessions, dataset.session_to_idx, mix=True)
            # 这里简单相加，有需要可加权
            def step_joint(b):
                return self._step_sup(b) + self._step_unsup(b)
            self._run_epoch(it, step_joint)
            self.total_epochs += 1
            if self.total_epochs % 5 == 0:
                save_checkpoint(
                    self.model,
                    self.proj,
                    self.opt,
                    total_epochs=self.total_epochs,
                    stage=3,
                    path=os.path.join("checkpoints", f"stage3_epoch{self.total_epochs}.pt"),
                )


def main() -> None:
    data_root = r"D:\Jiaqi\Datasets\Rats\TrainData_new"
    sessions = ["F3D5_outdoor", "F3D6_outdoor"]
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

    trainer = ThreeStageTrainer(num_feat_imu, num_feat_dlc, num_sessions)

    # 预处理多线程数：HDF5 写还是串行，线程只做裁剪与拼 batch
    n_workers_preproc = 1

    print(">>> Stage 1 (unsupervised)...")
    trainer.stage1(train_ds, batch_size, epochs=10, n_workers_preproc=n_workers_preproc)

    print(">>> Stage 2 (frozen encoder: unsup + sup)...")
    trainer.stage2(train_ds, batch_size, epochs=1, n_workers_preproc=n_workers_preproc)

    print(">>> Stage 3 (joint training)...")
    trainer.stage3(train_ds, batch_size, epochs=1, n_workers_preproc=n_workers_preproc)


if __name__ == "__main__":
    main()
