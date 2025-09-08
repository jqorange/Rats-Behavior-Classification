"""Simplified three-stage training script using the new window dataset.

This script is **not** a full reproduction of the specification but provides
clean code structure with minimal working components.  The heavy losses and
optimisation details are left as TODO items.
"""

from __future__ import annotations

from typing import Sequence

import torch

from utils.window_dataset import RatsWindowDataset
from utils.preprocess import preprocess_dataset, load_preprocessed_batches
from models.fusion import EncoderFusion
from models.losses import (
    hierarchical_contrastive_loss,
    multilabel_supcon_loss_bt,
)
from tqdm import tqdm

class ThreeStageTrainer:
    """Minimal trainer implementing the three-stage schedule.

    The backbone :class:`EncoderFusion` is reused from the original project.
    Only a subset of the losses is implemented for brevity.
    """

    def __init__(self, num_features_imu: int, num_features_dlc: int, num_sessions: int, device: str = "cpu") -> None:
        self.device = device
        self.model = EncoderFusion(
            N_feat_A=num_features_imu,
            N_feat_B=num_features_dlc,
            mask_type="binomial",
            d_model=64,
            nhead=4,
            num_sessions=num_sessions,
        ).to(device)
        self.proj = torch.nn.Linear(64, 64).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scaler = torch.cuda.amp.GradScaler()

    def _step_unsup(self, batch: dict) -> torch.Tensor:
        imu = batch["imu"].to(self.device)
        dlc = batch["dlc"].to(self.device)
        session_idx = batch["session_idx"].to(self.device, dtype=torch.long)
        emb, _, _ = self.model(imu, dlc, session_idx=session_idx)
        loss = hierarchical_contrastive_loss(emb, emb)
        return loss

    def _step_sup(self, batch: dict) -> torch.Tensor:
        imu = batch["imu"].to(self.device)
        dlc = batch["dlc"].to(self.device)
        labels = batch["label"].to(self.device)
        session_idx = batch["session_idx"].to(self.device)
        emb, _, _ = self.model(imu, dlc, session_idx=session_idx)
        loss = multilabel_supcon_loss_bt(emb, labels)
        return loss

    def stage1(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1) -> None:
        """Unsupervised contrastive learning within each session."""

        self.model.train()
        for _ in range(epochs):
            preprocess_dataset(dataset, batch_size)
            for batch in tqdm(
                load_preprocessed_batches(
                    dataset.sessions, dataset.session_to_idx, mix=False
                )
            ):
                self.opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = self._step_unsup(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

    def stage2(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1) -> None:
        """Supervised learning with frozen encoders.

        The unsupervised batches are restricted to single sessions while the
        supervised batches may mix sessions within a batch.
        """

        for p in self.model.encoderA.parameters():
            p.requires_grad = False
        for p in self.model.encoderB.parameters():
            p.requires_grad = False
        self.model.train()
        for _ in range(epochs):
            preprocess_dataset(dataset, batch_size)
            for batch in load_preprocessed_batches(
                dataset.sessions, dataset.session_to_idx, mix=False
            ):
                self.opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = self._step_unsup(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            for batch in load_preprocessed_batches(
                dataset.sessions, dataset.session_to_idx, mix=True
            ):
                self.opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = self._step_sup(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

    def stage3(self, dataset: RatsWindowDataset, batch_size: int, epochs: int = 1) -> None:
        """Joint supervised and unsupervised learning across sessions."""

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()
        for _ in range(epochs):
            preprocess_dataset(dataset, batch_size)
            for batch in load_preprocessed_batches(
                dataset.sessions, dataset.session_to_idx, mix=True
            ):
                self.opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = self._step_sup(batch) + self._step_unsup(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()


def main() -> None:  # pragma: no cover - entry point
    data_root = "D:\\Jiaqi\\Datasets\\Rats\\TrainData_new"
    sessions = ["F3D5_outdoor", "F3D6_outdoor"]
    batch_size = 256
    session_ranges = None  # e.g. {"F3D5_outdoor": (0, 10000)} to limit data
    train_ds = RatsWindowDataset(
        data_root, sessions, split="train", session_ranges=session_ranges
    )
    num_feat_imu = train_ds.data[sessions[0]].imu.shape[1]
    num_feat_dlc = train_ds.data[sessions[0]].dlc.shape[1]
    num_sessions = len(sessions)
    trainer = ThreeStageTrainer(num_feat_imu, num_feat_dlc, num_sessions)
    trainer.stage1(train_ds, batch_size, epochs=1)
    trainer.stage2(train_ds, batch_size, epochs=1)
    trainer.stage3(train_ds, batch_size, epochs=1)


if __name__ == "__main__":  # pragma: no cover
    main()
