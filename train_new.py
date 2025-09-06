"""Simplified three-stage training script using the new window dataset.

This script is **not** a full reproduction of the specification but provides
clean code structure with minimal working components.  The heavy losses and
optimisation details are left as TODO items.
"""

from __future__ import annotations

import os
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from utils.window_dataset import RatsWindowDataset, collate_fn
from models.fusion import EncoderFusion
from models.losses import (
    hierarchical_contrastive_loss,
    multilabel_supcon_loss_bt,
)


class ThreeStageTrainer:
    """Minimal trainer implementing the three-stage schedule.

    The backbone :class:`EncoderFusion` is reused from the original project.
    Only a subset of the losses is implemented for brevity.
    """

    def __init__(self, num_features_imu: int, num_features_dlc: int, num_classes: int, device: str = "cpu") -> None:
        self.device = device
        self.model = EncoderFusion(
            N_feat_A=num_features_imu,
            N_feat_B=num_features_dlc,
            mask_type="binomial",
            d_model=64,
            nhead=4,
            num_sessions=0,
        ).to(device)
        self.proj = torch.nn.Linear(64, 64).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _step_unsup(self, batch: dict) -> torch.Tensor:
        imu = batch["imu"].to(self.device)
        dlc = batch["dlc"].to(self.device)
        emb, _, _ = self.model(imu, dlc, session_idx=None)
        loss = hierarchical_contrastive_loss(emb, emb)
        return loss

    def _step_sup(self, batch: dict) -> torch.Tensor:
        imu = batch["imu"].to(self.device)
        dlc = batch["dlc"].to(self.device)
        labels = batch["label"].to(self.device)
        emb, _, _ = self.model(imu, dlc, session_idx=None)
        loss = multilabel_supcon_loss_bt(emb, labels)
        return loss

    def stage1(self, loader: DataLoader, epochs: int = 1) -> None:
        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                self.opt.zero_grad()
                loss = self._step_unsup(batch)
                loss.backward()
                self.opt.step()

    def stage2(self, loader: DataLoader, epochs: int = 1) -> None:
        for p in self.model.encoderA.parameters():
            p.requires_grad = False
        for p in self.model.encoderB.parameters():
            p.requires_grad = False
        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                self.opt.zero_grad()
                loss = self._step_sup(batch)
                loss.backward()
                self.opt.step()

    def stage3(self, loader: DataLoader, epochs: int = 1) -> None:
        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                self.opt.zero_grad()
                loss = self._step_sup(batch) + self._step_unsup(batch)
                loss.backward()
                self.opt.step()


def build_loaders(root: str, sessions: Sequence[str], batch_size: int = 8) -> tuple[DataLoader, DataLoader]:
    train_ds = RatsWindowDataset(root, sessions, split="train")
    test_ds = RatsWindowDataset(root, sessions, split="test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader


def main() -> None:  # pragma: no cover - entry point
    data_root = os.environ.get("RATS_DATA_ROOT", "./data")
    sessions = ["session1"]
    train_loader, _ = build_loaders(data_root, sessions)
    num_feat_imu = train_loader.dataset.data[sessions[0]].imu.shape[1]
    num_feat_dlc = train_loader.dataset.data[sessions[0]].dlc.shape[1]
    num_classes = train_loader.dataset.num_labels
    trainer = ThreeStageTrainer(num_feat_imu, num_feat_dlc, num_classes)
    trainer.stage1(train_loader, epochs=1)
    trainer.stage2(train_loader, epochs=1)
    trainer.stage3(train_loader, epochs=1)


if __name__ == "__main__":  # pragma: no cover
    main()
