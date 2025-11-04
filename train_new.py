# train_two_stage.py
"""Two-stage training script using the new window dataset.

Stage 1 focuses purely on unsupervised contrastive objectives. Stage 2
continues joint training with alternating unsupervised and supervised
batches. The original session-aware adapters and the previous stage 1
have been removed for a leaner workflow.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.single_modal import SingleModalModel
from models.losses import (
    gaussian_kl_divergence,
    gaussian_kl_divergence_masked,
    hierarchical_contrastive_loss,
    multilabel_supcon_loss_bt,
    sequential_next_step_nll,
)
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.preprocess import load_preprocessed_batches, preprocess_dataset
from utils.tools import take_per_row
from utils.window_dataset import RatsWindowDataset


class TwoStageTrainer:
    def __init__(
        self,
        num_features_imu: int,
        num_features_dlc: int,
        num_sessions: int,
        device: Optional[str] = None,
        *,
        stage_lrs: Optional[Dict[int, float]] = None,
        loss_weights: Optional[Dict[str, Dict[str, float]]] = None,
        modality: str = "imu",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = 64
        modality = modality.lower()
        if modality not in {"imu", "dlc"}:
            raise ValueError("modality must be either 'imu' or 'dlc'")
        self.modality = modality
        feature_dim = num_features_imu if modality == "imu" else num_features_dlc
        self.model = SingleModalModel(
            feature_dim,
            mask_type="binomial",
            d_model=self.d_model,
        ).to(self.device)

        self.stage_lrs: Dict[int, float] = {1: 1e-3, 2: 5e-4}
        if stage_lrs:
            self.stage_lrs.update(stage_lrs)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.stage_lrs[1])
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))
        self.temporal_unit = 3
        self.unsup_loss_weights = {
            "contrast": 1.0,
            "kl": 0.1,
            "recon": 0.1,
        }
        self.sup_loss_weights = {
            "contrast": 1.0,
            "kl": 0.1,
            "recon": 0.1,
        }
        if loss_weights:
            unsup_cfg = loss_weights.get("unsup", {})
            unsup_updates = {
                ("kl" if k == "cs" else k): float(v)
                for k, v in unsup_cfg.items()
            }
            self.unsup_loss_weights.update(unsup_updates)
            sup_cfg = loss_weights.get("sup", {})
            sup_updates = {
                ("kl" if k == "cs" else k): float(v) for k, v in sup_cfg.items()
            }
            self.sup_loss_weights.update(sup_updates)

        self.total_epochs = 0
        self.current_stage = 1
        self.class_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _configure_optimizer(self, stage: int) -> None:
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.stage_lrs[stage])
        self.current_stage = stage

    def _sanitize(self, t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        if loss is None:
            return
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

    def _record_metrics(self, stats: Dict[str, float], counts: Dict[str, int], metrics: Dict[str, torch.Tensor], prefix: str) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu())
            stats[prefix + key] += float(value)
            counts[prefix + key] += 1

    def _select_modality(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        key = "imu" if self.modality == "imu" else "dlc"
        return self._sanitize(batch[key].to(self.device))

    def _prepare_batches(
        self,
        dataset: RatsWindowDataset,
        batch_size: int,
        *,
        out_dir: str,
        use_unlabeled: bool,
        mix: bool,
        seed: int,
        n_workers_preproc: int,
    ) -> List[Dict[str, torch.Tensor]]:
        preprocess_dataset(
            dataset,
            batch_size,
            out_dir=out_dir,
            group_mode="by_session",
            assign_T="round_robin",
            device=self.device,
            num_workers=n_workers_preproc,
            seed=seed,
            use_unlabeled=use_unlabeled,
        )
        return list(
            load_preprocessed_batches(
                dataset.sessions,
                dataset.session_to_idx,
                out_dir=out_dir,
                mix=mix,
            )
        )

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------
    def _step_unsup(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        data = self._select_modality(batch)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)

        B, T, _ = data.shape
        if T <= 5:
            zero = data.new_tensor(0.0)
            metrics = {
                "reconstruction_nll": zero,
                "reconstruction_kl": zero,
                "unsupervised_contrastive": zero,
            }
            return None, metrics

        min_crop = 2 ** (self.temporal_unit + 1)
        crop_l = np.random.randint(low=min_crop, high=T + 1)
        crop_left = np.random.randint(T - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=T + 1)
        crop_offset = torch.randint(
            low=-crop_eleft,
            high=T - crop_eright + 1,
            size=(B,),
            device=data.device,
        )

        crop1 = take_per_row(data, crop_offset + crop_eleft, crop_right - crop_eleft)
        crop2 = take_per_row(data, crop_offset + crop_left, crop_eright - crop_left)

        crop_l = int(crop_right - crop_left)
        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            out1 = self.model(crop1)
            out2 = self.model(crop2)

            emb1 = out1.embedding[:, -crop_l:]
            emb2 = out2.embedding[:, :crop_l]
            jitter_std = 0.01
            emb1 = F.normalize(emb1 + torch.randn_like(emb1) * jitter_std, dim=-1)
            emb2 = F.normalize(emb2 + torch.randn_like(emb2) * jitter_std, dim=-1)

            loss_contrast = hierarchical_contrastive_loss(emb1, emb2, temporal_unit=self.temporal_unit)
            recon1 = out1.reconstruction[:, -crop_l:]
            recon2 = out2.reconstruction[:, :crop_l]
            target1 = crop1[:, -crop_l:]
            target2 = crop2[:, :crop_l]
            loss_recon_nll = 0.5 * (
                sequential_next_step_nll(recon1, target1)
                + sequential_next_step_nll(recon2, target2)
            )
            loss_recon_kl = 0.5 * (
                gaussian_kl_divergence(recon1, target1)
                + gaussian_kl_divergence(recon2, target2)
            )

            unsup_w = self.unsup_loss_weights
            loss_unsup = (
                unsup_w.get("contrast", 1.0) * loss_contrast
                + unsup_w.get("recon", 1.0) * loss_recon_nll
                + unsup_w.get("kl", 1.0) * loss_recon_kl
            )

            loss_total = loss_unsup

        metrics = {
            "reconstruction_nll": loss_recon_nll.detach(),
            "reconstruction_kl": loss_recon_kl.detach(),
            "unsupervised_contrastive": loss_contrast.detach(),
        }
        return loss_total, metrics

    def _step_sup(self, batch: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        data = self._select_modality(batch)
        labels = batch["label"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)

        scale = torch.empty(data.size(0), 1, 1, device=data.device).uniform_(0.8, 1.2)
        data = data * scale

        if self.class_weights is not None:
            weights = self.class_weights.to(labels.device)
            sample_w = (labels * weights).max(dim=1).values
            mix_idx = torch.multinomial(sample_w, labels.size(0), replacement=True)
            partner_idx = torch.randint(0, labels.size(0), (len(mix_idx),), device=labels.device)
            lam = 0.5
            data_mix = lam * data[mix_idx] + (1 - lam) * data[partner_idx]
            label_mix = torch.clamp(labels[mix_idx] + labels[partner_idx], 0, 1)
            if mask is not None:
                mask_mix = mask[mix_idx] & mask[partner_idx]
                mask = torch.cat([mask, mask_mix], dim=0)
            data = torch.cat([data, data_mix], dim=0)
            labels = torch.cat([labels, label_mix], dim=0)

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            out = self.model(data, mask=mask)
            jitter_std = 0.01
            emb = out.embedding + torch.randn_like(out.embedding) * jitter_std
            emb = F.normalize(emb, dim=-1)

            loss_sup = multilabel_supcon_loss_bt(emb, labels)
            loss_recon_nll = 0.5 * (
                sequential_next_step_nll(out.reconstruction, data, mask)
            )
            loss_recon_kl = 0.5 * (
                gaussian_kl_divergence_masked(out.reconstruction, data, mask)
            )
            sup_w = self.sup_loss_weights
            loss_total = (
                sup_w.get("contrast", 1.0) * loss_sup
                + sup_w.get("recon", 1.0) * loss_recon_nll
                + sup_w.get("kl", 1.0) * loss_recon_kl
            )

        metrics = {
            "reconstruction_nll": loss_recon_nll.detach(),
            "reconstruction_kl": loss_recon_kl.detach(),
            "supervised_contrastive": loss_sup.detach(),
        }
        return loss_total, metrics

    def _train_epoch(
        self,
        unsup_batches: List[Dict[str, torch.Tensor]],
        sup_batches: List[Dict[str, torch.Tensor]],
        *,
        stage_name: str,
        include_supervised: bool,
    ) -> Dict[str, float]:
        self.model.train()
        stats: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)

        if include_supervised:
            if len(unsup_batches) > 0:
                total_steps = len(unsup_batches)
            else:
                total_steps = len(sup_batches)
        else:
            total_steps = len(unsup_batches)
        if total_steps == 0:
            return {}

        progress = tqdm(
            range(total_steps),
            desc=f"[{stage_name}] Epoch {self.total_epochs + 1}",
            leave=False,
        )
        for step in progress:
            if step < len(unsup_batches):
                loss_u, metrics_u = self._step_unsup(unsup_batches[step])
                if loss_u is not None:
                    self._optimizer_step(loss_u)
                self._record_metrics(stats, counts, metrics_u, prefix="unsup_")
                progress.set_postfix(
                    unsup_contrast=float(metrics_u.get("unsupervised_contrastive", 0.0))
                )

            if include_supervised and sup_batches:
                if step < len(sup_batches):
                    sup_idx = step
                else:
                    sup_idx = int(np.random.randint(0, len(sup_batches)))
                loss_s, metrics_s = self._step_sup(sup_batches[sup_idx])
                if loss_s is not None:
                    self._optimizer_step(loss_s)
                self._record_metrics(stats, counts, metrics_s, prefix="sup_")
                progress.set_postfix(
                    unsup_contrast=float(
                        stats.get("unsup_unsupervised_contrastive", 0.0)
                        / max(counts.get("unsup_unsupervised_contrastive", 1), 1)
                    ),
                    sup_contrast=float(metrics_s.get("supervised_contrastive", 0.0)),
                )

        recon_nll_sum = stats.get("unsup_reconstruction_nll", 0.0) + stats.get("sup_reconstruction_nll", 0.0)
        recon_nll_count = counts.get("unsup_reconstruction_nll", 0) + counts.get("sup_reconstruction_nll", 0)
        recon_kl_sum = stats.get("unsup_reconstruction_kl", 0.0) + stats.get("sup_reconstruction_kl", 0.0)
        recon_kl_count = counts.get("unsup_reconstruction_kl", 0) + counts.get("sup_reconstruction_kl", 0)
        unsup_contrast_sum = stats.get("unsup_unsupervised_contrastive", 0.0)
        unsup_contrast_count = counts.get("unsup_unsupervised_contrastive", 0)
        sup_contrast_sum = stats.get("sup_supervised_contrastive", 0.0)
        sup_contrast_count = counts.get("sup_supervised_contrastive", 0)

        final_metrics: Dict[str, float] = {}
        final_metrics["reconstructionNLL_loss"] = (
            recon_nll_sum / recon_nll_count if recon_nll_count > 0 else 0.0
        )
        final_metrics["reconstructionKL_loss"] = (
            recon_kl_sum / recon_kl_count if recon_kl_count > 0 else 0.0
        )
        final_metrics["unsupervised_contrastive_loss"] = (
            unsup_contrast_sum / unsup_contrast_count if unsup_contrast_count > 0 else 0.0
        )

        if include_supervised:
            final_metrics["supervised_contrastive_loss"] = (
                sup_contrast_sum / sup_contrast_count if sup_contrast_count > 0 else 0.0
            )

        return final_metrics

    def _log_epoch(self, stage_name: str, metrics: Dict[str, float]) -> None:
        ordered = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(f"[{stage_name}][Epoch {self.total_epochs}] {ordered}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_from(self, path: str, expected_stage: int, *, load_optimizer: bool = True) -> int:
        opt = self.opt if load_optimizer else None
        self.total_epochs, stage = load_checkpoint(
            self.model,
            self.model.projection,
            opt,
            path,
            expected_stage=expected_stage,
        )
        self.current_stage = stage
        return stage

    def stage1(
        self,
        dataset: RatsWindowDataset,
        batch_size: int,
        epochs: int = 1,
        *,
        n_workers_preproc: int = 0,
        save_gap: int = 1,
    ) -> None:
        if self.current_stage != 1:
            self._configure_optimizer(1)

        for ep in range(epochs):
            unsup_batches = self._prepare_batches(
                dataset,
                batch_size,
                out_dir="Dataset_unsup",
                use_unlabeled=True,
                mix=False,
                seed=42 + ep,
                n_workers_preproc=n_workers_preproc,
            )
            metrics = self._train_epoch(
                unsup_batches,
                [],
                stage_name="Stage1",
                include_supervised=False,
            )

            self.total_epochs += 1
            self._log_epoch("Stage1", metrics)
            if self.total_epochs % save_gap == 0:
                save_checkpoint(
                    self.model,
                    self.model.projection,
                    self.opt,
                    total_epochs=self.total_epochs,
                    stage=1,
                    path=os.path.join("checkpoints", f"stage1_epoch{self.total_epochs}.pt"),
                )

    def stage2(
        self,
        dataset: RatsWindowDataset,
        batch_size: int,
        epochs: int = 1,
        *,
        n_workers_preproc: int = 0,
        save_gap: int = 1,
    ) -> None:
        if self.current_stage != 2:
            self._configure_optimizer(2)

        if self.class_weights is None:
            counts = torch.zeros(dataset.num_labels)
            for _, lab, _ in dataset.samples:
                counts += lab
            weights = 1.0 / (counts + 1e-6)
            self.class_weights = (weights / weights.sum()).to(self.device)

        for ep in range(epochs):
            unsup_batches = self._prepare_batches(
                dataset,
                batch_size,
                out_dir="Dataset_unsup",
                use_unlabeled=True,
                mix=True,
                seed=5678 + ep,
                n_workers_preproc=n_workers_preproc,
            )
            sup_batches = self._prepare_batches(
                dataset,
                batch_size,
                out_dir="Dataset_sup",
                use_unlabeled=False,
                mix=True,
                seed=5678 + ep,
                n_workers_preproc=n_workers_preproc,
            )

            metrics = self._train_epoch(
                unsup_batches,
                sup_batches,
                stage_name="Stage2",
                include_supervised=True,
            )
            self.total_epochs += 1
            self._log_epoch("Stage2", metrics)
            if self.total_epochs % save_gap == 0:
                save_checkpoint(
                    self.model,
                    self.model.projection,
                    self.opt,
                    total_epochs=self.total_epochs,
                    stage=2,
                    path=os.path.join("checkpoints", f"stage2_epoch{self.total_epochs}.pt"),
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-epoch", type=int, default=0, help="Resume training from given epoch")
    parser.add_argument("--lr-stage1", type=float, default=1e-4, help="Learning rate for stage 1")
    parser.add_argument("--lr-stage2", type=float, default=5e-5, help="Learning rate for stage 2")
    parser.add_argument("--split-seed", type=int, default=0, help="Random seed for segment-level train/test split")
    parser.add_argument("--cross-folds", type=int, default=1, help="Total number of cross-validation folds")
    parser.add_argument(
        "--cross-index",
        type=int,
        default=0,
        help="Index of the current cross-validation fold (0-based)",
    )
    parser.add_argument(
        "--modality",
        choices=["imu", "dlc"],
        default="imu",
        help="Which modality to use for single-encoder training",
    )
    args = parser.parse_args()

    data_root = r"D:\\Jiaqi\\Datasets\\Rats\\TrainData_new"
    sessions = ["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor",  "F5D7_outdoor"]
    batch_size = 512
    session_ranges = None

    if args.cross_folds <= 0:
        raise ValueError("--cross-folds must be positive")
    if not (0 <= args.cross_index < args.cross_folds):
        raise ValueError("--cross-index must satisfy 0 <= index < --cross-folds")

    train_ds = RatsWindowDataset(
        data_root,
        sessions,
        split="train",
        session_ranges=session_ranges,
        test_ratio = 0.3,
        split_seed=args.split_seed,
        num_folds=args.cross_folds,
        fold_index=args.cross_index,
    )

    num_feat_imu = train_ds.data[sessions[0]].imu.shape[1]
    num_feat_dlc = train_ds.data[sessions[0]].dlc.shape[1]
    num_sessions = len(sessions)

    trainer = TwoStageTrainer(
        num_feat_imu,
        num_feat_dlc,
        num_sessions,
        stage_lrs={1: args.lr_stage1, 2: args.lr_stage2},
        modality=args.modality,
    )

    n_workers_preproc = 1

    stage1_epochs = 30
    stage2_epochs = 200

    start_epoch = args.resume_epoch
    start_stage = 1
    if start_epoch > stage1_epochs:
        start_stage = 2

    if start_epoch > 0:
        ckpt_stage = 1 if start_epoch <= stage1_epochs else 2
        ckpt_path = os.path.join("checkpoints", f"stage{ckpt_stage}_epoch{start_epoch}.pt")
        if os.path.exists(ckpt_path):
            trainer._configure_optimizer(ckpt_stage)
            trainer.load_from(ckpt_path, expected_stage=ckpt_stage, load_optimizer=(ckpt_stage == start_stage))
            if start_stage == 2:
                trainer._configure_optimizer(2)
        else:
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

    if start_stage <= 1:
        remaining1 = max(stage1_epochs - start_epoch, 0)
        if remaining1 > 0:
            print(">>> Stage 1 (unsupervised)...")
            trainer.stage1(
                train_ds,
                batch_size,
                epochs=remaining1,
                n_workers_preproc=n_workers_preproc,
                save_gap=1,
            )

    if start_stage <= 2:
        offset = max(0, start_epoch - stage1_epochs)
        remaining2 = max(stage2_epochs - offset, 0)
        if remaining2 > 0:
            print(">>> Stage 2 (joint training)...")
            trainer.stage2(
                train_ds,
                batch_size,
                epochs=remaining2,
                n_workers_preproc=n_workers_preproc,
                save_gap=1,
            )


if __name__ == "__main__":
    main()
