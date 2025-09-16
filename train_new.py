# train_two_stage.py
"""Two-stage training script using the new window dataset.

Stage 1 combines unsupervised contrastive objectives with prototype
regularisation. Stage 2 continues joint training with alternating
unsupervised and supervised batches. The original session-aware adapters
and the previous stage 1 have been removed for a leaner workflow.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.fusion import EncoderFusion
from models.losses import (
    gaussian_cs_divergence,
    hierarchical_contrastive_loss,
    multilabel_supcon_loss_bt,
    prototype_loss,
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
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = 64
        self.model = EncoderFusion(
            N_feat_A=num_features_imu,
            N_feat_B=num_features_dlc,
            mask_type="binomial",
            d_model=self.d_model,
            nhead=4,
            num_sessions=num_sessions,
        ).to(self.device)

        self.stage_lrs: Dict[int, float] = {1: 1e-3, 2: 5e-4}
        if stage_lrs:
            self.stage_lrs.update(stage_lrs)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.stage_lrs[1])
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))
        self.temporal_unit = 3
        self.recon_weight = 1.0
        self.proto_weight = 0.01

        self.total_epochs = 0
        self.current_stage = 1
        self.prototypes: Optional[torch.Tensor] = None
        self.class_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _configure_optimizer(self, stage: int) -> None:
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.stage_lrs[stage])
        self.current_stage = stage

    def _sanitize(self, t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _masked_l2(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = (pred - target) ** 2
        if mask is not None:
            mask_f = mask.to(pred.device, dtype=pred.dtype)
            while mask_f.dim() < diff.dim():
                mask_f = mask_f.unsqueeze(-1)
            diff = diff * mask_f
            denom = mask_f.sum() * target.size(-1)
            if denom.item() == 0:
                return diff.new_tensor(0.0)
            return diff.sum() / denom
        return diff.mean()

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

    def _compute_prototypes(
        self,
        batches: Iterable[Dict[str, torch.Tensor]],
        extra_feats: Optional[torch.Tensor] = None,
        extra_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in batches:
                imu = self._sanitize(batch["imu"].to(self.device))
                dlc = self._sanitize(batch["dlc"].to(self.device))
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(self.device)
                output = self.model(imu, dlc, mask=mask)
                pooled = output.fused.max(dim=1).values
                feats.append(pooled)
                labels.append(batch["label"].to(self.device).argmax(dim=1))
        if extra_feats is not None and extra_labels is not None:
            feats.append(extra_feats)
            labels.append(extra_labels)
        feats_cat = torch.cat(feats, dim=0)
        labels_cat = torch.cat(labels, dim=0)
        n_classes = int(labels_cat.max().item()) + 1
        protos = []
        for k in range(n_classes):
            mask = labels_cat == k
            protos.append(feats_cat[mask].mean(dim=0))
        self.model.train()
        return torch.stack(protos, dim=0)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------
    def _step_unsup(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        prototypes: Optional[torch.Tensor] = None,
        proto_weight: float = 0.0,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        imu = self._sanitize(batch["imu"].to(self.device))
        dlc = self._sanitize(batch["dlc"].to(self.device))
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        session_idx = batch.get("session_idx")
        if session_idx is not None:
            session_idx = session_idx.to(self.device, dtype=torch.long)

        B, T, _ = imu.shape
        if T <= 5:
            zero = imu.new_tensor(0.0)
            metrics = {
                "align_l2": zero,
                "reconstruction_l2": zero,
                "unsupervised_contrastive": zero,
            }
            if prototypes is not None:
                metrics["prototype_loss"] = zero
            return None, metrics, None, None

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
            device=imu.device,
        )

        imu_crop1 = take_per_row(imu, crop_offset + crop_eleft, crop_right - crop_eleft)
        dlc_crop1 = take_per_row(dlc, crop_offset + crop_eleft, crop_right - crop_eleft)
        imu_crop2 = take_per_row(imu, crop_offset + crop_left, crop_eright - crop_left)
        dlc_crop2 = take_per_row(dlc, crop_offset + crop_left, crop_eright - crop_left)

        crop_l = int(crop_right - crop_left)
        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            out1 = self.model(imu_crop1, dlc_crop1, session_idx=session_idx)
            out2 = self.model(imu_crop2, dlc_crop2, session_idx=session_idx)

            emb1 = out1.fused[:, -crop_l:]
            emb2 = out2.fused[:, :crop_l]
            jitter_std = 0.01
            emb1 = F.normalize(emb1 + torch.randn_like(emb1) * jitter_std, dim=-1)
            emb2 = F.normalize(emb2 + torch.randn_like(emb2) * jitter_std, dim=-1)

            imu_self = out1.imu_self[:, -crop_l:]
            dlc_self = out1.dlc_self[:, -crop_l:]
            imu_to_dlc = out1.imu_to_dlc[:, -crop_l:]
            dlc_to_imu = out1.dlc_to_imu[:, -crop_l:]

            loss_contrast = hierarchical_contrastive_loss(emb1, emb2, temporal_unit=self.temporal_unit)
            loss_cs = gaussian_cs_divergence(imu_to_dlc, dlc_self) + gaussian_cs_divergence(dlc_to_imu, imu_self)
            loss_align = (
                torch.norm(imu_to_dlc - dlc_self, dim=-1).mean()
                + torch.norm(dlc_to_imu - imu_self, dim=-1).mean()
            )

            recon_a = out1.imu_recon[:, -crop_l:]
            recon_b = out1.dlc_recon[:, -crop_l:]
            recon_a2 = out2.imu_recon[:, :crop_l]
            recon_b2 = out2.dlc_recon[:, :crop_l]
            loss_recon = 0.25 * (
                self._masked_l2(recon_a, imu_crop1[:, -crop_l:])
                + self._masked_l2(recon_b, dlc_crop1[:, -crop_l:])
                + self._masked_l2(recon_a2, imu_crop2[:, :crop_l])
                + self._masked_l2(recon_b2, dlc_crop2[:, :crop_l])
            )

            loss_unsup = loss_contrast + loss_cs + loss_align + self.recon_weight * loss_recon

            proto_loss = None
            pseudo_feats = None
            pseudo_labels = None
            if prototypes is not None:
                full_out = self.model(imu, dlc, session_idx=session_idx, mask=mask)
                proto_loss, feats, pseudo = prototype_loss(full_out.fused, prototypes)
                if feats is not None and pseudo is not None:
                    pseudo_feats = feats.detach()
                    pseudo_labels = pseudo.detach()
                loss_total = loss_unsup + proto_weight * proto_loss
            else:
                loss_total = loss_unsup

        metrics = {
            "align_l2": loss_align.detach(),
            "reconstruction_l2": loss_recon.detach(),
            "unsupervised_contrastive": loss_contrast.detach(),
            "cs_divergence": loss_cs.detach(),
        }
        if proto_loss is not None:
            metrics["prototype_loss"] = proto_loss.detach()
        return loss_total, metrics, pseudo_feats, pseudo_labels

    def _step_sup(self, batch: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        imu = self._sanitize(batch["imu"].to(self.device))
        dlc = self._sanitize(batch["dlc"].to(self.device))
        labels = batch["label"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        session_idx = batch.get("session_idx")
        if session_idx is not None:
            session_idx = session_idx.to(self.device, dtype=torch.long)

        scale = torch.empty(imu.size(0), 1, 1, device=imu.device).uniform_(0.8, 1.2)
        imu = imu * scale
        dlc = dlc * scale

        if self.class_weights is not None:
            weights = self.class_weights.to(labels.device)
            sample_w = (labels * weights).max(dim=1).values
            mix_idx = torch.multinomial(sample_w, labels.size(0), replacement=True)
            partner_idx = torch.randint(0, labels.size(0), (len(mix_idx),), device=labels.device)
            lam = 0.5
            imu_mix = lam * imu[mix_idx] + (1 - lam) * imu[partner_idx]
            dlc_mix = lam * dlc[mix_idx] + (1 - lam) * dlc[partner_idx]
            label_mix = torch.clamp(labels[mix_idx] + labels[partner_idx], 0, 1)
            session_mix = session_idx[mix_idx] if session_idx is not None else None
            if mask is not None:
                mask_mix = mask[mix_idx] & mask[partner_idx]
                mask = torch.cat([mask, mask_mix], dim=0)
            imu = torch.cat([imu, imu_mix], dim=0)
            dlc = torch.cat([dlc, dlc_mix], dim=0)
            labels = torch.cat([labels, label_mix], dim=0)
            if session_idx is not None:
                session_idx = torch.cat([session_idx, session_mix], dim=0)

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            out = self.model(imu, dlc, session_idx=session_idx, mask=mask)
            jitter_std = 0.01
            emb = out.fused + torch.randn_like(out.fused) * jitter_std
            emb = F.normalize(emb, dim=-1)

            loss_sup = multilabel_supcon_loss_bt(emb, labels)
            loss_cs = gaussian_cs_divergence(out.imu_to_dlc, out.dlc_self) + gaussian_cs_divergence(out.dlc_to_imu, out.imu_self)
            loss_align = (
                torch.norm(out.imu_to_dlc - out.dlc_self, dim=-1).mean()
                + torch.norm(out.dlc_to_imu - out.imu_self, dim=-1).mean()
            )
            loss_recon = 0.5 * (
                self._masked_l2(out.imu_recon, imu, mask)
                + self._masked_l2(out.dlc_recon, dlc, mask)
            )
            loss_total = loss_sup + loss_cs + loss_align + self.recon_weight * loss_recon

        metrics = {
            "align_l2": loss_align.detach(),
            "reconstruction_l2": loss_recon.detach(),
            "supervised_contrastive": loss_sup.detach(),
            "cs_divergence": loss_cs.detach(),
        }
        return loss_total, metrics

    def _train_epoch(
        self,
        unsup_batches: List[Dict[str, torch.Tensor]],
        sup_batches: List[Dict[str, torch.Tensor]],
        *,
        stage_name: str,
        prototypes: Optional[torch.Tensor],
        proto_weight: float,
        include_supervised: bool,
    ) -> Tuple[Dict[str, float], List[torch.Tensor], List[torch.Tensor]]:
        self.model.train()
        stats: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        pseudo_feats: List[torch.Tensor] = []
        pseudo_labels: List[torch.Tensor] = []

        if include_supervised:
            total_steps = max(len(unsup_batches), len(sup_batches))
        else:
            total_steps = len(unsup_batches)
        if total_steps == 0:
            return {}, pseudo_feats, pseudo_labels

        progress = tqdm(
            range(total_steps),
            desc=f"[{stage_name}] Epoch {self.total_epochs + 1}",
            leave=False,
        )
        for step in progress:
            if step < len(unsup_batches):
                loss_u, metrics_u, feats, pseudo = self._step_unsup(
                    unsup_batches[step], prototypes=prototypes, proto_weight=proto_weight
                )
                if loss_u is not None:
                    self._optimizer_step(loss_u)
                self._record_metrics(stats, counts, metrics_u, prefix="unsup_")
                if feats is not None and pseudo is not None:
                    pseudo_feats.append(feats.detach())
                    pseudo_labels.append(pseudo.detach())
                progress.set_postfix(
                    unsup_contrast=float(metrics_u.get("unsupervised_contrastive", 0.0))
                )

            if include_supervised and step < len(sup_batches):
                loss_s, metrics_s = self._step_sup(sup_batches[step])
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

        align_sum = stats.get("unsup_align_l2", 0.0) + stats.get("sup_align_l2", 0.0)
        align_count = counts.get("unsup_align_l2", 0) + counts.get("sup_align_l2", 0)
        recon_sum = stats.get("unsup_reconstruction_l2", 0.0) + stats.get("sup_reconstruction_l2", 0.0)
        recon_count = counts.get("unsup_reconstruction_l2", 0) + counts.get("sup_reconstruction_l2", 0)
        unsup_contrast_sum = stats.get("unsup_unsupervised_contrastive", 0.0)
        unsup_contrast_count = counts.get("unsup_unsupervised_contrastive", 0)
        sup_contrast_sum = stats.get("sup_supervised_contrastive", 0.0)
        sup_contrast_count = counts.get("sup_supervised_contrastive", 0)
        cs_sum = stats.get("unsup_cs_divergence", 0.0) + stats.get("sup_cs_divergence", 0.0)
        cs_count = counts.get("unsup_cs_divergence", 0) + counts.get("sup_cs_divergence", 0)
        proto_sum = stats.get("unsup_prototype_loss", 0.0)
        proto_count = counts.get("unsup_prototype_loss", 0)

        final_metrics: Dict[str, float] = {}
        final_metrics["alignL2_loss"] = align_sum / align_count if align_count > 0 else 0.0
        final_metrics["reconstructionL2_loss"] = (
            recon_sum / recon_count if recon_count > 0 else 0.0
        )
        final_metrics["unsupervised_contrastive_loss"] = (
            unsup_contrast_sum / unsup_contrast_count if unsup_contrast_count > 0 else 0.0
        )
        final_metrics["cs_divergence_loss"] = cs_sum / cs_count if cs_count > 0 else 0.0

        if include_supervised:
            final_metrics["supervised_contrastive_loss"] = (
                sup_contrast_sum / sup_contrast_count if sup_contrast_count > 0 else 0.0
            )
        else:
            final_metrics["prototype_loss"] = (
                proto_sum / proto_count if proto_count > 0 else 0.0
            )

        return final_metrics, pseudo_feats, pseudo_labels

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
            sup_batches = self._prepare_batches(
                dataset,
                batch_size,
                out_dir="Dataset_sup",
                use_unlabeled=False,
                mix=True,
                seed=42 + ep,
                n_workers_preproc=n_workers_preproc,
            )
            self.prototypes = self._compute_prototypes(sup_batches)
            metrics, pseudo_feats, pseudo_labels = self._train_epoch(
                unsup_batches,
                sup_batches,
                stage_name="Stage1",
                prototypes=self.prototypes,
                proto_weight=self.proto_weight,
                include_supervised=False,
            )
            if pseudo_feats:
                pf = torch.cat(pseudo_feats, dim=0)
                pl = torch.cat(pseudo_labels, dim=0)
                self.prototypes = self._compute_prototypes(sup_batches, pf, pl)

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

            metrics, _, _ = self._train_epoch(
                unsup_batches,
                sup_batches,
                stage_name="Stage2",
                prototypes=None,
                proto_weight=0.0,
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
    args = parser.parse_args()

    data_root = r"D:\\Jiaqi\\Datasets\\Rats\\TrainData_new"
    sessions = ["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_2"]
    batch_size = 512
    session_ranges = None

    train_ds = RatsWindowDataset(
        data_root,
        sessions,
        split="train",
        session_ranges=session_ranges,
    )

    num_feat_imu = train_ds.data[sessions[0]].imu.shape[1]
    num_feat_dlc = train_ds.data[sessions[0]].dlc.shape[1]
    num_sessions = len(sessions)

    trainer = TwoStageTrainer(
        num_feat_imu,
        num_feat_dlc,
        num_sessions,
        stage_lrs={1: args.lr_stage1, 2: args.lr_stage2},
    )

    n_workers_preproc = 1

    stage1_epochs = 30
    stage2_epochs = 60

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
            print(">>> Stage 1 (unsupervised + prototypes)...")
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
