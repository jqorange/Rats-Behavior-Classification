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
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.classifier import MLPClassifier
from models.fusion import EncoderFusion
from models.losses import (
    gaussian_kl_divergence,
    gaussian_kl_divergence_masked,
    hierarchical_contrastive_loss,
    multilabel_supcon_loss_bt,
    binary_focal_loss,
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
        self.multi_scales = [16, 32, 64, 128]
        self.unsup_loss_weights = {
            "contrast": 1.0,
            "kl": 0.1,
            "align": 0.1,
            "recon": 0.1,
        }
        self.sup_loss_weights = {
            "contrast": 1.0,
            "kl": 0.1,
            "align": 0.1,
            "recon": 0.1,
            "focal": 1.0,
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
        self.mlp: Optional[MLPClassifier] = None
        self.num_labels: int = 0

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _configure_optimizer(self, stage: int) -> None:
        params = list(self.model.parameters())
        if stage == 2 and self.mlp is not None:
            params += list(self.mlp.parameters())
        self.opt = torch.optim.Adam(params, lr=self.stage_lrs[stage])
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

    def _ensure_mlp(self, num_labels: int) -> None:
        if num_labels <= 0:
            raise ValueError("Number of labels must be positive for supervised stage")
        input_dim = len(self.multi_scales) * self.d_model
        hidden_dim = max(256, input_dim * 2)
        reinit = False
        if self.mlp is None:
            reinit = True
        elif num_labels != self.num_labels:
            reinit = True

        if reinit:
            self.num_labels = num_labels
            self.mlp = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_labels).to(self.device)
            if self.current_stage == 2:
                self._configure_optimizer(2)

    def load_mlp(self, path: str) -> None:
        if self.mlp is None:
            raise RuntimeError("MLP classifier must be initialised before loading weights")
        state = torch.load(path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        missing, unexpected = self.mlp.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[load_mlp] missing_keys: {missing}, unexpected_keys: {unexpected}")

    def _extract_session_windows(
        self,
        dataset: RatsWindowDataset,
        session: str,
        centres: List[int],
        T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = dataset.data[session]
        imu_rows = data.imu
        dlc_rows = data.dlc
        device = imu_rows.device
        centres_t = torch.tensor(centres, dtype=torch.long, device=device)
        starts = centres_t - T // 2
        ar = torch.arange(T, device=device)
        idx = starts.unsqueeze(1) + ar.view(1, -1)
        imu_mask = (idx >= 0) & (idx < imu_rows.size(0))
        idx_clamped = idx.clamp(0, imu_rows.size(0) - 1)
        imu = imu_rows.index_select(0, idx_clamped.reshape(-1)).reshape(len(centres), T, -1)
        dlc = dlc_rows.index_select(0, idx_clamped.reshape(-1)).reshape(len(centres), T, -1)
        imu = imu * imu_mask.unsqueeze(-1)
        dlc = dlc * imu_mask.unsqueeze(-1)
        return imu.float(), dlc.float(), imu_mask.bool()

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
        multi_scale: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        if multi_scale:
            return self._prepare_batches_multi_scale(
                dataset,
                batch_size,
                use_unlabeled=use_unlabeled,
                mix=mix,
                seed=seed,
            )
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

    def _prepare_batches_multi_scale(
        self,
        dataset: RatsWindowDataset,
        batch_size: int,
        *,
        use_unlabeled: bool,
        mix: bool,
        seed: int,
    ) -> List[Dict[str, torch.Tensor]]:
        rng = random.Random(seed)
        samples = dataset.unsup_samples if use_unlabeled else dataset.samples
        if not samples:
            return []

        session_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, (session, _label, _centre) in enumerate(samples):
            session_to_indices[session].append(idx)

        batches: List[Dict[str, torch.Tensor]] = []
        for session, indices in session_to_indices.items():
            rng.shuffle(indices)
            groups = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
            groups = [g for g in groups if len(g) == batch_size]
            for group in groups:
                centres = [samples[i][2] for i in group]
                labels = torch.stack([samples[i][1] for i in group], dim=0).float()
                session_idx = torch.full((batch_size,), dataset.session_to_idx[session], dtype=torch.long)
                imu_list: List[torch.Tensor] = []
                dlc_list: List[torch.Tensor] = []
                mask_list: List[torch.Tensor] = []
                for T in self.multi_scales:
                    imu_win, dlc_win, mask = self._extract_session_windows(dataset, session, centres, T)
                    imu_list.append(imu_win)
                    dlc_list.append(dlc_win)
                    mask_list.append(mask)
                batches.append(
                    {
                        "imu": imu_list,
                        "dlc": dlc_list,
                        "mask": mask_list,
                        "label": labels,
                        "session": session,
                        "session_idx": session_idx,
                    }
                )

        if mix:
            rng.shuffle(batches)

        return batches

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------
    def _step_unsup(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(batch.get("imu"), (list, tuple)):
            return self._step_unsup_multi(batch)
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
                "align_nll": zero,
                "align_kl": zero,
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
            loss_align_nll = 0.5 * (
                sequential_next_step_nll(imu_to_dlc, dlc_self)
                + sequential_next_step_nll(dlc_to_imu, imu_self)
            )
            loss_align_kl = 0.5 * (
                gaussian_kl_divergence(imu_to_dlc, dlc_self)
                + gaussian_kl_divergence(dlc_to_imu, imu_self)
            )

            recon_a = out1.imu_recon[:, -crop_l:]
            recon_b = out1.dlc_recon[:, -crop_l:]
            recon_a2 = out2.imu_recon[:, :crop_l]
            recon_b2 = out2.dlc_recon[:, :crop_l]
            loss_recon_nll = 0.25 * (
                sequential_next_step_nll(recon_a, imu_crop1[:, -crop_l:])
                + sequential_next_step_nll(recon_b, dlc_crop1[:, -crop_l:])
                + sequential_next_step_nll(recon_a2, imu_crop2[:, :crop_l])
                + sequential_next_step_nll(recon_b2, dlc_crop2[:, :crop_l])
            )
            loss_recon_kl = 0.25 * (
                gaussian_kl_divergence(recon_a, imu_crop1[:, -crop_l:])
                + gaussian_kl_divergence(recon_b, dlc_crop1[:, -crop_l:])
                + gaussian_kl_divergence(recon_a2, imu_crop2[:, :crop_l])
                + gaussian_kl_divergence(recon_b2, dlc_crop2[:, :crop_l])
            )

            unsup_w = self.unsup_loss_weights
            loss_unsup = (
                unsup_w.get("contrast", 1.0) * loss_contrast
                + unsup_w.get("align", 1.0) * loss_align_nll
                + unsup_w.get("kl", 1.0) * loss_align_kl
                + unsup_w.get("recon", 1.0) * (loss_recon_nll + loss_recon_kl)
            )

            loss_total = loss_unsup

        metrics = {
            "align_nll": loss_align_nll.detach(),
            "align_kl": loss_align_kl.detach(),
            "reconstruction_nll": loss_recon_nll.detach(),
            "reconstruction_kl": loss_recon_kl.detach(),
            "unsupervised_contrastive": loss_contrast.detach(),
        }
        return loss_total, metrics

    def _step_unsup_multi(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        imu_list = [self._sanitize(t.to(self.device)) for t in batch["imu"]]
        dlc_list = [self._sanitize(t.to(self.device)) for t in batch["dlc"]]
        session_idx = batch.get("session_idx")
        if session_idx is not None:
            session_idx = session_idx.to(self.device, dtype=torch.long)

        pooled_view1: List[torch.Tensor] = []
        pooled_view2: List[torch.Tensor] = []
        loss_align_nll = torch.tensor(0.0, device=self.device)
        loss_align_kl = torch.tensor(0.0, device=self.device)
        loss_recon_nll = torch.tensor(0.0, device=self.device)
        loss_recon_kl = torch.tensor(0.0, device=self.device)
        valid_scales = 0
        jitter_std = 0.01

        for imu, dlc in zip(imu_list, dlc_list):
            B, T, _ = imu.shape
            if T <= 5:
                continue

            crop_min = min(2 ** (self.temporal_unit + 1), T)
            crop_max = T
            if crop_max <= crop_min:
                crop_l = crop_max
            else:
                crop_l = int(np.random.randint(low=crop_min, high=crop_max + 1))
            crop_left = int(np.random.randint(0, max(T - crop_l + 1, 1)))
            crop_right = crop_left + crop_l
            crop_eleft = int(np.random.randint(0, crop_left + 1))
            crop_eright = int(np.random.randint(low=crop_right, high=T + 1))
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

            crop_len = int(crop_right - crop_left)
            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                out1 = self.model(imu_crop1, dlc_crop1, session_idx=session_idx)
                out2 = self.model(imu_crop2, dlc_crop2, session_idx=session_idx)

                emb1 = out1.fused[:, -crop_len:]
                emb2 = out2.fused[:, :crop_len]
                emb1 = F.normalize(emb1 + torch.randn_like(emb1) * jitter_std, dim=-1)
                emb2 = F.normalize(emb2 + torch.randn_like(emb2) * jitter_std, dim=-1)

                pooled1 = torch.amax(emb1, dim=1, keepdim=True)
                pooled2 = torch.amax(emb2, dim=1, keepdim=True)
                pooled1 = F.normalize(pooled1, dim=-1)
                pooled2 = F.normalize(pooled2, dim=-1)
                pooled_view1.append(pooled1)
                pooled_view2.append(pooled2)

                loss_align_nll = loss_align_nll + 0.5 * (
                    sequential_next_step_nll(out1.imu_to_dlc, out1.dlc_self)
                    + sequential_next_step_nll(out1.dlc_to_imu, out1.imu_self)
                )
                loss_align_kl = loss_align_kl + 0.5 * (
                    gaussian_kl_divergence(out1.imu_to_dlc, out1.dlc_self)
                    + gaussian_kl_divergence(out1.dlc_to_imu, out1.imu_self)
                )

                recon_a = out1.imu_recon[:, -crop_len:]
                recon_b = out1.dlc_recon[:, -crop_len:]
                recon_a2 = out2.imu_recon[:, :crop_len]
                recon_b2 = out2.dlc_recon[:, :crop_len]

                loss_recon_nll = loss_recon_nll + 0.25 * (
                    sequential_next_step_nll(recon_a, imu_crop1[:, -crop_len:])
                    + sequential_next_step_nll(recon_b, dlc_crop1[:, -crop_len:])
                    + sequential_next_step_nll(recon_a2, imu_crop2[:, :crop_len])
                    + sequential_next_step_nll(recon_b2, dlc_crop2[:, :crop_len])
                )
                loss_recon_kl = loss_recon_kl + 0.25 * (
                    gaussian_kl_divergence(recon_a, imu_crop1[:, -crop_len:])
                    + gaussian_kl_divergence(recon_b, dlc_crop1[:, -crop_len:])
                    + gaussian_kl_divergence(recon_a2, imu_crop2[:, :crop_len])
                    + gaussian_kl_divergence(recon_b2, dlc_crop2[:, :crop_len])
                )

            valid_scales += 1

        if valid_scales == 0 or not pooled_view1:
            zero = torch.tensor(0.0, device=self.device)
            metrics = {
                "align_nll": zero,
                "align_kl": zero,
                "reconstruction_nll": zero,
                "reconstruction_kl": zero,
                "unsupervised_contrastive": zero,
            }
            return None, metrics

        emb1_cat = torch.cat(pooled_view1, dim=-1)
        emb2_cat = torch.cat(pooled_view2, dim=-1)
        loss_contrast = hierarchical_contrastive_loss(emb1_cat, emb2_cat, temporal_unit=0)

        loss_align_nll = loss_align_nll / valid_scales
        loss_align_kl = loss_align_kl / valid_scales
        loss_recon_nll = loss_recon_nll / valid_scales
        loss_recon_kl = loss_recon_kl / valid_scales

        unsup_w = self.unsup_loss_weights
        loss_total = (
            unsup_w.get("contrast", 1.0) * loss_contrast
            + unsup_w.get("align", 1.0) * loss_align_nll
            + unsup_w.get("kl", 1.0) * loss_align_kl
            + unsup_w.get("recon", 1.0) * (loss_recon_nll + loss_recon_kl)
        )

        metrics = {
            "align_nll": loss_align_nll.detach(),
            "align_kl": loss_align_kl.detach(),
            "reconstruction_nll": loss_recon_nll.detach(),
            "reconstruction_kl": loss_recon_kl.detach(),
            "unsupervised_contrastive": loss_contrast.detach(),
        }
        return loss_total, metrics

    def _step_sup(self, batch: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(batch.get("imu"), (list, tuple)):
            return self._step_sup_multi(batch)
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
            loss_align_nll = 0.5 * (
                sequential_next_step_nll(out.imu_to_dlc, out.dlc_self)
                + sequential_next_step_nll(out.dlc_to_imu, out.imu_self)
            )
            loss_align_kl = 0.5 * (
                gaussian_kl_divergence(out.imu_to_dlc, out.dlc_self)
                + gaussian_kl_divergence(out.dlc_to_imu, out.imu_self)
            )
            loss_recon_nll = 0.5 * (
                sequential_next_step_nll(out.imu_recon, imu, mask)
                + sequential_next_step_nll(out.dlc_recon, dlc, mask)
            )
            loss_recon_kl = 0.5 * (
                gaussian_kl_divergence_masked(out.imu_recon, imu, mask)
                + gaussian_kl_divergence_masked(out.dlc_recon, dlc, mask)
            )
            sup_w = self.sup_loss_weights
            loss_total = (
                sup_w.get("contrast", 1.0) * loss_sup
                + sup_w.get("align", 1.0) * loss_align_nll
                + sup_w.get("kl", 1.0) * loss_align_kl
                + sup_w.get("recon", 1.0) * (loss_recon_nll + loss_recon_kl)
            )

        metrics = {
            "align_nll": loss_align_nll.detach(),
            "align_kl": loss_align_kl.detach(),
            "reconstruction_nll": loss_recon_nll.detach(),
            "reconstruction_kl": loss_recon_kl.detach(),
            "supervised_contrastive": loss_sup.detach(),
        }
        return loss_total, metrics

    def _step_sup_multi(self, batch: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.mlp is None:
            raise RuntimeError("MLP classifier has not been initialised")

        imu_list = [self._sanitize(t.to(self.device)) for t in batch["imu"]]
        dlc_list = [self._sanitize(t.to(self.device)) for t in batch["dlc"]]
        mask_data = batch.get("mask")
        if not isinstance(mask_data, (list, tuple)):
            mask_list = [None for _ in imu_list]
        else:
            mask_list = [m.to(self.device) if m is not None else None for m in mask_data]

        labels = batch["label"].to(self.device)
        session_idx = batch.get("session_idx")
        if session_idx is not None:
            session_idx = session_idx.to(self.device, dtype=torch.long)

        if self.class_weights is not None:
            weights = self.class_weights.to(labels.device)
            sample_w = (labels * weights).max(dim=1).values
            mix_idx = torch.multinomial(sample_w, labels.size(0), replacement=True)
            partner_idx = torch.randint(0, labels.size(0), (len(mix_idx),), device=labels.device)
            lam = 0.5
            label_mix = torch.clamp(labels[mix_idx] + labels[partner_idx], 0, 1)
            labels = torch.cat([labels, label_mix], dim=0)
            if session_idx is not None:
                session_mix = session_idx[mix_idx]
                session_idx = torch.cat([session_idx, session_mix], dim=0)

            new_imu: List[torch.Tensor] = []
            new_dlc: List[torch.Tensor] = []
            new_mask: List[Optional[torch.Tensor]] = []
            for imu, dlc, mask in zip(imu_list, dlc_list, mask_list):
                imu_mix = lam * imu[mix_idx] + (1 - lam) * imu[partner_idx]
                dlc_mix = lam * dlc[mix_idx] + (1 - lam) * dlc[partner_idx]
                imu = torch.cat([imu, imu_mix], dim=0)
                dlc = torch.cat([dlc, dlc_mix], dim=0)
                if mask is not None:
                    mask_mix = mask[mix_idx] & mask[partner_idx]
                    mask = torch.cat([mask, mask_mix], dim=0)
                new_imu.append(imu)
                new_dlc.append(dlc)
                new_mask.append(mask)
            imu_list = new_imu
            dlc_list = new_dlc
            mask_list = new_mask

        batch_size = labels.size(0)
        scale = torch.empty(batch_size, 1, 1, device=self.device).uniform_(0.8, 1.2)
        for i in range(len(imu_list)):
            imu_list[i] = self._sanitize(imu_list[i] * scale)
            dlc_list[i] = self._sanitize(dlc_list[i] * scale)

        loss_align_nll = torch.tensor(0.0, device=self.device)
        loss_align_kl = torch.tensor(0.0, device=self.device)
        loss_recon_nll = torch.tensor(0.0, device=self.device)
        loss_recon_kl = torch.tensor(0.0, device=self.device)
        pooled_embeddings: List[torch.Tensor] = []
        mlp_features: List[torch.Tensor] = []

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            for imu, dlc, mask in zip(imu_list, dlc_list, mask_list):
                out = self.model(imu, dlc, session_idx=session_idx, mask=mask)
                jitter_std = 0.01
                seq = out.fused + torch.randn_like(out.fused) * jitter_std
                seq = F.normalize(seq, dim=-1)
                pooled = torch.amax(seq, dim=1, keepdim=True)
                pooled = F.normalize(pooled, dim=-1)
                pooled_embeddings.append(pooled)
                mlp_features.append(torch.amax(out.fused, dim=1))

                loss_align_nll = loss_align_nll + 0.5 * (
                    sequential_next_step_nll(out.imu_to_dlc, out.dlc_self)
                    + sequential_next_step_nll(out.dlc_to_imu, out.imu_self)
                )
                loss_align_kl = loss_align_kl + 0.5 * (
                    gaussian_kl_divergence(out.imu_to_dlc, out.dlc_self)
                    + gaussian_kl_divergence(out.dlc_to_imu, out.imu_self)
                )
                loss_recon_nll = loss_recon_nll + 0.5 * (
                    sequential_next_step_nll(out.imu_recon, imu, mask)
                    + sequential_next_step_nll(out.dlc_recon, dlc, mask)
                )
                loss_recon_kl = loss_recon_kl + 0.5 * (
                    gaussian_kl_divergence_masked(out.imu_recon, imu, mask)
                    + gaussian_kl_divergence_masked(out.dlc_recon, dlc, mask)
                )

            emb = torch.cat(pooled_embeddings, dim=-1)
            loss_sup = multilabel_supcon_loss_bt(emb, labels)

            mlp_input = torch.cat(mlp_features, dim=-1)
            logits = self.mlp(mlp_input)
            loss_focal = binary_focal_loss(logits, labels)

        num_scales = max(len(pooled_embeddings), 1)
        loss_align_nll = loss_align_nll / num_scales
        loss_align_kl = loss_align_kl / num_scales
        loss_recon_nll = loss_recon_nll / num_scales
        loss_recon_kl = loss_recon_kl / num_scales

        sup_w = self.sup_loss_weights
        loss_total = (
            sup_w.get("contrast", 1.0) * loss_sup
            + sup_w.get("align", 1.0) * loss_align_nll
            + sup_w.get("kl", 1.0) * loss_align_kl
            + sup_w.get("recon", 1.0) * (loss_recon_nll + loss_recon_kl)
            + sup_w.get("focal", 1.0) * loss_focal
        )

        metrics = {
            "align_nll": loss_align_nll.detach(),
            "align_kl": loss_align_kl.detach(),
            "reconstruction_nll": loss_recon_nll.detach(),
            "reconstruction_kl": loss_recon_kl.detach(),
            "supervised_contrastive": loss_sup.detach(),
            "focal_loss": loss_focal.detach(),
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
        if include_supervised and self.mlp is not None:
            self.mlp.train()
        elif self.mlp is not None:
            self.mlp.eval()
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

        align_nll_sum = stats.get("unsup_align_nll", 0.0) + stats.get("sup_align_nll", 0.0)
        align_nll_count = counts.get("unsup_align_nll", 0) + counts.get("sup_align_nll", 0)
        align_kl_sum = stats.get("unsup_align_kl", 0.0) + stats.get("sup_align_kl", 0.0)
        align_kl_count = counts.get("unsup_align_kl", 0) + counts.get("sup_align_kl", 0)
        recon_nll_sum = stats.get("unsup_reconstruction_nll", 0.0) + stats.get("sup_reconstruction_nll", 0.0)
        recon_nll_count = counts.get("unsup_reconstruction_nll", 0) + counts.get("sup_reconstruction_nll", 0)
        recon_kl_sum = stats.get("unsup_reconstruction_kl", 0.0) + stats.get("sup_reconstruction_kl", 0.0)
        recon_kl_count = counts.get("unsup_reconstruction_kl", 0) + counts.get("sup_reconstruction_kl", 0)
        unsup_contrast_sum = stats.get("unsup_unsupervised_contrastive", 0.0)
        unsup_contrast_count = counts.get("unsup_unsupervised_contrastive", 0)
        sup_contrast_sum = stats.get("sup_supervised_contrastive", 0.0)
        sup_contrast_count = counts.get("sup_supervised_contrastive", 0)

        final_metrics: Dict[str, float] = {}
        final_metrics["alignNLL_loss"] = (
            align_nll_sum / align_nll_count if align_nll_count > 0 else 0.0
        )
        final_metrics["alignKL_loss"] = (
            align_kl_sum / align_kl_count if align_kl_count > 0 else 0.0
        )
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
        self._ensure_mlp(dataset.num_labels)
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
                multi_scale=True,
            )
            sup_batches = self._prepare_batches(
                dataset,
                batch_size,
                out_dir="Dataset_sup",
                use_unlabeled=False,
                mix=True,
                seed=5678 + ep,
                n_workers_preproc=n_workers_preproc,
                multi_scale=True,
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
                if self.mlp is not None:
                    os.makedirs("checkpoints_mlp", exist_ok=True)
                    torch.save(
                        {
                            "state_dict": self.mlp.state_dict(),
                            "total_epochs": self.total_epochs,
                        },
                        os.path.join("checkpoints_mlp", f"stage2_epoch{self.total_epochs}.pt"),
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-epoch", type=int, default=0, help="Resume training from given epoch")
    parser.add_argument("--lr-stage1", type=float, default=1e-4, help="Learning rate for stage 1")
    parser.add_argument("--lr-stage2", type=float, default=5e-5, help="Learning rate for stage 2")
    parser.add_argument("--split-seed", type=int, default=0, help="Random seed for segment-level train/test split")
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
        test_ratio = 0.8,
        split_seed=args.split_seed,
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
    stage2_epochs = 200

    start_epoch = args.resume_epoch
    start_stage = 1
    if start_epoch > stage1_epochs:
        start_stage = 2

    if start_epoch > 0:
        ckpt_stage = 1 if start_epoch <= stage1_epochs else 2
        ckpt_path = os.path.join("checkpoints", f"stage{ckpt_stage}_epoch{start_epoch}.pt")
        if os.path.exists(ckpt_path):
            if ckpt_stage == 2:
                trainer._ensure_mlp(train_ds.num_labels)
            trainer._configure_optimizer(ckpt_stage)
            trainer.load_from(ckpt_path, expected_stage=ckpt_stage, load_optimizer=(ckpt_stage == start_stage))
            if ckpt_stage == 2:
                mlp_path = os.path.join("checkpoints_mlp", f"stage2_epoch{start_epoch}.pt")
                if os.path.exists(mlp_path):
                    trainer.load_mlp(mlp_path)
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
