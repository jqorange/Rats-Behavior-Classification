import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tqdm
from models.losses import (
    compute_contrastive_losses,
    CenterLoss,
    prototype_repulsion_loss,
)
import random
from models.fusion import EncoderFusion
from models.classifier import MLPClassifier
from models.losses import multilabel_supcon_loss_bt, hierarchical_contrastive_loss
from utils.tools import take_per_row
class FusionTrainer:
    """Multi-modal fusion trainer with separate contrastive and MLP training phases"""

    def __init__(
            self,
            N_feat_A,
            N_feat_B,
            num_classes,
            mask_type='binomial',
            d_model=64,
            nhead=4,
            hidden_dim=128,
            device='cuda',
            lr_encoder=0.001,
            lr_classifier=0.001,
            batch_size=16,
            temporal_unit=0,
            contrastive_epochs=100,  # 对比学习的epoch数
            mlp_epochs=10,  # MLP训练的epoch数
            save_path=None,
            save_gap=5,
            n_stable=1,
            n_adapted=2,
            n_all=3,
            use_amp=False,
            num_sessions=0,
            projection_mode="aware",
            proto_repulsion_weight=0.1,
    ):
        """
        Args:
            N_feat_A: Input dimension for modality A
            N_feat_B: Input dimension for modality B
            num_classes: Number of output classes
            mask_type: Type of masking ('binomial' or 'continuous')
            d_model: Model dimension
            nhead: Number of attention heads
            hidden_dim: Hidden dimension for MLP classifier
            device: Device for training
            lr_encoder: Learning rate for encoder
            lr_classifier: Learning rate for classifier
            batch_size: Batch size
            temporal_unit: Minimum unit for temporal contrast
            contrastive_epochs: Number of epochs for contrastive learning phase
            mlp_epochs: Number of epochs for MLP training phase
            projection_mode: Adapter mode for the final projection layer
        """
        self.device = device
        self.batch_size = batch_size
        self.temporal_unit = temporal_unit
        self.num_classes = num_classes
        self.contrastive_epochs = contrastive_epochs
        self.mlp_epochs = mlp_epochs
        self.path_prefix = save_path
        self.save_gap = save_gap
        self.n_stable = n_stable
        self.n_adapted = n_adapted
        self.n_all = n_all
        self.d_model = d_model
        self.projection_mode = projection_mode
        self.proto_repulsion_weight = proto_repulsion_weight

        # AMP settings
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Models
        self.encoder_fusion = EncoderFusion(
            N_feat_A=N_feat_A,
            N_feat_B=N_feat_B,
            mask_type=mask_type,
            d_model=d_model,
            nhead=nhead,
            num_sessions=num_sessions,
            projection_mode=projection_mode,
        ).to(device)



        self.classifier = MLPClassifier(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=num_classes
        ).to(device)

        # Optimizers
        self.optimizer_encoder = torch.optim.AdamW(
            self.encoder_fusion.parameters(),
            lr=lr_encoder
        )
        self.optimizer_classifier = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr_classifier
        )




        # Loss function
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Use temperature < 1 to sharpen prototype soft labels
        self.center_loss_fn = CenterLoss(num_classes, d_model).to(device)
        self.optimizer_center = torch.optim.AdamW(self.center_loss_fn.parameters(), lr=lr_encoder)

        self.n_epochs = 0
        self.n_iters = 0

    def train_contrastive_phase(
            self,
            train_data_A,
            train_data_B,
            train_ids,
            train_data_sup_A=None,
            train_data_sup_B=None,
            sup_ids=None,
            labels_sup=None,
            verbose=True,
            stage="unsup",
            unsup_by_session=None,
    ):
        """Train contrastive phase for different stages.

        Args:
            train_data_A: concatenated unsupervised IMU data
            train_data_B: concatenated unsupervised DLC data
            train_data_sup_A: supervised IMU data
            train_data_sup_B: supervised DLC data
            labels_sup: supervised labels
            verbose: print progress
            stage: one of ``'unsup'``, ``'adapt'`` or ``'all'``
            unsup_by_session: optional dict mapping session name to (imu, dlc)
                arrays. When provided and ``stage=='adapt'`` the unsupervised
                batches are drawn from a single session at a time.
        """

        unsup_ds = TensorDataset(
            torch.from_numpy(train_data_A).float(),
            torch.from_numpy(train_data_B).float(),
        )

        if stage != 'unsup':
            sup_ds = TensorDataset(
                torch.from_numpy(train_data_sup_A).float(),
                torch.from_numpy(train_data_sup_B).float(),
                torch.from_numpy(labels_sup).long(),
                torch.from_numpy(sup_ids).long()
            )
            label_counts = labels_sup.sum(axis=0) + 1e-6
            label_freq = label_counts / label_counts.sum()
            sample_weights = (labels_sup @ (1.0 / label_freq)).astype(np.float32)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            sup_loader = DataLoader(sup_ds,
                                    batch_size=min(self.batch_size, len(sup_ds)),
                                    sampler=sampler,
                                    drop_last=True)
        else:
            sup_loader = None

        if stage == 'adapt' and unsup_by_session:
            unsup_loaders = []
            for imu, dlc, ids in unsup_by_session.values():
                ds = TensorDataset(
                    torch.from_numpy(imu).float(),
                    torch.from_numpy(dlc).float(),
                    torch.from_numpy(ids).long(),
                )
                loader = DataLoader(
                    ds,
                    batch_size=min(self.batch_size, len(ds)),
                    shuffle=True,
                    drop_last=True,
                )
                unsup_loaders.append(loader)
            total_unsup_batches = sum(len(l) for l in unsup_loaders)
        else:
            unsup_loader = DataLoader(
                unsup_ds,
                batch_size=min(self.batch_size, len(unsup_ds)),
                shuffle=True,
                drop_last=True,
            )
            unsup_loaders = [unsup_loader]
            total_unsup_batches = len(unsup_loader)

        contrastive_losses = []
        # ====== Stage 3 Training ======
        # This phase discards the unsupervised contrastive cropping used in
        # previous stages. Instead, it relies purely on supervised contrastive
        # learning with heavy Gaussian noise.  Pseudo labels are generated in a
        # FixMatch style using class prototypes.
        for epoch in range(self.contrastive_epochs):
            epoch_losses = {
                'sup': 0.0,
                'proto': 0.0,
                'repul': 0.0,
                'total': 0.0,
            }
            if stage == 'adapt':
                pseudo_feats_epoch = []
                pseudo_labels_epoch = []
                if not hasattr(self, 'stage2_prototypes'):
                    self.stage2_prototypes = self.compute_prototypes(
                        train_data_sup_A, train_data_sup_B, labels_sup
                    )
                prototypes = self.stage2_prototypes
            n_batches = 0
            if sup_loader is not None:
                sup_iter = iter(sup_loader)

            unsup_iters = [iter(l) for l in unsup_loaders]

            for _ in tqdm.tqdm(
                    range(total_unsup_batches),
                    desc=f'Contrastive Epoch {epoch + 1}/{self.contrastive_epochs}',
            ):
                idx = np.random.randint(len(unsup_loaders))
                try:
                    xA_u, xB_u, id_u = next(unsup_iters[idx])
                except StopIteration:
                    unsup_iters[idx] = iter(unsup_loaders[idx])
                    xA_u, xB_u, id_u = next(unsup_iters[idx])
                if sup_loader is not None:
                    try:
                        xA_s, xB_s, y_s, id_s = next(sup_iter)
                    except StopIteration:
                        sup_iter = iter(sup_loader)
                        xA_s, xB_s, y_s, id_s = next(sup_iter)
                    xA_s, xB_s, y_s, id_s = xA_s.to(self.device), xB_s.to(self.device), y_s.to(self.device), id_s.to(
                        self.device)
                    # 移动到设备
                xA_u, xB_u, id_u = xA_u.to(self.device), xB_u.to(self.device), id_u.to(self.device)

                # Forward pass with AMP
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    f_u = self.encoder_fusion(xA_u, xB_u, id_u)
                    if sup_loader is not None:
                        f_s = self.encoder_fusion(xA_s, xB_s, id_s)

                    if stage == 'all':
                        sup_loss = compute_contrastive_losses(
                            self, xA_s, xB_s, y_s, f_s, id_s, is_supervised=True
                        )
                        unsup_loss = compute_contrastive_losses(
                            self, xA_u, xB_u, None, f_u, id_u, is_supervised=False
                        )
                        loss = sup_loss + unsup_loss
                    elif stage == 'adapt':
                        unsup_loss = compute_contrastive_losses(
                            self, xA_u, xB_u, None, f_u, id_u, is_supervised=False, stage=2
                        )
                        if sup_loader is not None:
                            pooled_s = f_s.max(dim=1).values
                            logits_sup = torch.matmul(F.normalize(pooled_s, dim=-1), F.normalize(prototypes, dim=-1).T)
                            target_sup = y_s.argmax(dim=1)
                            proto_sup = F.cross_entropy(logits_sup, target_sup)
                        else:
                            proto_sup = torch.tensor(0.0, device=self.device)
                            pooled_s = None

                        pooled_u = f_u.max(dim=1).values
                        sims = torch.matmul(F.normalize(pooled_u, dim=-1), F.normalize(prototypes, dim=-1).T)
                        max_sims, pseudo = sims.max(dim=1)
                        mask_h = max_sims > 0.9
                        if mask_h.any():
                            proto_unsup = F.cross_entropy(sims[mask_h], pseudo[mask_h])
                            pseudo_feats_epoch.append(pooled_u[mask_h].detach())
                            pseudo_labels_epoch.append(pseudo[mask_h].detach())
                        else:
                            proto_unsup = torch.tensor(0.0, device=self.device)

                        proto_loss = proto_sup + proto_unsup
                        loss = 0.7*unsup_loss + 0.3*proto_loss
                    else:
                        unsup_loss = compute_contrastive_losses(
                            self, xA_u, xB_u, None, f_u, id_u, is_supervised=False
                        )
                        loss = unsup_loss

                # 反向更新
                self.optimizer_encoder.zero_grad()
                self.optimizer_center.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer_encoder)
                    if stage == "all":
                        self.scaler.step(self.optimizer_center)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer_encoder.step()
                    if stage == "all":
                        self.optimizer_center.step()

                # 记录
                if stage == 'all':
                    epoch_losses['sup'] += sup_loss.item()
                    epoch_losses['unsup'] += unsup_loss.item()
                    epoch_losses['total'] += loss.item()
                elif stage == 'adapt':
                    epoch_losses['unsup'] += unsup_loss.item()
                    epoch_losses['proto'] += proto_loss.item()
                    epoch_losses['total'] += loss.item()
                else:
                    epoch_losses['unsup'] += unsup_loss.item()
                    epoch_losses['total'] += loss.item()
                n_batches += 1
                self.n_iters += 1

            # 平均
            for k in epoch_losses:
                epoch_losses[k] /= n_batches
            contrastive_losses.append(epoch_losses)

            if verbose:
                print(f"Epoch {epoch + 1}: Total={epoch_losses['total']:.8f}")
                if stage == 'all':
                    print(f"Sup={epoch_losses['sup']:.8f}, Unsup={epoch_losses['unsup']:.8f}")
                elif stage == 'adapt':
                    print(f"Unsup={epoch_losses['unsup']:.8f}, Proto={epoch_losses['proto']:.8f}")
            if stage == 'adapt' and pseudo_feats_epoch:
                pseudo_feats_cat = torch.cat(pseudo_feats_epoch, dim=0)
                pseudo_labels_cat = torch.cat(pseudo_labels_epoch, dim=0)
                self.stage2_prototypes = self.compute_prototypes(
                    train_data_sup_A, train_data_sup_B, labels_sup,
                    extra_feats=pseudo_feats_cat,
                    extra_labels=pseudo_labels_cat,
                )
            self.n_epochs += 1

        return contrastive_losses

    def amplitude_scale(self, x, scale_range=(0.8, 1.2)):
        scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*scale_range)
        return x * scale

    def amplitude_shift(self, x, shift_range=(-0.2, 0.2)):
        shift = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*shift_range)
        return x + shift

    def strong_augment(self, x):
        # 高斯噪声
        x = x + torch.randn_like(x) * 0.3
        # 幅度扰动
        if random.random() < 0.5:
            x = self.amplitude_scale(x)
        # 偏置扰动
        if random.random() < 0.5:
            x = self.amplitude_shift(x)
        return x
    def train_stage3(self, train_data_A, train_data_B, train_ids,
                     train_data_sup_A, train_data_sup_B, sup_ids, labels_sup,
                     verbose=True):
        """Stage 3 training with supervised and unsupervised losses."""

        unsup_ds = TensorDataset(
            torch.from_numpy(train_data_A).float(),
            torch.from_numpy(train_data_B).float(),
            torch.from_numpy(train_ids).long(),
        )
        unsup_loader = DataLoader(
            unsup_ds,
            batch_size=min(self.batch_size, len(unsup_ds)),
            shuffle=True,
            drop_last=True,
        )

        if train_data_sup_A is not None:
            sup_ds = TensorDataset(
                torch.from_numpy(train_data_sup_A).float(),
                torch.from_numpy(train_data_sup_B).float(),
                torch.from_numpy(labels_sup).long(),
                torch.from_numpy(sup_ids).long(),
            )
            label_counts = labels_sup.sum(axis=0) + 1e-6
            label_freq = label_counts / label_counts.sum()
            sample_weights = (labels_sup @ (1.0 / label_freq)).astype(np.float32)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            sup_loader = DataLoader(
                sup_ds,
                batch_size=min(self.batch_size, len(sup_ds)),
                sampler=sampler,
                drop_last=True,
            )
            sup_iter = iter(sup_loader)
        else:
            sup_loader = None

        contrastive_losses = []

        for epoch in range(self.contrastive_epochs):
            epoch_losses = {
                'sup': 0.0,
                'proto': 0.0,
                'repul':0.0,
                'total': 0.0,
            }
            pseudo_feats_epoch = []
            pseudo_labels_epoch = []
            if not hasattr(self, 'stage2_prototypes'):
                self.stage2_prototypes = self.compute_prototypes(
                    train_data_sup_A, train_data_sup_B, labels_sup
                )
            prototypes = self.stage2_prototypes
            repulsion = prototype_repulsion_loss(prototypes)
            unsup_iter = iter(unsup_loader)

            for _ in tqdm.tqdm(range(len(unsup_loader)), desc=f'Stage3 Epoch {epoch+1}/{self.contrastive_epochs}'):
                try:
                    xA_u, xB_u, id_u = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(unsup_loader)
                    xA_u, xB_u, id_u = next(unsup_iter)

                xA_u, xB_u, id_u = xA_u.to(self.device), xB_u.to(self.device), id_u.to(self.device)

                if sup_loader is not None:
                    try:
                        xA_s, xB_s, y_s, id_s = next(sup_iter)
                    except StopIteration:
                        sup_iter = iter(sup_loader)
                        xA_s, xB_s, y_s, id_s = next(sup_iter)
                    xA_s, xB_s, y_s, id_s = xA_s.to(self.device), xB_s.to(self.device), y_s.to(self.device), id_s.to(self.device)
                else:
                    xA_s = xB_s = y_s = id_s = None

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # ===== Supervised batch =====
                    if sup_loader is not None:
                        noise_A_s = torch.randn_like(xA_s) * 0.2
                        noise_B_s = torch.randn_like(xB_s) * 0.2
                        f_s = self.encoder_fusion(xA_s + noise_A_s, xB_s + noise_B_s, id_s)
                        sup_loss = compute_contrastive_losses(
                            self, None, None, y_s, f_s, id_s, is_supervised=True, stage=3
                        )
                        pooled_s = f_s.max(dim=1).values
                        logits_sup = torch.matmul(
                            F.normalize(pooled_s, dim=-1),
                            F.normalize(prototypes, dim=-1).T,
                        )
                        target_sup = y_s.argmax(dim=1)
                        proto_sup = F.cross_entropy(logits_sup, target_sup)
                    else:
                        sup_loss = torch.tensor(0.0, device=self.device)
                        proto_sup = torch.tensor(0.0, device=self.device)

                    # ===== Unsupervised batch: FixMatch pseudo labelling =====
                    weak_std = 0.05
                    f_w = self.encoder_fusion(
                        xA_u + torch.randn_like(xA_u) * weak_std,
                        xB_u + torch.randn_like(xB_u) * weak_std,
                        id_u,
                    )
                    f_su = self.encoder_fusion(
                        self.strong_augment(xA_u),
                        self.strong_augment(xB_u),
                        id_u,
                    )

                    pooled_w = f_w.max(dim=1).values
                    sims = torch.matmul(
                        F.normalize(pooled_w, dim=-1),
                        F.normalize(prototypes, dim=-1).T,
                    )
                    max_sims, pseudo = sims.max(dim=1)
                    mask_h = max_sims > 0.95
                    if mask_h.any():
                        pseudo_onehot = F.one_hot(pseudo[mask_h], num_classes=self.num_classes).float()
                        sup_pseudo = compute_contrastive_losses(
                            self, None, None, pseudo_onehot, f_su[mask_h], id_u[mask_h], is_supervised=True, stage=3
                        )
                        logits_strong = torch.matmul(
                            F.normalize(f_su[mask_h].max(dim=1).values, dim=-1),
                            F.normalize(prototypes, dim=-1).T,
                        )
                        proto_unsup = F.cross_entropy(logits_strong, pseudo[mask_h])
                        pseudo_feats_epoch.append(f_su[mask_h].max(dim=1).values.detach())
                        pseudo_labels_epoch.append(pseudo[mask_h].detach())
                    else:
                        sup_pseudo = torch.tensor(0.0, device=self.device)
                        proto_unsup = torch.tensor(0.0, device=self.device)

                    sup_total = sup_loss + sup_pseudo
                    proto_loss = proto_sup + proto_unsup

                loss = 0.5 * sup_total + 0.5 * proto_loss + self.proto_repulsion_weight * repulsion

                self.optimizer_encoder.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer_encoder)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer_encoder.step()


                epoch_losses['sup'] += sup_total.item()
                epoch_losses['proto'] += proto_loss.item()
                epoch_losses['repul'] += repulsion.item()
                epoch_losses['total'] += loss.item()
                self.n_iters += 1

            for k in epoch_losses:
                epoch_losses[k] /= len(unsup_loader)
            contrastive_losses.append(epoch_losses)

            if verbose:
                print(
                    f"Stage3 Epoch {epoch + 1}: Total={epoch_losses['total']:.8f}, "
                    f"Sup={epoch_losses['sup']:.8f}, Proto={epoch_losses['proto']:.8f}, "
                    f"Repul={epoch_losses['repul']:.8f}"
                )

            if pseudo_feats_epoch:
                pseudo_feats_cat = torch.cat(pseudo_feats_epoch, dim=0)
                pseudo_labels_cat = torch.cat(pseudo_labels_epoch, dim=0)
                self.stage2_prototypes = self.compute_prototypes(
                    train_data_sup_A, train_data_sup_B, labels_sup,
                    extra_feats=pseudo_feats_cat,
                    extra_labels=pseudo_labels_cat,
                )

            self.n_epochs += 1

        return contrastive_losses
    def train_contrastive_multi_session(self, session_data, verbose=True):
        """Unsupervised contrastive training mixing sessions per batch."""

        loaders = []
        for imu, dlc, ids in session_data.values():
            ds = TensorDataset(
                torch.from_numpy(imu).float(),
                torch.from_numpy(dlc).float(),
                torch.from_numpy(ids).long(),
            )
            loader = DataLoader(
                ds,
                batch_size=min(self.batch_size, len(ds)),
                shuffle=True,
                drop_last=True,
            )
            loaders.append(loader)

        if not loaders:
            return []

        total_batches = sum(len(l) for l in loaders)
        contrastive_losses = []

        for epoch in range(self.contrastive_epochs):
            epoch_losses = {'unsup': 0.0, 'total': 0.0}
            iters = [iter(l) for l in loaders]

            for _ in tqdm.tqdm(range(total_batches)):
                idx = np.random.randint(len(loaders))
                try:
                    xA_u, xB_u, id_u = next(iters[idx])
                except StopIteration:
                    iters[idx] = iter(loaders[idx])
                    xA_u, xB_u, id_u = next(iters[idx])

                xA_u, xB_u, id_u = xA_u.to(self.device), xB_u.to(self.device), id_u.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = compute_contrastive_losses(
                        self, xA_u, xB_u, None, fused_repr=None, session_idx=id_u, is_supervised=False
                    )

                self.optimizer_encoder.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer_encoder)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer_encoder.step()

                epoch_losses['unsup'] += loss.item()
                epoch_losses['total'] += loss.item()
                self.n_iters += 1

            for k in epoch_losses:
                epoch_losses[k] /= total_batches
            contrastive_losses.append(epoch_losses)

            if verbose:
                print(
                    f"Epoch {epoch + 1}: Total={epoch_losses['total']:.8f}"
                )
            self.n_epochs += 1

        return contrastive_losses
    def train_mlp_phase(self, train_data_A, train_data_B, train_labels,
                        test_data_A, test_data_B, test_labels, verbose=True):
        """Train MLP phase and evaluate on test data for DWA update"""

        # Create dataset for MLP training
        mlp_dataset = TensorDataset(
            torch.from_numpy(train_data_A).float(),
            torch.from_numpy(train_data_B).float(),
            torch.from_numpy(train_labels).float()
        )
        mlp_loader = DataLoader(mlp_dataset, batch_size=min(self.batch_size, len(mlp_dataset)),
                                shuffle=True, drop_last=True)

        # Create test dataset
        test_dataset = TensorDataset(
            torch.from_numpy(test_data_A).float(),
            torch.from_numpy(test_data_B).float(),
            torch.from_numpy(test_labels).float()
        )
        test_loader = DataLoader(test_dataset, batch_size=min(self.batch_size, len(test_dataset)),
                                 shuffle=False, drop_last=False)
        mlp_losses = []

        # Freeze encoder
        for param in self.encoder_fusion.parameters():
            param.requires_grad = False

        for epoch in range(self.mlp_epochs):
            self.classifier.train()
            epoch_losses = {'bce': 0.0, 'total': 0.0}
            n_batches = 0

            for batch in tqdm.tqdm(mlp_loader, desc=f'MLP Epoch {epoch + 1}/{self.mlp_epochs}'):
                xA, xB, y = [b.to(self.device) for b in batch]

                with torch.no_grad():
                    fused_repr = self.encoder_fusion(xA, xB)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    predictions = self.classifier(fused_repr)  # (B, T, C)
                    y_expanded = y.unsqueeze(1).expand(-1, predictions.size(1), -1)
                    bce_loss = self.bce_loss(predictions, y_expanded)

                self.optimizer_classifier.zero_grad()
                if self.use_amp:
                    self.scaler.scale(bce_loss).backward()
                    self.scaler.step(self.optimizer_classifier)
                    self.scaler.update()
                else:
                    bce_loss.backward()
                    self.optimizer_classifier.step()

                epoch_losses['bce'] += bce_loss.item()
                epoch_losses['total'] += bce_loss.item()
                n_batches += 1

            for key in epoch_losses:
                epoch_losses[key] /= n_batches

            mlp_losses.append(epoch_losses)
            self.n_epochs += 1

            if verbose:
                print(f"MLP Epoch {epoch + 1}: Total={epoch_losses['total']:.4f}, BCE={epoch_losses['bce']:.4f}")

        # === Evaluation ===
        self.encoder_fusion.eval()
        self.classifier.eval()

        all_preds = []
        all_labels = []

        test_bce_losses = []

        with torch.no_grad():
            for batch in test_loader:
                xA, xB, y = [b.to(self.device) for b in batch]

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    fused_repr = self.encoder_fusion(xA, xB)
                    predictions = self.classifier(fused_repr)

                y_expanded = y.unsqueeze(1).expand(-1, predictions.size(1), -1)
                test_bce = self.bce_loss(predictions, y_expanded)
                test_bce_losses.append(test_bce.item())

                # ==== For metrics ====
                preds_np = (torch.sigmoid(predictions) > 0.5).cpu().numpy()  # binary predictions
                labels_np = y_expanded.cpu().numpy()

                all_preds.append(preds_np)
                all_labels.append(labels_np)

        # Aggregate results
        avg_test_bce = np.mean(test_bce_losses)
        y_pred_all = np.concatenate(all_preds, axis=0).reshape(-1, test_labels.shape[-1])
        y_true_all = np.concatenate(all_labels, axis=0).reshape(-1, test_labels.shape[-1])

        acc = accuracy_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

        if verbose:
            print(f"Test BCE: {avg_test_bce:.4f}")
            print(f"Test Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # Unfreeze encoder
        for param in self.encoder_fusion.parameters():
            param.requires_grad = True

        return mlp_losses, {
            'test_bce': avg_test_bce,
            'acc': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }

    def fit(self, unsup_sessions, train_data_A, train_data_B, train_ids,
            train_data_sup_A, train_data_sup_B, sup_ids, labels_sup,
            test_data_A, test_data_B, test_labels, verbose=True, start_epoch: int = 0):
        """Train the model following the three-stage curriculum."""
        assert train_data_A.shape[1] == train_data_B.shape[1]
        assert test_data_A.ndim == 3 and test_data_B.ndim == 3 and test_labels.ndim == 2
        assert test_data_A.shape[0] == test_data_B.shape[0] == test_labels.shape[0]
        assert test_data_A.shape[1] == test_data_B.shape[1]

        all_losses = {'contrastive': [], 'mlp': [], 'test_performance': []}

        # Unsupervised data uses all samples
        train_data_A_80 = train_data_A
        train_data_B_80 = train_data_B
        train_ids_80 = train_ids

        if train_data_sup_A is not None and len(train_data_sup_A):
            # Take the first 80% from each session individually for supervised data
            unique_ids = np.unique(sup_ids)
            sup_A_parts, sup_B_parts, label_parts, id_parts = [], [], [], []
            for sid in unique_ids:
                idxs = np.where(sup_ids == sid)[0]
                cut = int(0.8 * len(idxs))
                if cut == 0:
                    continue
                sel = idxs[:cut]
                sup_A_parts.append(train_data_sup_A[sel])
                sup_B_parts.append(train_data_sup_B[sel])
                label_parts.append(labels_sup[sel])
                id_parts.append(sup_ids[sel])
            if len(sup_A_parts):
                train_sup_A_80 = np.concatenate(sup_A_parts, axis=0)
                train_sup_B_80 = np.concatenate(sup_B_parts, axis=0)
                labels_sup_80 = np.concatenate(label_parts, axis=0)
                sup_ids_80 = np.concatenate(id_parts, axis=0)
            else:
                train_sup_A_80 = np.empty((0, *train_data_sup_A.shape[1:]), dtype=train_data_sup_A.dtype)
                train_sup_B_80 = np.empty((0, *train_data_sup_B.shape[1:]), dtype=train_data_sup_B.dtype)
                labels_sup_80 = np.empty((0, labels_sup.shape[1]), dtype=labels_sup.dtype)
                sup_ids_80 = np.empty((0,), dtype=sup_ids.dtype)
        else:
            train_sup_A_80 = train_data_sup_A
            train_sup_B_80 = train_data_sup_B
            sup_ids_80 = sup_ids
            labels_sup_80 = labels_sup

        # Unsupervised sessions keep all samples
        unsup_sessions_80 = unsup_sessions

        for epoch in range(start_epoch, self.n_all):
            if verbose:
                print(f"\n=== Epoch {epoch + 1}/{self.n_all} ===")

            if epoch < self.n_stable:
                for param in self.encoder_fusion.parameters():
                    param.requires_grad = True

                if verbose:
                    print("[Stage1] Mixing sessions")
                self.train_contrastive_multi_session(unsup_sessions, verbose=verbose)

            elif epoch < self.n_adapted:
                if epoch == self.n_stable+1:
                    self.init_stage2(unsup_sessions_80, train_data_A_80, train_data_B_80, train_ids_80,
                                    train_sup_A_80, train_sup_B_80, sup_ids_80, labels_sup_80,
                                    verbose=True)

                contrastive_losses = self.train_contrastive_phase(
                    train_data_A_80,
                    train_data_B_80,
                    train_ids_80,
                    train_sup_A_80,
                    train_sup_B_80,
                    sup_ids_80,
                    labels_sup_80,
                    verbose,
                    stage="adapt",
                    unsup_by_session=unsup_sessions_80,
                )
            else:
                if epoch == self.n_adapted + 1:
                    self.init_stage3(unsup_sessions_80, train_data_A_80, train_data_B_80, train_ids_80,
                                     train_sup_A_80, train_sup_B_80, sup_ids_80, labels_sup_80,
                                     verbose=True)

                for param in self.encoder_fusion.parameters():
                    param.requires_grad = True

                contrastive_losses = self.train_stage3(
                    train_data_A_80, train_data_B_80, train_ids_80,
                    train_sup_A_80, train_sup_B_80,
                    sup_ids_80, labels_sup_80, verbose
                )
                all_losses['contrastive'].extend(contrastive_losses)

                # mlp_losses, test_performance = self.train_mlp_phase(
                #     train_data_sup_A, train_data_sup_B, labels_sup,
                #     test_data_A, test_data_B, test_labels, verbose)
                # all_losses['mlp'].extend(mlp_losses)
                # all_losses['test_performance'].append(test_performance)

            if epoch % self.save_gap == 0:
                self.save(epoch)

        return all_losses

    def init_stage2(self, unsup_sessions, train_data_A, train_data_B, train_ids,
                   train_data_sup_A, train_data_sup_B, sup_ids, labels_sup,
                   verbose: bool = True):

        # Freeze entire model
        for p in self.encoder_fusion.parameters():
            p.requires_grad = False
        # Enable adapters
        for module in [self.encoder_fusion.encoderA.adapter,
                       self.encoder_fusion.encoderB.adapter,
                       self.encoder_fusion.projection]:
            for p in module.parameters():
                p.requires_grad = True

            # Ensure proper adapter modes for stage2
        self.encoder_fusion.encoderA.adapter.set_mode("align")
        self.encoder_fusion.encoderB.adapter.set_mode("align")
        self.encoder_fusion.projection.set_mode("align")

        # Optimizer over adapters only
        self.optimizer_encoder = torch.optim.AdamW(
            list(self.encoder_fusion.encoderA.adapter.parameters()) +
            list(self.encoder_fusion.encoderB.adapter.parameters()) +
            list(self.encoder_fusion.projection.parameters()),
            lr=self.optimizer_encoder.defaults['lr']
        )
        self.load_stage2(self.n_stable)

        return
    def init_stage3(self, unsup_sessions, train_data_A, train_data_B, train_ids,
                   train_data_sup_A, train_data_sup_B, sup_ids, labels_sup,
                   verbose: bool = True):

        # Freeze entire model
        for p in self.encoder_fusion.parameters():
            p.requires_grad = True
        # Enable adapters
        for module in [self.encoder_fusion.encoderA.adapter,
                       self.encoder_fusion.encoderB.adapter,
                       self.encoder_fusion.projection]:
            for p in module.parameters():
                p.requires_grad = True

            # Ensure proper adapter modes for stage2
        self.encoder_fusion.encoderA.adapter.set_mode("align")
        self.encoder_fusion.encoderB.adapter.set_mode("align")
        self.encoder_fusion.projection.set_mode("align")

        self.load(self.n_adapted)

        return

    def encode(self, data_A, data_B, mode="align", batch_size=None, pool=False):
        """
        Encode data using the trained fusion model

        Args:
            data_A: (N, T, feat_A) - Modality A data
            data_B: (N, T, feat_B) - Modality B data
            batch_size: Batch size for inference

        Returns:
            If pool is False: fused representations (N, T, D)
            If pool is True:  pooled features (N, 2, D)
        """
        if batch_size is None:
            batch_size = self.batch_size

        dataset = TensorDataset(
            torch.from_numpy(data_A).float(),
            torch.from_numpy(data_B).float()
        )
        loader = DataLoader(dataset, batch_size=batch_size)
        self.encoder_fusion.mask_type = None
        self.encoder_fusion.eval()
        self.encoder_fusion.encoderA.adapter.set_mode(mode)
        self.encoder_fusion.encoderB.adapter.set_mode(mode)
        self.encoder_fusion.projection.set_mode(mode)

        with torch.no_grad():
            outputs = []
            for batch in loader:
                xA, xB = batch
                xA = xA.to(self.device)
                xB = xB.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.encoder_fusion(xA, xB)
                if pool:
                    # global max pooling
                    global_feat = out.max(dim=1).values

                    out = global_feat.view(-1, self.d_model)
                outputs.append(out.cpu())

            output = torch.cat(outputs, dim=0)

        self.encoder_fusion.train()

        return output.numpy()
    def encode_state1(self, data_A, data_B, idx, batch_size=None, pool=False):
        """
                Encode data using the trained fusion model

                Args:
                    data_A: (N, T, feat_A) - Modality A data
                    data_B: (N, T, feat_B) - Modality B data
                    batch_size: Batch size for inference

                Returns:
                    If pool is False: fused representations (N, T, D)
                    If pool is True:  pooled features (N, 2, D)
                """
        if batch_size is None:
            batch_size = self.batch_size
        dataset = TensorDataset(
            torch.from_numpy(data_A).float(),
            torch.from_numpy(data_B).float()
        )
        loader = DataLoader(dataset, batch_size=batch_size)
        self.encoder_fusion.mask_type = None
        self.encoder_fusion.eval()

        with torch.no_grad():
            outputs = []
            for batch in loader:
                xA, xB = batch
                idxs = torch.full((len(xA),), idx, dtype=torch.long, device=self.device)
                xA = xA.to(self.device)
                xB = xB.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.encoder_fusion(xA, xB, idxs)
                if pool:
                    # global max pooling
                    global_feat = out.max(dim=1).values

                    out = global_feat.view(-1, self.d_model)
                outputs.append(out.cpu())

            output = torch.cat(outputs, dim=0)

        self.encoder_fusion.train()

        return output.numpy()
    def predict(self, data_A, data_B, batch_size=None):
        """
        Make predictions using the trained model

        Args:
            data_A: (N, T, feat_A) - Modality A data
            data_B: (N, T, feat_B) - Modality B data
            batch_size: Batch size for inference

        Returns:
            Predictions: (N, T, C) - Class probabilities
        """
        if batch_size is None:
            batch_size = self.batch_size

        dataset = TensorDataset(
            torch.from_numpy(data_A).float(),
            torch.from_numpy(data_B).float()
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        self.encoder_fusion.eval()
        self.classifier.eval()

        with torch.no_grad():
            outputs = []
            for batch in loader:
                xA, xB = batch
                xA = xA.to(self.device)
                xB = xB.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Encode
                    fused_repr = self.encoder_fusion(xA, xB)

                    # Classify
                    predictions = self.classifier(fused_repr)
                    predictions = torch.sigmoid(predictions)

                outputs.append(predictions.cpu())

            output = torch.cat(outputs, dim=0)

        self.encoder_fusion.train()
        self.classifier.train()
        return output.numpy()

    def compute_prototypes(self, data_A, data_B, labels, extra_feats=None, extra_labels=None, batch_size=None):
        """Compute class prototypes from labelled data and optional extra features."""
        if batch_size is None:
            batch_size = self.batch_size

        dataset = TensorDataset(
            torch.from_numpy(data_A).float(),
            torch.from_numpy(data_B).float(),
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        feats = []
        with torch.no_grad():
            for xA, xB in loader:
                xA, xB = xA.to(self.device), xB.to(self.device)
                out = self.encoder_fusion(xA, xB)
                pooled = out.max(dim=1).values
                feats.append(pooled.cpu())

        feats = torch.cat(feats, dim=0)
        labels_t = torch.from_numpy(labels).float()

        if extra_feats is not None and extra_labels is not None and len(extra_feats) > 0:
            feats = torch.cat([feats, extra_feats.cpu()], dim=0)
            extra_onehot = F.one_hot(extra_labels, num_classes=self.num_classes).float()
            labels_t = torch.cat([labels_t, extra_onehot.cpu()], dim=0)

        protos = []
        for c in range(self.num_classes):
            mask = labels_t[:, c] > 0
            if mask.any():
                protos.append(feats[mask].mean(0))
            else:
                protos.append(torch.zeros(self.d_model))

        return torch.stack(protos).to(self.device)

    def save(self, num):
        """Save the trained models"""
        torch.save({
            # 保存 encoderA 部分（包括 adapter + TCN + trans）
            'adapterA': self.encoder_fusion.encoderA.adapter.state_dict(),

            # 保存 encoderB 部分
            'adapterB': self.encoder_fusion.encoderB.adapter.state_dict(),

            'encoderA_rest': {
                k: v for k, v in self.encoder_fusion.encoderA.state_dict().items()
                if not k.startswith("adapter.")
            },
            'encoderB_rest': {
                k: v for k, v in self.encoder_fusion.encoderB.state_dict().items()
                if not k.startswith("adapter.")
            },
            'cross_attn': self.encoder_fusion.cross_attn.state_dict(),
            'gate': self.encoder_fusion.gate.state_dict(),
            'norm': self.encoder_fusion.norm.state_dict(),
            # 保存最终投影层
            'projection': self.encoder_fusion.projection.state_dict(),
        }, f"./{self.path_prefix}/encoder_{num}.pkl")


    def load(self, num):
        """Load the saved model parts into encoder_fusion"""
        import torch
        state_path = f"./{self.path_prefix}/encoder_{num}.pkl"
        state = torch.load(state_path, map_location='cpu')

        print(f"[INFO] Keys in loaded state_dict from {state_path}:")
        for key in state.keys():
            print(f"  - {key}: {type(state[key])}")

        # === 打印 adapterA / adapterB keys ===
        if 'adapterA' in state:
            print(f"[DEBUG] adapterA keys: {list(state['adapterA'].keys())}")
        if 'adapterB' in state:
            print(f"[DEBUG] adapterB keys: {list(state['adapterB'].keys())}")

        # === 加载 adapterA ===
        try:
            self.encoder_fusion.encoderA.adapter.load_state_dict(state['adapterA'], strict=False)
            # 检查 session_embed 是否加载成功
            if 'session_embed.weight' in state['adapterA']:
                saved_shape = state['adapterA']['session_embed.weight'].shape
                model_shape = self.encoder_fusion.encoderA.adapter.session_embed.weight.shape
                print(f"[CHECK] session_embed.weight shape - saved: {saved_shape}, model: {model_shape}")
                if saved_shape != model_shape:
                    print("[WARNING] session_embed weight shape mismatch – may cause invalid memory access.")
                else:
                    print("[INFO] session_embed.weight successfully loaded.")
            else:
                print("[WARNING] session_embed.weight not found in adapterA checkpoint.")
        except Exception as e:
            print(f"[ERROR] Failed to load adapterA: {e}")

        # === 加载 adapterB ===
        try:
            self.encoder_fusion.encoderB.adapter.load_state_dict(state['adapterB'], strict=False)
        except Exception as e:
            print(f"[ERROR] Failed to load adapterB: {e}")

        # === 加载 encoder 主体（不包括 adapter）===
        try:
            self.encoder_fusion.encoderA.load_state_dict(state['encoderA_rest'], strict=False)
        except Exception as e:
            print(f"[WARNING] Failed to load encoderA_rest: {e}")
        try:
            self.encoder_fusion.encoderB.load_state_dict(state['encoderB_rest'], strict=False)
        except Exception as e:
            print(f"[WARNING] Failed to load encoderB_rest: {e}")

        # === 加载跨模态融合部分 ===
        try:
            self.encoder_fusion.cross_attn.load_state_dict(state['cross_attn'], strict=False)
            self.encoder_fusion.gate.load_state_dict(state['gate'], strict=False)
            self.encoder_fusion.norm.load_state_dict(state['norm'], strict=False)
            if 'projection' in state:
                self.encoder_fusion.projection.load_state_dict(state['projection'], strict=False)
        except Exception as e:
            print(f"[WARNING] Failed to load fusion parts: {e}")

        print(f"[INFO] Successfully loaded model from encoder_{num}.pkl")

    def load_stage2(self, num):
        """Load pretrained weights for stage 2 without adapters or input projections."""
        import torch
        state_path = f"./{self.path_prefix}/encoder_{num}.pkl"
        state = torch.load(state_path, map_location="cpu")

        # Load encoder body excluding adapters
        try:
            encA_state = {k: v for k, v in state['encoderA_rest'].items() if not k.startswith('adapter.')}
            encB_state = {k: v for k, v in state['encoderB_rest'].items() if not k.startswith('adapter.')}
            self.encoder_fusion.encoderA.load_state_dict(encA_state, strict=False)
            self.encoder_fusion.encoderB.load_state_dict(encB_state, strict=False)
        except Exception as e:
            print(f"[WARNING] Failed to load encoder body: {e}")

        # Load fusion modules
        for part in ['cross_attn', 'gate', 'norm']:
            try:
                getattr(self.encoder_fusion, part).load_state_dict(state[part], strict=False)
            except Exception as e:
                print(f"[WARNING] Failed to load {part}: {e}")

        # Reinitialize adapters for stage 2
        for adapter in [self.encoder_fusion.encoderA.adapter,
                        self.encoder_fusion.encoderB.adapter,
                        self.encoder_fusion.projection]:
            for m in adapter.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

        # Stage2 uses session-align adapters before each encoder and an
        # alignment-only projection adapter without session embeddings.
        self.encoder_fusion.encoderA.adapter.set_mode("align")
        self.encoder_fusion.encoderB.adapter.set_mode("align")
        self.encoder_fusion.projection.set_mode("align")

        print(f"[INFO] Loaded stage1 weights for stage2 training from encoder_{num}.pkl")


