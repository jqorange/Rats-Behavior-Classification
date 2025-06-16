import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from collections import deque
import tqdm
from models.losses import compute_contrastive_losses, compute_smoothness_loss
from models.fusion import EncoderFusion
from models.classifier import MLPClassifier
from models.dwa import DWAOptimizer
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
            smooth_window=5,
            dwa_temp=2.0,
            dwa_window=5,
            contrastive_epochs=100,  # 对比学习的epoch数
            mlp_epochs=10,  # MLP训练的epoch数
            save_path=None,
            save_gap=5,
            n_cycles=1,
            n_stable=1
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
            smooth_window: Window size for smoothness loss
            dwa_temp: Temperature for DWA
            dwa_window: Window size for DWA loss history
            contrastive_epochs: Number of epochs for contrastive learning phase
            mlp_epochs: Number of epochs for MLP training phase
        """
        self.device = device
        self.batch_size = batch_size
        self.temporal_unit = temporal_unit
        self.smooth_window = smooth_window
        self.num_classes = num_classes
        self.contrastive_epochs = contrastive_epochs
        self.mlp_epochs = mlp_epochs
        self.path_prefix = save_path
        self.save_gap = save_gap
        self.n_cycles = n_cycles
        self.n_stable = n_stable

        # Models
        self.encoder_fusion = EncoderFusion(
            N_feat_A=N_feat_A,
            N_feat_B=N_feat_B,
            mask_type=mask_type,
            out_dim=None,  # Keep sequence representation
            d_model=d_model,
            nhead=nhead
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

        # DWA optimizer for loss weights
        self.dwa = DWAOptimizer(num_tasks=2, temp=dwa_temp, window_size=dwa_window)

        # Loss function
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.n_epochs = 0
        self.n_iters = 0



    def train_contrastive_phase(self,
                                train_data_A, train_data_B,
                                train_data_sup_A, train_data_sup_B, labels_sup,
                                verbose=True, is_stable=False):
        """Train contrastive phase: unsup-driven, sup sampled per batch"""
        # 构造 Dataset 与 Loader
        sup_ds = TensorDataset(
            torch.from_numpy(train_data_sup_A).float(),
            torch.from_numpy(train_data_sup_B).float(),
            torch.from_numpy(labels_sup).long()
        )
        unsup_ds = TensorDataset(
            torch.from_numpy(train_data_A).float(),
            torch.from_numpy(train_data_B).float()
        )
        sup_loader = DataLoader(sup_ds,
                                batch_size=min(self.batch_size, len(sup_ds)),
                                shuffle=True,
                                drop_last=True)
        unsup_loader = DataLoader(unsup_ds,
                                  batch_size=min(self.batch_size, len(unsup_ds)),
                                  shuffle=True,
                                  drop_last=True)

        contrastive_losses = []
        for epoch in range(self.contrastive_epochs):
            epoch_losses = {'sup': 0.0, 'unsup': 0.0, 'total': 0.0}
            n_batches = 0
            sup_iter = iter(sup_loader)

            for xA_u, xB_u in tqdm.tqdm(unsup_loader,
                                        desc=f'Contrastive Epoch {epoch + 1}/{self.contrastive_epochs}'):
                # 每个无监督 batch 配一个有监督 batch
                try:
                    xA_s, xB_s, y_s = next(sup_iter)
                except StopIteration:
                    sup_iter = iter(sup_loader)
                    xA_s, xB_s, y_s = next(sup_iter)

                # 移动到设备
                xA_u, xB_u = xA_u.to(self.device), xB_u.to(self.device)
                xA_s, xB_s, y_s = xA_s.to(self.device), xB_s.to(self.device), y_s.to(self.device)



                # 前向：融合编码
                f_u = self.encoder_fusion(xA_u, xB_u)  # (B, T, D)
                f_s = self.encoder_fusion(xA_s, xB_s)  # (B, T, D)
                if is_stable:
                    # 计算两种对比损失
                    sup_loss= compute_contrastive_losses(self, xA_s, xB_s, y_s, f_s,is_supervised=True)
                    unsup_loss = compute_contrastive_losses(self, xA_u, xB_u, None, f_u,is_supervised=False)

                    alpha1, alpha2 = self.dwa.weights
                    loss = alpha1 * sup_loss + alpha2 * unsup_loss
                else:
                    unsup_loss = compute_contrastive_losses(self, xA_u, xB_u, None, f_u, is_supervised=False)
                    sup_loss = torch.tensor(0., device=unsup_loss.device, dtype=unsup_loss.dtype)
                    alpha1, alpha2 = self.dwa.weights
                    loss = unsup_loss
                # 反向更新
                self.optimizer_encoder.zero_grad()
                loss.backward()
                self.optimizer_encoder.step()

                # 记录
                epoch_losses['sup'] += sup_loss.item()
                epoch_losses['unsup'] += unsup_loss.item()
                epoch_losses['total'] += loss.item()
                n_batches += 1
                self.n_iters += 1

            # 平均
            for k in epoch_losses:
                epoch_losses[k] /= n_batches
            contrastive_losses.append(epoch_losses)

            if verbose:
                print(f"Epoch {epoch + 1}: Total={epoch_losses['total']:.8f}, "
                      f"Sup={epoch_losses['sup']:.8f}, Unsup={epoch_losses['unsup']:.8f}, "
                      f"α1={alpha1:.3f}, α2={alpha2:.3f}")

            self.n_epochs += 1

        return contrastive_losses

    def train_mlp_phase(self, train_data_A, train_data_B, train_labels,
                        test_data_A, test_data_B, test_labels, verbose=True):
        """Train MLP phase and evaluate on test data for DWA update"""

        # Create dataset for MLP training (same as contrastive data)
        mlp_dataset = TensorDataset(
            torch.from_numpy(train_data_A).float(),
            torch.from_numpy(train_data_B).float(),
            torch.from_numpy(train_labels).float()
        )
        mlp_loader = DataLoader(
            mlp_dataset,
            batch_size=min(self.batch_size, len(mlp_dataset)),
            shuffle=True,
            drop_last=True
        )

        # Create test dataset for evaluation
        test_dataset = TensorDataset(
            torch.from_numpy(test_data_A).float(),
            torch.from_numpy(test_data_B).float(),
            torch.from_numpy(test_labels).float()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(self.batch_size, len(test_dataset)),
            shuffle=False,
            drop_last=False
        )

        mlp_losses = []

        # Freeze encoder during MLP training
        for param in self.encoder_fusion.parameters():
            param.requires_grad = False

        for epoch in range(self.mlp_epochs):
            # Training phase
            epoch_losses = {'bce': 0.0, 'smooth': 0.0, 'total': 0.0}
            n_batches = 0

            self.classifier.train()
            for batch in tqdm.tqdm(mlp_loader, desc=f'MLP Epoch {epoch + 1}/{self.mlp_epochs}'):
                xA, xB, y = batch
                xA = xA.to(self.device)
                xB = xB.to(self.device)
                y = y.to(self.device)


                # Forward pass through frozen encoder
                with torch.no_grad():
                    fused_repr = self.encoder_fusion(xA, xB)  # (B, T, D)

                # Classification with MLP
                predictions = self.classifier(fused_repr)  # (B, T, C)

                # BCE loss
                y_expanded = y.unsqueeze(1).expand(-1, predictions.size(1), -1)  # (B, T, C)
                bce_loss = self.bce_loss(predictions, y_expanded)

                # Smoothness loss
                smooth_loss = compute_smoothness_loss(self, predictions)

                # Total MLP loss
                mlp_loss = bce_loss + 0.1 * smooth_loss

                # Backward pass - Classifier only
                self.optimizer_classifier.zero_grad()
                mlp_loss.backward()
                self.optimizer_classifier.step()

                # Logging
                epoch_losses['bce'] += bce_loss.item()
                try:
                    epoch_losses['smooth'] += smooth_loss.item()
                except:
                    print('smooth', smooth_loss)
                epoch_losses['total'] += mlp_loss.item()
                n_batches += 1

            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= n_batches

            mlp_losses.append(epoch_losses)

            if verbose:
                print(f"MLP Epoch {epoch + 1}: Total={epoch_losses['total']:.4f}, "
                      f"BCE={epoch_losses['bce']:.4f}, Smooth={epoch_losses['smooth']:.4f}")

            self.n_epochs += 1

        # Evaluate on test set to get performance for DWA update
        self.encoder_fusion.eval()
        self.classifier.eval()

        test_bce_losses = []
        test_smooth_losses = []

        with torch.no_grad():
            for batch in test_loader:
                xA, xB, y = batch
                xA = xA.to(self.device)
                xB = xB.to(self.device)
                y = y.to(self.device)

                # Forward pass
                fused_repr = self.encoder_fusion(xA, xB)
                predictions = self.classifier(fused_repr)

                # Compute test losses
                y_expanded = y.unsqueeze(1).expand(-1, predictions.size(1), -1)
                test_bce = self.bce_loss(predictions, y_expanded)
                test_smooth = compute_smoothness_loss(self, predictions)

                test_bce_losses.append(test_bce.item())
                test_smooth_losses.append(test_smooth.item() if isinstance(test_smooth, torch.Tensor) else float(test_smooth))


        # Average test performance
        avg_test_bce = np.mean(test_bce_losses)
        avg_test_smooth = np.mean(test_smooth_losses)



        # Unfreeze encoder after MLP training
        for param in self.encoder_fusion.parameters():
            param.requires_grad = True

        # Update DWA weights based on TEST performance
        dwa_weights = self.dwa.update([avg_test_bce, avg_test_smooth])

        if verbose:
            print(f"Test Performance - BCE: {avg_test_bce:.4f}, Smooth: {avg_test_smooth:.4f}")
            print(f"Updated DWA weights based on test performance: α1={dwa_weights[0]:.3f}, α2={dwa_weights[1]:.3f}")

        return mlp_losses, {'test_bce': avg_test_bce, 'test_smooth': avg_test_smooth}

    def fit(self, train_data_A, train_data_B,train_data_sup_A, train_data_sup_B, labels_sup,
            test_data_A, test_data_B, test_labels, verbose=True, start_cycle: int = 0):
        """
        Train the fusion model with alternating phases

        Args:
            train_data_A: (N, T, feat_A) - Modality A data for training (both contrastive and MLP)
            train_data_B: (N, T, feat_B) - Modality B data for training (both contrastive and MLP)
            labels: (N, C) - Multi-label binary targets for training
            test_data_A: (M, T, feat_A) - Modality A data for testing MLP performance
            test_data_B: (M, T, feat_B) - Modality B data for testing MLP performance
            test_labels: (M, C) - Multi-label binary targets for testing
            n_cycles: Number of contrastive+MLP cycles
            verbose: Whether to print training progress
        """
        assert train_data_A.shape[1] == train_data_B.shape[1]  # Same sequence length

        assert test_data_A.ndim == 3 and test_data_B.ndim == 3 and test_labels.ndim == 2
        assert test_data_A.shape[0] == test_data_B.shape[0] == test_labels.shape[0]
        assert test_data_A.shape[1] == test_data_B.shape[1]  # Same sequence length

        all_losses = {'contrastive': [], 'mlp': [], 'test_performance': []}
        is_stable = False
        for cycle in range(start_cycle, self.n_cycles):
            if cycle >= self.n_stable:
                is_stable = True

            if verbose:
                print(f"\n=== Training Cycle {cycle + 1}/{self.n_cycles} ===")

            # Phase 1: Contrastive Learning (using train data)
            if verbose:
                print("Phase 1: Contrastive Learning")
            contrastive_losses = self.train_contrastive_phase(
                train_data_A, train_data_B, train_data_sup_A, train_data_sup_B, labels_sup, verbose, is_stable
            )
            all_losses['contrastive'].extend(contrastive_losses)
            if is_stable:
            # Phase 2: MLP Training (using same train data) + Test Evaluation
                if verbose:
                    print("Phase 2: MLP Training + Test Evaluation")
                mlp_losses, test_performance = self.train_mlp_phase(
                    train_data_sup_A, train_data_sup_B, labels_sup,  # Same training data
                    test_data_A, test_data_B, test_labels,  # Different test data
                    verbose
                )
                all_losses['mlp'].extend(mlp_losses)
                all_losses['test_performance'].append(test_performance)
            if cycle % self.save_gap == 0:
                self.save(cycle)

        return all_losses

    def encode(self, data_A, data_B, batch_size=None):
        """
        Encode data using the trained fusion model

        Args:
            data_A: (N, T, feat_A) - Modality A data
            data_B: (N, T, feat_B) - Modality B data
            batch_size: Batch size for inference

        Returns:
            Fused representations: (N, T, D)
        """
        if batch_size is None:
            batch_size = self.batch_size

        dataset = TensorDataset(
            torch.from_numpy(data_A).float(),
            torch.from_numpy(data_B).float()
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        self.encoder_fusion.eval()

        with torch.no_grad():
            outputs = []
            for batch in loader:
                xA, xB = batch
                xA = xA.to(self.device)
                xB = xB.to(self.device)

                out = self.encoder_fusion(xA, xB)
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

    def save(self, num):
        """Save the trained models"""
        torch.save(self.encoder_fusion.state_dict(), f"./{self.path_prefix}/encoder_{num}.pkl")
        torch.save(self.classifier.state_dict(), f"./{self.path_prefix}/classifier_{num}.pkl")

    def load(self, num):
        """Load the trained models"""
        self.encoder_fusion.load_state_dict(
            torch.load(f"./{self.path_prefix}/encoder_{num}.pkl", map_location=self.device)
        )
        self.classifier.load_state_dict(
            torch.load(f"./{self.path_prefix}/classifier_{num}.pkl", map_location=self.device)
        )