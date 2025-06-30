import argparse
import os
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from utils.data_loader import DataLoader as SessionLoader
from utils.trainer import FusionTrainer
from models.losses import compute_contrastive_losses

def load_new_session(data_path: str, session: str):
    loader = SessionLoader([session], data_path)
    loader.load_all_data()
    imu_u = loader.train_IMU.get(session)
    dlc_u = loader.train_DLC.get(session)
    sup_imu = loader.train_sup_IMU.get(session)
    sup_dlc = loader.train_sup_DLC.get(session)
    labels = loader.train_labels.get(session)

    if imu_u is None or dlc_u is None:
        raise RuntimeError("Unsupervised data not found for session " + session)
    min_len = min(len(imu_u), len(dlc_u))
    imu_u = imu_u[:min_len].astype(np.float32)
    dlc_u = dlc_u[:min_len].astype(np.float32)

    if sup_imu is not None and sup_dlc is not None and labels is not None:
        min_sup = min(len(sup_imu), len(sup_dlc), len(labels))
        sup_imu = sup_imu[:min_sup].astype(np.float32)
        sup_dlc = sup_dlc[:min_sup].astype(np.float32)
        labels = labels[:min_sup].astype(np.int64)
    else:
        raise RuntimeError("Supervised data not found for session " + session)

    return imu_u, dlc_u, sup_imu, sup_dlc, labels


def freeze_except_adapters(trainer: FusionTrainer):
    for p in trainer.encoder_fusion.parameters():
        p.requires_grad = False
    for module in [trainer.encoder_fusion.encoderA.adapter,
                   trainer.encoder_fusion.encoderB.adapter]:
        for p in module.parameters():
            p.requires_grad = True
    for p in trainer.classifier.parameters():
        p.requires_grad = False
    trainer.optimizer_encoder = torch.optim.AdamW(
        list(trainer.encoder_fusion.encoderA.adapter.parameters()) +
        list(trainer.encoder_fusion.encoderB.adapter.parameters()),
        lr=trainer.optimizer_encoder.defaults.get('lr', 1e-4)
    )
    trainer.encoder_fusion.encoderA.adapter.set_mode("align")
    trainer.encoder_fusion.encoderB.adapter.set_mode("align")
    trainer.encoder_fusion.projection.set_mode("align")


def finetune(trainer: FusionTrainer, imu_u: np.ndarray, dlc_u: np.ndarray,
             sup_imu: np.ndarray, sup_dlc: np.ndarray, labels: np.ndarray,
             epochs: int, save_gap: int, save_dir: str):
    device = trainer.device
    os.makedirs(save_dir, exist_ok=True)
    trainer.path_prefix = save_dir
    trainer.contrastive_epochs = 1  # run one epoch per outer loop

    ids_unsup = np.zeros(len(imu_u), dtype=np.int64)
    ids_sup = np.zeros(len(sup_imu), dtype=np.int64)

    unsup_ds = TensorDataset(torch.from_numpy(imu_u).float(),
                             torch.from_numpy(dlc_u).float(),
                             torch.from_numpy(ids_unsup).long())
    unsup_loader = DataLoader(unsup_ds,
                              batch_size=min(trainer.batch_size, len(unsup_ds)),
                              shuffle=True, drop_last=True)

    label_counts = labels.sum(axis=0) + 1e-6
    label_freq = label_counts / label_counts.sum()
    sample_weights = (labels @ (1.0 / label_freq)).astype(np.float32)
    sup_ds = TensorDataset(torch.from_numpy(sup_imu).float(),
                           torch.from_numpy(sup_dlc).float(),
                           torch.from_numpy(labels).long(),
                           torch.from_numpy(ids_sup).long())
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)
    sup_loader = DataLoader(sup_ds,
                            batch_size=min(trainer.batch_size, len(sup_ds)),
                            sampler=sampler, drop_last=True)
    sup_iter = iter(sup_loader)

    for epoch in range(epochs):
        epoch_losses = {'sup': 0.0, 'unsup': 0.0, 'proto': 0.0, 'total': 0.0}
        unsup_iter = iter(unsup_loader)
        for _ in range(len(unsup_loader)):
            try:
                xA_u, xB_u, id_u = next(unsup_iter)
            except StopIteration:
                unsup_iter = iter(unsup_loader)
                xA_u, xB_u, id_u = next(unsup_iter)
            xA_u, xB_u, id_u = xA_u.to(device), xB_u.to(device), id_u.to(device)
            try:
                xA_s, xB_s, y_s, id_s = next(sup_iter)
            except StopIteration:
                sup_iter = iter(sup_loader)
                xA_s, xB_s, y_s, id_s = next(sup_iter)
            xA_s, xB_s, y_s, id_s = (xA_s.to(device), xB_s.to(device),
                                     y_s.to(device), id_s.to(device))

            with torch.cuda.amp.autocast(enabled=trainer.use_amp):
                f_u = trainer.encoder_fusion(xA_u, xB_u, id_u)
                unsup_loss = compute_contrastive_losses(
                    trainer, xA_u, xB_u, None, f_u, id_u, is_supervised=False)
                f_s = trainer.encoder_fusion(xA_s, xB_s, id_s)
                sup_loss = compute_contrastive_losses(
                    trainer, xA_s, xB_s, y_s, f_s, id_s, is_supervised=True, stage=3)
                pooled_s = f_s.max(dim=1).values
                pooled_u = f_u.max(dim=1).values
                if not trainer.prototype_memory.initialized:
                    trainer.prototype_memory.update(pooled_s.detach(), y_s.detach())
                pseudo = trainer.prototype_memory.assign_labels(pooled_u.detach())
                proto_loss = trainer.prototype_memory(pooled_u, pseudo)
                loss = 0.1 * sup_loss + 0.7 * unsup_loss + 0.2 * proto_loss

            trainer.optimizer_encoder.zero_grad()
            if trainer.use_amp:
                trainer.scaler.scale(loss).backward()
                trainer.scaler.step(trainer.optimizer_encoder)
                trainer.scaler.update()
            else:
                loss.backward()
                trainer.optimizer_encoder.step()

            with torch.no_grad():
                trainer.prototype_memory.update(pooled_s.detach(), y_s.detach(),
                                                pooled_u.detach(), pseudo)
            epoch_losses['unsup'] += unsup_loss.item()
            epoch_losses['proto'] += proto_loss.item()
            epoch_losses['total'] += loss.item()
            epoch_losses['sup'] += sup_loss.item()

        for k in epoch_losses:
            epoch_losses[k] /= len(unsup_loader)
        print(f"Epoch {epoch+1}/{epochs}: Total={epoch_losses['total']:.6f}, "
              f"Sup={epoch_losses['sup']:.6f}, Unsup={epoch_losses['unsup']:.6f}, "
              f"Proto={epoch_losses['proto']:.6f}")

        if (epoch + 1) % save_gap == 0:
            trainer.save(epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune adapters on new session")
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='directory containing pretrained checkpoints')
    parser.add_argument('--epoch', type=int, required=True, help='which checkpoint epoch to load')
    parser.add_argument('--data_path', type=str, required=True, help='path to dataset root')
    parser.add_argument('--session', type=str, required=True, help='session name to finetune on')
    parser.add_argument('--epochs', type=int, default=10, help='finetune epochs')
    parser.add_argument('--save_gap', type=int, default=5, help='save interval')
    parser.add_argument('--output_dir', default='checkpoints_Finetune', help='directory to save finetuned models')
    args = parser.parse_args()

    imu_u, dlc_u, sup_imu, sup_dlc, labels = load_new_session(args.data_path, args.session)
    num_classes = labels.shape[1]
    d_model = 64

    # determine num_sessions from checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, f"encoder_{args.epoch}.pkl")
    state = torch.load(ckpt_path, map_location='cpu')
    num_sessions = state['adapterA']['session_embed.weight'].shape[0]

    trainer = FusionTrainer(
        N_feat_A=imu_u.shape[2],
        N_feat_B=dlc_u.shape[2],
        num_classes=num_classes,
        d_model=d_model,
        nhead=4,
        hidden_dim=128,
        lr_encoder=1e-4,
        lr_classifier=1e-4,
        batch_size=32,
        contrastive_epochs=1,
        save_path=args.output_dir,
        save_gap=args.save_gap,
        n_stable=1,
        n_adapted=2,
        n_all=3,
        use_amp=True,
        num_sessions=num_sessions,
        projection_mode="align",
    )

    trainer.load(args.epoch)
    freeze_except_adapters(trainer)

    finetune(trainer, imu_u, dlc_u, sup_imu, sup_dlc, labels,
             epochs=args.epochs, save_gap=args.save_gap,
             save_dir=args.output_dir)

