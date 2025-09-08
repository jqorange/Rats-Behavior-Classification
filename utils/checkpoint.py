import os
import warnings
from typing import Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    projector: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    stage: int,
    path: str,
) -> None:
    """保存模型、优化器和训练状态。

    Args:
        model: 主模型。
        projector: 最后的 projector 模块。
        optimizer: 当前使用的优化器。
        total_epochs: 目前的总训练轮数。
        stage: 当前阶段编号 (1/2/3)。
        path: 保存路径。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "projector_state": projector.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "total_epochs": total_epochs,
            "stage": stage,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    projector: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    *,
    expected_stage: Optional[int] = None,
) -> tuple[int, int]:
    """加载模型、优化器参数，并返回保存时的总轮数与阶段。

    Args:
        model: 主模型。
        projector: projector 模块。
        optimizer: 若提供，将加载优化器状态。
        path: checkpoint 路径。
        expected_stage: 当前训练阶段，可用于检测错载。

    Returns:
        tuple[int, int]: (total_epochs, stage)
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", {}))
    projector.load_state_dict(ckpt.get("projector_state", {}))
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    total_epochs = int(ckpt.get("total_epochs", 0))
    stage = int(ckpt.get("stage", 1))

    if expected_stage is not None and expected_stage > 1 and stage == 1:
        warnings.warn(
            f"Checkpoint from stage1 (total_epochs={total_epochs}) loaded for stage{expected_stage}.",
            RuntimeWarning,
        )

    return total_epochs, stage
