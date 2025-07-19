import numpy as np
import matplotlib.pyplot as plt
import torch

# 加载 attn_weights，如果是 tensor 就直接用
# attn_weights = torch.load("attn_weights.pt")   # 如果你保存的是 pt 文件
# 或者
attn_weights = np.load("../attn_map.npy")     # 如果你保存的是 numpy 文件

# 如果 attn_weights 是 Tensor
if isinstance(attn_weights, torch.Tensor):
    attn_weights = attn_weights.cpu().numpy()  # 转成 numpy

# attn_weights: (512, 41, 41)

def plot_single_attention(attn, idx=0, save_path=None):
    """
    attn: shape (batch, T_q, T_k)
    idx: 选择 batch/sample 的索引
    """
    A = attn[idx]  # (41, 41)
    plt.figure(figsize=(6,5))
    plt.imshow(A, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Attention Weight')
    plt.xlabel("Time in Modality B (Key/Value)")
    plt.ylabel("Time in Modality A (Query)")
    plt.title(f"Cross-Attention Map (Sample {idx})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# 例如画第0个样本
plot_single_attention(attn_weights, idx=30)
