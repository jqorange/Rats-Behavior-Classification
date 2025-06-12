import torch
import numpy as np


def take_per_row(x, offset, length):
    """
    从每个样本（行）中提取指定位置和长度的子序列

    Args:
        x: torch.Tensor of shape (B, T, D) - 输入张量，B是batch大小，T是序列长度，D是特征维度
        offset: int or array-like of shape (B,) - 每个样本的起始位置偏移
        length: int - 要提取的序列长度

    Returns:
        torch.Tensor of shape (B, length, D) - 提取的子序列
    """
    B, T, D = x.shape
    device = x.device

    # 确保offset是tensor格式
    if isinstance(offset, (int, np.integer)):
        # 如果offset是单个整数，应用到所有样本
        offset = torch.full((B,), offset, dtype=torch.long, device=device)
    elif isinstance(offset, np.ndarray):
        # 如果是numpy数组，转换为tensor
        offset = torch.from_numpy(offset).long().to(device)
    elif isinstance(offset, (list, tuple)):
        # 如果是列表或元组，转换为tensor
        offset = torch.tensor(offset, dtype=torch.long, device=device)

    # 确保offset的形状正确
    if offset.shape[0] != B:
        raise ValueError(f"offset shape {offset.shape} doesn't match batch size {B}")

    # 创建索引矩阵
    # 为每个样本创建相对索引 [0, 1, 2, ..., length-1]
    relative_indices = torch.arange(length, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)  # (B, length)

    # 为每个样本添加对应的offset
    absolute_indices = relative_indices + offset.unsqueeze(1)  # (B, length)

    # 创建batch索引
    batch_indices = torch.arange(B, dtype=torch.long, device=device).unsqueeze(1).expand(-1, length)  # (B, length)

    # 使用高级索引提取数据
    # 限制索引范围，防止越界
    absolute_indices = torch.clamp(absolute_indices, 0, T - 1)

    # 提取数据
    result = x[batch_indices, absolute_indices]  # (B, length, D)

    return result


def take_per_row_safe(x, offset, length):
    """
    安全版本的take_per_row，会处理边界情况

    Args:
        x: torch.Tensor of shape (B, T, D)
        offset: int or array-like of shape (B,)
        length: int

    Returns:
        torch.Tensor of shape (B, length, D)
    """
    B, T, D = x.shape
    device = x.device

    # 确保offset是tensor格式
    if isinstance(offset, (int, np.integer)):
        offset = torch.full((B,), offset, dtype=torch.long, device=device)
    elif isinstance(offset, np.ndarray):
        offset = torch.from_numpy(offset).long().to(device)
    elif isinstance(offset, (list, tuple)):
        offset = torch.tensor(offset, dtype=torch.long, device=device)

    # 限制offset范围，确保不会越界
    offset = torch.clamp(offset, 0, max(0, T - length))

    # 创建结果张量
    result = torch.zeros(B, length, D, dtype=x.dtype, device=device)

    for i in range(B):
        start_idx = offset[i].item()
        end_idx = min(start_idx + length, T)
        actual_length = end_idx - start_idx

        if actual_length > 0:
            result[i, :actual_length] = x[i, start_idx:end_idx]
            # 如果实际长度小于所需长度，用最后一个值填充
            if actual_length < length:
                result[i, actual_length:] = x[i, end_idx - 1:end_idx].expand(length - actual_length, -1)

    return result


def take_per_row_vectorized(x, offset, length):
    """
    完全向量化的版本，性能更好
    """
    B, T, D = x.shape
    device = x.device

    # 处理offset
    if isinstance(offset, (int, np.integer)):
        offset = torch.full((B,), offset, dtype=torch.long, device=device)
    elif isinstance(offset, np.ndarray):
        offset = torch.from_numpy(offset).long().to(device)
    elif isinstance(offset, (list, tuple)):
        offset = torch.tensor(offset, dtype=torch.long, device=device)

    # 创建索引
    batch_idx = torch.arange(B, device=device)[:, None]  # (B, 1)
    seq_idx = torch.arange(length, device=device)[None, :]  # (1, length)

    # 计算绝对索引
    abs_idx = offset[:, None] + seq_idx  # (B, length)

    # 处理边界情况
    abs_idx = torch.clamp(abs_idx, 0, T - 1)

    # 提取数据
    result = x[batch_idx, abs_idx]  # (B, length, D)

    return result


# 测试函数
def test_take_per_row():
    """测试take_per_row函数"""
    # 创建测试数据
    B, T, D = 3, 10, 4
    x = torch.randn(B, T, D)

    print("Original tensor shape:", x.shape)
    print("Original tensor:")
    print(x)

    # 测试1: 单个offset
    offset1 = 2
    length1 = 5
    result1 = take_per_row(x, offset1, length1)
    print(f"\nTest 1 - Single offset {offset1}, length {length1}:")
    print("Result shape:", result1.shape)
    print("Result[0]:")
    print(result1[0])
    print("Expected (x[0, 2:7]):")
    print(x[0, 2:7])

    # 测试2: 不同的offset
    offset2 = [1, 3, 0]  # 每个样本不同的起始位置
    length2 = 4
    result2 = take_per_row(x, offset2, length2)
    print(f"\nTest 2 - Different offsets {offset2}, length {length2}:")
    print("Result shape:", result2.shape)
    for i in range(B):
        print(f"Result[{i}] (offset={offset2[i]}):")
        print(result2[i])
        print(f"Expected (x[{i}, {offset2[i]}:{offset2[i] + length2}]):")
        print(x[i, offset2[i]:offset2[i] + length2])
        print()


if __name__ == "__main__":
    test_take_per_row()