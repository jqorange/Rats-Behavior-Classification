import os
import pandas as pd
import numpy as np
def threshold(session_name):
    input_path = f'../prediction_prob/{session_name}.csv'
    output_path = f'../prediction_prob/{session_name}_pred_t.csv'

    # -----------------------------------------
    # Step 0: 读取原始预测文件
    # -----------------------------------------
    df_raw = pd.read_csv(input_path)
    df = df_raw.copy()

    # -----------------------------------------
    # Step 1: 第一轮 - 基础阈值二值化
    # -----------------------------------------
    high_thresh_cols = ["walk", "jump", "local_search"]
    for col in df.columns[1:]:
        if col in high_thresh_cols:
            df[col] = df[col].apply(lambda x: 1 if x > 0.95 else 0)
        if col in ["turn_left", "turn_right"]:
            df[col] = df[col].apply(lambda x: 1 if x > 0.8 else 0)
        else:
            df[col] = df[col].apply(lambda x: 1 if x > 0.9 else 0)

    # -----------------------------------------
    # Step 2: 第二轮 - 修复空行和仅 scratch 行
    # -----------------------------------------
    behavior_cols = df.columns[1:]  # 忽略 Index 列
    raw_behavior = df_raw[behavior_cols]

    # 2.1 找出：全为 0 或者只有 scratch 为 1 的行
    only_rearing_rows = (df[behavior_cols].sum(axis=1) == df["rearing"])
    zero_rows = df[behavior_cols].sum(axis=1) == 0
    target_rows = zero_rows | only_rearing_rows

    # 2.2 对这些行，仅对 backup_cols 中 >0.2 的置为 1，其余不变
    backup_cols = ["stand_up", "grooming","scratch","still","rearing", "aiming"]
    for col in backup_cols:
        df.loc[target_rows, col] = raw_behavior.loc[target_rows, col].apply(
            lambda x: 1 if x > 0.75 else 0
        )
    # 2.3 如果一行动作多过两个，则只取概率最高的两个
    bin_matrix = df[behavior_cols].values  # shape (N, C)，0/1 矩阵
    raw_matrix = raw_behavior.values  # shape (N, C)，原始概率

    # 找出哪些行超过 2 个 1
    row_sums = bin_matrix.sum(axis=1)
    mask = row_sums > 2
    rows_to_fix = np.where(mask)[0]

    # 对这些行，保留概率最高的两个
    for i in rows_to_fix:
        probs = raw_matrix[i]
        top2_idx = np.argsort(probs)[-2:]  # 概率最高的两个索引
        # 先清零，再保留 top2
        bin_matrix[i, :] = 0
        bin_matrix[i, top2_idx] = 1

    # 回写到 DataFrame
    df.loc[:, behavior_cols] = bin_matrix


    # -----------------------------------------
    # Step 3: 保存处理后的结果
    # -----------------------------------------
    df.to_csv(output_path, index=False)
    print(f"[✔] Processed: {session_name}")

# -----------------------------------------
# Step 4: 批量处理所有 session
# -----------------------------------------
session_names = [
    "F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_2"
]
for session in session_names:
    threshold(session)
