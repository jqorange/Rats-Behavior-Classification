import pandas as pd

# ==== 输入路径（改成你的实际路径） ====
path_state = r"D:\Jiaqi\Projects\Rats-Behavior-Classification\representations\F3D5_outdoor_slds.csv"   # 列：center,state（center 就是 Index）
path_label = "D:\Jiaqi\Datasets\Rats\TrainData_new\labels\F3D5_outdoor\label_F3D5_outdoor.csv"          # 列：Index, walk, jump, ..., not_in_frame, unknown

# 读取
df_state = pd.read_csv(path_state)
df_label = pd.read_csv(path_label)

# 与标签合并（center 对应标签表里的 Index）
df = df_state.merge(df_label, left_on="center", right_on="Index", how="inner")

# 需要的动作列（按你给的顺序；去掉 not_in_frame 与 unknown）
all_cols = [
    "walk","jump","aiming","scratch","rearing","stand_up","still",
    "eating","grooming","local_search","turn_left","turn_right"
]
# （也可自动从 df_label 中推：）
# label_cols = [c for c in df_label.columns if c not in ["Index","not_in_frame","unknown"]]

# 对 state 分组，计算每个动作列的均值（= 占比）
grouped = df.groupby("state")

proportions = grouped[all_cols].mean()        # 0~1 之间的占比
counts = grouped.size().rename("n_samples")   # 每个 state 的样本数

# 汇总结果
result = counts.to_frame().join(proportions).reset_index().sort_values("state")

# 可选：也给出百分比形式
result_pct = result.copy()
for col in all_cols:
    result_pct[col] = (result_pct[col] * 100).round(2)

# 保存（可选）
result.to_csv("state_action_proportions_frac.csv", index=False)  # 占比(0~1)
result_pct.to_csv("state_action_proportions_pct.csv", index=False)  # 百分比(0~100%)

# 打印预览
print(result.head())