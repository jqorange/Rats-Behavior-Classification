import os
import pandas as pd
def threshold(session_name):
    # 读取原始CSV文件
    input_path = f'../probs_csv/{session_name}_probs.csv'
    output_path = f'../Predictions_results/{session_name}_pred.csv'

    df_raw = pd.read_csv(input_path)
    df = df_raw.copy()

    # 应用不同类别的阈值
    for col in df.columns[1:]:
        sub_thresh = 0.3
        df[col] = df[col].apply(lambda x: 1 if x > sub_thresh else 0)

    # === 找出所有行为标签全为 0 的行 ===
    label_cols = df.columns[1:]
    zero_rows = df[label_cols].sum(axis=1) == 0

    # === 将原始df中这些行中值 > 0.3 的列设置为1 ===
    df.loc[zero_rows, label_cols] = df_raw.loc[zero_rows, label_cols].applymap(lambda x: 1 if x > 0.2 else 0)

    # 保存结果
    df.to_csv(output_path, index=False)
session_names = [
    "F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor",
    "F5D10_outdoor", "F6D5_outdoor_2", "F6D5_outdoor_1"
]
for session in session_names:
    threshold(session)