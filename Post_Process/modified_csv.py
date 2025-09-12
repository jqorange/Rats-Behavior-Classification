import os
import pandas as pd
import numpy as np


def smooth_behavior_column(series, min_duration=10):
    """平滑一个二值行为列，去除短于 min_duration 的片段"""
    arr = series.values.copy()
    change_points = np.where(np.diff(arr) != 0)[0] + 1
    change_points = np.concatenate(([0], change_points, [len(arr)]))

    for i in range(1, len(change_points) - 1):
        start = change_points[i]
        end = change_points[i + 1]
        duration = end - start
        if duration < min_duration:
            arr[start:end] = arr[change_points[i - 1]]

    return pd.Series(arr, index=series.index)


def modify_all_labels(session_names, behavior_names, input_dir="../prediction_prob", output_dir="../prediction_prob"):
    for session in session_names:
        input_path = os.path.join(input_dir, f"{session}_pred_t.csv")
        output_path = os.path.join(output_dir, f"{session}_modified.csv")

        df = pd.read_csv(input_path)

        for name in behavior_names:
            df[name] = smooth_behavior_column(df[name], min_duration=5)

        # 不用重写 Index，除非真的需要
        df.to_csv(output_path, index=False)
        print(f"✅ Modified labels saved to {output_path}")


# === 用法 ===
session_names = [
    "F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_2"
]
behavior_names = [
    'walk', 'jump', 'aiming', 'scratch', 'rearing', 'stand_up',
    'still', 'eating', 'grooming', 'local_search',
    'turn_left', 'turn_right'
]

modify_all_labels(session_names, behavior_names)
