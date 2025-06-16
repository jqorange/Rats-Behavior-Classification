import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from tqdm import tqdm

from utils.data_loader import DataLoader


def load_sup_data(base_path, sessions, split_num):
    """Load supervised train/test data using DataLoader."""
    loader = DataLoader(sessions, base_path)
    loader.load_all_data()

    train_sessions = sessions[:split_num]
    test_sessions = sessions[split_num:]

    def gather(sessions_list):
        imu_list = []
        dlc_list = []
        label_list = []
        for s in sessions_list:
            sup_imu = loader.train_sup_IMU.get(s)
            sup_dlc = loader.train_sup_DLC.get(s)
            lbl = loader.train_labels.get(s)
            if sup_imu is None or sup_dlc is None or lbl is None:
                continue
            imu_list.append(sup_imu)
            dlc_list.append(sup_dlc)
            label_list.append(lbl)
        if not imu_list:
            empty3 = np.empty((0, 0, 0))
            return empty3, empty3, np.empty((0, 0))
        return (
            np.concatenate(imu_list, axis=0),
            np.concatenate(dlc_list, axis=0),
            np.concatenate(label_list, axis=0),
        )

    train_sup_imu, train_sup_dlc, train_labels = gather(train_sessions)
    test_sup_imu, test_sup_dlc, test_labels = gather(test_sessions)

    return (
        train_sup_imu,
        train_sup_dlc,
        train_labels,
        test_sup_imu,
        test_sup_dlc,
        test_labels,
    )


def flatten_data(data):
    """Average temporal dimension to obtain (N, F) features."""
    if data.size == 0:
        return np.empty((0, 0))
    return data.mean(axis=1)


def label_indices(labels):
    """Convert multi-hot labels to single index, -1 if none."""
    if labels.ndim == 1:
        return labels
    idx = []
    for row in labels:
        ones = np.where(row == 1)[0]
        idx.append(ones[0] if len(ones) > 0 else -1)
    return np.array(idx)


def plot_pca(train_sup, train_labels, test_sup, test_labels, title):
    """Perform PCA on train/test data and plot in 3D."""
    train_feat = flatten_data(train_sup)
    test_feat = flatten_data(test_sup)
    all_feat = np.vstack([train_feat, test_feat])

    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(all_feat)

    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat_scaled)

    train_pca = feat_pca[: len(train_feat)]
    test_pca = feat_pca[len(train_feat) :]

    train_idx = label_indices(train_labels)
    test_idx = label_indices(test_labels)
    all_idx = np.hstack([train_idx, test_idx])

    n_classes = int(max(all_idx.max(), 0)) + 1
    df = pd.read_csv(r"D:\Jiaqi\TrainData\labels\F5D2_outdoor\label_F5D2_outdoor.csv")

    # Extract column names, excluding the first one, and append 'NAN'
    channel_names = df.columns[1:].tolist() + ["NAN"]

    custom_colors = [
        "rgb(255, 0, 0)",
        "rgb(0, 255, 0)",
        "rgb(0, 0, 255)",
        "rgb(255, 255, 0)",
        "rgb(0, 255, 255)",
        "rgb(255, 0, 255)",
        "rgb(128, 128, 128)",
        "rgb(255, 165, 0)",
        "rgb(75, 0, 130)",
        "rgb(255, 105, 180)",
        "rgb(34, 139, 34)",
        "rgb(255, 20, 147)",
        "rgb(238, 130, 238)",
        "rgb(255, 69, 0)",
        "rgb(60, 179, 113)",
        "rgb(186, 85, 211)",
        "rgb(128, 0, 128)",
        "rgb(0, 191, 255)",
        "rgb(139, 69, 19)",
        "rgb(255, 222, 173)",
    ]
    color_map = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(channel_names)}

    scatter_data = {name: [[], [], []] for name in channel_names}
    all_pca = np.vstack([train_pca, test_pca])
    for i in range(all_pca.shape[0]):
        idx = all_idx[i]
        name = channel_names[idx] if idx >= 0 and idx < len(channel_names) - 1 else "NAN"
        scatter_data[name][0].append(all_pca[i, 0])
        scatter_data[name][1].append(all_pca[i, 1])
        scatter_data[name][2].append(all_pca[i, 2])

    fig = go.Figure()
    for name in channel_names:
        if scatter_data[name][0]:
            fig.add_trace(
                go.Scatter3d(
                    x=scatter_data[name][0],
                    y=scatter_data[name][1],
                    z=scatter_data[name][2],
                    mode="markers",
                    marker=dict(size=2, color=color_map[name]),
                    name=name,
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="PCA 1",
            yaxis_title="PCA 2",
            zaxis_title="PCA 3",
        ),
        title=title,
        legend=dict(x=1, y=1),
    )
    fig.show()

#["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor","F5D10_outdoor", "F6D5_outdoor_1", "F6D5_outdoor_2"]
def main():
    parser = argparse.ArgumentParser(description="Visualize supervised data using PCA")
    parser.add_argument("--data_path", required=False, default="D:\Jiaqi\TrainData", help="Base data directory")
    parser.add_argument("--sessions", nargs="+", required=False, default=["F3D5_outdoor","F5D10_outdoor"], help="Session names")
    parser.add_argument("--split", type=int, default=1, help="Number of sessions used for training")
    args = parser.parse_args()

    (
        train_sup_imu,
        train_sup_dlc,
        train_labels,
        test_sup_imu,
        test_sup_dlc,
        test_labels,
    ) = load_sup_data(args.data_path, args.sessions, args.split)

    plot_pca(
        train_sup_imu,
        train_labels,
        test_sup_imu,
        test_labels,
        title="3D PCA of Supervised IMU Data",
    )

    plot_pca(
        train_sup_dlc,
        train_labels,
        test_sup_dlc,
        test_labels,
        title="3D PCA of Supervised DLC Data",
    )


if __name__ == "__main__":
    main()