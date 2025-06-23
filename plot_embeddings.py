import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


def load_channel_names(base_path, session):
    csv_path = os.path.join(base_path, "labels", session, f"label_{session}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.columns[1:13].tolist()
    return []


def main(args):
    sessions = args.sessions
    rep_list = []
    label_list = []
    channel_names = []
    for s in sessions:
        rep_file = os.path.join(args.rep_dir, f"{s}_repr.npy")
        if not os.path.exists(rep_file):
            print(f"Representation for {s} not found")
            continue
        reps = np.load(rep_file)
        rep_list.append(reps)

        label_file = os.path.join(args.data_path, "labels", f"{s}",f"label_{s}.csv")
        if os.path.exists(label_file):
            labels = pd.read_csv(label_file).values[:, 1:13]
            if len(labels) > len(reps):
                labels = labels[: len(reps)]
            elif len(labels) < len(reps):
                reps = reps[: len(labels)]
            label_list.append(labels)
        if not channel_names:
            channel_names = load_channel_names(args.data_path, s)

    if not rep_list:
        print("No representations loaded")
        return

    data = np.concatenate(rep_list, axis=0)
    labels = np.concatenate(label_list, axis=0) if label_list else None

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_scaled)

    channel_names = channel_names or [str(i) for i in range(labels.shape[1] if labels is not None else 0)]
    channel_names.append("NAN")

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
    scatter_data = {name: ([], [], []) for name in channel_names}

    if labels is not None:
        for i in range(len(data_pca)):
            row_label = labels[i]
            idx = np.where(row_label == 1)[0]
            if len(idx) > 0:
                name = channel_names[idx[0]]
            else:
                name = "NAN"
            scatter_data[name][0].append(data_pca[i, 0])
            scatter_data[name][1].append(data_pca[i, 1])
            scatter_data[name][2].append(data_pca[i, 2])
    else:
        scatter_data["NAN"] = [data_pca[:, 0].tolist(), data_pca[:, 1].tolist(), data_pca[:, 2].tolist()]

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
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        title="3D PCA of Representations",
        legend=dict(x=1, y=1),
    )
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot representations")
    parser.add_argument("--data_path", default="D:\Homework//NLP project\ACC_DATA\ACC_DATA\TrainData", help="Base data path")
    parser.add_argument("--sessions", nargs="+", default=["F3D6_outdoor"], help="Session names")
    parser.add_argument("--rep_dir", default="representations", help="Representation directory")
    args = parser.parse_args()
    main(args)