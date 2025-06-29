import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import plotly.graph_objects as go


def load_channel_names(base_path, session):
    csv_path = os.path.join(base_path, "labels", session, f"label_{session}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.columns[1:13].tolist()
    return []


def compute_cluster_metrics(data_pca, labels, channel_names):
    results = {}
    valid_indices = np.array([np.argmax(row) if row.sum() > 0 else -1 for row in labels])

    # Remove NANs
    mask = valid_indices != -1
    filtered_data = data_pca[mask]
    filtered_labels = valid_indices[mask]

    # 1. Silhouette Score
    if len(np.unique(filtered_labels)) > 1:
        sil_score = silhouette_score(filtered_data, filtered_labels, metric="cosine")
    else:
        sil_score = float("nan")

    # 2. Center-to-sample cosine similarity
    label_to_samples = defaultdict(list)
    for i, label in enumerate(filtered_labels):
        label_to_samples[label].append(filtered_data[i])

    cos_sims = []
    for label, samples in label_to_samples.items():
        samples = np.vstack(samples)
        center = samples.mean(axis=0, keepdims=True)
        sims = cosine_similarity(samples, center)
        cos_sims.extend(sims.flatten())
    avg_cos_sim = np.mean(cos_sims)

    # 3. Intra-class and inter-class distance
    centers = {}
    for label, samples in label_to_samples.items():
        samples = np.vstack(samples)
        centers[label] = samples.mean(axis=0)

    # Intra-class
    intra_dist = 0.0
    total_count = 0
    for label, samples in label_to_samples.items():
        samples = np.vstack(samples)
        dists = np.linalg.norm(samples - centers[label], axis=1)
        intra_dist += dists.sum()
        total_count += len(samples)
    intra_avg = intra_dist / total_count

    # Inter-class
    center_list = list(centers.values())
    inter_dists = []
    for i in range(len(center_list)):
        for j in range(i + 1, len(center_list)):
            dist = np.linalg.norm(center_list[i] - center_list[j])
            inter_dists.append(dist)
    inter_avg = np.mean(inter_dists) if inter_dists else float("nan")

    results["Silhouette Score"] = sil_score
    results["Avg Center-to-Sample Cosine Sim"] = avg_cos_sim
    results["Intra/Inter Distance Ratio"] = intra_avg / inter_avg if inter_avg > 0 else float("nan")

    return results


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

        label_file = os.path.join(args.data_path, "labels", f"{s}", f"label_{s}.csv")
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
        "rgb(255, 0, 0)", "rgb(0, 255, 0)", "rgb(0, 0, 255)", "rgb(255, 255, 0)",
        "rgb(0, 255, 255)", "rgb(255, 0, 255)", "rgb(128, 128, 128)", "rgb(255, 165, 0)",
        "rgb(75, 0, 130)", "rgb(255, 105, 180)", "rgb(34, 139, 34)", "rgb(255, 20, 147)",
        "rgb(238, 130, 238)", "rgb(255, 69, 0)", "rgb(60, 179, 113)", "rgb(186, 85, 211)",
        "rgb(128, 0, 128)", "rgb(0, 191, 255)", "rgb(139, 69, 19)", "rgb(255, 222, 173)",
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

    # ✅ 打印结构性评估指标
    if labels is not None:
        metrics = compute_cluster_metrics(data_pca, labels, channel_names[:-1])  # exclude "NAN"
        print("\n=== Cluster Evaluation Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot representations with evaluation")
    parser.add_argument("--data_path", default="D:\Jiaqi\Datasets\Rats\TrainData_1",
                        help="Base data path")
    parser.add_argument("--sessions", nargs="+", default=["F5D10_outdoor"], help="Session names")
    parser.add_argument("--rep_dir", default="representations", help="Representation directory")
    args = parser.parse_args()
    main(args)
