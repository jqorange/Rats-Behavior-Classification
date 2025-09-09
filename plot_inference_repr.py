import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


LABEL_COLUMNS = [
    "walk",
    "jump",
    "aiming",
    "scratch",
    "rearing",
    "stand_up",
    "still",
    "eating",
    "local_search",
    "turn_left",
    "turn_right",
]


def load_label_ranges(results_txt_path: str):
    """Load labeled index ranges from results.txt."""
    session_ranges = {}
    with open(results_txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            csv_path, range_str = parts[0], parts[1]
            session_name = os.path.basename(csv_path).replace("label_", "").replace(
                ".csv", ""
            )
            ranges = eval(range_str)
            session_ranges[session_name] = ranges
    return session_ranges


def compute_cluster_metrics(data_pca, labels):
    """Compute clustering quality metrics for labeled data."""
    results = {}
    valid_idx = np.array([np.argmax(row) if row.sum() > 0 else -1 for row in labels])
    mask = valid_idx != -1
    filtered_data = data_pca[mask]
    filtered_labels = valid_idx[mask]

    if len(np.unique(filtered_labels)) > 1:
        sil_score = silhouette_score(filtered_data, filtered_labels, metric="cosine")
    else:
        sil_score = float("nan")

    label_to_samples = defaultdict(list)
    for i, lab in enumerate(filtered_labels):
        label_to_samples[lab].append(filtered_data[i])

    cos_sims = []
    centers = {}
    for lab, samples in label_to_samples.items():
        samples = np.vstack(samples)
        center = samples.mean(axis=0, keepdims=True)
        sims = cosine_similarity(samples, center)
        cos_sims.extend(sims.flatten())
        centers[lab] = center[0]
    avg_cos_sim = np.mean(cos_sims)

    intra = 0.0
    total = 0
    for lab, samples in label_to_samples.items():
        samples = np.vstack(samples)
        dists = np.linalg.norm(samples - centers[lab], axis=1)
        intra += dists.sum()
        total += len(samples)
    intra_avg = intra / total

    center_list = list(centers.values())
    inter_dists = []
    for i in range(len(center_list)):
        for j in range(i + 1, len(center_list)):
            inter_dists.append(
                np.linalg.norm(center_list[i] - center_list[j])
            )
    inter_avg = np.mean(inter_dists) if inter_dists else float("nan")

    results["Silhouette Score"] = sil_score
    results["Avg Center-to-Sample Cosine Sim"] = avg_cos_sim
    results["Intra/Inter Distance Ratio"] = (
        intra_avg / inter_avg if inter_avg > 0 else float("nan")
    )
    return results


def main(args: argparse.Namespace) -> None:
    rep_list = []
    label_list = []

    label_ranges = {}
    if args.if_split:
        label_ranges = load_label_ranges(
            os.path.join(args.data_path, "labels", "results.txt")
        )

    for sess in args.sessions:
        rep_file = os.path.join(args.rep_dir, f"{sess}.pt")
        if not os.path.exists(rep_file):
            print(f"Representation for {sess} not found")
            continue
        data = torch.load(rep_file)
        reps = data["features"].numpy()

        label_file = os.path.join("predictions", f"{sess}_pred_t.csv")
        if os.path.exists(label_file):
            labels = pd.read_csv(label_file)[LABEL_COLUMNS].to_numpy()
            min_len = min(len(labels), len(reps))
            labels = labels[:min_len]
            reps = reps[:min_len]

            if args.if_split and sess in label_ranges:
                indices = []
                for start, end in label_ranges[sess]:
                    start = max(0, start)
                    end = min(len(labels), end)
                    indices.extend(range(start, end))
                if not indices:
                    print(f"[WARN] No labeled range for session {sess}")
                    continue
                split_start = int(0.8 * len(indices))
                sel = indices[split_start:]
                labels = labels[sel]
                reps = reps[sel]

            label_list.append(labels)
        rep_list.append(reps)

    if not rep_list:
        print("No representations loaded")
        return

    data = np.concatenate(rep_list, axis=0)
    labels = np.concatenate(label_list, axis=0) if label_list else None

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_scaled)

    channel_names = LABEL_COLUMNS + ["NAN"]
    colors = [
        "rgb(255, 0, 0)",
        "rgb(0, 255, 0)",
        "rgb(0, 0, 255)",
        "rgb(255, 255, 0)",
        "rgb(0, 255, 255)",
        "rgb(186, 85, 211)",
        "rgb(128, 128, 128)",
        "rgb(255, 165, 0)",
        "rgb(75, 0, 130)",
        "rgb(255, 105, 180)",
        "rgb(255, 0, 255)",
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
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(channel_names)}
    scatter_data = {name: ([], [], []) for name in channel_names}

    if labels is not None:
        for i, row in enumerate(data_pca):
            idx = np.where(labels[i] == 1)[0]
            name = channel_names[idx[0]] if len(idx) > 0 else "NAN"
            scatter_data[name][0].append(row[0])
            scatter_data[name][1].append(row[1])
            scatter_data[name][2].append(row[2])
    else:
        scatter_data["NAN"] = (
            data_pca[:, 0].tolist(),
            data_pca[:, 1].tolist(),
            data_pca[:, 2].tolist(),
        )

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
        title="3D PCA of Inferred Representations",
        legend=dict(x=1, y=1),
    )
    fig.write_html("pca_plot.html", auto_open=True)

    if labels is not None:
        metrics = compute_cluster_metrics(data_pca, labels)
        print("\n=== Cluster Evaluation Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot inferred representations with optional evaluation"
    )
    parser.add_argument("--data_path", required=True, help="Base data path")
    parser.add_argument("--sessions", nargs="+", required=True, help="Session names")
    parser.add_argument(
        "--rep_dir", default="representations", help="Directory of .pt representation files"
    )
    parser.add_argument(
        "--if_split", action="store_true", help="Use last 20% of labeled ranges"
    )
    args = parser.parse_args()
    main(args)
