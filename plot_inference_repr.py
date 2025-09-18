import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from utils.segments import (
    SegmentInfo,
    collect_segment_centres,
    compute_segments,
    extract_label_arrays,
    split_segments_by_action,
)


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

    split_centres: Dict[str, List[int]] = {}
    if args.if_split:
        segments_by_session: Dict[str, Dict[str, List[SegmentInfo]]] = {}
        label_columns_ref: Optional[List[str]] = None
        for sess in args.sessions:
            label_file = os.path.join(args.data_path, "labels", sess, f"label_{sess}.csv")
            if not os.path.exists(label_file):
                segments_by_session[sess] = {}
                continue

            label_df = pd.read_csv(label_file)
            indices, labels_arr, label_cols = extract_label_arrays(
                label_df, label_columns=label_columns_ref
            )
            if label_columns_ref is None and label_cols:
                label_columns_ref = label_cols
            elif label_columns_ref is not None and label_cols and label_cols != label_columns_ref:
                raise ValueError("Label columns mismatch across sessions")

            columns_for_session = label_cols if label_cols else (label_columns_ref or [])
            segments_by_session[sess] = compute_segments(
                indices, labels_arr, columns_for_session
            )

        assignments = split_segments_by_action(
            segments_by_session, test_ratio=args.segment_test_ratio, seed=args.split_seed
        )
        split_centres = collect_segment_centres(
            segments_by_session, assignments, args.segment_split
        )

    for sess in args.sessions:
        rep_file = os.path.join(args.rep_dir, f"{sess}.pt")
        if not os.path.exists(rep_file):
            print(f"Representation for {sess} not found")
            continue
        data = torch.load(rep_file)
        reps = data["features"].numpy()
        centers = np.array(data.get("centers", np.arange(len(reps))))

        label_file = os.path.join(args.data_path, "labels", sess, f"label_{sess}.csv")
        if os.path.exists(label_file):
            labels_all = pd.read_csv(label_file)[LABEL_COLUMNS].to_numpy()

            # Align labels with representation centers
            mask = centers < len(labels_all)
            centers = centers[mask]
            reps = reps[mask]
            labels = labels_all[centers]

            if args.if_split:
                keep_centres = set(split_centres.get(sess, []))
                if not keep_centres:
                    print(f"[WARN] No selected segments for session {sess}")
                    continue
                mask = np.isin(centers, list(keep_centres))
                if not mask.any():
                    print(f"[WARN] Selected centres not present in representations for {sess}")
                    continue
                labels = labels[mask]
                reps = reps[mask]

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
    parser.add_argument("--data_path", default="D:\Jiaqi\Datasets\Rats\TrainData_new", help="Base data path")
    parser.add_argument("--sessions", nargs="+", default=["F3D5_outdoor", "F3D6_outdoor", "F5D2_outdoor", "F5D10_outdoor", "F6D5_outdoor_2"], help="Session names")
    parser.add_argument(
        "--rep_dir", default="representations", help="Directory of .pt representation files"
    )
    parser.add_argument(
        "--if_split", action="store_true", help="Restrict to selected segment split"
    )
    parser.add_argument(
        "--segment_split", choices=["train", "test"], default="test", help="Segment split to visualise"
    )
    parser.add_argument(
        "--split_seed", type=int, default=0, help="Random seed for segment splitting"
    )
    parser.add_argument(
        "--segment_test_ratio", type=float, default=0.2, help="Test ratio for segment splitting"
    )
    args = parser.parse_args()
    main(args)
