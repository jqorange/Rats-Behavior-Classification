# -*- coding: utf-8 -*-
import argparse, os, random
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import imageio.v2 as imageio
import cv2

LABEL_COLUMNS = [
    "walk","jump","aiming","scratch","rearing","stand_up",
    "still","eating","local_search","turn_left","turn_right",
]

# BGR colors（OpenCV用BGR）
COLORS = np.array([
    [  0,  0,255],[  0,255,  0],[255,  0,  0],[  0,255,255],
    [255,255,  0],[211, 85,186],[128,128,128],[  0,165,255],
    [130,  0, 75],[180,105,255],[255,  0,255]
], dtype=np.uint8)

def color_for(idx: int):
    return COLORS[idx % len(COLORS)]

def load_reps_and_labels_drop_nan(data_path, rep_dir, sessions):
    rep_kept, lab_kept = [], []
    for sess in sessions:
        rep_file = os.path.join(rep_dir, f"{sess}.pt")
        rep_file = os.path.join(rep_dir, f"{sess}.pt")
        lab_file = os.path.join(data_path, "labels", sess, f"label_{sess}.csv")
        if not os.path.exists(rep_file) or not os.path.exists(lab_file):
            print(f"[WARN] skip {sess} (missing file)"); continue
        p = torch.load(rep_file, map_location="cpu")
        reps = p["features"].detach().cpu().numpy()
        centers = np.array(p.get("centers", np.arange(len(reps))))
        labs_all = pd.read_csv(lab_file)[LABEL_COLUMNS].to_numpy()
        m = centers < len(labs_all)
        reps = reps[m]; centers = centers[m]
        labs = labs_all[centers]
        keep = labs.sum(axis=1) > 0
        if not keep.any():
            print(f"[WARN] {sess} all NAN"); continue
        rep_kept.append(reps[keep]); lab_kept.append(labs[keep])
        print(f"[INFO] {sess}: kept {keep.sum()} rows")
    if not rep_kept: raise RuntimeError("No labeled samples loaded")
    X = np.concatenate(rep_kept, 0).astype(np.float32, copy=False)
    Y = np.concatenate(lab_kept, 0).astype(np.int8, copy=False)
    return X, Y

def maybe_subsample(X, Y, max_points, seed):
    if (max_points is None) or (X.shape[0] <= max_points): return X, Y
    rng = random.Random(seed); idx = list(range(X.shape[0])); rng.shuffle(idx)
    keep = np.array(idx[:max_points], dtype=np.int64)
    print(f"[INFO] subsample {X.shape[0]} -> {keep.shape[0]}")
    return X[keep], Y[keep]

def compute_pca3(X):
    Xs = StandardScaler().fit_transform(X.astype(np.float32, copy=False))
    return PCA(n_components=3).fit_transform(Xs).astype(np.float32, copy=False)

def labels_to_class(Y):
    return (Y == 1).argmax(1).astype(np.int32)

def precompute_bounds_for_stable_scale(P, rotate_axis: str):
    """
    为绕指定轴旋转预计算稳定边界：
      - 绕 x 轴：max_x, max_sqrt(y^2+z^2)
      - 绕 y 轴：max_y, max_sqrt(x^2+z^2)
      - 绕 z 轴：max_z, max_sqrt(x^2+y^2)
    """
    pad = 1e-6
    if rotate_axis == "x":
        max_along = np.max(np.abs(P[:,0])) + pad
        max_perp  = np.max(np.sqrt(P[:,1]**2 + P[:,2]**2)) + pad
    elif rotate_axis == "y":
        max_along = np.max(np.abs(P[:,1])) + pad
        max_perp  = np.max(np.sqrt(P[:,0]**2 + P[:,2]**2)) + pad
    else:  # 'z'
        max_along = np.max(np.abs(P[:,2])) + pad
        max_perp  = np.max(np.sqrt(P[:,0]**2 + P[:,1]**2)) + pad
    return float(max_along), float(max_perp)

def rotation_matrix(axis: str, theta: float):
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
    elif axis == "y":
        return np.array([[ c,0, s],[0,1,0],[-s,0, c]], dtype=np.float32)
    else:  # 'z'
        return np.array([[c,-s,0],[s, c,0],[0,0,1]], dtype=np.float32)

def draw_legend(height, legend_width, font_scale, font_thickness):
    """在白底上画左侧图例，返回 (H, legend_width, 3) 的BGR图像"""
    legend = np.full((height, legend_width, 3), 255, dtype=np.uint8)
    margin_top = 20
    line_gap = int(24 * font_scale) + 8
    y = margin_top
    circle_r = max(3, int(5 * font_scale))

    for i, name in enumerate(LABEL_COLUMNS):
        color = color_for(i).tolist()
        cy = y + circle_r + 2
        cx = 16  # 圆点x
        cv2.circle(legend, (cx, cy), circle_r, color, -1, lineType=cv2.LINE_AA)
        text_x = cx + circle_r + 10
        cv2.putText(
            legend, name, (text_x, cy + int(6 * font_scale)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0),
            font_thickness, lineType=cv2.LINE_AA
        )
        y += line_gap
        if y + circle_r + 20 > height:
            break

    cv2.putText(
        legend, "", (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.9, (80,80,80), font_thickness, cv2.LINE_AA
    )
    cv2.line(legend, (legend_width-1, 0), (legend_width-1, height-1), (200,200,200), 1, cv2.LINE_AA)
    return legend

def make_frame(P_centered, cls, plot_w, height, angle_rad,
               rotate_axis, max_along, max_perp,
               projection="ortho", radius=2.0, point_size=1,
               plot_center_x=0.45, plot_center_y=0.50, margin_scale=0.48):
    """
    plot_center_x / plot_center_y 控制绘图区里的中心位置（0~1），
    margin_scale 控制两侧留白比例（默认 0.48 ≈ 留 4% 边距）。
    """
    # 旋转
    R = rotation_matrix(rotate_axis, angle_rad)
    Q = P_centered @ R.T  # (N,3)
    x, y, z = Q[:,0], Q[:,1], Q[:,2]

    # 投影坐标（正交或弱透视；弱透视用固定参考深度，稳）
    if projection == "weak":
        f = 1.2 * radius
        scale = f / (f + 0.0)
        xp, yp, zp = x*scale, y*scale, z
    else:
        xp, yp, zp = x, y, z

    # 依据旋转轴选择“沿轴量”和“垂直平面量”，用于稳定缩放映射
    if rotate_axis == "x":
        along  = xp; perp_y = yp; denom_along = max_along; denom_perp = max_perp
    elif rotate_axis == "y":
        # 绕 y 轴时，水平跨度主要来自 x/z 平面，这里用 xp 作为横向; 垂直仍用 y
        along  = yp; perp_y = yp  # 沿轴=Y，用于竖直尺度；横向用 max_perp
        denom_along = max_along; denom_perp = max_perp
    else:  # 'z'
        along  = zp; perp_y = yp
        denom_along = max_along; denom_perp = max_perp

    # 将水平坐标用“垂直于旋转轴的最大范围”归一，竖直坐标用“沿旋转轴的最大范围”归一
    px = ((xp / denom_perp) * margin_scale + plot_center_x) * (plot_w - 1)
    py = ((-perp_y / denom_along) * margin_scale + plot_center_y) * (height - 1)

    # 深度排序（近盖远）
    order = np.argsort(zp)[::-1]
    px = px[order].astype(np.int32); py = py[order].astype(np.int32); cls = cls[order]

    img = np.full((height, plot_w, 3), 255, dtype=np.uint8)
    H, W = img.shape[:2]
    for xpix, ypix, k in zip(px, py, cls):
        if 0 <= xpix < W and 0 <= ypix < H:
            cv2.circle(img, (int(xpix), int(ypix)), point_size,
                       color_for(k).tolist(), -1, lineType=cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=r"D:\Jiaqi\Datasets\Rats\TrainData_new")
    ap.add_argument("--sessions", nargs="+",
        default=["F3D5_outdoor","F3D6_outdoor","F5D2_outdoor","F5D10_outdoor","F6D5_outdoor_2"])
    ap.add_argument("--rep_dir", type=str, default="representations")
    ap.add_argument("--gif_out", type=str, default="pca_fast.gif")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=900)          # 总宽度 = 图例 + 绘图
    ap.add_argument("--height", type=int, default=680)
    ap.add_argument("--legend_width", type=int, default=160)   # 左侧图例宽度
    ap.add_argument("--legend_font_scale", type=float, default=0.6)
    ap.add_argument("--legend_font_thickness", type=int, default=1)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--max_points", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point_size", type=int, default=1)
    ap.add_argument("--projection", choices=["ortho","weak"], default="ortho",
                    help="orthographic (stable) or weak perspective (stable)")
    ap.add_argument("--rotate_axis", choices=["x","y","z"], default="y",
                    help="axis to rotate around; y = vertical axis")
    ap.add_argument("--plot_center_x", type=float, default=0.45,
                    help="horizontal center inside the right plot area (0~1). Lower = move left.")
    ap.add_argument("--plot_center_y", type=float, default=0.50,
                    help="vertical center inside the right plot area (0~1).")
    ap.add_argument("--margin_scale", type=float, default=0.48,
                    help="effective half-width/height fraction; lower = more margin.")
    args = ap.parse_args()

    # 数据
    X, Y = load_reps_and_labels_drop_nan(args.data_path, args.rep_dir, args.sessions)
    X, Y = maybe_subsample(X, Y, args.max_points, args.seed)
    P = compute_pca3(X).astype(np.float32, copy=False)
    P -= np.median(P, axis=0, keepdims=True)
    cls = labels_to_class(Y)

    # 稳定缩放边界（依据旋转轴）
    max_along, max_perp = precompute_bounds_for_stable_scale(P, args.rotate_axis)

    # 左侧图例
    legend_w = max(80, min(args.legend_width, args.width-200))
    legend_img = draw_legend(args.height, legend_w, args.legend_font_scale, args.legend_font_thickness)

    # 右侧绘图区宽度
    plot_w = args.width - legend_w
    if plot_w < 200:
        raise ValueError(f"plot width too small ({plot_w}). Increase --width or reduce --legend_width.")

    # 合成 GIF
    with imageio.get_writer(args.gif_out, mode="I", fps=args.fps, loop=0) as w:
        for i in range(args.frames):
            ang = 2*np.pi * (i/args.frames)
            plot_img = make_frame(
                P, cls, plot_w, args.height, ang,
                rotate_axis=args.rotate_axis,
                max_along=max_along, max_perp=max_perp,
                projection=args.projection, radius=args.radius,
                point_size=args.point_size,
                plot_center_x=args.plot_center_x, plot_center_y=args.plot_center_y,
                margin_scale=args.margin_scale
            )
            frame = np.concatenate([legend_img, plot_img], axis=1)
            w.append_data(frame)

    print("[OK] GIF ->", args.gif_out)

if __name__ == "__main__":
    main()
