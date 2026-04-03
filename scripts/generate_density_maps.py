"""
generate_density_maps.py
Converts dot annotations (.mat) → Gaussian density maps (.npy).
  - Part A: adaptive kernel (k-NN based sigma)
  - Part B: fixed kernel (uniform sigma)
Density map sum ≈ ground truth head count per image.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import yaml
from tqdm import tqdm


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_annotation(mat_path: str) -> np.ndarray:
    """Return (N, 2) array of (col, row) = (x, y) head positions."""
    mat = sio.loadmat(mat_path)
    points = mat["image_info"][0, 0]["location"][0, 0]
    return points.astype(np.float32)


def make_density_map_adaptive(
    img_shape: tuple,
    points: np.ndarray,
    k: int,
    beta: float,
    min_sigma: float,
) -> np.ndarray:
    """
    Adaptive Gaussian kernel (Part A — dense crowds).
    Sigma per head = beta * mean distance to k nearest neighbours.
    """
    h, w = img_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    # Build KDTree over all head positions (x, y)
    tree = KDTree(points)

    for pt in points:
        x, y = int(min(pt[0], w - 1)), int(min(pt[1], h - 1))

        if len(points) <= 1:
            sigma = min_sigma
        else:
            # k+1 because the point itself is the nearest neighbour
            k_query = min(k + 1, len(points))
            dists, _ = tree.query(pt, k=k_query)
            avg_dist = np.mean(dists[1:])          # exclude self
            sigma = max(beta * avg_dist, min_sigma)

        # Place a single Gaussian at (y, x)
        dot = np.zeros((h, w), dtype=np.float32)
        if 0 <= y < h and 0 <= x < w:
            dot[y, x] = 1.0
        density += gaussian_filter(dot, sigma=sigma, mode="constant")

    return density


def make_density_map_fixed(
    img_shape: tuple,
    points: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Fixed Gaussian kernel (Part B — sparse crowds).
    All heads share the same sigma.
    """
    h, w = img_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    for pt in points:
        x, y = int(min(pt[0], w - 1)), int(min(pt[1], h - 1))
        dot = np.zeros((h, w), dtype=np.float32)
        if 0 <= y < h and 0 <= x < w:
            dot[y, x] = 1.0
        density += gaussian_filter(dot, sigma=sigma, mode="constant")

    return density


# ── Per-split processor ───────────────────────────────────────────────────────

def process_split(
    img_dir: str,
    gt_dir: str,
    out_dir: str,
    method: str,
    cfg_dm: dict,
) -> dict:
    """
    Generate and save density maps for one split.
    Returns stats dict: {count_errors, gt_counts, pred_counts}.
    """
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if not img_files:
        print(f"  [WARN] No images in {img_dir}")
        return {}

    gt_counts, pred_counts, count_errors = [], [], []

    for fname in tqdm(img_files, desc=f"  {os.path.basename(out_dir)}", ncols=80):
        img_id   = os.path.splitext(fname)[0]          # IMG_1
        mat_path = os.path.join(gt_dir, f"GT_{img_id}.mat")

        if not os.path.exists(mat_path):
            print(f"  [WARN] Missing annotation: {mat_path}")
            continue

        # Load image shape without decoding pixels (faster)
        import cv2
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            print(f"  [WARN] Cannot read image: {fname}")
            continue
        img_shape = img.shape          # (H, W, 3)

        points = load_annotation(mat_path)
        gt_count = len(points)

        if method == "adaptive":
            dm = make_density_map_adaptive(
                img_shape,
                points,
                k=cfg_dm["k_nearest"],
                beta=cfg_dm["beta"],
                min_sigma=cfg_dm["min_sigma"],
            )
        else:
            dm = make_density_map_fixed(
                img_shape,
                points,
                sigma=cfg_dm["fixed_sigma"],
            )

        pred_count = float(dm.sum())
        error = abs(pred_count - gt_count)

        gt_counts.append(gt_count)
        pred_counts.append(pred_count)
        count_errors.append(error)

        # Save density map
        out_path = os.path.join(out_dir, f"{img_id}.npy")
        np.save(out_path, dm)

    return {
        "gt_counts":    np.array(gt_counts),
        "pred_counts":  np.array(pred_counts),
        "count_errors": np.array(count_errors),
    }


def print_stats(label: str, stats: dict) -> None:
    if not stats:
        return
    errs = stats["count_errors"]
    gt   = stats["gt_counts"]
    pred = stats["pred_counts"]
    print(f"\n  [{label}]")
    print(f"    Images processed : {len(gt)}")
    print(f"    GT   count range : {gt.min():.0f} – {gt.max():.0f}  (mean {gt.mean():.1f})")
    print(f"    Pred count range : {pred.min():.1f} – {pred.max():.1f}  (mean {pred.mean():.1f})")
    print(f"    Count MAE        : {errs.mean():.4f}")
    print(f"    Count max error  : {errs.max():.4f}  ← should be < 1.0")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--part", choices=["A", "B", "both"], default="both",
        help="Which dataset part to process"
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    ds_root = cfg["dataset"]["root"]
    dm_root = cfg["density_maps"]["output_dir"]

    parts = ["A", "B"] if args.part == "both" else [args.part]

    for part in parts:
        part_key = f"part_{part}"
        dm_cfg   = cfg["density_maps"][part_key]
        method   = dm_cfg["method"]

        print(f"\n{'='*55}")
        print(f"  Generating density maps — Part {part}  [{method} kernel]")
        print(f"{'='*55}")

        for split in ["train_data", "test_data"]:
            img_dir = os.path.join(ds_root, part_key, split, "images")
            gt_dir  = os.path.join(ds_root, part_key, split, "ground_truth")
            out_dir = os.path.join(dm_root, part_key, split)

            stats = process_split(img_dir, gt_dir, out_dir, method, dm_cfg)
            print_stats(f"{part_key}/{split}", stats)

    print(f"\n✅  All density maps saved to: {dm_root}")


if __name__ == "__main__":
    main()