"""
inspect_dataset.py
Loads a .mat annotation file, overlays dot annotations on the image,
prints head count, and saves the visualisation to outputs/evaluation/.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_annotation(mat_path: str) -> np.ndarray:
    """Return (N, 2) array of (col, row) head positions."""
    mat = sio.loadmat(mat_path)
    # ShanghaiTech stores annotations under 'image_info' > 'location'
    points = mat["image_info"][0, 0]["location"][0, 0]
    return points.astype(np.float32)


def visualise_sample(img_path: str, mat_path: str, save_path: str) -> None:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    points = load_annotation(mat_path)
    count = len(points)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Draw dots
    vis = img_rgb.copy()
    for x, y in points:
        cv2.circle(vis, (int(x), int(y)), radius=3,
                   color=(255, 0, 0), thickness=-1)

    plt.figure(figsize=(10, 6))
    plt.imshow(vis)
    plt.title(f"Head Count: {count}  |  {os.path.basename(img_path)}")
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[OK] Saved visualisation → {save_path}")
    print(f"     Image shape : {img_bgr.shape}")
    print(f"     Head count  : {count}")


def dataset_stats(cfg: dict) -> None:
    """Print min/max/mean count for all splits."""
    for part in ["part_A", "part_B"]:
        for split in ["train_data", "test_data"]:
            gt_dir = os.path.join(
                cfg["dataset"]["root"], part, split, "ground_truth"
            )
            if not os.path.isdir(gt_dir):
                print(f"[WARN] Not found: {gt_dir}")
                continue
            counts = []
            for fname in sorted(os.listdir(gt_dir)):
                if fname.endswith(".mat"):
                    pts = load_annotation(os.path.join(gt_dir, fname))
                    counts.append(len(pts))
            counts = np.array(counts)
            print(
                f"[{part}/{split}]  n={len(counts):4d} | "
                f"min={counts.min():5.0f}  max={counts.max():5.0f}  "
                f"mean={counts.mean():6.1f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--part", choices=["A", "B"], default="A",
        help="Which dataset part to sample from"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    part_key = f"part_{args.part}"
    img_dir = os.path.join(
        cfg["dataset"]["root"], part_key, "train_data", "images"
    )
    gt_dir = os.path.join(
        cfg["dataset"]["root"], part_key, "train_data", "ground_truth"
    )

    # Pick first image as sample
    img_files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    )
    if not img_files:
        print("[ERROR] No images found. Check dataset path in config.yaml.")
        sys.exit(1)

    sample_img = img_files[0]
    img_id = os.path.splitext(sample_img)[0]          # e.g. IMG_1
    mat_name = f"GT_{img_id}.mat"

    img_path = os.path.join(img_dir, sample_img)
    mat_path = os.path.join(gt_dir, mat_name)

    save_path = os.path.join(
        cfg["paths"]["output_dir"], "evaluation",
        f"inspect_part{args.part}_sample.png"
    )

    print(f"\n=== Dataset Inspection: Part {args.part} ===")
    visualise_sample(img_path, mat_path, save_path)

    print("\n=== Dataset Statistics (all splits) ===")
    dataset_stats(cfg)


if __name__ == "__main__":
    main()