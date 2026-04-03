"""
validate_density_maps.py
Overlays predicted density map (jet colormap) on original image.
Shows: original | density overlay | GT count vs predicted count.
Saves validation panels to outputs/evaluation/.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_annotation(mat_path):
    mat = sio.loadmat(mat_path)
    return mat["image_info"][0, 0]["location"][0, 0].astype(np.float32)


def make_overlay(img_rgb, density_map, alpha=0.5):
    """Blend jet colormap density map onto image."""
    dm_norm = density_map / (density_map.max() + 1e-8)
    cmap    = plt.get_cmap("jet")
    heatmap = (cmap(dm_norm)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap


def validate_sample(img_path, mat_path, dm_path, save_path, alpha=0.5):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    points   = load_annotation(mat_path)
    gt_count = len(points)

    dm           = np.load(dm_path)
    pred_count   = dm.sum()
    overlay, heatmap = make_overlay(img_rgb, dm, alpha=alpha)

    # Dot annotation image
    dot_img = img_rgb.copy()
    for x, y in points:
        cv2.circle(dot_img, (int(x), int(y)), 3, (255, 0, 0), -1)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(dot_img)
    ax1.set_title(f"GT Annotations\nCount: {gt_count}", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(heatmap)
    ax2.set_title("Density Map (jet)", fontsize=11)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(overlay)
    ax3.set_title(
        f"Overlay\nPredicted: {pred_count:.1f}  |  GT: {gt_count}  |  Error: {abs(pred_count - gt_count):.3f}",
        fontsize=11
    )
    ax3.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] {os.path.basename(img_path)}  GT={gt_count}  Pred={pred_count:.2f}  Error={abs(pred_count-gt_count):.4f}")
    print(f"     Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--part",    choices=["A", "B"], default="A")
    parser.add_argument("--n",       type=int, default=3,
                        help="Number of samples to validate")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    part_key = f"part_{args.part}"
    ds_root  = cfg["dataset"]["root"]
    dm_root  = cfg["density_maps"]["output_dir"]
    alpha    = cfg["inference"]["heatmap_alpha"]

    img_dir = os.path.join(ds_root, part_key, "train_data", "images")
    gt_dir  = os.path.join(ds_root, part_key, "train_data", "ground_truth")
    dm_dir  = os.path.join(dm_root, part_key, "train_data")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    print(f"\n=== Density Map Validation: Part {args.part} ({args.n} samples) ===\n")

    for fname in img_files[:args.n]:
        img_id   = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, fname)
        mat_path = os.path.join(gt_dir, f"GT_{img_id}.mat")
        dm_path  = os.path.join(dm_dir, f"{img_id}.npy")

        if not os.path.exists(dm_path):
            print(f"[SKIP] Density map not found: {dm_path}")
            continue

        save_path = os.path.join(
            cfg["paths"]["output_dir"], "evaluation",
            f"validate_part{args.part}_{img_id}.png"
        )
        validate_sample(img_path, mat_path, dm_path, save_path, alpha=alpha)


if __name__ == "__main__":
    main()