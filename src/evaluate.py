"""
evaluate.py
Loads best checkpoints, runs full test split evaluation on MPS.
Computes MAE and MSE per part, generates:
  - MAE/MSE results table (printed + saved as figure)
  - Count accuracy scatter plot (predicted vs GT)
  - Density map comparison panels (original | GT dm | pred dm)
  - Sparse vs dense crowd side-by-side comparison panel
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from src.model import build_model
from src.utils import load_config, load_checkpoint, compute_mae_mse


# ── Helpers ───────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_annotation(mat_path):
    mat = sio.loadmat(mat_path)
    return mat["image_info"][0, 0]["location"][0, 0].astype(np.float32)


def preprocess_image(img_bgr):
    """BGR → normalised float tensor (1, 3, H, W)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    img_t   = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
    return img_t


def predict(model, img_bgr, device):
    """Run inference, return density map as numpy (H, W) and predicted count."""
    img_t = preprocess_image(img_bgr).to(device)
    with torch.no_grad():
        out = model(img_t)
    dm_pred = out.squeeze().cpu().numpy()
    count   = float(dm_pred.sum())
    return dm_pred, count


def make_overlay(img_rgb, dm, alpha=0.5):
    dm_norm = dm / (dm.max() + 1e-8)
    cmap    = plt.get_cmap("jet")
    heatmap = (cmap(dm_norm)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap


# ── Full test split evaluation ────────────────────────────────────────────────

def evaluate_split(model, img_dir, gt_dir, device):
    img_files   = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    pred_counts = []
    gt_counts   = []

    for fname in tqdm(img_files, desc=f"  Evaluating", ncols=80):
        img_id   = os.path.splitext(fname)[0]
        mat_path = os.path.join(gt_dir, f"GT_{img_id}.mat")

        img_bgr  = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None or not os.path.exists(mat_path):
            continue

        points   = load_annotation(mat_path)
        gt_count = len(points)

        _, pred_count = predict(model, img_bgr, device)

        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

    mae, mse = compute_mae_mse(pred_counts, gt_counts)
    return mae, mse, pred_counts, gt_counts


# ── Scatter plot ──────────────────────────────────────────────────────────────

def plot_scatter(pred_A, gt_A, pred_B, gt_B, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Predicted vs Ground Truth Count", fontsize=14)

    for ax, pred, gt, label, color in zip(
        axes,
        [pred_A, pred_B],
        [gt_A,   gt_B],
        ["Part A (Dense)", "Part B (Sparse)"],
        ["steelblue", "darkorange"],
    ):
        pred_arr = np.array(pred)
        gt_arr   = np.array(gt)
        ax.scatter(gt_arr, pred_arr, alpha=0.5, s=18, color=color)
        mn, mx = min(gt_arr.min(), pred_arr.min()), max(gt_arr.max(), pred_arr.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Ground Truth Count")
        ax.set_ylabel("Predicted Count")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "scatter_pred_vs_gt.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Scatter plot → {path}")


# ── Results table figure ──────────────────────────────────────────────────────

def plot_results_table(results, out_dir):
    """
    results = {
      'Part A': {'MAE': ..., 'MSE': ..., 'N': ...},
      'Part B': {'MAE': ..., 'MSE': ..., 'N': ...},
    }
    """
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")

    col_labels = ["Dataset", "Split", "Images", "MAE ↓", "RMSE ↓"]
    rows = []
    for part_label, v in results.items():
        rows.append([part_label, "Test", str(v["N"]),
                     f"{v['MAE']:.2f}", f"{v['MSE']:.2f}"])

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 2.0)

    # Header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row styling
    for i, _ in enumerate(rows):
        color = "#eaf0fb" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(color)

    plt.title("CSRNet — Test Set Results", fontsize=13,
              fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "results_table.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Results table → {path}")


# ── Density map comparison panels ────────────────────────────────────────────

def plot_dm_comparison(model, img_dir, gt_dir, dm_dir,
                       device, out_dir, part, n=5, alpha=0.5):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    saved = 0

    for fname in img_files:
        if saved >= n:
            break
        img_id   = os.path.splitext(fname)[0]
        mat_path = os.path.join(gt_dir, f"GT_{img_id}.mat")
        dm_path  = os.path.join(dm_dir, f"{img_id}.npy")

        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None or not os.path.exists(mat_path) \
                or not os.path.exists(dm_path):
            continue

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        points   = load_annotation(mat_path)
        gt_count = len(points)
        dm_gt    = np.load(dm_path)

        dm_pred, pred_count = predict(model, img_bgr, device)

        # Upsample pred dm to original image size for display
        dm_pred_up = cv2.resize(
            dm_pred, (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        _, heatmap_gt   = make_overlay(img_rgb, dm_gt,      alpha=alpha)
        _, heatmap_pred = make_overlay(img_rgb, dm_pred_up, alpha=alpha)
        overlay_pred, _ = make_overlay(img_rgb, dm_pred_up, alpha=alpha)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            f"Part {part} | {fname} | GT: {gt_count}  Pred: {pred_count:.1f}",
            fontsize=12
        )

        axes[0].imshow(img_rgb);         axes[0].set_title("Original");           axes[0].axis("off")
        axes[1].imshow(heatmap_gt);      axes[1].set_title("GT Density Map");     axes[1].axis("off")
        axes[2].imshow(heatmap_pred);    axes[2].set_title("Predicted Density");  axes[2].axis("off")
        axes[3].imshow(overlay_pred);    axes[3].set_title("Overlay");            axes[3].axis("off")

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"dm_comparison_part{part}_{img_id}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        saved += 1

    print(f"[OK] {saved} density map panels saved → {out_dir}")


# ── Sparse vs Dense comparison panel ─────────────────────────────────────────

def plot_sparse_vs_dense(
    model_A, model_B,
    cfg, device, out_dir, alpha=0.5
):
    ds_root = cfg["dataset"]["root"]
    dm_root = cfg["density_maps"]["output_dir"]

    def get_sample(part):
        part_key = f"part_{part}"
        img_dir  = os.path.join(ds_root, part_key, "test_data", "images")
        gt_dir   = os.path.join(ds_root, part_key, "test_data", "ground_truth")
        dm_dir   = os.path.join(dm_root, part_key, "test_data")
        model    = model_A if part == "A" else model_B

        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        fname    = img_files[0]
        img_id   = os.path.splitext(fname)[0]

        img_bgr  = cv2.imread(os.path.join(img_dir, fname))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mat_path = os.path.join(gt_dir, f"GT_{img_id}.mat")
        dm_path  = os.path.join(dm_dir, f"{img_id}.npy")

        points   = load_annotation(mat_path)
        gt_count = len(points)
        dm_gt    = np.load(dm_path)

        dm_pred, pred_count = predict(model, img_bgr, device)
        dm_pred_up = cv2.resize(
            dm_pred, (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        overlay, _ = make_overlay(img_rgb, dm_pred_up, alpha=alpha)
        return img_rgb, overlay, gt_count, pred_count

    img_A, ov_A, gt_A, pred_A = get_sample("A")
    img_B, ov_B, gt_B, pred_B = get_sample("B")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Sparse vs Dense Crowd — Density Map Comparison", fontsize=14)

    axes[0, 0].imshow(img_A); axes[0, 0].set_title("Part A — Dense (Original)");   axes[0, 0].axis("off")
    axes[0, 1].imshow(ov_A);  axes[0, 1].set_title(f"Part A — Density Overlay\nGT: {gt_A}  Pred: {pred_A:.1f}"); axes[0, 1].axis("off")
    axes[1, 0].imshow(img_B); axes[1, 0].set_title("Part B — Sparse (Original)");  axes[1, 0].axis("off")
    axes[1, 1].imshow(ov_B);  axes[1, 1].set_title(f"Part B — Density Overlay\nGT: {gt_B}  Pred: {pred_B:.1f}"); axes[1, 1].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "sparse_vs_dense_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Sparse vs dense panel → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args   = parser.parse_args()
    cfg    = load_config(args.config)

    device  = torch.device(cfg["evaluation"]["device"])
    out_dir = cfg["evaluation"]["output_dir"]
    ds_root = cfg["dataset"]["root"]
    dm_root = cfg["density_maps"]["output_dir"]
    alpha   = cfg["inference"]["heatmap_alpha"]
    os.makedirs(out_dir, exist_ok=True)

    results    = {}
    pred_all   = {}
    gt_all     = {}
    models     = {}

    for part in ["A", "B"]:
        print(f"\n{'='*55}")
        print(f"  Evaluating Part {part} on {str(device).upper()}")
        print(f"{'='*55}")

        part_key  = f"part_{part}"
        ckpt_path = cfg["training"][f"part_{part}"]["best_checkpoint"]

        model = build_model(cfg).to(device)
        load_checkpoint(ckpt_path, model, device=str(device))
        model.eval()
        models[part] = model

        img_dir = os.path.join(ds_root, part_key, "test_data", "images")
        gt_dir  = os.path.join(ds_root, part_key, "test_data", "ground_truth")

        mae, mse, pred_counts, gt_counts = evaluate_split(
            model, img_dir, gt_dir, device
        )

        results[f"Part {part}"] = {
            "MAE": mae, "MSE": mse, "N": len(gt_counts)
        }
        pred_all[part] = pred_counts
        gt_all[part]   = gt_counts

        print(f"\n  Part {part} Test Results:")
        print(f"    MAE  : {mae:.2f}")
        print(f"    RMSE : {mse:.2f}")
        print(f"    N    : {len(gt_counts)} images")

        # Density map comparison panels (5 test samples per part)
        dm_dir = os.path.join(dm_root, part_key, "test_data")
        plot_dm_comparison(
            model, img_dir, gt_dir, dm_dir,
            device, out_dir, part, n=5, alpha=alpha
        )

    # ── Cross-part figures ────────────────────────────────────────────────
    plot_scatter(
        pred_all["A"], gt_all["A"],
        pred_all["B"], gt_all["B"],
        out_dir
    )
    plot_results_table(results, out_dir)
    plot_sparse_vs_dense(
        models["A"], models["B"],
        cfg, device, out_dir, alpha=alpha
    )

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Final Test Results")
    print(f"{'='*55}")
    for label, v in results.items():
        print(f"  {label} | MAE={v['MAE']:.2f}  RMSE={v['MSE']:.2f}  N={v['N']}")
    print(f"\n  All figures saved → {out_dir}")


if __name__ == "__main__":
    main()