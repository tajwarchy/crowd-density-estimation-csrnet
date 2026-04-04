"""
plot_training_curves.py
Plots training loss and validation MAE/MSE from the CSV logs.
Saves figures to outputs/evaluation/.
"""

import os
import sys
import argparse
import csv
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.abspath("."))
from src.utils import load_config


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def plot_curves(cfg, part):
    part_cfg = cfg["training"][f"part_{part}"]
    log_path = part_cfg["log_csv"]
    out_dir  = cfg["paths"]["output_dir"] + "/evaluation"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(log_path):
        print(f"[WARN] Log not found: {log_path}")
        return

    rows   = read_csv(log_path)
    epochs = [int(r["epoch"])      for r in rows]
    losses = [float(r["train_loss"]) for r in rows]

    val_epochs = [int(r["epoch"])    for r in rows if r["val_mae"]]
    val_maes   = [float(r["val_mae"]) for r in rows if r["val_mae"]]
    val_mses   = [float(r["val_mse"]) for r in rows if r["val_mse"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"CSRNet Training — Part {part}", fontsize=14)

    # Loss curve
    axes[0].plot(epochs, losses, color="steelblue", linewidth=1.2)
    axes[0].set_title("Training Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # MAE / MSE curve
    axes[1].plot(val_epochs, val_maes, label="MAE",
                 color="darkorange", linewidth=1.5, marker="o", markersize=3)
    axes[1].plot(val_epochs, val_mses, label="RMSE",
                 color="crimson",    linewidth=1.5, marker="s", markersize=3)
    axes[1].set_title("Validation MAE & RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Count Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_path = os.path.join(out_dir, f"training_curves_part{part}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Curves saved → {save_path}")

    if val_maes:
        best_idx = val_maes.index(min(val_maes))
        print(f"  Best MAE  : {val_maes[best_idx]:.2f}  at epoch {val_epochs[best_idx]}")
        print(f"  Best RMSE : {val_mses[best_idx]:.2f}  at epoch {val_epochs[best_idx]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--part", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    parts = ["A", "B"] if args.part == "both" else [args.part]
    for p in parts:
        plot_curves(cfg, p)


if __name__ == "__main__":
    main()