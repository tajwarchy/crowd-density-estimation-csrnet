"""
utils.py
Shared utilities: config loading, checkpoint save/load, metric computation,
training CSV logger.
"""

import os
import csv
import yaml
import torch
import numpy as np


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_mae", float("inf"))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_mae_mse(pred_counts: list, gt_counts: list) -> tuple:
    pred = np.array(pred_counts, dtype=np.float32)
    gt   = np.array(gt_counts,   dtype=np.float32)
    mae  = float(np.mean(np.abs(pred - gt)))
    mse  = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return mae, mse


# ── CSV logger ────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: str, fieldnames: list):
        self.path       = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write header if file doesn't exist
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)