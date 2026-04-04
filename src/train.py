"""
train.py
CSRNet training loop for ShanghaiTech Part A and Part B.
  - MSE loss on density maps
  - Adam optimiser + StepLR scheduler
  - Best checkpoint saved by validation MAE
  - All params from config.yaml — no hardcoded values
  - Training on M1 CPU (batch_size=1)
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from src.dataset import CrowdDataset
from src.model   import build_model
from src.utils   import (
    load_config, save_checkpoint, load_checkpoint,
    compute_mae_mse, CSVLogger
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(cfg, part):
    part_key  = f"part_{part}"
    ds_root   = cfg["dataset"]["root"]
    dm_root   = cfg["density_maps"]["output_dir"]
    crop_size = cfg["training"]["crop_size"]
    val_split = cfg["training"]["val_split"]
    seed      = cfg["training"]["seed"]
    nw        = cfg["training"]["num_workers"]
    bs        = cfg["training"]["batch_size"]

    img_dir = os.path.join(ds_root, part_key, "train_data", "images")
    dm_dir  = os.path.join(dm_root, part_key, "train_data")

    full_ds = CrowdDataset(img_dir, dm_dir, crop_size=crop_size, augment=True)
    val_n   = max(1, int(len(full_ds) * val_split))
    train_n = len(full_ds) - val_n

    train_ds, val_ds = random_split(
        full_ds, [train_n, val_n],
        generator=torch.Generator().manual_seed(seed)
    )

    # Val dataset — no augmentation
    val_img_dir = os.path.join(ds_root, part_key, "train_data", "images")
    val_dm_dir  = os.path.join(dm_root, part_key, "train_data")
    val_ds_noaug = CrowdDataset(
        val_img_dir, val_dm_dir, crop_size=crop_size, augment=False
    )
    # Restrict val_ds_noaug to same indices as val_ds
    val_ds_noaug = torch.utils.data.Subset(val_ds_noaug, val_ds.indices)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw
    )
    val_loader = DataLoader(
        val_ds_noaug, batch_size=1, shuffle=False, num_workers=nw
    )

    print(f"  [Part {part}] train={train_n}  val={val_n}")
    return train_loader, val_loader


def validate(model, loader, device):
    model.eval()
    pred_counts, gt_counts = [], []
    with torch.no_grad():
        for img_t, dm_t in loader:
            img_t = img_t.to(device)
            out   = model(img_t)
            # Full image inference — no crop, so sum over full output
            pred_counts.append(out.sum().item())
            gt_counts.append(dm_t.sum().item())
    mae, mse = compute_mae_mse(pred_counts, gt_counts)
    model.train()
    return mae, mse


def train_one_part(cfg, part):
    print(f"\n{'='*55}")
    print(f"  Training CSRNet — Part {part}")
    print(f"{'='*55}\n")

    t_cfg    = cfg["training"]
    device   = torch.device(t_cfg["device"])
    epochs   = t_cfg["epochs"]
    lr       = t_cfg["learning_rate"]
    wd       = t_cfg["weight_decay"]
    part_cfg = t_cfg[f"part_{part}"]

    set_seed(t_cfg["seed"])

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    model.train()

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=t_cfg["lr_scheduler"]["step_size"],
        gamma=t_cfg["lr_scheduler"]["gamma"],
    )
    criterion = nn.MSELoss()

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(cfg, part)

    # ── Logger ───────────────────────────────────────────────────────────
    logger = CSVLogger(
        part_cfg["log_csv"],
        fieldnames=["epoch", "train_loss", "val_mae", "val_mse", "lr"]
    )

    best_mae  = float("inf")
    ckpt_path = part_cfg["best_checkpoint"]

    # ── Resume if checkpoint exists ──────────────────────────────────────
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"  Resuming from checkpoint: {ckpt_path}")
        start_epoch, best_mae = load_checkpoint(
            ckpt_path, model, optimizer, device=str(device)
        )
        print(f"  Resumed at epoch {start_epoch}  best_mae={best_mae:.2f}\n")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"  Epoch [{epoch+1:03d}/{epochs}]",
            ncols=80,
            leave=False,
        )
        for img_t, dm_t in pbar:
            img_t = img_t.to(device)
            dm_t  = dm_t.to(device)

            optimizer.zero_grad()
            out  = model(img_t)

            # Downscale GT density map to match output resolution (÷8)
            dm_down = nn.functional.interpolate(
                dm_t, size=out.shape[2:], mode="bilinear", align_corners=False
            )
            # Scale density values so sum is preserved after downscaling
            scale   = (dm_t.shape[2] * dm_t.shape[3]) / \
                      (out.shape[2]  * out.shape[3])
            dm_down = dm_down * scale

            loss = criterion(out, dm_down)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Validate every 5 epochs ──────────────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_mae, val_mse = validate(model, val_loader, device)

            print(
                f"  Epoch {epoch+1:03d}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"MAE={val_mae:.2f} | "
                f"MSE={val_mse:.2f} | "
                f"lr={current_lr:.2e}"
            )

            logger.log({
                "epoch":      epoch + 1,
                "train_loss": round(avg_loss, 6),
                "val_mae":    round(val_mae, 4),
                "val_mse":    round(val_mse, 4),
                "lr":         current_lr,
            })

            if val_mae < best_mae:
                best_mae = val_mae
                save_checkpoint(
                    {
                        "epoch":           epoch + 1,
                        "model_state":     model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_mae":        best_mae,
                        "val_mse":         val_mse,
                        "part":            part,
                    },
                    ckpt_path,
                )
                print(f"  ✅ Best checkpoint saved (MAE={best_mae:.2f}) → {ckpt_path}")
        else:
            # Still log loss every epoch
            logger.log({
                "epoch":      epoch + 1,
                "train_loss": round(avg_loss, 6),
                "val_mae":    "",
                "val_mse":    "",
                "lr":         current_lr,
            })

    print(f"\n  Training complete — Part {part}")
    print(f"  Best validation MAE : {best_mae:.2f}")
    print(f"  Checkpoint saved at : {ckpt_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--part", choices=["A", "B"], required=True,
        help="Which dataset part to train on"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_one_part(cfg, args.part)


if __name__ == "__main__":
    main()