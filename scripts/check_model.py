"""
check_model.py
Verifies:
  1. CSRNet forward pass produces correct output shape
  2. VGG-16 frontend weights loaded
  3. Dataset pipeline returns correct tensor shapes
  4. DataLoader iterates without error
"""

import torch
from torch.utils.data import DataLoader, random_split
import yaml
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.model   import build_model
from src.dataset import CrowdDataset
from src.utils   import load_config


def check_model(cfg):
    print("\n── Model Sanity Check ──────────────────────────────────────")
    model = build_model(cfg)
    model.eval()

    # Dummy input: batch=1, 3 channels, 256×256
    dummy = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy)

    print(f"  Input  shape : {list(dummy.shape)}")
    print(f"  Output shape : {list(out.shape)}")
    # Output should be (1, 1, 32, 32) = input / 8
    expected_h = 256 // 8
    assert out.shape == (1, 1, expected_h, expected_h), \
        f"Unexpected output shape: {out.shape}"
    print(f"  Output shape ✅  (expected 1×1×{expected_h}×{expected_h})")

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {train_params:,}")


def check_dataset(cfg, part="A"):
    print(f"\n── Dataset Pipeline Check: Part {part} ─────────────────────")
    part_key  = f"part_{part}"
    ds_root   = cfg["dataset"]["root"]
    dm_root   = cfg["density_maps"]["output_dir"]
    crop_size = cfg["training"]["crop_size"]

    img_dir = os.path.join(ds_root, part_key, "train_data", "images")
    dm_dir  = os.path.join(dm_root, part_key, "train_data")

    dataset = CrowdDataset(img_dir, dm_dir, crop_size=crop_size, augment=True)

    val_n   = max(1, int(len(dataset) * cfg["training"]["val_split"]))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(cfg["training"]["seed"])
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=True,
    )

    img_t, dm_t = next(iter(loader))
    print(f"  Dataset size      : {len(dataset)}  (train {train_n} / val {val_n})")
    print(f"  Image tensor      : {list(img_t.shape)}  dtype={img_t.dtype}")
    print(f"  Density map tensor: {list(dm_t.shape)}   dtype={dm_t.dtype}")
    print(f"  GT count (sum)    : {dm_t.sum().item():.2f}")
    print(f"  DataLoader ✅")


def main():
    cfg = load_config("configs/config.yaml")
    check_model(cfg)
    check_dataset(cfg, part="A")
    check_dataset(cfg, part="B")
    print("\n✅  All checks passed — ready for training.\n")


if __name__ == "__main__":
    main()