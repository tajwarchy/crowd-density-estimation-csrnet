"""
dataset.py
PyTorch Dataset for ShanghaiTech Part A and Part B.
Loads pre-generated .npy density maps.
Augmentations: random crop + horizontal flip (train only).
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import yaml


# ImageNet normalisation stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CrowdDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        dm_dir: str,
        crop_size: int = 256,
        augment: bool = True,
    ):
        self.img_dir   = img_dir
        self.dm_dir    = dm_dir
        self.crop_size = crop_size
        self.augment   = augment

        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        )
        assert len(self.img_files) > 0, f"No images found in {img_dir}"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname  = self.img_files[idx]
        img_id = os.path.splitext(fname)[0]

        # ── Load image ────────────────────────────────────────────────────
        img_bgr = cv2.imread(os.path.join(self.img_dir, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ── Load density map ──────────────────────────────────────────────
        dm = np.load(os.path.join(self.dm_dir, f"{img_id}.npy")).astype(np.float32)

        # ── Augmentation (train only) ─────────────────────────────────────
        if self.augment:
            img_rgb, dm = self._random_crop(img_rgb, dm)
            if random.random() > 0.5:
                img_rgb = np.fliplr(img_rgb).copy()
                dm      = np.fliplr(dm).copy()

        # ── Normalise image ───────────────────────────────────────────────
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        img_rgb = (img_rgb - mean) / std

        # ── To tensors ───────────────────────────────────────────────────
        # Image: (H, W, 3) → (3, H, W)
        img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1))
        # Density map: (H, W) → (1, H, W)
        dm_t  = torch.from_numpy(dm).unsqueeze(0)

        return img_t, dm_t

    def _random_crop(self, img, dm):
        h, w = img.shape[:2]
        cs   = self.crop_size

        # If image is smaller than crop size, resize up
        if h < cs or w < cs:
            scale = max(cs / h, cs / w) + 0.01
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            dm  = cv2.resize(dm,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w

        top  = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        img  = img[top:top + cs, left:left + cs]
        dm   = dm [top:top + cs, left:left + cs]
        return img, dm