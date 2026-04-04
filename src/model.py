"""
model.py
CSRNet architecture:
  Frontend : VGG-16 first 10 conv layers (pretrained, pooling retained
             for blocks 1-3 only — as per original paper)
  Backend  : 6 dilated conv layers for multi-scale context aggregation
  Output   : single-channel density map at 1/8 input resolution
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CSRNet(nn.Module):
    def __init__(
        self,
        pretrained_frontend: bool = True,
        dilation_rates: list = None,
    ):
        super().__init__()

        if dilation_rates is None:
            dilation_rates = [2, 2, 2, 2, 2, 2]

        # ── Frontend: VGG-16 first 10 conv layers ────────────────────────
        # VGG-16 features layout:
        #   0-4   : block1 (conv1_1, relu, conv1_2, relu, maxpool)
        #   5-9   : block2 (conv2_1, relu, conv2_2, relu, maxpool)
        #   10-16 : block3 (conv3_1, relu, conv3_2, relu, conv3_3, relu, maxpool)
        #   17-23 : block4 (conv4_1 ... maxpool)   ← NOT used
        # We take layers 0–16 (first 3 blocks with their pooling layers)
        # = 10 conv layers total as stated in the paper
        vgg = models.vgg16(weights="IMAGENET1K_V1" if pretrained_frontend else None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # ── Backend: dilated conv layers ──────────────────────────────────
        # Input channels = 256 (output of VGG block3)
        # Each layer: 512 channels, 3×3 kernel, specified dilation
        layers = []
        in_ch  = 512
        for d in dilation_rates:
            layers += [
                nn.Conv2d(in_ch, 256, kernel_size=3,
                          padding=d, dilation=d),
                nn.ReLU(inplace=True),
            ]
            in_ch = 256

        # Final 1×1 conv → single-channel density map
        layers.append(nn.Conv2d(256, 1, kernel_size=1))
        self.backend = nn.Sequential(*layers)

        # ── Weight initialisation (backend only) ─────────────────────────
        self._init_backend_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

    def _init_backend_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def build_model(cfg: dict) -> CSRNet:
    return CSRNet(
        pretrained_frontend=cfg["model"]["frontend_pretrained"],
        dilation_rates=cfg["model"]["backend_dilation_rates"],
    )