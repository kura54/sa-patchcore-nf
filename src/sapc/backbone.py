"""Frozen WideResNet-50-2 feature extractor (layer2 and layer3)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class FeatureExtractor(nn.Module):
    """Returns the outputs of ``layer2`` and ``layer3``.

    For a 256x128 input this gives (B, 512, 32, 16) and (B, 1024, 16, 8).
    The network is frozen and always runs in eval mode.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        net = wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # noqa: D102 - always frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        return f2, f3
