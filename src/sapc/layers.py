"""Small building blocks: self-attention and 2D positional encoding."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def spatial_self_attention(x: torch.Tensor, scale: bool = False) -> torch.Tensor:
    """Parameter-free self-attention over spatial positions.

    Query, key and value are all the feature map itself, so every position
    becomes a softmax-weighted mix of all positions in the image.

    Args:
        x: (B, C, H, W) feature map.
        scale: divide the logits by sqrt(C) as in a Transformer. The thesis
            runs without it; with C = 1024 the unscaled softmax is close to
            one-hot on the diagonal (see "Findings" in the README).
    """
    b, c, h, w = x.shape
    flat = x.flatten(2)  # (B, C, HW)
    logits = torch.bmm(flat.transpose(1, 2), flat)  # (B, HW, HW)
    if scale:
        logits = logits / math.sqrt(c)
    attn = logits.softmax(dim=-1)
    out = torch.bmm(flat, attn.transpose(1, 2))  # (B, C, HW)
    return out.view(b, c, h, w)


def sincos_2d(dim: int, height: int, width: int, device=None) -> torch.Tensor:
    """2D sine-cosine positional encoding, shape (dim, H, W).

    The first half of the channels encodes the x coordinate and the second
    half the y coordinate, each with the usual Transformer frequencies.
    """
    if dim % 4:
        raise ValueError(f"dim must be a multiple of 4, got {dim}")
    pe = torch.zeros(dim, height, width, device=device)
    half = dim // 2
    freq = torch.exp(torch.arange(0, half, 2, device=device).float() * -(math.log(10000.0) / half))
    pos_w = torch.arange(width, device=device).float().unsqueeze(1)  # (W, 1)
    pos_h = torch.arange(height, device=device).float().unsqueeze(1)  # (H, 1)
    pe[0:half:2] = torch.sin(pos_w * freq).T.unsqueeze(1).expand(-1, height, -1)
    pe[1:half:2] = torch.cos(pos_w * freq).T.unsqueeze(1).expand(-1, height, -1)
    pe[half::2] = torch.sin(pos_h * freq).T.unsqueeze(2).expand(-1, -1, width)
    pe[half + 1 :: 2] = torch.cos(pos_h * freq).T.unsqueeze(2).expand(-1, -1, width)
    return pe


def to_patches(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B*H*W, C), row-major over positions."""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b * h * w, c)


def from_patches(x: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    """(B*H*W, C) -> (B, C, H, W). Inverse of :func:`to_patches`."""
    return x.view(b, h, w, -1).permute(0, 3, 1, 2)


def resize_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def pool_channels(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Average-pool the channel axis of (N, C) patch vectors down to ``dim``."""
    if x.shape[1] == dim:
        return x
    if x.device.type == "mps" and x.shape[1] % dim:
        # MPS only supports evenly divisible adaptive pooling.
        return F.adaptive_avg_pool1d(x.cpu().unsqueeze(1), dim).squeeze(1).to(x.device)
    return F.adaptive_avg_pool1d(x.unsqueeze(1), dim).squeeze(1)
