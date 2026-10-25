"""Greedy k-center coreset subsampling (PatchCore, Roth et al. 2022)."""

from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def greedy_coreset(
    feats: torch.Tensor,
    ratio: float,
    projection_dim: int = 128,
    start_points: int = 10,
    seed: int = 0,
    device: str | torch.device = "cpu",
    progress: bool = True,
) -> torch.Tensor:
    """Return indices of a subset that covers ``feats`` well.

    Distances are computed in a random linear projection to save time, which
    is the approximation used by the reference PatchCore implementation.
    Each step adds the point farthest from the current subset.
    """
    n = len(feats)
    k = max(1, int(n * ratio))
    if k >= n:
        return torch.arange(n)

    gen = torch.Generator().manual_seed(seed)
    if projection_dim and projection_dim < feats.shape[1]:
        proj = torch.randn(feats.shape[1], projection_dim, generator=gen) / projection_dim**0.5
        x = feats.float() @ proj
    else:
        x = feats.float()
    x = x.to(device)

    start = torch.randperm(n, generator=gen)[: min(start_points, n)].to(device)
    min_dist = torch.cdist(x, x[start]).mean(dim=1)

    chosen = torch.empty(k, dtype=torch.long)
    for i in tqdm(range(k), desc="coreset", disable=not progress, leave=False):
        idx = int(torch.argmax(min_dist))
        chosen[i] = idx
        min_dist = torch.minimum(min_dist, torch.cdist(x, x[idx : idx + 1]).squeeze(1))
    return chosen
