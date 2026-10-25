"""Learnable positional encoding (LPE).

A fixed sine-cosine code says *where* a patch is on the grid, but organs move
with body shape and patient position. The LPE learns a code for *what body
region* a patch looks like instead:

1. Cluster the frozen backbone features of all training patches with k-means.
   With k = 7 the clusters line up with head, arms, torso and legs.
2. Train a small MLP on ``[feature, one_hot(cluster)]`` with a supervised
   InfoNCE loss: patches from the same cluster are positives, the rest of the
   batch are negatives.
3. Concatenate the MLP output to the flow output before the memory bank.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn


class LPEHead(nn.Module):
    def __init__(self, feat_dim: int, clusters: int, out_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.clusters = clusters
        self.net = nn.Sequential(
            nn.Linear(feat_dim + clusters, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feats: torch.Tensor, cluster_code: torch.Tensor) -> torch.Tensor:
        """``cluster_code`` is a one-hot (N, k) or a soft (N, k) distribution."""
        return self.net(torch.cat([feats, cluster_code], dim=1))

    def uniform_code(self, n: int, device) -> torch.Tensor:
        return torch.full((n, self.clusters), 1.0 / self.clusters, device=device)


def supervised_info_nce(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Multi-positive InfoNCE.

    For each anchor i, the positive term is the log-mean-exp of the similarity
    to every other sample with the same label; the denominator is over the
    whole batch. Anchors without a positive in the batch are skipped.

    Vectorised version of the per-anchor loop in the original notebook.
    """
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau  # (B, B)
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye

    log_denom = torch.logsumexp(sim, dim=1)
    pos_sim = sim.masked_fill(~pos, float("-inf"))
    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    log_pos = torch.logsumexp(pos_sim[valid], dim=1) - n_pos[valid].float().log()
    return (log_denom[valid] - log_pos).mean()


def fit_kmeans(feats: np.ndarray, k: int, max_samples: int, seed: int) -> KMeans:
    rng = np.random.default_rng(seed)
    if len(feats) > max_samples:
        feats = feats[rng.choice(len(feats), max_samples, replace=False)]
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(feats)


def one_hot_clusters(kmeans: KMeans, feats: torch.Tensor) -> torch.Tensor:
    ids = kmeans.predict(feats.detach().cpu().numpy().astype(np.float32))
    return F.one_hot(torch.from_numpy(ids).long(), kmeans.n_clusters).float().to(feats.device)
