import math

import numpy as np
import pytest
import torch

from sapc import layers
from sapc.coreset import greedy_coreset
from sapc.lpe import supervised_info_nce
from sapc.metrics import evaluate


def test_sincos_shape_and_values():
    pe = layers.sincos_2d(8, 3, 5)
    assert pe.shape == (8, 3, 5)
    # x-encoding is constant along y, y-encoding constant along x
    assert torch.allclose(pe[0, 0], pe[0, 2])
    assert torch.allclose(pe[4, :, 0], pe[4, :, 3])
    assert pe[0, 0, 0] == 0 and pe[1, 0, 0] == 1  # sin(0), cos(0)


def test_sincos_rejects_bad_dim():
    with pytest.raises(ValueError):
        layers.sincos_2d(6, 2, 2)


def test_patch_roundtrip():
    x = torch.randn(2, 3, 4, 5)
    p = layers.to_patches(x)
    assert p.shape == (2 * 4 * 5, 3)
    assert torch.equal(layers.from_patches(p, 2, 4, 5), x)


def test_self_attention_preserves_shape_and_mixes():
    x = torch.randn(2, 16, 4, 2)
    y = layers.spatial_self_attention(x, scale=True)
    assert y.shape == x.shape
    # each output vector is a convex combination of inputs
    assert torch.all(y.flatten(2).abs().amax(-1) <= x.flatten(2).abs().amax(-1) + 1e-5)


def test_unscaled_attention_is_nearly_identity_for_large_c():
    torch.manual_seed(0)
    x = torch.rand(1, 1024, 4, 2)
    y = layers.spatial_self_attention(x, scale=False)
    assert torch.allclose(x, y, atol=1e-3)


def test_info_nce_prefers_clustered_embeddings():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    good = torch.eye(3).repeat_interleave(2, dim=0)
    bad = torch.randn(6, 3)
    assert supervised_info_nce(good, labels) < supervised_info_nce(bad, labels)
    assert math.isfinite(supervised_info_nce(good, labels).item())


def test_coreset_picks_spread_points():
    pts = torch.cat([torch.zeros(50, 2), torch.ones(50, 2) * 10, torch.tensor([[100.0, 100.0]])])
    idx = greedy_coreset(pts, ratio=0.03, projection_dim=0, progress=False)
    chosen = pts[idx]
    assert len(idx) == 3
    assert (chosen == torch.tensor([100.0, 100.0])).all(dim=1).any()


def test_metrics_perfect_separation():
    m = evaluate(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert m.auroc == 1.0 and m.accuracy == 1.0
    assert (m.tp, m.tn, m.fp, m.fn) == (2, 2, 0, 0)
