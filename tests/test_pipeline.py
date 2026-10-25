"""End-to-end smoke test on synthetic images.

Runs every variant with a randomly initialised backbone and tiny settings, so
it checks wiring and shapes, not accuracy.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from sapc import PatchCoreModel, evaluate
from sapc.config import Config, CoresetConfig, DataConfig, FlowConfig, LPEConfig, ModelConfig
from sapc.data import build_loaders

VARIANTS = {
    "patchcore": dict(self_attention=False, normalizing_flow=False, positional_encoding="none"),
    "sa": dict(self_attention=True, normalizing_flow=False, positional_encoding="none"),
    "sa_nf": dict(self_attention=True, normalizing_flow=True, positional_encoding="sincos"),
    "sa_nf_lpe": dict(self_attention=True, normalizing_flow=True, positional_encoding="learnable"),
}


def _write_images(folder: Path, n: int, bright_spot: bool, rng) -> None:
    folder.mkdir(parents=True)
    for i in range(n):
        img = (rng.random((64, 32)) * 2000).astype(np.uint16)
        if bright_spot:
            y, x = rng.integers(8, 56), rng.integers(4, 28)
            img[y - 3 : y + 3, x - 3 : x + 3] = 65535
        cv2.imwrite(str(folder / f"{i}.tif"), img)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("mip")
    _write_images(root / "train" / "normal", 8, False, rng)
    _write_images(root / "test" / "normal", 4, False, rng)
    _write_images(root / "test" / "abnormal", 4, True, rng)
    return root


def _config(root: Path, variant: str, tmp_path: Path) -> Config:
    return Config(
        name=variant,
        output_dir=str(tmp_path),
        device="cpu",
        data=DataConfig(root=str(root), image_size=(64, 32), batch_size=4),
        model=ModelConfig(pretrained=False, embed_dim=64, **VARIANTS[variant]),
        flow=FlowConfig(condition_dim=8, coupling_blocks=2, epochs=2, fiber_batch=16),
        lpe=LPEConfig(clusters=3, embed_dim=16, hidden_dim=32, epochs=1, batch_size=16),
        coreset=CoresetConfig(ratio=0.5, projection_dim=32),
    )


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_fit_predict_save_load(variant, data_root, tmp_path):
    torch.manual_seed(0)
    cfg = _config(data_root, variant, tmp_path)
    train_loader, test_loader, test_set = build_loaders(cfg.data)
    model = PatchCoreModel(cfg)
    model.fit(train_loader, train_loader.eval_loader)

    n_train_patches = 8 * 8 * 4  # images * layer2 grid (64/8 x 32/8)
    assert model.memory_bank.shape == (n_train_patches // 2, 64)

    pred = model.predict(test_loader)
    assert pred.image_scores.shape == (8,)
    assert pred.patch_scores.shape == (8, 8, 4)
    assert np.isfinite(pred.image_scores).all()
    evaluate(pred.image_scores, pred.labels)

    path = tmp_path / "model.pt"
    model.save(path)
    restored = PatchCoreModel(cfg)
    restored.load(path)
    again = restored.predict(test_loader)
    np.testing.assert_allclose(again.image_scores, pred.image_scores, rtol=1e-4, atol=1e-5)
