"""MIP image datasets.

Expected layout::

    data/
      train/normal/*.tif
      test/normal/*.tif
      test/abnormal/*.tif

Images are single-channel MIPs (8- or 16-bit). They are scaled to [0, 1] and
repeated to three channels for the ImageNet backbone.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def list_images(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing image folder: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError(f"No images with {extensions} in {folder}")
    return files


def load_image(path: Path, cfg: DataConfig) -> np.ndarray:
    """Read an image as float32 (H, W, 3) in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Could not read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    else:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / (hi - lo + 1e-8)
    h, w = cfg.image_size
    if img.shape != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if cfg.rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return np.repeat(img[..., None], 3, axis=2)


class MIPDataset(Dataset):
    def __init__(self, files: list[Path], labels: list[int], cfg: DataConfig):
        assert len(files) == len(labels)
        self.files = files
        self.labels = labels
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = load_image(self.files[idx], self.cfg)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        if self.cfg.imagenet_norm:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, self.labels[idx], idx


def build_loaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, MIPDataset]:
    root = Path(cfg.root)
    train_files = list_images(root / "train" / "normal", cfg.extensions)
    normal = list_images(root / "test" / "normal", cfg.extensions)
    abnormal = list_images(root / "test" / "abnormal", cfg.extensions)

    train_set = MIPDataset(train_files, [0] * len(train_files), cfg)
    test_set = MIPDataset(normal + abnormal, [0] * len(normal) + [1] * len(abnormal), cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    # Unshuffled copy for building the memory bank and fitting k-means.
    train_eval_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    train_loader.eval_loader = train_eval_loader  # type: ignore[attr-defined]
    return train_loader, test_loader, test_set
