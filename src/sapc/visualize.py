"""Anomaly-map overlays and score plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def overlay(image: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0, 1] heat map (H, W) onto an RGB float image in [0, 1].

    Returns uint8 BGR, ready for ``cv2.imwrite``.
    """
    h, w = image.shape[:2]
    heat = cv2.resize(heat.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base, alpha, color, 1 - alpha, 0)


def save_anomaly_maps(
    out_dir: Path,
    images: list[np.ndarray],
    patch_scores: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    names: list[str],
) -> None:
    """Write one overlay per image, grouped into TP / FP / TN / FN folders.

    All maps share one colour scale (global min/max) so they can be compared.
    """
    lo, hi = float(patch_scores.min()), float(patch_scores.max())
    for img, score, y, p, name in zip(images, patch_scores, labels, preds, names):
        group = {(1, 1): "TP", (0, 1): "FP", (0, 0): "TN", (1, 0): "FN"}[(int(y), int(p))]
        folder = out_dir / group
        folder.mkdir(parents=True, exist_ok=True)
        heat = (score - lo) / (hi - lo + 1e-8)
        cv2.imwrite(str(folder / f"{name}.png"), overlay(img, heat))


def plot_scores(path: Path, scores: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="normal")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="abnormal")
    ax.axvline(threshold, color="k", ls="--", lw=1, label="Youden threshold")
    ax.set_xlabel("image anomaly score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
