"""Experiment configuration.

One dataclass covers every variant in the ablation. The four YAML files in
``configs/`` only flip the switches in ``ModelConfig``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

PositionalEncoding = Literal["none", "sincos", "learnable"]


@dataclass
class DataConfig:
    root: str = "data"
    """Folder with ``train/normal``, ``test/normal`` and ``test/abnormal``."""
    image_size: tuple[int, int] = (256, 128)
    """(height, width) of the MIP images fed to the backbone."""
    rotate_180: bool = True
    """The source MIPs are stored upside down; flip them to head-up."""
    imagenet_norm: bool = False
    """Apply ImageNet mean/std. The original PatchCore baseline used it."""
    extensions: tuple[str, ...] = (".tif", ".tiff", ".png")
    batch_size: int = 10
    num_workers: int = 0


@dataclass
class ModelConfig:
    self_attention: bool = False
    """Parameter-free self-attention over the layer3 feature map."""
    attention_scale: bool = False
    """Divide logits by sqrt(C). Off reproduces the thesis; see README."""
    normalizing_flow: bool = False
    positional_encoding: PositionalEncoding = "none"
    """``sincos``: fixed 2D encoding. ``learnable``: sincos for the flow plus
    the cluster-contrastive LPE branch."""
    embed_dim: int = 1536
    """Channel size of every patch vector stored in the memory bank."""
    pretrained: bool = True
    search_backend: Literal["auto", "faiss", "torch"] = "auto"
    """Nearest-neighbour search. ``auto`` avoids faiss on macOS."""


@dataclass
class FlowConfig:
    condition_dim: int = 1024
    coupling_blocks: int = 25
    clamp_alpha: float = 2.8
    epochs: int = 100
    lr: float = 1e-4
    fiber_batch: int = 1024
    """Number of patch vectors per optimisation step."""
    patience: int = 5


@dataclass
class LPEConfig:
    clusters: int = 7
    """k for k-means. 7 roughly matches head / arms / torso / legs."""
    embed_dim: int = 256
    hidden_dim: int = 512
    tau: float = 0.07
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 512
    kmeans_max_samples: int = 200_000
    """Subsample patch vectors before k-means to keep it fast."""
    test_onehot: Literal["uniform", "kmeans"] = "uniform"
    """How cluster IDs are given at test time. The thesis uses ``uniform``."""


@dataclass
class CoresetConfig:
    ratio: float = 0.4
    projection_dim: int = 1024
    start_points: int = 10


@dataclass
class Config:
    name: str = "experiment"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    lpe: LPEConfig = field(default_factory=LPEConfig)
    coreset: CoresetConfig = field(default_factory=CoresetConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, values: dict[str, Any] | None):
    values = values or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        default = getattr(cls(), key)
        if hasattr(default, "__dataclass_fields__"):
            kwargs[key] = _build(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return _build(Config, raw)
