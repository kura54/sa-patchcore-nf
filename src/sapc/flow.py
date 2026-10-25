"""Conditional normalizing flow that Gaussianises patch features.

One flow per backbone layer. Each patch vector (features concatenated with its
positional encoding) is mapped to z, and the flow is trained so that z follows
a standard normal distribution. The positional encoding is also passed as the
condition of every coupling block, following CFLOW-AD.
"""

from __future__ import annotations

import math

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
from torch import nn

from .config import FlowConfig

_LOG_2PI_HALF = 0.5 * math.log(2 * math.pi)


def _subnet(dims_in: int, dims_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dims_in, 2 * dims_in),
        nn.ReLU(),
        nn.Linear(2 * dims_in, dims_out),
    )


def build_flow(input_dim: int, cfg: FlowConfig) -> Ff.SequenceINN:
    flow = Ff.SequenceINN(input_dim)
    for _ in range(cfg.coupling_blocks):
        flow.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(cfg.condition_dim,),
            subnet_constructor=_subnet,
            affine_clamping=cfg.clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=True,
        )
    return flow


def log_likelihood(z: torch.Tensor, log_jac_det: torch.Tensor) -> torch.Tensor:
    """log p(x) under a standard normal prior, per sample."""
    dim = z.shape[1]
    return -dim * _LOG_2PI_HALF - 0.5 * z.pow(2).sum(dim=1) + log_jac_det


def flow_loss(z: torch.Tensor, log_jac_det: torch.Tensor) -> torch.Tensor:
    """Negative log-sigmoid of the per-dimension log-likelihood.

    Equivalent to the KL objective in the paper up to constants, but bounded,
    which kept training stable with 25 coupling blocks.
    """
    per_dim = log_likelihood(z, log_jac_det) / z.shape[1]
    return -nn.functional.logsigmoid(per_dim).mean()
