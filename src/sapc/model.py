"""PatchCore and its three extensions behind one class.

    variant                     self_attention  normalizing_flow  positional_encoding
    PatchCore                   off             off               none
    Self-Attention PatchCore    on              off               none
    + Normalizing Flow          on              on                sincos
    + Learnable PE (proposed)   on              on                learnable

Pipeline for one image (proposed model)::

    WRN50 layer2 (512x32x16) ──────────────┐
                                           ├─ + sincos PE ─ flow ─ z2 ─┐
    WRN50 layer3 (1024x16x8) ─ self-attn ──┘                           │
                     │       └─ + sincos PE ─ flow ─ z3 ─ upsample ────┤
                     │                                                 ├─ concat ─ pool to 1536 ─ memory bank
    layer2/3 copies ─┴─ k-means id ─ LPE MLP ─ e2, e3 ─ upsample ──────┘
"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import layers
from .backbone import FeatureExtractor
from .config import Config
from .coreset import greedy_coreset
from .flow import build_flow, flow_loss
from .lpe import LPEHead, fit_kmeans, one_hot_clusters, supervised_info_nce

log = logging.getLogger(__name__)

LAYER_CHANNELS = (512, 1024)


@dataclass
class Prediction:
    image_scores: np.ndarray  # (N,)  max patch distance per image
    patch_scores: np.ndarray  # (N, H, W)
    labels: np.ndarray  # (N,)
    indices: np.ndarray  # (N,)  dataset indices, to map back to files


class PatchCoreModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        if m.positional_encoding != "none" and not m.normalizing_flow:
            raise ValueError("positional_encoding is only used together with normalizing_flow")

        self.backbone = FeatureExtractor(pretrained=m.pretrained)
        self.use_flow = m.normalizing_flow
        self.use_lpe = m.positional_encoding == "learnable"

        self.flows = nn.ModuleList()
        if self.use_flow:
            p = cfg.flow.condition_dim
            self.flows = nn.ModuleList(build_flow(c + p, cfg.flow) for c in LAYER_CHANNELS)

        self.lpe_heads = nn.ModuleList()
        self.kmeans: list = []
        if self.use_lpe:
            self.lpe_heads = nn.ModuleList(
                LPEHead(c, cfg.lpe.clusters, cfg.lpe.embed_dim, cfg.lpe.hidden_dim) for c in LAYER_CHANNELS
            )

        self.register_buffer("memory_bank", torch.empty(0), persistent=True)
        self._pe_cache: dict = {}

    # ------------------------------------------------------------------ utils
    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _pe(self, h: int, w: int) -> torch.Tensor:
        key = (h, w, self.device)
        if key not in self._pe_cache:
            self._pe_cache[key] = layers.sincos_2d(self.cfg.flow.condition_dim, h, w, self.device)
        return self._pe_cache[key]

    def _branch_inputs(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return (processed features, raw backbone features) for layer2/layer3.

        layer2 is used as is. layer3 goes through self-attention when enabled.
        The original notebooks also called ``adaptive_avg_pool2d`` with the
        input size, which is a no-op and is dropped here.
        """
        f2, f3 = self.backbone(x)
        if self.cfg.model.self_attention:
            f3_proc = layers.spatial_self_attention(f3, scale=self.cfg.model.attention_scale)
        else:
            f3_proc = f3
        return [f2, f3_proc], [f2, f3]

    def _flow_inputs(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = feat.shape
        pe = self._pe(h, w).unsqueeze(0).expand(b, -1, -1, -1)
        return layers.to_patches(torch.cat([feat, pe], dim=1)), layers.to_patches(pe)

    # -------------------------------------------------------------- embedding
    @torch.no_grad()
    def embed(self, x: torch.Tensor, cluster_mode: str = "uniform") -> tuple[torch.Tensor, tuple[int, int]]:
        """Patch vectors for a batch: (B*H*W, embed_dim) on the layer2 grid."""
        b = x.shape[0]
        feats, raws = self._branch_inputs(x)
        grid = tuple(feats[0].shape[-2:])
        maps = []
        for i, (feat, raw) in enumerate(zip(feats, raws)):
            _, _, h, w = feat.shape
            out = feat
            if self.use_flow:
                inp, cond = self._flow_inputs(feat)
                z, _ = self.flows[i](inp, c=[cond])
                out = layers.from_patches(z, b, h, w)
            if self.use_lpe:
                raw_p = layers.to_patches(raw)
                if cluster_mode == "kmeans":
                    code = one_hot_clusters(self.kmeans[i], raw_p)
                else:
                    code = self.lpe_heads[i].uniform_code(len(raw_p), raw_p.device)
                e = self.lpe_heads[i](raw_p, code)
                out = torch.cat([out, layers.from_patches(e, b, h, w)], dim=1)
            maps.append(layers.resize_to(out, grid))
        patches = layers.to_patches(torch.cat(maps, dim=1))
        return layers.pool_channels(patches, self.cfg.model.embed_dim), grid

    # ---------------------------------------------------------------- training
    def train_flows(self, loader: DataLoader) -> list[float]:
        cfg = self.cfg.flow
        opt = torch.optim.Adam(self.flows.parameters(), lr=cfg.lr)
        history: list[float] = []
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, cfg.epochs + 1):
            self.flows.train()
            total, steps = 0.0, 0
            for x, _, _ in tqdm(loader, desc=f"flow {epoch}/{cfg.epochs}", leave=False):
                x = x.to(self.device)
                feats, _ = self._branch_inputs(x)
                for flow, feat in zip(self.flows, feats):
                    inp, cond = self._flow_inputs(feat)
                    perm = torch.randperm(len(inp), device=inp.device)
                    fb = min(cfg.fiber_batch, len(inp))
                    for k in range(len(inp) // fb):
                        idx = perm[k * fb : (k + 1) * fb]
                        z, log_jac = flow(inp[idx], c=[cond[idx]])
                        loss = flow_loss(z, log_jac)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        total += loss.item()
                        steps += 1
            avg = total / steps
            history.append(avg)
            log.info("flow epoch %d: loss %.4f", epoch, avg)
            if avg < best:
                best, best_state, bad = avg, copy.deepcopy(self.flows.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    log.info("early stop after %d epochs without improvement", bad)
                    break
        if best_state is not None:
            self.flows.load_state_dict(best_state)
        self.flows.eval()
        return history

    @torch.no_grad()
    def _collect_raw_patches(self, loader: DataLoader) -> list[torch.Tensor]:
        """Raw layer2/layer3 patch vectors from the training set (subsampled)."""
        cap = self.cfg.lpe.kmeans_max_samples
        per_layer: list[list[torch.Tensor]] = [[], []]
        for x, _, _ in tqdm(loader, desc="collect patches", leave=False):
            _, raws = self._branch_inputs(x.to(self.device))
            for i, raw in enumerate(raws):
                per_layer[i].append(layers.to_patches(raw).cpu())
        gen = torch.Generator().manual_seed(self.cfg.seed)
        out = []
        for chunks in per_layer:
            allp = torch.cat(chunks)
            if len(allp) > cap:
                allp = allp[torch.randperm(len(allp), generator=gen)[:cap]]
            out.append(allp)
        return out

    def train_lpe(self, loader: DataLoader) -> list[list[float]]:
        cfg = self.cfg.lpe
        patches = self._collect_raw_patches(loader)
        self.kmeans = []
        histories = []
        for i, (head, feats) in enumerate(zip(self.lpe_heads, patches)):
            km = fit_kmeans(feats.numpy(), cfg.clusters, cfg.kmeans_max_samples, self.cfg.seed)
            self.kmeans.append(km)
            sizes = np.bincount(km.labels_, minlength=cfg.clusters)
            log.info("layer%d k-means cluster sizes: %s", i + 2, sizes.tolist())

            ids = torch.from_numpy(km.predict(feats.numpy())).long()
            ds = torch.utils.data.TensorDataset(feats, ids)
            dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
            opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
            head.train()
            hist = []
            for epoch in range(1, cfg.epochs + 1):
                total, steps = 0.0, 0
                for f, cid in dl:
                    f, cid = f.to(self.device), cid.to(self.device)
                    code = torch.nn.functional.one_hot(cid, cfg.clusters).float()
                    loss = supervised_info_nce(head(f, code), cid, cfg.tau)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += loss.item()
                    steps += 1
                hist.append(total / max(steps, 1))
                log.info("lpe layer%d epoch %d: InfoNCE %.4f", i + 2, epoch, hist[-1])
            head.eval()
            histories.append(hist)
        return histories

    @torch.no_grad()
    def build_memory_bank(self, loader: DataLoader) -> None:
        bank = []
        for x, _, _ in tqdm(loader, desc="memory bank", leave=False):
            emb, _ = self.embed(x.to(self.device), cluster_mode="kmeans")
            bank.append(emb.cpu())
        bank = torch.cat(bank)
        c = self.cfg.coreset
        idx = greedy_coreset(bank, c.ratio, c.projection_dim, c.start_points, self.cfg.seed, self.device)
        self.memory_bank = bank[idx].contiguous()
        log.info("memory bank: %d -> %d patch vectors", len(bank), len(idx))

    def fit(self, train_loader: DataLoader, eval_loader: DataLoader) -> dict:
        history: dict = {}
        if self.use_flow:
            history["flow"] = self.train_flows(train_loader)
        if self.use_lpe:
            history["lpe"] = self.train_lpe(eval_loader)
        self.build_memory_bank(eval_loader)
        return history

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Prediction:
        if self.memory_bank.numel() == 0:
            raise RuntimeError("Call fit() or load a checkpoint first")
        index = NearestNeighbour(self.memory_bank, backend=self.cfg.model.search_backend)
        scores, labels, indices = [], [], []
        mode = self.cfg.lpe.test_onehot
        for x, y, idx in tqdm(loader, desc="predict", leave=False):
            emb, (h, w) = self.embed(x.to(self.device), cluster_mode=mode)
            dist = index.search(emb.cpu()).view(len(x), h, w)
            scores.append(dist)
            labels.append(y)
            indices.append(idx)
        patch = torch.cat(scores).numpy()
        return Prediction(
            image_scores=patch.reshape(len(patch), -1).max(axis=1),
            patch_scores=patch,
            labels=torch.cat(labels).numpy(),
            indices=torch.cat(indices).numpy(),
        )

    # ------------------------------------------------------------- checkpoint
    def save(self, path) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "kmeans": self.kmeans, "config": self.cfg.to_dict()}, path
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.memory_bank = ckpt["state_dict"]["memory_bank"]
        self.load_state_dict(ckpt["state_dict"])
        self.kmeans = ckpt.get("kmeans", [])


class NearestNeighbour:
    """L2 distance to the closest memory-bank vector.

    ``backend="auto"`` uses faiss on Linux/Windows and plain PyTorch on macOS,
    where faiss-cpu and PyTorch ship two OpenMP runtimes that crash together.
    """

    def __init__(self, bank: torch.Tensor, backend: str = "auto", chunk: int = 4096):
        self.chunk = chunk
        self.index = None
        self.bank = bank.float()
        if backend == "auto":
            backend = "torch" if sys.platform == "darwin" else "faiss"
        if backend == "faiss":
            try:
                import faiss
            except ImportError:
                log.warning("faiss not installed, falling back to torch search")
            else:
                self.index = faiss.IndexFlatL2(bank.shape[1])
                self.index.add(self.bank.numpy())

    def search(self, query: torch.Tensor) -> torch.Tensor:
        query = query.float()
        if self.index is not None:
            d, _ = self.index.search(query.numpy(), 1)
            return torch.from_numpy(d[:, 0]).clamp_min(0).sqrt()
        out = [torch.cdist(q, self.bank).min(dim=1).values for q in query.split(self.chunk)]
        return torch.cat(out)
