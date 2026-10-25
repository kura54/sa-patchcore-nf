"""Project memory-bank patch vectors to 2D and colour them by body region.

Used to check that the learnable positional encoding groups patches by
anatomy. Region masks come from LabelMe polygon files whose labels are one of
``head, left arm, right arm, upper body, lower body, left foot, right foot``.

    python scripts/feature_atlas.py \
        --config configs/sa_patchcore_nf_lpe.yaml \
        --checkpoint outputs/sa_patchcore_nf_lpe/model.pt \
        --annotations path/to/labelme_jsons --method tsne
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sapc import PatchCoreModel, load_config  # noqa: E402
from sapc.data import load_image  # noqa: E402
from sapc.utils import resolve_device, seed_everything  # noqa: E402

REGIONS = [
    "background",
    "head",
    "left arm",
    "right arm",
    "upper body",
    "lower body",
    "right foot",
    "left foot",
]


def region_map(
    annotation: Path, image_hw: tuple[int, int], grid_hw: tuple[int, int], rotate_180: bool
) -> np.ndarray:
    data = json.loads(annotation.read_text())
    h, w = data.get("imageHeight", image_hw[0]), data.get("imageWidth", image_hw[1])
    mask = np.zeros((h, w), np.int32)
    for shape in data["shapes"]:
        if shape["label"] in REGIONS:
            pts = np.asarray(shape["points"], np.int32)
            cv2.fillPoly(mask, [pts], REGIONS.index(shape["label"]))
    if rotate_180:
        mask = cv2.rotate(mask, cv2.ROTATE_180)
    gh, gw = grid_hw
    return cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_NEAREST).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--annotations", required=True, help="folder of LabelMe .json files")
    ap.add_argument("--images", help="image folder (default: <data.root>/train/normal)")
    ap.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    ap.add_argument("--max-points", type=int, default=20000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    model = PatchCoreModel(cfg).to(device)
    model.load(args.checkpoint)

    image_dir = Path(args.images or Path(cfg.data.root) / "train" / "normal")
    feats, regions = [], []
    for ann in sorted(Path(args.annotations).glob("*.json")):
        name = Path(json.loads(ann.read_text())["imagePath"]).name
        img = load_image(image_dir / name, cfg.data)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        emb, grid = model.embed(x, cluster_mode=cfg.lpe.test_onehot)
        r = region_map(ann, img.shape[:2], grid, cfg.data.rotate_180)
        keep = r > 0
        feats.append(emb.cpu().numpy()[keep])
        regions.append(r[keep])
        if sum(len(f) for f in feats) >= args.max_points:
            break
    feats_np, regions_np = np.concatenate(feats), np.concatenate(regions)
    print(f"{len(feats_np)} labelled patch vectors from {len(feats)} images")

    if args.method == "tsne":
        from sklearn.manifold import TSNE

        xy = TSNE(n_components=2, perplexity=30, random_state=cfg.seed).fit_transform(feats_np)
    else:
        import umap  # optional dependency

        xy = umap.UMAP(n_components=2, random_state=cfg.seed).fit_transform(feats_np)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for rid in np.unique(regions_np):
        sel = regions_np == rid
        ax.scatter(xy[sel, 0], xy[sel, 1], s=4, alpha=0.6, label=REGIONS[rid])
    ax.legend(markerscale=4, fontsize=8)
    ax.set_title(f"{cfg.name}: patch embeddings by body region ({args.method})")
    ax.set_xticks([])
    ax.set_yticks([])
    out = Path(args.out or cfg.run_dir / f"feature_atlas_{args.method}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
