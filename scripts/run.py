"""Fit a model on normal MIPs and evaluate it on the test split.

    python scripts/run.py --config configs/sa_patchcore_nf_lpe.yaml
    python scripts/run.py --config configs/patchcore.yaml --set data.root=/path/to/data
    python scripts/run.py --config configs/sa_patchcore_nf.yaml --checkpoint outputs/x/model.pt

Writes to ``outputs/<name>/``: config.yaml, model.pt, metrics.json,
scores.csv, score_histogram.png and (with --maps) anomaly-map overlays.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sapc import PatchCoreModel, evaluate, load_config  # noqa: E402
from sapc.data import build_loaders, load_image  # noqa: E402
from sapc.utils import resolve_device, seed_everything, setup_logging  # noqa: E402
from sapc.visualize import plot_scores, save_anomaly_maps  # noqa: E402

log = logging.getLogger("run")


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--checkpoint", help="skip training and load this model.pt")
    ap.add_argument("--maps", action="store_true", help="save anomaly-map overlays")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    log.info("run %s on %s", cfg.name, device)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    train_loader, test_loader, test_set = build_loaders(cfg.data)
    log.info("train %d normal | test %d images", len(train_loader.dataset), len(test_set))

    model = PatchCoreModel(cfg).to(device)
    if args.checkpoint:
        model.load(args.checkpoint)
    else:
        history = model.fit(train_loader, train_loader.eval_loader)
        model.save(run_dir / "model.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    pred = model.predict(test_loader)
    metrics = evaluate(pred.image_scores, pred.labels)
    log.info(metrics.summary())
    (run_dir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    plot_scores(run_dir / "score_histogram.png", pred.image_scores, pred.labels, metrics.threshold)

    with open(run_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "score"])
        for i, y, s in zip(pred.indices, pred.labels, pred.image_scores):
            w.writerow([int(i), int(y), f"{s:.6f}"])

    if args.maps:
        preds = (pred.image_scores >= metrics.threshold).astype(int)
        images = [load_image(test_set.files[i], cfg.data) for i in pred.indices]
        names = [f"{int(i):04d}" for i in pred.indices]
        save_anomaly_maps(run_dir / "maps", images, pred.patch_scores, pred.labels, preds, names)
        log.info("anomaly maps saved to %s", run_dir / "maps")


if __name__ == "__main__":
    main()
