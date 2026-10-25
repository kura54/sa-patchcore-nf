"""Image-level evaluation: AUROC plus accuracy / sensitivity / specificity at
the threshold that maximises Youden's J."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class Metrics:
    auroc: float
    accuracy: float
    sensitivity: float
    specificity: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"AUROC {self.auroc:.3f} | ACC {self.accuracy:.3f} | "
            f"SEN {self.sensitivity:.3f} | SPE {self.specificity:.3f}"
        )


def evaluate(scores: np.ndarray, labels: np.ndarray) -> Metrics:
    """``labels``: 1 = abnormal. Higher score = more anomalous.

    Note that the threshold is chosen on the same scores it is evaluated on,
    as in the thesis. Use a separate validation split for an unbiased estimate.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best = int(np.argmax(tpr - fpr))
    thr = float(thresholds[best])
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return Metrics(
        auroc=float(roc_auc_score(labels, scores)),
        accuracy=(tp + tn) / len(labels),
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
        threshold=thr,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )
