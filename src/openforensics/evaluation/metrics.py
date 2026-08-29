"""Prediction, calibration and threshold selection.

Accuracy at a 0.5 threshold is the least useful number a forensic classifier
can report. What a caller needs is a chosen operating point, a confidence
value that means what it says, and a statement of which degradations the
model survives. That is what lives here.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)

EPS = 1e-7


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------
def predict(model, ds, tta: bool = False) -> np.ndarray:
    """Predict P(Real). With `tta`, average over the image and its mirror.

    Horizontal flip is the only safe test-time transform here: it is the one
    augmentation that provably preserves the label for faces, so averaging
    over it reduces variance without shifting the distribution.
    """
    probs = model.predict(ds, verbose=0).ravel()
    if not tta:
        return probs
    flipped = ds.map(lambda x, y: (tf.image.flip_left_right(x), y))
    return (probs + model.predict(flipped, verbose=0).ravel()) / 2.0


def labels_of(ds) -> np.ndarray:
    return np.concatenate([y.numpy().ravel() for _, y in ds])


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_temperature(y: np.ndarray, probs: np.ndarray) -> float:
    """Fit a single temperature on held-out data by minimising NLL.

    Networks trained with cross-entropy are systematically overconfident.
    Dividing the logit by one scalar fixes most of that without touching
    accuracy -- the ranking is unchanged, so AUC is identical before and
    after; only the numbers attached to the decision move.
    """
    logits = _to_logit(probs)
    best_t, best_nll = 1.0, np.inf
    for t in np.geomspace(0.25, 8.0, 200):
        q = 1.0 / (1.0 + np.exp(-logits / t))
        q = np.clip(q, EPS, 1 - EPS)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_to_logit(probs) / t))


def expected_calibration_error(y, probs, bins: int = 15) -> float:
    """Average gap between stated confidence and observed accuracy."""
    conf = np.maximum(probs, 1 - probs)
    correct = (probs >= 0.5).astype(int) == y
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# --------------------------------------------------------------------------
# threshold selection
# --------------------------------------------------------------------------
def select_threshold(y, probs, criterion: str = "youden",
                     target: float = 0.95) -> tuple[float, dict]:
    """Choose an operating point on validation, never on test.

    criterion:
      youden          maximise TPR - FPR (balanced)
      f1              maximise F1
      target_recall   cheapest threshold meeting recall >= target on Real
      min_fpr         highest threshold whose false-positive rate <= 1-target
    """
    fpr, tpr, thr = roc_curve(y, probs)
    if criterion == "youden":
        i = int(np.argmax(tpr - fpr))
        t = float(thr[i])
    elif criterion == "f1":
        grid = np.linspace(0.01, 0.99, 197)
        t = float(grid[int(np.argmax([f1_score(y, probs >= g) for g in grid]))])
    elif criterion == "target_recall":
        ok = np.where(tpr >= target)[0]
        t = float(thr[ok[0]]) if len(ok) else 0.5
    elif criterion == "min_fpr":
        ok = np.where(fpr <= (1 - target))[0]
        t = float(thr[ok[-1]]) if len(ok) else 0.5
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    t = float(np.clip(t, 0.01, 0.99))
    return t, summarise(y, probs, t)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def summarise(y, probs, threshold: float = 0.5) -> dict:
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(set(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, probs)) if len(set(y)) > 1 else None,
        "confusion_matrix": cm.tolist(),
        # Named explicitly: "false positive" is ambiguous when the positive
        # class is Real. This is a genuine image accused of being fake.
        "real_called_fake": int(fn),
        "fake_called_real": int(fp),
        "false_accusation_rate": float(fn / max(fn + tp, 1)),
        "ece": expected_calibration_error(y, probs),
        "n": int(len(y)),
    }
    for name, (a, b) in {"Fake": (tn, fp), "Real": (tp, fn)}.items():
        out[f"{name.lower()}_recall"] = float(a / max(a + b, 1))
    return out


def risk_coverage(y, probs, points: int = 20) -> list[dict]:
    """Accuracy as a function of how many predictions you keep.

    Abstaining on the least confident cases is the only honest answer a
    forensic tool can give for a borderline image, and this curve says what
    that abstention buys.
    """
    conf = np.maximum(probs, 1 - probs)
    order = np.argsort(-conf)
    y_s, p_s = np.asarray(y)[order], probs[order]
    rows = []
    for frac in np.linspace(1.0, 0.1, points):
        k = max(int(len(y_s) * frac), 1)
        rows.append({
            "coverage": round(float(frac), 3),
            "n": k,
            "accuracy": float(accuracy_score(y_s[:k], (p_s[:k] >= 0.5).astype(int))),
            "min_confidence": float(np.maximum(p_s[:k], 1 - p_s[:k]).min()),
        })
    return rows
