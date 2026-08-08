"""Scoring.

Strict pixel overlap is the wrong yardstick for this task. OSM centrelines in the
High Sierra sit ~1.4 m from the true tread (84% within 3 m), and the tread itself
is 1-1.5 m wide, so a perfect prediction can miss the label by two or three
pixels at 0.5 m. Relaxed precision/recall (Wiedemann et al.) allows a tolerance
radius, which is what the JOSM reviewer actually cares about: did we draw a
highlight on top of the trail, within a couple of metres?

Everything respects the weight plane, so ignore regions never score.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6

#: Tolerance in metres for relaxed scoring, matching the ignore-ring radius.
TOLERANCE_M = 5.0


def _dilate(x: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    return F.max_pool2d(x, k, stride=1, padding=radius)


def relaxed(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
            radius: int) -> tuple[float, float]:
    """Relaxed (precision, recall) for binary masks at a tolerance radius."""
    valid = (w > 0).to(pred.dtype)
    pred = pred * valid
    y_v = y * valid

    y_near = _dilate(y, radius) * valid
    p_near = _dilate(pred, radius) * valid

    tp_p = (pred * y_near).sum()
    tp_r = (y_v * p_near).sum()
    precision = (tp_p / (pred.sum() + EPS)).item()
    recall = (tp_r / (y_v.sum() + EPS)).item()
    return precision, recall


def sweep(prob: torch.Tensor, y: torch.Tensor, w: torch.Tensor, res: float,
          thresholds=(0.3, 0.5, 0.7)) -> dict:
    """Relaxed P/R/F1 at several operating points."""
    radius = max(int(round(TOLERANCE_M / res)), 1)
    out = {}
    for t in thresholds:
        p, r = relaxed((prob >= t).to(prob.dtype), y, w, radius)
        out[f"p@{t}"] = round(p, 4)
        out[f"r@{t}"] = round(r, 4)
        out[f"f1@{t}"] = round(2 * p * r / (p + r + EPS), 4)
    return out


def average_precision(prob: np.ndarray, y: np.ndarray, w: np.ndarray,
                      max_pixels: int = 4_000_000) -> float:
    """Pixel-wise AP over unignored pixels.

    Reported for comparability with the survey baselines (hand-crafted terrain
    indices scored 0.51-0.56 AUC, TrailScan 0.649). Do not use it for model
    selection -- it rewards fattening predictions, which relaxed F1 does not.
    """
    from sklearn.metrics import average_precision_score

    m = w.ravel() > 0
    p, t = prob.ravel()[m], (y.ravel()[m] > 0.5)
    if t.sum() == 0 or t.all():
        return float("nan")
    if len(p) > max_pixels:
        idx = np.random.default_rng(0).choice(len(p), max_pixels, replace=False)
        p, t = p[idx], t[idx]
    return float(average_precision_score(t, p))


def false_positive_rate(prob: np.ndarray, w: np.ndarray,
                        threshold: float = 0.5) -> float:
    """Fraction of valid pixels fired on. For control tiles this is the whole
    story: every positive there is wrong by construction."""
    m = w.ravel() > 0
    if not m.any():
        return float("nan")
    return float((prob.ravel()[m] >= threshold).mean())
