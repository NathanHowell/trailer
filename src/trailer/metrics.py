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

from . import osm

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


#: Operating points searched when scoring per class. Wider than sweep()'s three
#: because the best threshold is being looked for, not reported at.
SELECTION_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class Stratified:
    """Relaxed scoring split by visibility class, accumulated over a whole set.

    Two things this fixes about selecting on pooled ``f1@0.5``.

    *Pooled recall is length-weighted.* Whichever class has the most labelled
    kilometres in the validation band decides the score, and the model can buy
    its way to a better number by getting better at the class we already have.
    Faint trail is the one the project exists for and the one with the worst SNR
    (median 1.97, p10 0.60 against 5.61 for active); it must not be able to
    disappear into an average.

    *0.5 is not a fixed point.* The output bias is set from the sampled prior and
    the loss reweights positives, so where the sigmoid sits moves between runs
    for reasons that have nothing to do with whether the model ranks pixels well.
    Selection takes the best F1 over a threshold grid per class, which measures
    the ranking; calibration is a deployment choice made later.

    Counts are summed across batches and reduced once, rather than scored per
    batch and averaged. A 256 px crop often holds no pixel of a given class at
    all, so per-batch recall is frequently 0/0; averaging those makes the score
    depend on how crops fell into batches, and lets every batch pick its own
    threshold.

    Precision is deliberately *not* stratified. A predicted pixel does not belong
    to a class -- and a highlight drawn over a lifecycle trail is not a false
    positive just because active recall is being measured. So each class's F1
    combines that class's recall with the pooled precision, which is the honest
    reading: of everything we drew, how much was trail of any kind.
    """

    def __init__(self, res: float, classes: tuple[str, ...] | None = None,
                 thresholds: tuple[float, ...] = SELECTION_THRESHOLDS):
        self.radius = max(int(round(TOLERANCE_M / res)), 1)
        self.code = ({c: osm.CLASS_CODE[c] for c in classes} if classes
                     else dict(osm.CLASS_CODE))
        self.thresholds = tuple(thresholds)
        self.y_sum = {c: 0.0 for c in self.code}
        self.pred_sum = {t: 0.0 for t in self.thresholds}
        self.tp_p = {t: 0.0 for t in self.thresholds}
        self.tp_r = {t: {c: 0.0 for c in self.code} for t in self.thresholds}

    @torch.no_grad()
    def update(self, prob: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
               cls: torch.Tensor) -> None:
        valid = (w > 0).to(prob.dtype)
        # Dilated over all classes: a prediction near any trail is on trail.
        y_near = _dilate(y, self.radius) * valid
        per_class = {c: y * valid * (cls == code).to(prob.dtype)
                     for c, code in self.code.items()}
        for c, a in per_class.items():
            self.y_sum[c] += float(a.sum())
        for t in self.thresholds:
            pred = (prob >= t).to(prob.dtype) * valid
            p_near = _dilate(pred, self.radius) * valid
            self.pred_sum[t] += float(pred.sum())
            self.tp_p[t] += float((pred * y_near).sum())
            for c, a in per_class.items():
                self.tp_r[t][c] += float((a * p_near).sum())

    def result(self) -> dict:
        """Per-class best F1, and their mean as the selection score.

        Classes with no labelled pixel in the set are dropped, not scored zero:
        a validation band that happens to contain no lifecycle way says nothing
        about lifecycle recall, and scoring it zero would make the checkpoint
        choice depend on that accident.
        """
        out: dict[str, dict] = {}
        for c, ysum in self.y_sum.items():
            if ysum <= 0:
                continue
            best = None
            for t in self.thresholds:
                p = self.tp_p[t] / (self.pred_sum[t] + EPS)
                r = self.tp_r[t][c] / (ysum + EPS)
                f1 = 2 * p * r / (p + r + EPS)
                if best is None or f1 > best[0]:
                    best = (f1, p, r, t)
            f1, p, r, t = best
            # Labelled pixels behind this class's recall. Carried so a consumer
            # can tell a measurement from a rumour: several held-out tiles hold
            # a class on a few hundred metres of way, and a spread that pools
            # those with kilometre-scale ones reads as model instability when it
            # is sample size.
            out[c] = {"f1": round(f1, 4), "p": round(p, 4),
                      "r": round(r, 4), "t": t, "px": int(ysum)}
        score = float(np.mean([v["f1"] for v in out.values()])) if out else 0.0
        return {"by_class": out, "score": round(score, 4),
                "classes": sorted(out)}


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


#: Labelled body pixels a class needs in a tile before its F1 there is treated
#: as a measurement. A trail is 4-9 m of bench at 1 m output, so this is
#: roughly a kilometre of way -- the same floor the held-out tiles were
#: selected on.
MIN_CLASS_PX = 4000


def held_out_spread(held_out: dict, min_px: int = MIN_CLASS_PX) -> dict:
    """Per-class held-out F1 across eval AOIs, as a spread rather than a number.

    A single held-out tile per class is not an estimate. The same checkpoint
    scores per-tile lifecycle F1 anywhere from 0.00 to 0.94 across this corpus,
    so one draw says nothing about the model -- which is exactly how a topo-map
    trace with no measurable tread came to be read as a capability gap.

    Tiles carrying an ``advisory`` are excluded from the aggregate but not from
    the report: their number is not evidence, and averaging it in would launder
    that back into one. See ``aois.Aoi.advisory``.

    A wide spread here is information, not noise. Some held-out tiles hold a
    class at a tread the QA transects can barely see, and a class that swings by
    0.5 across AOIs is a fact about how much a deployment claim can lean on any
    one of them.
    """
    out: dict[str, dict] = {}
    for variant, tiles in held_out.items():
        per_class: dict[str, list] = {}
        thin: dict[str, list] = {}
        for name, rec in sorted(tiles.items()):
            if rec.get("advisory"):
                continue
            for c, v in rec.get("strat", {}).get("by_class", {}).items():
                # A class present on a few hundred metres says nothing about
                # recall for it. Dropped from the aggregate, and counted so the
                # report can say how many were dropped rather than going quiet.
                if v.get("px", min_px) < min_px:
                    thin.setdefault(c, []).append(name)
                    continue
                per_class.setdefault(c, []).append((name, v["f1"]))
        summary = {}
        for c, pairs in sorted(per_class.items()):
            vals = sorted(f1 for _, f1 in pairs)
            summary[c] = {
                "n": len(vals),
                "median": round(float(np.median(vals)), 4),
                "min": round(vals[0], 4),
                "max": round(vals[-1], 4),
                "tiles": {n: f1 for n, f1 in sorted(pairs, key=lambda t: t[1])},
                "too_thin": sorted(thin.get(c, [])),
            }
        out[variant] = summary
    return out
