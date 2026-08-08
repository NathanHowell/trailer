"""Segmentation losses for thin, connected structures.

Three terms, each covering a failure the others miss:

* **Weighted BCE** -- per-pixel calibration. Alone it collapses to "predict
  background" at 0.7% positive rate.
* **Tversky** (alpha=0.3, beta=0.7) -- region overlap with false negatives
  penalised harder than false positives. A human reviewing proposals in JOSM can
  dismiss a false positive in a second; a trail we never draw is invisible to
  them, so recall is worth more than precision here.
* **clDice** (Shit et al., CVPR 2021) -- topology. Dice is nearly indifferent to
  a one-pixel gap in a 1 m wide trail, but that gap is exactly what makes a
  proposal unusable. clDice scores the *skeleton*, so breaking a line is
  expensive no matter how few pixels it costs.

All three consume the same per-pixel weight plane, which carries both
trail_visibility grading and the ignore regions (weight 0).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def masked_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
               pos_weight: float = 8.0) -> torch.Tensor:
    pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw,
                                             reduction="none")
    return (raw * w).sum() / (w.sum() + EPS)


def tversky(prob: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
            alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
    """Soft Tversky. alpha weights false positives, beta false negatives."""
    dims = (1, 2, 3)
    tp = (w * prob * y).sum(dims)
    fp = (w * prob * (1 - y)).sum(dims)
    fn = (w * (1 - prob) * y).sum(dims)
    return (1 - (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)).mean()


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    a = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    b = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(a, b)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))


def soft_skeleton(x: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Differentiable morphological skeleton via iterative erosion."""
    opened = _soft_dilate(_soft_erode(x))
    skel = F.relu(x - opened)
    for _ in range(iters):
        x = _soft_erode(x)
        opened = _soft_dilate(_soft_erode(x))
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice(prob: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
           iters: int = 8) -> torch.Tensor:
    """Centreline Dice.

    The weight plane cannot simply multiply here: zeroing the ignore ring would
    punch holes through the very lines whose connectivity is being scored. So
    inside ignored pixels the prediction is replaced by the target, which leaves
    the skeleton intact while making the loss blind to what was predicted there.
    """
    keep = (w > 0).to(prob.dtype)
    prob = prob * keep + y * (1 - keep)

    skel_p = soft_skeleton(prob, iters)
    skel_y = soft_skeleton(y, iters)
    dims = (1, 2, 3)
    # precision: how much of the predicted centreline lies on true trail
    tprec = (skel_p * y).sum(dims).add(EPS) / (skel_p.sum(dims) + EPS)
    # sensitivity: how much of the true centreline the prediction covers
    tsens = (skel_y * prob).sum(dims).add(EPS) / (skel_y.sum(dims) + EPS)
    return (1 - 2 * tprec * tsens / (tprec + tsens + EPS)).mean()


class TrailLoss(torch.nn.Module):
    """Weighted sum of the three terms, with a clDice warm-up.

    clDice on random early predictions skeletonises noise and gives gradients
    that fight the region terms, so it is ramped in only once the model produces
    something line-shaped.
    """

    def __init__(self, bce: float = 1.0, tversky_w: float = 1.0,
                 cldice_w: float = 0.5, pos_weight: float = 8.0,
                 alpha: float = 0.3, beta: float = 0.7, iters: int = 8):
        super().__init__()
        self.bce = bce
        self.tversky_w = tversky_w
        self.cldice_w = cldice_w
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.beta = beta
        self.iters = iters

    def forward(self, logits, y, w, ramp: float = 1.0) -> tuple[torch.Tensor, dict]:
        prob = torch.sigmoid(logits)
        l_bce = masked_bce(logits, y, w, self.pos_weight)
        l_tv = tversky(prob, y, w, self.alpha, self.beta)
        parts = {"bce": l_bce.item(), "tversky": l_tv.item()}
        total = self.bce * l_bce + self.tversky_w * l_tv

        scale = self.cldice_w * ramp
        if scale > 0:
            l_cl = cldice(prob, y, w, self.iters)
            total = total + scale * l_cl
            parts["cldice"] = l_cl.item()
        return total, parts
