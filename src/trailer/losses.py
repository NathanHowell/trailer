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

**Tolerance.** The region terms are optionally relaxed at a radius, mirroring
``metrics.relaxed``. Without it we score with a 5 m tolerance but train demanding
pixel-exact agreement with geometry we know is imprecise -- and imprecise for a
structural reason, not a random one: most Sierra trail geometry in OSM was
digitised from satellite imagery, so it carries orthorectification and
relief-displacement error, metres of it in 200 m of relief, plus canopy occlusion
where the line was inferred under tree cover.

The relaxation is asymmetric, exactly as the metric is. For recall, the
*prediction* is dilated: a label pixel is satisfied by a confident prediction
anywhere within the radius. For precision, the *label* is dilated: a prediction
is only a false positive if no label lies within the radius. Max-pooling also
routes the gradient to the single best candidate in the neighbourhood rather
than smearing it, so the model is pushed to sharpen one line rather than raise
a whole disc.

clDice is deliberately left strict. It scores the skeleton, and dilating either
side would thicken the very structure whose connectivity is the point.

The cost, stated plainly: predictions get laterally fuzzier, up to about twice
the radius. Against a 5 m review tolerance on a 1 m output grid that is the same
trade already accepted when the head was fixed at 1 m.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def dilate(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Max over a (2r+1) square. Same operator ``metrics._dilate`` uses."""
    if radius <= 0:
        return x
    return F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
               pos_weight: float = 8.0, radius: int = 0) -> torch.Tensor:
    if radius <= 0:
        pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        raw = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw,
                                                 reduction="none")
        return (raw * w).sum() / (w.sum() + EPS)

    # Written on probabilities rather than logits because the dilation happens
    # between the sigmoid and the log, so the fused stable form does not apply.
    # Both logs are clamped instead.
    prob = torch.sigmoid(logits)
    p_near = dilate(prob, radius)
    y_near = dilate(y, radius)
    pos = -pos_weight * y * torch.log(p_near.clamp_min(EPS))
    neg = -(1 - y_near) * torch.log((1 - prob).clamp_min(EPS))
    return ((pos + neg) * w).sum() / (w.sum() + EPS)


def tversky(prob: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
            alpha: float = 0.3, beta: float = 0.7,
            radius: int = 0) -> torch.Tensor:
    """Soft Tversky. alpha weights false positives, beta false negatives."""
    dims = (1, 2, 3)
    p_hit = dilate(prob, radius) if radius > 0 else prob
    y_hit = dilate(y, radius) if radius > 0 else y
    tp = (w * p_hit * y).sum(dims)
    fp = (w * prob * (1 - y_hit)).sum(dims)
    fn = (w * (1 - p_hit) * y).sum(dims)
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
                 alpha: float = 0.3, beta: float = 0.7, iters: int = 8,
                 tolerance_m: float = 5.0, res: float = 1.0):
        super().__init__()
        # Radius in pixels at the resolution the head predicts at. 0 recovers the
        # strict, pixel-exact behaviour, which is what an A/B needs.
        self.radius = max(int(round(tolerance_m / res)), 0) if tolerance_m else 0
        self.tolerance_m = tolerance_m
        self.bce = bce
        self.tversky_w = tversky_w
        self.cldice_w = cldice_w
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.beta = beta
        self.iters = iters

    def forward(self, logits, y, w, ramp: float = 1.0) -> tuple[torch.Tensor, dict]:
        prob = torch.sigmoid(logits)
        l_bce = masked_bce(logits, y, w, self.pos_weight, self.radius)
        l_tv = tversky(prob, y, w, self.alpha, self.beta, self.radius)
        parts = {"bce": l_bce.item(), "tversky": l_tv.item()}
        total = self.bce * l_bce + self.tversky_w * l_tv

        scale = self.cldice_w * ramp
        if scale > 0:
            l_cl = cldice(prob, y, w, self.iters)
            total = total + scale * l_cl
            parts["cldice"] = l_cl.item()
        return total, parts
