"""TrailLoss: what it returns and what it back-propagates.

The module had no tests. These cover the parts of the contract train.py relies
on -- the warm-up actually withholding clDice, the reported breakdown matching
the total it came from, and gradients surviving the relaxed forms.
"""
from __future__ import annotations

import math

import torch

from trailer.losses import TrailLoss


def _fixture(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 1, 64, 64, generator=g, requires_grad=True)
    y = (torch.rand(2, 1, 64, 64, generator=g) > 0.97).float()
    w = torch.ones_like(y)
    return logits, y, w


def test_cldice_is_withheld_until_the_ramp_opens():
    """The warm-up omits the term, rather than including it at weight zero.

    train.py logs whatever keys come back, so an always-present cldice would
    report a number the total never contained.
    """
    crit = TrailLoss(cldice_w=0.5, tolerance_m=5.0, res=1.0)
    logits, y, w = _fixture()

    _, cold = crit(logits, y, w, ramp=0.0)
    _, warm = crit(logits, y, w, ramp=1.0)
    assert set(cold) == {"bce", "tversky"}
    assert set(warm) == {"bce", "tversky", "cldice"}


def test_reported_parts_add_up_to_the_total():
    """With clDice off and unit weights the total is exactly its parts."""
    crit = TrailLoss(bce=1.0, tversky_w=1.0, cldice_w=0.0,
                     tolerance_m=5.0, res=1.0)
    logits, y, w = _fixture()
    total, parts = crit(logits, y, w, ramp=0.0)
    assert math.isclose(float(total), parts["bce"] + parts["tversky"],
                        rel_tol=1e-6)


def test_the_ramp_scales_cldice_into_the_total():
    """A larger ramp must move the total by the reported clDice term."""
    crit = TrailLoss(bce=1.0, tversky_w=1.0, cldice_w=1.0,
                     tolerance_m=5.0, res=1.0)
    logits, y, w = _fixture()
    base, _ = crit(logits, y, w, ramp=0.0)
    half, parts = crit(logits, y, w, ramp=0.5)
    assert math.isclose(float(half) - float(base), 0.5 * parts["cldice"],
                        rel_tol=1e-5)


def test_gradients_reach_the_logits_through_the_relaxed_forms():
    """radius > 0 routes through max_pool; the gradient must survive it."""
    for tolerance in (0.0, 5.0):
        crit = TrailLoss(cldice_w=0.5, tolerance_m=tolerance, res=1.0)
        logits, y, w = _fixture()
        total, _ = crit(logits, y, w, ramp=1.0)
        total.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all(), tolerance
        assert logits.grad.abs().sum() > 0, tolerance


def test_ignored_pixels_out_of_dilation_reach_do_not_move_the_loss():
    """Weight 0 means ignored -- but only beyond the relaxation radius.

    The relaxation is deliberately asymmetric: a label pixel is satisfied by a
    confident prediction anywhere within the radius, and nothing stops that
    prediction sitting inside an ignore region. So a prediction within
    ``radius`` of the boundary legitimately does move the loss. This scrambles
    only the part of the band that no labelled pixel can reach.
    """
    crit = TrailLoss(cldice_w=0.5, tolerance_m=5.0, res=1.0)
    assert crit.radius == 5
    logits, y, w = _fixture()
    w = w.clone()
    w[:, :, :16, :] = 0.0                    # ignored: rows 0-15

    a, _ = crit(logits, y, w, ramp=1.0)
    scrambled = logits.detach().clone()
    scrambled[:, :, :8, :] += 50.0           # rows 0-7, >5 px from row 16
    b, _ = crit(scrambled.requires_grad_(True), y, w, ramp=1.0)
    assert math.isclose(float(a), float(b), rel_tol=1e-5)
