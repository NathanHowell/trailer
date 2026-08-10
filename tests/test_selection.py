"""Which epoch a run keeps.

The rule is not "the best model" but "the best model we can actually ship":
lidar05 needs canopy bands 3DEP cannot supply, export_onnx refuses it, and a
checkpoint chosen partly on its behalf is chosen for something nobody runs.
"""
from __future__ import annotations

import pytest

from trailer.train import selection_score


def _stats(**by_variant):
    return {k: {"strat": {"score": v}} for k, v in by_variant.items()}


def test_a_canopy_variant_does_not_vote():
    """dem1 alone decides, even when lidar05 disagrees loudly."""
    score, score_all, _ = selection_score(_stats(lidar05=0.9, dem1=0.5))
    assert score == pytest.approx(0.5)
    assert score_all == pytest.approx(0.7)


def test_the_kept_epoch_follows_dem1_not_the_mean():
    """The failure mode 440.24 was filed on: lidar05 carrying a worse dem1.

    Epoch 1 improves the all-variant mean while dem1 gets worse. Under the old
    rule that epoch was promoted to best; it must not be now.
    """
    epochs = [_stats(lidar05=0.50, dem1=0.60),
              _stats(lidar05=0.80, dem1=0.55)]
    sel = [selection_score(e)[0] for e in epochs]
    allv = [selection_score(e)[1] for e in epochs]
    assert allv[1] > allv[0], "fixture does not exercise the failure mode"
    assert sel[1] < sel[0]


def test_two_deployable_variants_are_averaged():
    """Deployable-only is a filter, not a hard-coded dem1."""
    score, _, dep = selection_score(_stats(dem1=0.4, lidar1=0.6, lidar05=0.9))
    assert len(dep) == 1, "lidar1 carries canopy, so only dem1 deploys"
    assert score == pytest.approx(0.4)


def test_an_all_canopy_run_falls_back_rather_than_dividing_by_zero():
    """A lidar05-only run still has to select something."""
    score, score_all, dep = selection_score(_stats(lidar05=0.8))
    assert dep == []
    assert score == pytest.approx(score_all) == pytest.approx(0.8)
