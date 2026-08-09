"""What the stratified selection metric is supposed to protect against.

Each test here is a model-selection mistake that the pooled ``f1@0.5`` score
would have made silently. A scoring function that returns a plausible number for
the wrong reason is worse than one that crashes, so the cases are written as
"pooled says X, stratified says Y" rather than as bare assertions on Y.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trailer import metrics, osm  # noqa: E402
from trailer.data import block_min_nonzero  # noqa: E402

SIZE = 128
RES = 1.0  # 5 m tolerance -> radius 5 px, so keep lines >10 px apart


def canvas():
    return np.zeros((1, 1, SIZE, SIZE), dtype="float32")


def line(y, cls, row, cols, klass):
    """Burn a horizontal run of trail of one visibility class."""
    y[0, 0, row, cols[0]:cols[1]] = 1.0
    cls[0, 0, row, cols[0]:cols[1]] = osm.CLASS_CODE[klass]


def tensors(*arrays):
    return [torch.from_numpy(a) for a in arrays]


def scene(rows):
    """rows: list of (row, (col0, col1), class). Returns y, w, cls tensors."""
    y, cls = canvas(), canvas()
    for row, cols, klass in rows:
        line(y, cls, row, cols, klass)
    w = np.ones_like(y)
    return tensors(y, w, cls)


def test_pooled_f1_hides_a_class_the_model_never_finds():
    """The failure this whole change exists for.

    Three kilometres of active trail and a couple hundred metres of faint, and a
    model that finds every active pixel and no faint one. Pooled recall is
    weighted by labelled length, so it reads 0.94 and the checkpoint looks
    excellent -- while being useless at the only class a mapper cannot already
    see in imagery.
    """
    y, w, cls = scene([(10, (10, 118), "active"),
                       (20, (10, 118), "active"),
                       (30, (10, 118), "active"),
                       (100, (50, 71), "faint")])
    prob = canvas()
    prob[0, 0, [10, 20, 30], 10:118] = 1.0
    prob = torch.from_numpy(prob)

    pooled = metrics.sweep(prob, y, w, RES)
    assert pooled["f1@0.5"] > 0.95

    strat = metrics.Stratified(RES)
    strat.update(prob, y, w, cls)
    out = strat.result()
    assert out["by_class"]["active"]["f1"] == pytest.approx(1.0, abs=1e-3)
    assert out["by_class"]["faint"]["f1"] == pytest.approx(0.0, abs=1e-3)
    assert out["score"] == pytest.approx(0.5, abs=1e-3)


def test_precision_is_pooled_so_another_class_is_not_a_false_positive():
    """Predicting a lifecycle trail must not be charged against active.

    A predicted pixel does not carry a class, and a highlight drawn over a real
    abandoned trail is correct however you slice the score. Stratifying precision
    as well as recall would call it a false positive two thirds of the time.
    """
    y, w, cls = scene([(20, (10, 118), "active"),
                       (60, (10, 118), "lifecycle")])
    prob = canvas()
    prob[0, 0, [20, 60], 10:118] = 1.0  # both trails found
    prob = torch.from_numpy(prob)

    strat = metrics.Stratified(RES)
    strat.update(prob, y, w, cls)
    out = strat.result()["by_class"]
    assert out["active"]["p"] == pytest.approx(1.0, abs=1e-3)
    assert out["active"]["f1"] == pytest.approx(1.0, abs=1e-3)
    assert out["lifecycle"]["f1"] == pytest.approx(1.0, abs=1e-3)


def test_threshold_search_survives_a_miscalibrated_head():
    """A perfect ranking whose probabilities all sit below 0.5.

    The output bias comes from the sampled positive prior and the loss reweights
    positives, so where the sigmoid lands drifts between runs for reasons
    unrelated to model quality. Reading only 0.5 would score this at zero.
    """
    y, w, cls = scene([(20, (10, 118), "faint")])
    prob = canvas()
    prob[0, 0, 20, 10:118] = 0.35
    prob = torch.from_numpy(prob)

    assert metrics.sweep(prob, y, w, RES)["f1@0.5"] == 0.0

    strat = metrics.Stratified(RES)
    strat.update(prob, y, w, cls)
    faint = strat.result()["by_class"]["faint"]
    assert faint["t"] <= 0.3
    assert faint["f1"] == pytest.approx(1.0, abs=1e-3)


def test_absent_class_is_dropped_rather_than_scored_zero():
    """A validation band with no lifecycle way says nothing about lifecycle.

    Scoring it zero would make the checkpoint choice depend on which tiles
    happened to fall in the split.
    """
    y, w, cls = scene([(20, (10, 118), "active")])
    prob = torch.from_numpy(np.zeros_like(y.numpy()))
    prob[0, 0, 20, 10:118] = 1.0

    out = metrics.Stratified(RES)
    out.update(prob, y, w, cls)
    res = out.result()
    assert res["classes"] == ["active"]
    assert res["score"] == pytest.approx(1.0, abs=1e-3)


def test_tolerance_is_honoured_per_class():
    """A prediction two metres off the label still counts, as relaxed scoring
    intends -- OSM centrelines here sit ~1.4 m from the true tread."""
    y, w, cls = scene([(20, (10, 118), "faint")])
    prob = canvas()
    prob[0, 0, 22, 10:118] = 1.0  # 2 px = 2 m off, inside the 5 m tolerance
    prob = torch.from_numpy(prob)

    strat = metrics.Stratified(RES)
    strat.update(prob, y, w, cls)
    assert strat.result()["by_class"]["faint"]["f1"] > 0.95


def test_ignored_pixels_never_score():
    """Weight zero means the label is unknown, not negative."""
    y, w, cls = scene([(20, (10, 118), "active")])
    w = torch.zeros_like(w)
    prob = torch.from_numpy(canvas())
    prob[0, 0, 60, 10:118] = 1.0  # fires in an ignored region

    strat = metrics.Stratified(RES)
    strat.update(prob, y, w, cls)
    assert strat.result()["classes"] == []


def test_block_min_nonzero_prefers_the_lower_code():
    """Reducing the class plane to body resolution must not hand every shared
    pixel to lifecycle, which block_max would."""
    a = np.array([[0.0, 3.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 3.0, 3.0],
                  [0.0, 0.0, 3.0, 0.0]], dtype="float32")
    out = block_min_nonzero(a, 2)
    assert out.tolist() == [[2.0, 0.0], [0.0, 3.0]]
    assert block_min_nonzero(a, 1) is a


def _held(**tiles):
    """A held_out block: {tile: (advisory, {class: f1})}."""
    return {"dem1": {
        name: {"advisory": adv,
               "strat": {"by_class": {c: {"f1": f} for c, f in by.items()}}}
        for name, (adv, by) in tiles.items()}}


def test_spread_reports_every_eval_aoi_not_just_the_middle():
    s = metrics.held_out_spread(_held(
        a=("", {"lifecycle": 0.10}),
        b=("", {"lifecycle": 0.60}),
        c=("", {"lifecycle": 0.80})))["dem1"]["lifecycle"]
    assert s["n"] == 3
    assert s["median"] == 0.60
    assert (s["min"], s["max"]) == (0.10, 0.80)
    # The per-tile values survive: a median hides which AOI is the bad one.
    assert set(s["tiles"]) == {"a", "b", "c"}


def test_an_advisory_tile_is_excluded_from_the_aggregate():
    # Averaging in a score that is explicitly not evidence would launder it
    # back into one -- the exact move this whole mechanism exists to block.
    s = metrics.held_out_spread(_held(
        good=("", {"lifecycle": 0.60}),
        also=("", {"lifecycle": 0.70}),
        bogus=("the label has no measurable tread", {"lifecycle": 0.00})))
    life = s["dem1"]["lifecycle"]
    assert life["n"] == 2, "advisory tile must not count toward the spread"
    assert life["min"] == 0.60
    assert "bogus" not in life["tiles"]


def test_a_class_absent_from_a_tile_does_not_score_it_zero():
    # A tile with no lifecycle way says nothing about lifecycle recall.
    s = metrics.held_out_spread(_held(
        a=("", {"active": 0.80}),
        b=("", {"active": 0.60, "lifecycle": 0.50})))["dem1"]
    assert s["active"]["n"] == 2
    assert s["lifecycle"]["n"] == 1
    assert s["lifecycle"]["min"] == 0.50


def _held_px(**tiles):
    """A held_out block carrying per-class pixel counts."""
    return {"dem1": {
        name: {"advisory": "",
               "strat": {"by_class": {c: {"f1": f, "px": px}
                                      for c, (f, px) in by.items()}}}
        for name, by in tiles.items()}}


def test_a_class_measured_on_a_few_hundred_metres_is_not_a_measurement():
    # 400 px is roughly 100 m of way. Pooling that with kilometre-scale tiles
    # makes small-sample noise read as model instability.
    s = metrics.held_out_spread(_held_px(
        solid=({"active": (0.90, 40000)}),
        also=({"active": (0.70, 20000)}),
        sliver=({"active": (0.10, 400)})))["dem1"]["active"]
    assert s["n"] == 2
    assert s["min"] == 0.70, "the sliver must not widen the range"
    assert s["too_thin"] == ["sliver"], "and it must be named, not hidden"


def test_the_thin_floor_is_adjustable_and_off_for_old_reports():
    # A report written before per-class pixel counts existed has no "px", and
    # must summarise rather than silently drop every tile.
    old = {"dem1": {"a": {"strat": {"by_class": {"active": {"f1": 0.5}}}}}}
    assert metrics.held_out_spread(old)["dem1"]["active"]["n"] == 1
    tiny = metrics.held_out_spread(_held_px(a={"active": (0.5, 100)}), min_px=50)
    assert tiny["dem1"]["active"]["n"] == 1
