"""The pieces of inference that also have to survive being exported to ONNX.

Everything here is arithmetic the JOSM plugin must *not* reimplement. The whole
point of baking the D4 average and the window taper into the graph is that there
is one definition; these tests pin that definition so a rewrite for the
exporter's benefit cannot quietly change what the model does.
"""
from __future__ import annotations

import torch

from trailer import infer


def _asym(h: int = 5, w: int = 3) -> torch.Tensor:
    """Asymmetric in both axes, so a transpose bug cannot pass by symmetry."""
    return torch.arange(float(h * w)).reshape(1, 1, h, w)


def test_rot90_matches_torch():
    # aten::rot90 does not export, so it is spelled with transpose and flip.
    # k=2 is easy and k=1/k=3 are where a wrong spelling hides.
    x = _asym()
    for k in range(-4, 5):
        assert torch.equal(infer._rot90(x, k), torch.rot90(x, k, dims=(-2, -1))), k


def test_d4_inverse_undoes_d4():
    x = _asym()
    for flip in (False, True):
        for k in range(4):
            back = infer._d4_inv(infer._d4(x, k, flip), k, flip)
            assert torch.equal(back, x), (k, flip)


def test_d4_gives_eight_distinct_orientations():
    # If two of the eight collide, the average is silently weighted.
    x = _asym(4, 3)
    seen = {tuple(infer._d4(x, k, f).flatten().tolist())
            for f in (False, True) for k in range(4)}
    assert len(seen) == 8


def test_hann_taper_is_positive_at_the_edges():
    # hann_window(size + 2) trimmed, not hann_window(size): a plain Hann is
    # exactly zero at both ends, so the outermost row and column of every tile
    # would contribute nothing and the 1e-3 floor would be doing all the work.
    t = infer.hann2d(8, "cpu")
    assert t.shape == (8, 8)
    assert float(t.min()) > 1e-3
    assert float(t[0, 0]) > 0
    # Symmetric, and heaviest in the middle -- that is what makes a window's
    # centre dominate the blend.
    assert torch.allclose(t, t.flip(-1))
    assert torch.allclose(t, t.flip(-2))
    assert float(t[4, 4]) == float(t.max())


def test_window_step_quantises_to_the_stride():
    # An origin that is not a multiple of the stride lands between output
    # pixels. This is the number the export sidecar carries to the plugin.
    assert infer.window_step(256, 0.5, 1) == 128
    assert infer.window_step(512, 0.5, 2) == 256
    assert infer.window_step(512, 0.7, 2) == 152      # not 153
    assert infer.window_step(512, 0.9, 2) == 50       # not 51
    assert infer.window_step(512, 1.0, 2) == 2        # never zero


def test_pad_to_covers_the_ragged_tail():
    for n, tile, step in ((100, 64, 32), (64, 64, 32), (50, 64, 32), (129, 64, 32)):
        pad = infer._pad_to(n, tile, step)
        padded = n + pad
        assert padded >= tile
        assert (padded - tile) % step == 0
        # Every input row is inside some window.
        assert padded - step < max(n, tile) + step
