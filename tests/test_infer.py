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


def test_pad_to_only_fires_below_one_window():
    # A ragged tail is no longer padded -- the last window is pulled flush
    # against it instead. All that is left for padding is a raster too small to
    # hold a single window, where there is nothing else to hand the model.
    for n, tile, step in ((100, 64, 32), (64, 64, 32), (129, 64, 32)):
        assert infer._pad_to(n, tile, step) == 0, (n, tile)
    assert infer._pad_to(50, 64, 32) == 14
    assert infer._pad_to(1, 64, 32) == 63


# Rasters at least one window across, at every stride the variant table has,
# ragged and exact.
_AXES = ((1121, 256, 128, 1), (1121, 512, 256, 2), (256, 256, 128, 1),
         (300, 128, 64, 1), (129, 64, 32, 1), (2242, 512, 152, 2))


def test_no_window_reads_a_pixel_that_is_not_there():
    """The last window sits flush against the far edge, not off the end of it.

    Reflect-padding the tail and letting a window hang over it manufactures a
    mirror seam, and a mirrored hillslope is a symmetric V a few metres wide --
    which is the tread cross-section the model is trained to fire on. Measured
    on runs/full-b/best.pt: 100% of the control tile's false positives sat in
    the outer four body pixels, and cropping the raster to a size needing no
    padding took the bottom rows from 0.22 mean probability to 0.00.

    So the invariant is not "the padding is cropped off the output" -- it was,
    and the frame was there anyway. It is that no window ever *reads* an
    invented pixel.
    """
    for n, tile, step, stride in _AXES:
        origins = infer.window_origins(n, tile, step, stride)
        assert origins[0] == 0, (n, tile, step, stride)
        # Flush, up to the stride quantisation: what the last window leaves
        # uncovered is less than one input pixel per body pixel, so no body
        # pixel is stranded (which the coverage test below states directly).
        assert 0 <= n - (origins[-1] + tile) < stride, (n, tile, step, stride)
        assert all(o % stride == 0 for o in origins), (n, tile, step, stride)
        assert origins == sorted(set(origins)), (n, tile, step, stride)


def test_every_body_pixel_falls_under_some_window():
    # A flush final window is only safe if pulling it back does not strand the
    # rows the regular stride would have reached.
    for n, tile, step, stride in _AXES:
        covered = set()
        for o in infer.window_origins(n, tile, step, stride):
            covered.update(range(o // stride, o // stride + tile // stride))
        assert covered == set(range(n // stride)), (n, tile, step, stride)


def test_window_origins_pins_the_known_cases():
    # The plugin reimplements this rule, so the values are pinned here as well
    # as carried into golden.json.
    assert infer.window_origins(1121, 256, 128, 1) == [
        0, 128, 256, 384, 512, 640, 768, 865]
    assert infer.window_origins(256, 256, 128, 1) == [0]
    # Shorter than one window: padded up, and the single window starts at zero.
    assert infer.window_origins(200, 256, 128, 1) == [0]
    # Stride 2, ragged: the flush origin is quantised down to the stride so it
    # still lands on a body pixel.
    assert infer.window_origins(1121, 512, 256, 2) == [0, 256, 512, 608]
