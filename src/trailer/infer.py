"""Sliding-window inference over whole tiles.

Tiled inference on curvilinear features leaves visible seams: a trail crossing a
window boundary gets half its context on each side and the model hedges. Two
mitigations, both of which the JOSM plugin will need to reproduce:

* 50% overlap with a 2-D Hann taper, so every pixel's final value is dominated
  by the window that had it nearest the centre;
* optional D4 test-time augmentation. Terrain has no canonical orientation, so
  averaging the eight dihedral transforms is nearly free accuracy -- at 8x the
  compute, which is why it is off by default.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hann2d(size: int, device, dtype=torch.float32) -> torch.Tensor:
    w = torch.hann_window(size + 2, periodic=False, device=device, dtype=dtype)[1:-1]
    return (w[:, None] * w[None, :]).clamp_min(1e-3)


def _rot90(x: torch.Tensor, k: int) -> torch.Tensor:
    """``torch.rot90(x, k, dims=(-2, -1))``, spelled in ops ONNX can export.

    ``aten::rot90`` has no symbolic at opset 17, and the D4 average is baked into
    the exported graph so the plugin never reimplements it. Transpose and flip
    both export, so the rotation is written out in terms of them. Equality with
    ``torch.rot90`` is asserted in the tests rather than argued for here — the
    two spellings are easy to get right for k=2 and subtly wrong for k=1 and 3.
    """
    k %= 4
    if k == 0:
        return x
    if k == 1:
        return torch.flip(x.transpose(-2, -1), dims=(-2,))
    if k == 2:
        return torch.flip(x, dims=(-2, -1))
    return torch.flip(x.transpose(-2, -1), dims=(-1,))


def _d4(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=(-1,))
    return _rot90(x, k)


def _d4_inv(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = _rot90(x, -k)
    return torch.flip(x, dims=(-1,)) if flip else x


def _pad_to(n: int, tile: int, step: int) -> int:
    """Extra rows/cols so windows of ``tile`` tile the axis exactly.

    Only reachable now for a raster narrower than a single window, where there
    is nothing to do but invent pixels. ``step`` no longer enters it: the ragged
    tail is handled by pulling the last window flush against the far edge --
    see :func:`window_origins` -- rather than by extending the axis to meet a
    window that hangs off the end.
    """
    return max(tile - n, 0)


def window_origins(n: int, tile: int, step: int, stride: int) -> list[int]:
    """Window origins along one axis of an ``n`` pixel raster, in input pixels.

    Stepped by ``step`` from zero, with the last window pulled back flush
    against the far edge instead of hanging off a padded one.

    This is not a tidiness preference. Reflect-padding the tail put a mirror
    seam one window-edge away from real output, and a mirrored hillslope is a
    symmetric V a few metres across -- the tread cross-section the model exists
    to find. It fired on it every time: on ``runs/full-b/best.pt`` at ``dem1``,
    **100%** of the trail-free control tile's predictions above 0.5 sat in the
    outer four body pixels, against an interior rate of zero, and re-cropping
    the same raster to a size needing no padding took the bottom rows from 0.22
    mean probability to 0.00. The padding was already cropped off the *output*;
    that never helped, because the damage was done to the *input* the window
    read.

    The flush origin is quantised down to ``stride`` for the same reason
    :func:`window_step` is: an origin between output pixels misregisters the
    whole window by half a body pixel. Quantising down rather than up cannot
    strand the tail -- ``n // stride`` body pixels are covered exactly, since
    the leftover is by construction less than one body pixel.

    Emitted into the export sidecar as a rule the plugin follows, not
    re-derived there; ``Tiler.origins`` is checked against these values.
    """
    last = max(((n - tile) // stride) * stride, 0)
    return sorted({*range(0, last + 1, step), last})


def window_step(tile: int, overlap: float, stride: int) -> int:
    """Distance between window origins, in the *input's* pixels.

    Quantised to the stem's stride, and floored at it. Both matter: an origin
    that is not a multiple of the stride lands between output pixels, so the
    window's predictions would be accumulated half a body pixel off from where
    they belong -- a misregistration that grows no worse at the tile's centre and
    is therefore invisible except as a general softening.

    Defined here, and exported into the model's sidecar by ``export_onnx``, so
    the plugin reads the number rather than reimplementing this line. It already
    reimplemented it wrongly once: Kotlin had neither the quantisation nor the
    floor, and disagreed at overlap 0.7 with a stride-2 stem.
    """
    return max(int(tile * (1 - overlap)) // stride * stride, stride)


@torch.no_grad()
def predict(model, z: np.ndarray, canopy: np.ndarray | None = None,
            variant: str | None = None, body_tile: int = 256,
            overlap: float = 0.5, device=None, batch: int = 8,
            tta: bool = False) -> np.ndarray:
    """Run a model over a (1, H, W) elevation raster in metres.

    Windows are stepped in the *input's* pixels but accumulated in the trunk's,
    since the model predicts at ``BODY_RES`` whatever it was fed. Returns
    ``(1, H // stride, W // stride)`` probabilities.
    """
    from . import variants as var_mod

    device = device or next(model.parameters()).device
    model.eval()
    v = var_mod.get(variant) if variant else next(iter(model.stems.values())).variant
    if isinstance(v, str):
        v = var_mod.get(v)
    stride = v.stride
    tile = body_tile * stride

    _, h, w = z.shape
    step = window_step(tile, overlap, stride)
    pad_h, pad_w = _pad_to(h, tile, step), _pad_to(w, tile, step)

    # Reflect-padding survives only for a raster smaller than one window, where
    # the model has to be handed something and there is no real ground to hand
    # it. Everywhere else the tail is covered by a flush final window instead --
    # a mirror seam beside real output is a manufactured trail, see
    # `window_origins`. NaN nodata reflects harmlessly either way: the model's
    # centring step drops it.
    t = torch.from_numpy(z).unsqueeze(0)
    cn = torch.from_numpy(canopy).unsqueeze(0) if v.canopy else None
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        if cn is not None:
            cn = F.pad(cn, (0, pad_w, 0, pad_h), mode="reflect")
    t = t.to(device)
    cn = cn.to(device) if cn is not None else None
    _, _, H, W = t.shape

    bt = tile // stride
    taper = hann2d(bt, device)
    acc = torch.zeros((1, 1, H // stride, W // stride), device=device)
    den = torch.zeros_like(acc)

    origins = [(r, c) for r in window_origins(H, tile, step, stride)
               for c in window_origins(W, tile, step, stride)]
    d4 = [(k, f) for f in (False, True) for k in range(4)] if tta else [(0, False)]

    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        crops = torch.cat([t[:, :, r:r + tile, c:c + tile] for r, c in chunk])
        ccrops = (torch.cat([cn[:, :, r:r + tile, c:c + tile] for r, c in chunk])
                  if cn is not None else None)
        prob = torch.zeros((len(chunk), 1, bt, bt), device=device)
        for k, flip in d4:
            cc = _d4(ccrops, k, flip) if ccrops is not None else None
            out = model(_d4(crops, k, flip), cc, variant=v.key)
            prob += torch.sigmoid(_d4_inv(out, k, flip))
        prob /= len(d4)
        for j, (r, c) in enumerate(chunk):
            br, bc = r // stride, c // stride
            acc[0, :, br:br + bt, bc:bc + bt] += prob[j] * taper
            den[0, :, br:br + bt, bc:bc + bt] += taper

    out = (acc / den.clamp_min(1e-6))[0, :, :h // stride, :w // stride]
    return out.cpu().numpy()
