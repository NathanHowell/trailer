"""Terrain derivatives as torch layers, so the exported model is self-contained.

The deployment target is a JOSM plugin, and what JOSM can fetch is the USGS 3DEP
ImageServer: ``pixelType F32``, one band, 1 m, bare earth. If the derivatives
stay in numpy/scipy then shipping the model means reimplementing a median
filter, a variance filter and four normalisation constants in Java and keeping
them in lockstep with every checkpoint. Putting them in the graph means the
plugin feeds one float32 DEM tile straight in.

That also closes a train/serve skew hole rather than opening one: there is a
single implementation, used by training and by export. The last bug of that
family in this project silently misregistered two bands across fourteen tiles.

Everything here is ONNX-expressible. The median is the only op without a native
equivalent; ``im2col`` + ``TopK`` reproduces ``scipy.ndimage.median_filter``
exactly -- verified bit-identical for k = 3, 4, 10, 20, including scipy's
lower-median convention for even k, and again after a round trip through
onnxruntime.

Two deliberate differences from the numpy version in ``rasters.py``:

* **Border handling is ``replicate``** (scipy's ``mode="nearest"``), not scipy's
  ``reflect`` default. Parity is therefore exact only in the interior. Crops are
  interior windows of much larger tiles, so this costs nothing real, and one
  padding convention across all window sizes is easier to keep honest.

* **The clip bounds scale with the window actually used.** ``rasters.py`` floors
  every window at 3 px, so at 1 m the 1.5 m roughness window becomes 3 m and the
  2 m micro-relief window becomes 3 m. Holding the old bounds there drove
  roughness to a 0.82 mean against a 1.0 clip -- most of the tile saturated, the
  band destroyed before the network sees it. Bounds now scale with the window's
  true extent in metres, which is an exact no-op at 0.5 m and keeps the 1 m
  bands informative.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

#: Window sizes in metres, and the floors ``rasters.py`` applies in pixels.
MRM2_M, MRM2_FLOOR = 2.0, 3
MRM10_M, MRM10_FLOOR = 10.0, 5
ROUGH_M, ROUGH_FLOOR = 1.5, 3

#: Clip bounds at the reference window extent. Signal of interest is tens of
#: millimetres, so these stay tight; a wide range would quantise it away.
MRM2_CLIP = 0.5
MRM10_CLIP = 1.0
ROUGH_CLIP = 0.5
SLOPE_CLIP = 60.0

#: Window for de-trending before the variance. See ``Roughness``.
DETREND_M, DETREND_FLOOR = 6.0, 5


def window_px(target_m: float, res: float, floor: int) -> int:
    """Pixel window for a physical size, matching ``rasters.py`` exactly."""
    return max(int(round(target_m / res)), floor)


def _pad(x: torch.Tensor, k: int) -> torch.Tensor:
    p = k // 2
    return F.pad(x, (p, k - 1 - p, p, k - 1 - p), mode="replicate")


def median_filter(x: torch.Tensor, k: int) -> torch.Tensor:
    """Exact k x k median, matching ``scipy.ndimage.median_filter(mode="nearest")``.

    ``im2col`` + ``TopK`` rather than a sort: both export, and TopK stops at the
    middle element instead of ordering the whole window.
    """
    if k <= 1:
        return x
    n, c, h, w = x.shape
    u = F.unfold(_pad(x, k), kernel_size=k)          # (n, c*k*k, h*w)
    u = u.reshape(n, c, k * k, h * w).transpose(2, 3)
    med = u.topk(k * k // 2 + 1, dim=-1, largest=False).values[..., -1]
    return med.reshape(n, c, h, w)


def box_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """k x k mean, matching ``scipy.ndimage.uniform_filter(mode="nearest")``."""
    if k <= 1:
        return x
    return F.avg_pool2d(_pad(x, k), kernel_size=k, stride=1)


def centre(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Strip absolute elevation, returning the centred field and a valid mask.

    ``NaN`` marks nodata, so the model takes one self-describing input and the
    plugin has only to map its source nodata to NaN.

    Every derivative below is a local difference, so a constant offset cancels
    algebraically -- but not in float32. At 3000 m the representable spacing is
    0.24 mm, which is 1.6% of a 15 mm tread. Centring makes the cancellation
    exact rather than merely close.
    """
    mask = torch.isfinite(z)
    zf = torch.where(mask, z, torch.zeros_like(z))
    n = mask.to(z.dtype).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    z = zf - (zf.sum(dim=(1, 2, 3), keepdim=True) / n)
    return torch.where(mask, z, torch.zeros_like(z)), mask


class FineDerivatives(nn.Module):
    """The three short-window bands, at the variant's own pixel size.

    These are what a 0.5 m stack buys over a 1 m one, so they are computed
    before any decimation. All three windows are a handful of pixels, so the
    im2col median stays cheap here.
    """

    def __init__(self, res: float):
        super().__init__()
        self.res = float(res)
        self.k2 = window_px(MRM2_M, res, MRM2_FLOOR)
        self.krough = window_px(ROUGH_M, res, ROUGH_FLOOR)
        self.kdetrend = window_px(DETREND_M, res, DETREND_FLOOR)
        # Clip bounds scale with the window's true extent (see module docstring).
        self.mrm2_clip = MRM2_CLIP * (self.k2 * res) / MRM2_M
        self.rough_clip = ROUGH_CLIP * (self.krough * res) / ROUGH_M

    def extra_repr(self) -> str:
        return (f"res={self.res}, mrm2={self.k2}px, rough={self.krough}px, "
                f"detrend={self.kdetrend}px")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mrm2 = (z - median_filter(z, self.k2)).clamp(-self.mrm2_clip,
                                                     self.mrm2_clip)

        # Central differences. np.gradient switches to a one-sided difference on
        # the outermost row and column; replicate padding halves it there
        # instead. Correcting that needs an indexed in-place write, which is an
        # ONNX export hazard for a one-pixel border that Hann-tapered sliding
        # window inference down-weights anyway -- so the border is left as is.
        pad = F.pad(z, (1, 1, 1, 1), mode="replicate")
        gx = (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]) / (2 * self.res)
        gy = (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]) / (2 * self.res)
        slope = torch.atan((gx * gx + gy * gy).sqrt()) * (180.0 / math.pi)

        # De-trend before the variance. `E[z^2] - E[z]^2` on raw elevation is
        # catastrophic cancellation: measured against a float64 reference it
        # drifts up to 1.2e-2, 7% of the band's own spread, and the drift grows
        # with elevation -- so the same terrain would score differently at 300 m
        # and at 3000 m. Removing a 6 m trend first leaves a decimetre-scale
        # field where float32 is exact to 1e-9, and it also stops roughness
        # doubling as a slope proxy, which `slope` already covers.
        u = z - box_mean(z, self.kdetrend)
        var = box_mean(u * u, self.krough) - box_mean(u, self.krough) ** 2
        rough = var.clamp(min=0.0).sqrt()

        return torch.cat([
            mrm2 / self.mrm2_clip,
            (slope / SLOPE_CLIP).clamp(0.0, 1.0),
            (rough / self.rough_clip).clamp(0.0, 1.0),
        ], dim=1)


class Background(nn.Module):
    """The 10 m micro-relief band, always at body resolution.

    This is a 10 m trend, so resolving it at 0.5 m buys nothing and costs a
    20x20 im2col -- 400 copies of the crop, ~1 GB per sample at a 768 px crop.
    Computing it once at 1 m for every variant is exact, cheap, and gives the
    shared trunk a band that means the same thing whatever fed it.

    Approximating it instead -- decimating the grid and upsampling the median --
    was measured at 0.246 mean absolute error against a band whose own spread is
    0.327, so that route is closed.
    """

    def __init__(self, res: float):
        super().__init__()
        self.res = float(res)
        self.k10 = window_px(MRM10_M, res, MRM10_FLOOR)
        self.clip = MRM10_CLIP * (self.k10 * res) / MRM10_M

    def extra_repr(self) -> str:
        return f"res={self.res}, mrm10={self.k10}px"

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mrm10 = (z - median_filter(z, self.k10)).clamp(-self.clip, self.clip)
        return mrm10 / self.clip
