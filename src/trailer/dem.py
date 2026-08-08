"""Fetch USGS 3DEP's published 1 m DEM -- the raster the plugin will actually see.

The ``dem1`` variant exists to be deployable, and what it deploys against is this
service. Training it on a 2x2 block-mean of our own PDAL gridding was a
measured train/serve skew, not a theoretical one: against the published product,
across four tiles, ``slope`` and ``mrm_10m`` agree well (r 0.79-0.92) but the
tread-scale ``mrm_2m`` band correlates at only **r = 0.10-0.18** with amplitude
around half. USGS grids from ground returns with its own interpolation and
hydro-flattening, so the fine-scale content is not a smoothed version of ours --
it is different content. A model trained on ours would read a band at inference
it had never seen.

So the 1 m variants train on this, and the block-mean proxy is retired.

Two things this module refuses to do quietly:

* **Accept a raster that disagrees with our own DTM.** Agreement on ``slope`` --
  a band that transfers well where both products describe the same ground -- is
  checked against the DTM we already hold, and the tile is rejected below
  ``MIN_SLOPE_CORR``. 13 of 64 tiles failed on the first run.

  The gate is on agreement, not on a diagnosis. Two obvious culprits were
  measured and RULED OUT: it is not a 1/3 arc-second fallback resampled up (the
  structure function D(1)/D(8) matches our own grid on failing tiles, where a
  10 m source would collapse it), and it is not misregistration (cross-
  correlating over +/-12 px moves whitney_switchbacks only 0.28 -> 0.32). The
  leading remaining hypothesis is genuine divergence where ground returns are
  sparse -- steep rock and cliffs, where both products are mostly interpolating
  and our IDW and USGS's method have nothing to agree about. Unconfirmed.

* **Trust HTTP 200.** The ImageServer returns errors as JSON with a 200 status,
  so the response is checked for a TIFF magic number rather than a status code.
"""
from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import rasterio
import requests

log = logging.getLogger(__name__)

URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation"
       "/ImageServer/exportImage")

USER_AGENT = "trailer/0.1 (OpenStreetMap trail detection; contact via OSM)"

#: Target pixel size in metres. Matches variants.BODY_RES.
RES_M = 1.0

#: Slope correlation against our own DTM below which the fetch is rejected.
#: Most tiles sit at 0.85-0.97. This gates on AGREEMENT, which is what matters
#: for training data, but it does not diagnose the cause -- see build_dem.
MIN_SLOPE_CORR = 0.70

#: Anything below this is nodata however the service labelled it. The service
#: has returned nodata=None on every tile tried, so the sentinel is not reliable.
ABSURD_M = -1e5


def fetch(bounds, epsg: int, width: int, height: int,
          attempts: int = 4, timeout: int = 180) -> np.ndarray:
    """Pull a float32 elevation raster for a projected bounding box.

    Requests bboxSR and imageSR directly in UTM, so the service does the
    reprojection once and we never round-trip through Web Mercator -- at 37 deg N
    Mercator's scale factor is ~1.25, and every window in the model is defined in
    metres.
    """
    x0, y0, x1, y1 = bounds
    params = {
        "bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": epsg, "imageSR": epsg,
        "size": f"{width},{height}", "format": "tiff", "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    last = None
    for i in range(attempts):
        try:
            r = requests.get(URL, params=params, timeout=timeout,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            if r.content[:2] not in (b"II", b"MM"):
                # Errors arrive as JSON with HTTP 200; surface the message.
                try:
                    msg = json.loads(r.content)
                except Exception:  # noqa: BLE001
                    msg = r.content[:200]
                raise RuntimeError(f"service returned no TIFF: {msg}")
            with rasterio.open(io.BytesIO(r.content)) as s:
                a = s.read(1).astype("float32")
                nodata = s.nodata
            if nodata is not None:
                a = np.where(a == nodata, np.nan, a)
            return np.where(a < ABSURD_M, np.nan, a)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < attempts - 1:
                time.sleep(2 ** i)
    raise RuntimeError(f"3DEP fetch failed after {attempts} attempts: {last}")


def _slope(z: np.ndarray, res: float) -> np.ndarray:
    z = np.nan_to_num(z, nan=float(np.nanmedian(z)))
    gy, gx = np.gradient(z.astype("float64"), res)
    return np.hypot(gx, gy)


def build_dem(d: Path, force: bool = False) -> dict | None:
    """Fetch and validate the published 1 m DEM for one built tile."""
    from .data import block_nanmean

    src = d / "dtm_clean.tif"
    if not src.exists():
        return None
    out = d / "dem1m.tif"
    if out.exists() and not force:
        return {"key": d.name, "skipped": True}

    with rasterio.open(src) as s:
        ours05 = s.read(1).astype("float64")
        bounds, crs, transform, native = s.bounds, s.crs, s.transform, s.res[0]

    k = int(round(RES_M / native))
    w = int(round((bounds.right - bounds.left) / RES_M))
    h = int(round((bounds.top - bounds.bottom) / RES_M))

    z = fetch((bounds.left, bounds.bottom, bounds.right, bounds.top),
              crs.to_epsg(), w, h)

    # Our own grid on the same footprint, purely to validate the fetch.
    a = np.where(ours05 != 0, ours05, np.nan)
    ours = block_nanmean(a, k)
    n0, n1 = min(ours.shape[0], z.shape[0]), min(ours.shape[1], z.shape[1])
    ours, zc = ours[:n0, :n1], z[:n0, :n1]

    m = np.zeros(zc.shape, dtype=bool)
    m[40:-40, 40:-40] = True
    m &= np.isfinite(zc) & np.isfinite(ours)
    if m.sum() < 1000:
        return {"key": d.name, "error": "too little overlap to validate"}

    so, su = _slope(ours, RES_M)[m], _slope(zc, RES_M)[m]
    corr = float(np.corrcoef(so, su)[0, 1])
    offset = float(np.nanmedian(zc[m] - ours[m]))

    rec = {"key": d.name, "shape": [int(n0), int(n1)],
           "slope_corr": round(corr, 3), "median_offset_m": round(offset, 3),
           "nan_frac": round(float(np.mean(~np.isfinite(zc))), 4)}
    if corr < MIN_SLOPE_CORR:
        # Cause deliberately not asserted. Resolution fallback and
        # misregistration were both measured and ruled out; see module docstring.
        rec["error"] = (f"slope correlation {corr:.2f} < {MIN_SLOPE_CORR} vs our "
                        "own DTM; not usable as dem1 training data")
        return rec

    meta = dict(driver="GTiff", height=n0, width=n1, count=1, dtype="float32",
                crs=crs, transform=transform * rasterio.Affine.scale(k, k),
                nodata=np.nan, compress="deflate", predictor=2, tiled=True)
    with rasterio.open(out, "w", **meta) as o:
        o.write(zc.astype("float32"), 1)
        o.descriptions = ("elevation_m",)
    log.info("%-22s %dx%d  slope r=%.3f  offset %+.2f m", d.name, n0, n1,
             corr, offset)
    return rec


def build_all(dirs: list[Path], force: bool = False,
              pause: float = 0.5) -> list[dict]:
    out = []
    for d in dirs:
        try:
            rec = build_dem(d, force=force)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: %s", d.name, exc)
            rec = {"key": d.name, "error": str(exc)}
        if rec is None:
            continue
        out.append(rec)
        if not rec.get("skipped"):
            time.sleep(pause)  # be polite to a public service
    return out
