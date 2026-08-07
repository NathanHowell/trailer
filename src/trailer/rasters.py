"""Point-cloud extraction and terrain-derivative rasters via PDAL.

Design notes worth keeping in mind:

* Everything works in a projected metric CRS (UTM), never in the EPT's native
  Web Mercator. At 37 deg N, Mercator's scale factor is ~1.25, so a "0.5 m"
  Mercator pixel is really 0.4 m on the ground -- enough to matter when the
  feature you want is 1 m wide.

* Absolute elevation is removed early. The network must key on ~10-60 mm of
  local relief; feeding it 2000 m of regional topography wastes capacity.

* 3DEP carries high-noise points (class 18) at absurd altitudes. They are
  filtered before any height-above-ground work or the CHM max blows out.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from scipy.ndimage import median_filter, uniform_filter

log = logging.getLogger(__name__)

NODATA = 0.0
NOISE_CLASSES = "Classification != 7 && Classification != 18"

BAND_NAMES = ("mrm_2m", "mrm_10m", "slope", "roughness", "chm", "vdi")


def utm_epsg(lat: float, lon: float) -> str:
    zone = int(math.floor((lon + 180) / 6)) + 1
    return f"EPSG:{(326 if lat >= 0 else 327)}{zone:02d}"


def run_pdal(stages: list[dict], workdir: Path, tag: str) -> None:
    path = workdir / f"_pipe_{tag}.json"
    path.write_text(json.dumps(stages, indent=1))
    proc = subprocess.run(["pdal", "pipeline", str(path)],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"pdal [{tag}] failed:\n{proc.stderr[:2000]}")


def extract_points(ept_url: str, lat: float, lon: float, size_m: int,
                   out_laz: Path, buffer_m: float = 60.0) -> str:
    """Pull a square AOI out of an EPT resource, reprojected to UTM."""
    epsg = utm_epsg(lat, lon)
    to_m = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    to_web = Transformer.from_crs(epsg, "EPSG:3857", always_xy=True)
    x, y = to_m.transform(lon, lat)
    half = size_m / 2 + buffer_m

    corners = [to_web.transform(px, py)
               for px in (x - half, x + half) for py in (y - half, y + half)]
    wx = [c[0] for c in corners]
    wy = [c[1] for c in corners]

    out_laz.parent.mkdir(parents=True, exist_ok=True)
    run_pdal([
        {"type": "readers.ept", "filename": ept_url,
         "bounds": f"([{min(wx):.1f},{max(wx):.1f}],[{min(wy):.1f},{max(wy):.1f}])"},
        {"type": "filters.reprojection", "out_srs": epsg},
        {"type": "filters.crop",
         "bounds": f"([{x-half},{x+half}],[{y-half},{y+half}])"},
        {"type": "writers.las", "filename": str(out_laz),
         "compression": "laszip", "a_srs": epsg},
    ], out_laz.parent, out_laz.stem)
    return epsg


def _gdal_stage(dst: Path, res: float, output_type: str, dimension: str,
                window: int, power: float = 1.0) -> dict:
    return {"type": "writers.gdal", "filename": str(dst), "resolution": res,
            "output_type": output_type, "power": power, "window_size": window,
            "data_type": "float32", "nodata": NODATA, "dimension": dimension}


def build_dtm(laz: Path, dst: Path, res: float) -> None:
    run_pdal([
        {"type": "readers.las", "filename": str(laz)},
        {"type": "filters.expression", "expression": "Classification == 2"},
        {"type": "filters.outlier", "method": "statistical",
         "mean_k": 8, "multiplier": 3.0},
        {"type": "filters.expression", "expression": "Classification == 2"},
        _gdal_stage(dst, res, "idw", "Z", window=12, power=6.0),
    ], dst.parent, dst.stem)


def build_chm(laz: Path, dst: Path, res: float) -> None:
    run_pdal([
        {"type": "readers.las", "filename": str(laz)},
        {"type": "filters.expression", "expression": NOISE_CLASSES},
        {"type": "filters.hag_nn"},
        _gdal_stage(dst, res, "max", "HeightAboveGround", window=3),
    ], dst.parent, dst.stem)


def build_veg_count(laz: Path, dst: Path, res: float, expr: str) -> None:
    run_pdal([
        {"type": "readers.las", "filename": str(laz)},
        {"type": "filters.expression", "expression": NOISE_CLASSES},
        {"type": "filters.hag_nn"},
        {"type": "filters.expression", "expression": expr},
        _gdal_stage(dst, res, "count", "HeightAboveGround", window=5),
    ], dst.parent, dst.stem)


def _fill_nodata(a: np.ndarray) -> np.ndarray:
    a = np.where(a == NODATA, np.nan, a)
    if np.all(np.isnan(a)):
        raise ValueError("raster is entirely nodata")
    return np.where(np.isnan(a), np.nanmedian(a), a)


def derive_features(dtm: np.ndarray, chm: np.ndarray, low: np.ndarray,
                    high: np.ndarray, res: float) -> np.ndarray:
    """Stack the terrain derivatives the model actually consumes.

    Two micro-relief scales: ~2 m isolates the tread itself, ~10 m picks up the
    bench-and-berm cross-section that survives on constructed trails. Both are
    clipped tight -- the signal of interest is tens of millimetres, so a wide
    range would quantise it away.
    """
    px = max(int(round(2.0 / res)), 3)
    px10 = max(int(round(10.0 / res)), 5)

    mrm2 = np.clip(dtm - median_filter(dtm, size=px), -0.5, 0.5)
    mrm10 = np.clip(dtm - median_filter(dtm, size=px10), -1.0, 1.0)

    gy, gx = np.gradient(dtm, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    win = max(int(round(1.5 / res)), 3)
    rough = np.sqrt(np.clip(
        uniform_filter(dtm ** 2, win) - uniform_filter(dtm, win) ** 2, 0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        vdi = np.divide(low, high, out=np.zeros_like(low), where=high != 0)
    vdi = np.clip(vdi, 0, 1)

    return np.stack([
        mrm2 / 0.5,
        mrm10 / 1.0,
        np.clip(slope / 60.0, 0, 1),
        np.clip(rough / 0.5, 0, 1),
        np.clip(chm / 40.0, 0, 1),
        vdi,
    ]).astype("float32")


def build_feature_stack(laz: Path, out_tif: Path, res: float) -> dict:
    """Full raster build for one AOI. Returns a small QA summary."""
    work = out_tif.parent
    dtm_p, chm_p = work / "dtm.tif", work / "chm.tif"
    low_p, high_p = work / "lowveg.tif", work / "highveg.tif"

    build_dtm(laz, dtm_p, res)
    build_chm(laz, chm_p, res)
    build_veg_count(laz, low_p, res, "HeightAboveGround < 0.8")
    build_veg_count(laz, high_p, res, "HeightAboveGround <= 12")

    with rasterio.open(dtm_p) as s:
        dtm_raw = s.read(1)
        transform, crs = s.transform, s.crs
    arrays = {"chm": chm_p, "low": low_p, "high": high_p}
    read = {}
    for k, p in arrays.items():
        with rasterio.open(p) as s:
            read[k] = s.read(1)

    rows = min([dtm_raw.shape[0]] + [v.shape[0] for v in read.values()])
    cols = min([dtm_raw.shape[1]] + [v.shape[1] for v in read.values()])
    dtm_raw = dtm_raw[:rows, :cols]
    read = {k: v[:rows, :cols] for k, v in read.items()}

    valid = dtm_raw != NODATA
    dtm = _fill_nodata(dtm_raw)
    feats = derive_features(dtm, read["chm"], read["low"], read["high"], res)
    feats[:, ~valid] = 0.0

    meta = dict(driver="GTiff", height=rows, width=cols, count=len(BAND_NAMES),
                dtype="float32", crs=crs, transform=transform,
                compress="deflate", predictor=2, tiled=True)
    with rasterio.open(out_tif, "w", **meta) as d:
        d.write(feats)
        d.descriptions = BAND_NAMES

    # keep the bare DTM alongside; hillshades for review come from it
    with rasterio.open(work / "dtm_clean.tif", "w", driver="GTiff",
                       height=rows, width=cols, count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=NODATA,
                       compress="deflate") as d:
        d.write(np.where(valid, dtm, NODATA).astype("float32"), 1)

    return {
        "shape": [rows, cols],
        "res": res,
        "valid_frac": float(valid.mean()),
        "elev_min": float(dtm[valid].min()),
        "elev_max": float(dtm[valid].max()),
        "relief": float(np.percentile(dtm[valid], 98) - np.percentile(dtm[valid], 2)),
    }
