"""Diagnostics: is there tread signal in this tile, and how strong?

The headline metric is berm-relative incision measured over many perpendicular
transects. Individual transects are useless -- terrain noise runs 65 mm in a
sandy meadow to 500 mm in alpine talus while tread is only 15-100 mm deep -- so
detectability is governed by the ratio of the two, integrated along the trail.
This is exactly why the task needs a large receptive field rather than a
per-pixel classifier: pixelwise AUC on these features sits near 0.55.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import map_coordinates, median_filter
from shapely.geometry import LineString, box

from . import osm

log = logging.getLogger(__name__)

HALF_WIDTH_M = 6.0
STEP_M = 2.0
BERM_M = 4.0


def transects(line: LineString, field: np.ndarray, inv_transform,
              offsets: np.ndarray) -> list[np.ndarray]:
    out = []
    length = line.length
    for s in np.arange(2.0, max(length - 2.0, 3.0), STEP_M):
        p = line.interpolate(s)
        q = line.interpolate(min(s + 1.0, length))
        dx, dy = q.x - p.x, q.y - p.y
        norm = np.hypot(dx, dy)
        if norm < 1e-6:
            continue
        nx, ny = -dy / norm, dx / norm
        cols, rows = inv_transform * (p.x + nx * offsets, p.y + ny * offsets)
        out.append(map_coordinates(field, [rows, cols], order=1, mode="nearest"))
    return out


def profile_stats(profiles: np.ndarray, offsets: np.ndarray) -> dict:
    """Incision from berm crest to tread centre, plus per-transect noise."""
    mean = np.nanmean(profiles, axis=0)
    centre = len(offsets) // 2
    span = int(BERM_M / (offsets[1] - offsets[0]))
    win = slice(max(centre - span, 0), centre + span + 1)

    incision = 1000.0 * (np.nanmax(mean[win]) - mean[centre])
    noise = 1000.0 * np.nanstd(profiles[:, centre] - np.nanmax(profiles[:, win], axis=1))
    snr = float(incision / noise * np.sqrt(len(profiles))) if noise > 0 else 0.0
    return {
        "n_transects": int(len(profiles)),
        "incision_mm": round(float(incision), 1),
        "noise_mm": round(float(noise), 1),
        "snr": round(snr, 2),
        "mean_profile": [round(float(v) * 1000, 2) for v in mean],
        "offsets_m": [round(float(v), 2) for v in offsets],
    }


#: Visibility grades that make a way "faint" for reporting. These are the ways
#: the project exists to find, so pooling them with clear trails -- as an
#: active/lifecycle split alone does -- hides the only comparison that matters.
FAINT_VISIBILITY = osm.FAINT_VISIBILITY

CLASSES = osm.VISIBILITY_CLASSES

_bucket = osm.visibility_class


def analyse(aoi_dir: Path) -> dict:
    """Measure tread signal separately per visibility class."""
    manifest = json.loads((aoi_dir / "manifest.json").read_text())
    epsg = manifest["epsg"]
    res = manifest["res"]

    with rasterio.open(aoi_dir / "dtm_clean.tif") as src:
        dtm = src.read(1)
        transform = src.transform
        bounds = box(*src.bounds)
    valid = dtm != 0
    if valid.sum() < 100:
        return {"error": "no valid DTM"}
    dtm = np.where(valid, dtm, np.nanmedian(dtm[valid]))
    mrm = dtm - median_filter(dtm, size=max(int(round(10.0 / res)), 5))

    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    elements = json.loads((aoi_dir / "osm.json").read_text())["elements"]
    groups: dict[str, list[LineString]] = {c: [] for c in CLASSES}
    for el in elements:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {})
        kind = osm.classify(tags)
        if kind is None or kind[0] != "trail":
            continue
        groups[_bucket(tags)].append(
            LineString([tf.transform(p["lon"], p["lat"]) for p in geom]))

    offsets = np.arange(-HALF_WIDTH_M, HALF_WIDTH_M + res, res)
    inv = ~transform
    result = {"aoi": manifest["aoi"]["key"]}
    for name, lines in groups.items():
        rows = []
        for line in lines:
            if not line.intersects(bounds):
                continue
            clipped = line.intersection(bounds)
            parts = clipped.geoms if hasattr(clipped, "geoms") else [clipped]
            for part in parts:
                if getattr(part, "length", 0) < 25:
                    continue
                rows.extend(transects(part, mrm, inv, offsets))
        if len(rows) < 40:
            result[name] = {"n_transects": len(rows), "note": "too few transects"}
            continue
        result[name] = profile_stats(np.array(rows), offsets)
    return result


def summarise(results: list[dict]) -> str:
    head = (f'{"aoi":22s} {"class":10s} {"n":>6} {"incision":>10} '
            f'{"noise":>9} {"SNR":>7}  verdict')
    lines = [head, "-" * len(head)]
    for r in results:
        if "error" in r:
            lines.append(f'{r.get("aoi","?"):22s} {r["error"]}')
            continue
        for cls in CLASSES:
            s = r.get(cls)
            if not s or "note" in s:
                continue
            snr = s["snr"]
            verdict = "STRONG" if snr > 8 else "usable" if snr > 3 else "weak"
            lines.append(f'{r["aoi"]:22s} {cls:10s} {s["n_transects"]:6,} '
                         f'{s["incision_mm"]:9.1f}mm {s["noise_mm"]:8.1f}mm '
                         f'{snr:7.2f}  {verdict}')
    return "\n".join(lines)
