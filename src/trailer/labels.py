"""Rasterise OSM ways into target / weight / ignore planes.

The ignore plane is the important part. Measured OSM-vs-LiDAR alignment in the
High Sierra is ~1.4 m median (84% within 3 m), which is good for OSM but still
1-3 pixels at 0.5 m. Treating everything outside a tight buffer as hard
negative would punish the model for firing on the true tread when the mapped
centreline is a metre off. So:

    <= 2 m from centreline   positive
    2-5 m                    ignored (alignment slop)
    > 5 m                    negative

Band 4 records which *kind* of way covers a pixel -- active, faint, or lifecycle
-- because nothing downstream could otherwise tell them apart. Without it,
metrics can only report a pooled number over a set that is 97% active trail by
length, and the crop sampler can only draw faint examples in proportion to that
same 3%. Both of those quietly optimise for the trails we are not looking for.

Boardwalk/paved ways and open water are ignored outright rather than marked
negative: they are places where a trail may genuinely exist but cannot leave a
signature, so neither answer is informative.
"""
from __future__ import annotations

import logging

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import LineString, box

from . import osm

log = logging.getLogger(__name__)

#: Band layout of labels.tif. Readers index positionally, so appending only.
BAND_NAMES = ("target", "weight", "ignore", "class")

POSITIVE_M = 2.0
IGNORE_M = 5.0
EXCLUDED_M = 4.0


def _to_lines(elements: list[dict],
              epsg: str) -> list[tuple[LineString, str, float, str]]:
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    out = []
    for el in elements:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {})
        kind = osm.classify(tags)
        if kind is None:
            continue
        cls, weight = kind
        line = LineString([tf.transform(p["lon"], p["lat"]) for p in geom])
        out.append((line, cls, weight, osm.visibility_class(tags)))
    return out


def build(elements: list[dict], reference_tif, out_tif, epsg: str) -> dict:
    """Write a 3-band label raster: target, weight, ignore."""
    with rasterio.open(reference_tif) as src:
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs
        extent = box(*src.bounds)

    lines = _to_lines(elements, epsg)
    trails = [(l, w, v) for l, c, w, v in lines if c == "trail"]
    excluded = [l for l, c, _, _ in lines if c == "excluded"]
    water = [l for l, c, _, _ in lines if c == "water"]

    def burn(geoms, value=1):
        if not geoms:
            return np.zeros(shape, dtype="float32")
        return rasterize([(g, value) for g in geoms], out_shape=shape,
                         transform=transform, dtype="float32", all_touched=True)

    target = burn([l.buffer(POSITIVE_M) for l, _, _ in trails])
    near = burn([l.buffer(IGNORE_M) for l, _, _ in trails])

    # Class plane, burned low code first so a rarer class wins any overlap.
    klass = np.zeros(shape, dtype="float32")
    by_class: dict[str, list] = {}
    for line, _, v in trails:
        by_class.setdefault(v, []).append(line.buffer(POSITIVE_M))
    for name in osm.VISIBILITY_CLASSES:
        if name in by_class:
            code = osm.CLASS_CODE[name]
            klass = np.maximum(klass, burn(by_class[name]) * code)

    # Weight plane: max weight of any trail covering the pixel. Group by value
    # first -- there are only a handful of distinct weights, and rasterising
    # once per way would mean hundreds of full-size passes.
    weight = np.zeros(shape, dtype="float32")
    by_weight: dict[float, list] = {}
    for line, w, _ in trails:
        by_weight.setdefault(w, []).append(line.buffer(POSITIVE_M))
    for w, geoms in sorted(by_weight.items()):
        weight = np.maximum(weight, burn(geoms) * w)

    ignore = ((near > 0) & (target == 0)).astype("float32")
    if excluded:
        ignore = np.maximum(ignore, burn([l.buffer(EXCLUDED_M) for l in excluded]))
    if water:
        ignore = np.maximum(ignore, burn([l.buffer(3.0) for l in water]))
    ignore[target > 0] = 0.0

    # negatives carry unit weight; positives carry their visibility weight
    weight = np.where(target > 0, np.maximum(weight, 0.05), 1.0).astype("float32")
    weight[ignore > 0] = 0.0

    klass[target == 0] = 0.0

    meta = dict(driver="GTiff", height=shape[0], width=shape[1], count=4,
                dtype="float32", crs=crs, transform=transform,
                compress="deflate", predictor=2, tiled=True)
    with rasterio.open(out_tif, "w", **meta) as d:
        d.write(target, 1)
        d.write(weight, 2)
        d.write(ignore, 3)
        d.write(klass, 4)
        d.descriptions = BAND_NAMES

    n_life = sum(1 for _, w, _ in trails if abs(w - osm.LIFECYCLE_WEIGHT) < 1e-9)
    # Report length *inside the tile*. Ways are fetched with a padded bbox and
    # routinely run for kilometres beyond it, so their full length says nothing
    # about how much supervision this tile actually carries.
    in_tile = sum(l.intersection(extent).length for l, _, _ in trails)
    # Per-class kilometres in the tile. The whole reason for harvesting is that
    # this breakdown was 34.2 active / 0.98 faint / 0.00 lifecycle across the
    # curated set, so it needs to be visible per tile rather than recomputed by
    # hand each time someone wonders.
    km_by_class = {c: 0.0 for c in osm.VISIBILITY_CLASSES}
    for line, _, v in trails:
        km_by_class[v] += line.intersection(extent).length / 1000.0
    return {
        "trail_km_by_class": {k: round(v, 3) for k, v in km_by_class.items()},
        "positive_frac_by_class": {
            c: round(float((klass == osm.CLASS_CODE[c]).mean()), 6)
            for c in osm.VISIBILITY_CLASSES},
        "ways_trail": len(trails),
        "ways_lifecycle": n_life,
        "ways_excluded": len(excluded),
        "positive_frac": float((target > 0).mean()),
        "ignore_frac": float((ignore > 0).mean()),
        "trail_km": round(in_tile / 1000, 2),
        "trail_km_untrimmed": round(sum(l.length for l, _, _ in trails) / 1000, 2),
    }


def mask_water_from_dtm(dtm_path, labels_path, flat_thresh: float = 0.02) -> float:
    """Flag interpolated lake surfaces as ignore.

    Lakes return no ground points, so the IDW fill produces implausibly flat
    plateaus. Rae Lakes is ~40% water; left alone the model would learn that
    perfectly smooth terrain is strongly negative, which is true but useless.
    """
    from scipy.ndimage import uniform_filter

    with rasterio.open(dtm_path) as s:
        dtm = s.read(1)
    valid = dtm != 0
    local_sd = np.sqrt(np.clip(
        uniform_filter(dtm ** 2, 15) - uniform_filter(dtm, 15) ** 2, 0, None))
    flat = valid & (local_sd < flat_thresh)

    with rasterio.open(labels_path) as s:
        bands = s.read()
        profile = s.profile
    h, w = min(bands.shape[1], flat.shape[0]), min(bands.shape[2], flat.shape[1])
    bands[2, :h, :w] = np.maximum(bands[2, :h, :w], flat[:h, :w])
    bands[1] = np.where(bands[2] > 0, 0.0, bands[1])
    with rasterio.open(labels_path, "w", **profile) as d:
        d.write(bands)
        d.descriptions = BAND_NAMES[:bands.shape[0]]
    return float(flat.mean())
