"""Per-AOI build driver: point cloud -> feature stack -> labels -> manifest."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer

from . import coverage, labels, osm, rasters
from .aois import Aoi

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"


def aoi_bbox(aoi: Aoi, epsg: str, pad_m: float = 80.0) -> tuple[float, float, float, float]:
    """Lat/lon bbox covering the AOI, padded so edge trails are labelled."""
    to_m = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    to_ll = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
    x, y = to_m.transform(aoi.lon, aoi.lat)
    half = aoi.size_m / 2 + pad_m
    corners = [to_ll.transform(px, py)
               for px in (x - half, x + half) for py in (y - half, y + half)]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lats), min(lons), max(lats), max(lons)


def point_stats(laz_path: Path, size_m: int) -> dict:
    """Density and classification summary, read in chunks.

    A 1 km tile at 3DEP density is 80-100M points; reading it whole costs
    several GB, and we only need counts plus ground elevation.
    """
    total = 0
    n_ground = 0
    ground_z: list[np.ndarray] = []
    with laspy.open(str(laz_path)) as fh:
        for chunk in fh.chunk_iterator(5_000_000):
            cls = np.asarray(chunk.classification)
            total += len(cls)
            g = cls == 2
            n_ground += int(g.sum())
            if g.any():
                # subsample; the median is stable and this bounds memory
                z = np.asarray(chunk.z)[g]
                ground_z.append(z[::20] if len(z) > 200_000 else z)
    if n_ground < 100:
        raise ValueError(f"only {n_ground} ground points -- likely a data hole")
    zg = np.concatenate(ground_z)
    area = float(size_m) ** 2
    return {
        "points": total,
        "density": round(total / area, 1),
        "ground_density": round(n_ground / area, 1),
        "ground_frac": round(100 * n_ground / total, 1),
        "elev_median": round(float(np.median(zg)), 1),
    }


def build_aoi(aoi: Aoi, root: Path, res: float = 0.5,
              force: bool = False, cache: Path | None = None) -> dict:
    """Build one AOI. Idempotent unless force=True."""
    out = root / aoi.slug
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / MANIFEST
    if manifest_path.exists() and not force:
        log.info("%-22s cached", aoi.key)
        return json.loads(manifest_path.read_text())

    cache = cache or (root / ".cache")
    t0 = time.time()
    rec: dict = {"aoi": asdict(aoi) | {"flags": sorted(aoi.flags)},
                 "res": res}

    project = coverage.find_project(aoi.lat, aoi.lon, cache)
    rec["project"] = project
    epsg = rasters.utm_epsg(aoi.lat, aoi.lon)
    rec["epsg"] = epsg
    log.info("%-22s project=%s %s", aoi.key, project, epsg)

    # Fetch OSM first. It is the cheapest step and the only one that depends on
    # a flaky third party -- doing it after the ~10 min raster build means a
    # transient Overpass outage throws all that work away.
    south, west, north, east = aoi_bbox(aoi, epsg)
    elements = osm.fetch(south, west, north, east,
                         cache_path=out / "osm.json", refresh=force)["elements"]
    log.info("%-22s %d OSM ways", aoi.key, len(elements))

    laz = out / "points.laz"
    if not laz.exists() or force:
        rasters.extract_points(coverage.ept_url(project), aoi.lat,
                               aoi.lon, aoi.size_m, laz)
    rec["points"] = point_stats(laz, aoi.size_m)
    log.info("%-22s %s pts, ground %.1f/m2", aoi.key,
             f"{rec['points']['points']:,}", rec["points"]["ground_density"])

    feats = out / "features.tif"
    raster_meta = out / "raster.json"
    if feats.exists() and raster_meta.exists() and not force:
        rec["raster"] = json.loads(raster_meta.read_text())
        log.info("%-22s reusing feature stack", aoi.key)
    else:
        rec["raster"] = rasters.build_feature_stack(laz, feats, res)
        raster_meta.write_text(json.dumps(rec["raster"], indent=1))

    rec["labels"] = labels.build(elements, feats, out / "labels.tif", epsg)

    if "water" in aoi.flags:
        frac = labels.mask_water_from_dtm(out / "dtm_clean.tif", out / "labels.tif")
        rec["labels"]["water_masked_frac"] = round(frac, 4)
        log.info("%-22s masked %.1f%% as water", aoi.key, 100 * frac)

    rec["seconds"] = round(time.time() - t0, 1)
    manifest_path.write_text(json.dumps(rec, indent=1))
    log.info("%-22s done in %.0fs -- %.2f km trail, %.2f%% positive",
             aoi.key, rec["seconds"], rec["labels"]["trail_km"],
             100 * rec["labels"]["positive_frac"])
    return rec


def build_all(aois, root: Path, res: float = 0.5, force: bool = False) -> list[dict]:
    out = []
    for aoi in aois:
        try:
            out.append(build_aoi(aoi, root, res=res, force=force))
        except Exception as exc:  # keep going; one bad tile shouldn't stop a run
            log.error("%-22s FAILED: %s", aoi.key, exc)
            out.append({"aoi": {"key": aoi.key}, "error": str(exc)})
    (root / "index.json").write_text(json.dumps(out, indent=1))
    return out
