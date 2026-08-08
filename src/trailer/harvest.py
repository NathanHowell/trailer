"""Find tiles worth building, instead of guessing coordinates.

The curated set in ``aois.py`` was chosen to span terrain -- canopy, elevation,
substrate, roughness -- and it does that well. What it does not do is contain
the thing the model is supposed to find: of 35 km of labelled trail in the
training tiles, 0.98 km is faint and none at all is lifecycle-tagged. A model
trained on that is a model trained on clear, well-constructed trail.

This module fixes the sampling, not the terrain coverage. It asks Overpass for
every faint and lifecycle-tagged way in a region, bins them onto a grid, and
ranks cells by how much of that kind of way they contain. Terrain diversity
comes along for free -- faint trails are scattered across the whole range.

Two constraints are load-bearing:

* **Held-out tiles are excluded with a wide buffer.** A harvested cell
  overlapping an eval or control tile would train on the test set, and the
  buffer has to exceed the model's context window, not just the tile edge.
* **Disk.** A built tile is ~420 MB, of which ~78 MB is worth keeping. Harvest
  runs assume ``evict_points``; without it a hundred tiles is 42 GB.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import LineString, box
from shapely.strtree import STRtree

from . import coverage, osm
from .aois import AOIS, Aoi

log = logging.getLogger(__name__)

#: High Sierra: Sequoia/Kings Canyon north through Yosemite. All of this sits
#: in UTM zone 11N, so one projected CRS covers the whole grid.
SIERRA_BBOX = (36.20, -119.40, 38.20, -117.90)
GRID_EPSG = "EPSG:32611"

#: Keep harvested cells this far from any curated tile. It must exceed the
#: model's context window (384 px at 0.5 m = 192 m), not merely the tile edge,
#: or a training crop can see pixels a held-out crop also sees.
EXCLUSION_BUFFER_M = 600.0

DEFAULT_MIN_FAINT_M = 400.0


def build_query(south: float, west: float, north: float, east: float) -> str:
    """Faint and lifecycle-tagged ways only.

    Deliberately not ``osm.build_query``: that fetches every highway and
    waterway, which over a 2-degree box is enormous and almost entirely ways we
    already have plenty of.
    """
    bbox = f"({south:.5f},{west:.5f},{north:.5f},{east:.5f})"
    vis = "|".join(sorted(osm.VISIBILITY_WEIGHT))
    parts = [f'  way["highway"]["trail_visibility"~"^({vis})$"]{bbox};']
    parts += [f'  way["{p}:highway"]{bbox};' for p in osm.LIFECYCLE_PREFIXES]
    body = "\n".join(parts)
    return f"[out:json][timeout:900];\n(\n{body}\n);\nout geom;"


def fetch(bbox=SIERRA_BBOX, cache_dir: Path | None = None,
          refresh: bool = False) -> list[dict]:
    cache = (cache_dir / "harvest_ways.json") if cache_dir else None
    if cache and cache.exists() and not refresh:
        return json.loads(cache.read_text())["elements"]

    query = build_query(*bbox)
    data = osm._request(query, attempts=4)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    return data["elements"]


def _interesting(tags: dict) -> str | None:
    """Classify a way as the kind of evidence we are short of."""
    if any(f"{p}:highway" in tags for p in osm.LIFECYCLE_PREFIXES):
        kind = osm.classify(tags)
        return "lifecycle" if kind and kind[0] == "trail" else None
    if tags.get("trail_visibility") in {"bad", "horrible", "no"}:
        kind = osm.classify(tags)
        return "faint" if kind and kind[0] == "trail" else None
    return None


def _exclusion_zones():
    """Curated tiles, buffered, in grid coordinates."""
    tf = Transformer.from_crs("EPSG:4326", GRID_EPSG, always_xy=True)
    zones = []
    for a in AOIS:
        x, y = tf.transform(a.lon, a.lat)
        half = a.size_m / 2 + EXCLUSION_BUFFER_M
        zones.append(box(x - half, y - half, x + half, y + half))
    return zones


def score_cells(elements: list[dict], size_m: int = 1000) -> list[dict]:
    """Accumulate faint/lifecycle way length into grid cells."""
    tf = Transformer.from_crs("EPSG:4326", GRID_EPSG, always_xy=True)
    zones = _exclusion_zones()
    tree = STRtree(zones) if zones else None

    cells: dict[tuple[int, int], dict] = {}
    for el in elements:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        kind = _interesting(el.get("tags", {}))
        if kind is None:
            continue
        line = LineString([tf.transform(p["lon"], p["lat"]) for p in geom])
        if line.length < 1.0:
            continue
        # Walk the way and attribute each step to the cell it falls in. Simply
        # binning endpoints would miss every way that crosses a cell without
        # starting in it, which is most of the long ones.
        step = min(20.0, line.length)
        n = max(int(line.length / step), 1)
        for i in range(n):
            p = line.interpolate((i + 0.5) * line.length / n)
            key = (int(math.floor(p.x / size_m)), int(math.floor(p.y / size_m)))
            rec = cells.setdefault(key, {"faint_m": 0.0, "lifecycle_m": 0.0,
                                         "ways": set()})
            rec[f"{kind}_m"] += line.length / n
            rec["ways"].add(el["id"])

    inv = Transformer.from_crs(GRID_EPSG, "EPSG:4326", always_xy=True)
    out = []
    dropped = 0
    for (gx, gy), rec in cells.items():
        cx, cy = (gx + 0.5) * size_m, (gy + 0.5) * size_m
        cell_box = box(cx - size_m / 2, cy - size_m / 2,
                       cx + size_m / 2, cy + size_m / 2)
        if tree is not None and any(zones[i].intersects(cell_box)
                                    for i in tree.query(cell_box)):
            dropped += 1
            continue
        lon, lat = inv.transform(cx, cy)
        out.append({
            "lat": round(lat, 5), "lon": round(lon, 5),
            "faint_m": round(rec["faint_m"]),
            "lifecycle_m": round(rec["lifecycle_m"]),
            "total_m": round(rec["faint_m"] + rec["lifecycle_m"]),
            "n_ways": len(rec["ways"]),
        })
    out.sort(key=lambda c: -c["total_m"])
    log.info("%d cells with faint/lifecycle way, %d dropped for overlapping "
             "a curated tile", len(out) + dropped, dropped)
    return out


def select_cells(cells: list[dict], limit: int, cache_dir: Path,
                 min_m: float = DEFAULT_MIN_FAINT_M) -> list[Aoi]:
    """Take the best cells that 3DEP actually covers."""
    chosen: list[Aoi] = []
    checked = uncovered = 0
    for cell in cells:
        if len(chosen) >= limit:
            break
        if cell["total_m"] < min_m:
            break
        checked += 1
        try:
            project = coverage.find_project(cell["lat"], cell["lon"], cache_dir)
        except LookupError:
            uncovered += 1
            continue
        key = f"h_{cell['lat']:.4f}_{cell['lon']:.4f}".replace(".", "").replace("-", "s")
        chosen.append(Aoi(
            key=key, name=f"harvest {cell['lat']:.4f},{cell['lon']:.4f}",
            lat=cell["lat"], lon=cell["lon"], role="harvest",
            notes=(f"auto-selected: {cell['faint_m']} m faint, "
                   f"{cell['lifecycle_m']} m lifecycle, {cell['n_ways']} ways, "
                   f"3DEP {project}"),
        ))
    log.info("checked %d candidate cells, %d had no 3DEP coverage, kept %d",
             checked, uncovered, len(chosen))
    return chosen


#: Vetting gates. Every one of these is a *data quality* test. Signal strength
#: is deliberately not among them: faint trails have low SNR by construction
#: (junction_pass measures 1.01), so rejecting weak-signal tiles would discard
#: precisely the examples harvest exists to gather. SNR is recorded instead, for
#: stratified evaluation.
MIN_GROUND_DENSITY = 2.0   # /m2; below this a 0.5 m grid is mostly interpolation
MIN_VALID_FRAC = 0.85      # fraction of the tile with real ground returns
MIN_TRAIL_KM = 0.30        # a way clipped to a sliver is not worth a tile


def vet_tile(d: Path) -> dict:
    """Judge one built tile on data quality. Returns a verdict record."""
    from . import qa

    rec: dict = {"key": d.name, "reasons": []}
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        return rec | {"accepted": False, "reasons": ["not built"]}
    m = json.loads(manifest_path.read_text())
    if "error" in m:
        return rec | {"accepted": False, "reasons": [f"build error: {m['error']}"]}

    points = m.get("points") or {}
    raster = m.get("raster") or {}
    labels = m.get("labels") or {}
    gd = points.get("ground_density")
    vf = raster.get("valid_frac")
    km = labels.get("trail_km", 0.0)
    rec |= {"ground_density": gd, "valid_frac": vf, "trail_km": km,
            "positive_frac": labels.get("positive_frac")}

    if gd is not None and gd < MIN_GROUND_DENSITY:
        rec["reasons"].append(f"ground density {gd:.1f}/m2 < {MIN_GROUND_DENSITY}")
    if vf is not None and vf < MIN_VALID_FRAC:
        rec["reasons"].append(f"valid fraction {vf:.2f} < {MIN_VALID_FRAC}")
    if km < MIN_TRAIL_KM:
        rec["reasons"].append(f"only {km:.2f} km of trail in tile")

    # Recorded, never a gate.
    try:
        signal = qa.analyse(d)
        rec["snr"] = {c: round(signal[c]["snr"], 2) for c in qa.CLASSES
                      if isinstance(signal.get(c), dict) and "snr" in signal[c]}
    except Exception as exc:
        rec["snr"] = {}
        log.debug("qa failed on %s: %s", d.name, exc)

    rec["accepted"] = not rec["reasons"]
    return rec


def vet(dirs: list[Path]) -> list[dict]:
    return [vet_tile(d) for d in dirs]


def write_registry(aois: list[Aoi], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [asdict(a) | {"flags": sorted(a.flags)} for a in aois], indent=1))
    return path
