"""USGS 3DEP point-cloud coverage lookup.

Resolves a lat/lon to an EPT (Entwine Point Tile) resource on the public
usgs-lidar bucket. Note this index is the Hobu-maintained entwine mirror, which
is not the whole of 3DEP -- some collections (Yosemite NP among them) exist in
the National Map but not here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import requests
from shapely.geometry import Point, shape

log = logging.getLogger(__name__)

RESOURCES_URL = "https://usgs.entwine.io/boundaries/resources.geojson"
EPT_BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"


def resources_path(cache_dir: Path) -> Path:
    return cache_dir / "3dep_resources.geojson"


def fetch_index(cache_dir: Path, refresh: bool = False) -> dict:
    """Download (and cache) the 3DEP entwine boundary index."""
    path = resources_path(cache_dir)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching 3DEP coverage index ...")
    r = requests.get(RESOURCES_URL, timeout=300)
    r.raise_for_status()
    path.write_text(r.text)
    return r.json()


def find_project(lat: float, lon: float, cache_dir: Path,
                 prefer: str | None = None) -> str:
    """Return the EPT project name covering a point.

    Prefers the project with the most points when several overlap, since the
    denser collection is nearly always the newer one.
    """
    index = fetch_index(cache_dir)
    p = Point(lon, lat)
    hits = [
        (f["properties"]["name"], f["properties"].get("count", 0))
        for f in index["features"]
        if shape(f["geometry"]).contains(p)
    ]
    if not hits:
        raise LookupError(
            f"no 3DEP EPT coverage at {lat:.5f},{lon:.5f}. The entwine mirror "
            f"is incomplete -- check the National Map before concluding the "
            f"data does not exist."
        )
    if prefer:
        for name, _ in hits:
            if name == prefer:
                return name
    return max(hits, key=lambda h: h[1])[0]


def ept_url(project: str) -> str:
    return f"{EPT_BASE}/{project}/ept.json"
