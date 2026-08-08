"""OSM trail labels via Overpass, including lifecycle-prefixed ways.

Two things here matter more than they look:

1. Lifecycle prefixes (``abandoned:highway``, ``disused:highway``, ...) must be
   queried. LiDAR responds to trail *construction*, not traffic, so an
   abandoned bench cut is still plainly a trail on the ground. Omitting these
   ways means scoring correct detections as false positives.

2. ``trail_visibility`` is carried through as a per-pixel loss weight. It is a
   rare thing to have: a human-assigned difficulty grade over the whole label
   set, which makes stratified evaluation possible.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from . import atomic

log = logging.getLogger(__name__)

ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

#: Overpass rejects the default requests User-Agent outright -- 406 from the
#: main instance, 429 from the mirrors, both of which look like load-shedding
#: rather than the client fault they are. Identifying yourself is required
#: etiquette, not optional.
HEADERS = {
    "User-Agent": (
        "trailer/0.1 (LiDAR trail detection for OSM; "
        "https://github.com/NathanHowell/trailer)"
    ),
}

#: Ways whose tread is genuinely invisible in a bare-earth DTM. Boardwalk sits
#: on the ground and paved paths are graded flush -- measured cross-profiles are
#: flat. Training on them injects pure noise.
INVISIBLE_SURFACES = frozenset({"wood", "asphalt", "concrete", "paved", "metal"})

ACTIVE_HIGHWAYS = frozenset({
    "path", "footway", "track", "bridleway", "cycleway", "steps",
})

LIFECYCLE_PREFIXES = (
    "abandoned", "disused", "was", "razed", "demolished", "removed",
)

#: Visibility grades that make a way "faint". These are the ways the project
#: exists to find, so pooling them with clear trails hides the only comparison
#: that matters.
FAINT_VISIBILITY = frozenset({"bad", "horrible", "no"})

#: Ordered so a higher code wins where ways overlap: the rarer, harder class is
#: the one we want a pixel attributed to.
VISIBILITY_CLASSES = ("active", "faint", "lifecycle")
CLASS_CODE = {c: i + 1 for i, c in enumerate(VISIBILITY_CLASSES)}


def visibility_class(tags: dict[str, str]) -> str:
    """Which kind of evidence a way represents: active, faint, or lifecycle."""
    if any(f"{p}:highway" in tags for p in LIFECYCLE_PREFIXES):
        return "lifecycle"
    if tags.get("trail_visibility") in FAINT_VISIBILITY:
        return "faint"
    return "active"

#: Loss weight by trail_visibility. Faint trails are real but the label is less
#: certain and the terrain evidence is weaker, so they are down-weighted rather
#: than dropped.
VISIBILITY_WEIGHT = {
    "excellent": 1.00,
    "good": 1.00,
    "intermediate": 0.90,
    "bad": 0.70,
    "horrible": 0.50,
    "no": 0.35,
}
DEFAULT_WEIGHT = 1.00
LIFECYCLE_WEIGHT = 0.60


def build_query(south: float, west: float, north: float, east: float) -> str:
    bbox = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    parts = [f'  way["highway"]{bbox};']
    parts += [f'  way["{p}:highway"]{bbox};' for p in LIFECYCLE_PREFIXES]
    parts.append(f'  way["waterway"]{bbox};')
    body = "\n".join(parts)
    return f"[out:json][timeout:180];\n(\n{body}\n);\nout geom;"


def fetch(south: float, west: float, north: float, east: float,
          cache_path: Path | None = None, refresh: bool = False,
          attempts: int = 4) -> dict:
    """Run an Overpass query with retry/backoff across mirrors."""
    if cache_path and cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())

    data = _request(build_query(south, west, north, east), attempts)
    if cache_path:
        # Staged: an interrupted write here leaves a half-JSON file that the
        # branch above would hand straight back on the next run. That one fails
        # loudly rather than silently -- unlike a truncated LAZ -- but it is a
        # cache nothing cleans up, so a tile stays broken until someone deletes
        # the file by hand.
        atomic.write_text(cache_path, json.dumps(data))
    return data


def _request(query: str, attempts: int = 4) -> dict:
    """POST a query, rotating mirrors and backing off."""
    # Keep one error per endpoint. Reporting only the last one hides which
    # mirror actually broke, and they fail for different reasons.
    errors: dict[str, str] = {}
    for attempt in range(attempts):
        for url in ENDPOINTS:
            host = url.split("/")[2]
            try:
                r = requests.post(url, data=query.encode(), timeout=240,
                                  headers=HEADERS)
            except requests.RequestException as exc:
                errors[host] = f"{type(exc).__name__}: {exc}"[:160]
                continue
            # Overpass signals load-shedding with an HTML error page, not a 5xx.
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                return r.json()
            snippet = r.text[:160].replace("\n", " ")
            errors[host] = f"HTTP {r.status_code}: {snippet}"
        if attempt < attempts - 1:
            wait = 8 * (attempt + 1)
            log.warning("overpass unavailable, retrying in %ds (%s)", wait,
                        "; ".join(f"{h}: {e[:60]}" for h, e in errors.items()))
            time.sleep(wait)
    detail = "\n  ".join(f"{h}: {e}" for h, e in errors.items())
    raise RuntimeError(f"overpass failed after {attempts} attempts:\n  {detail}")


def classify(tags: dict[str, str]) -> tuple[str, float] | None:
    """Map a way's tags to (class, loss weight), or None to ignore it.

    Classes:
        trail     -- positive, terrain-visible
        excluded  -- a real trail whose tread cannot appear in a DTM; these
                     become ignore-region rather than negative, so the model is
                     not punished for firing where a boardwalk sits on a
                     genuine old alignment
        water     -- used to mask lakes and streams
    """
    if "waterway" in tags:
        return ("water", 0.0) if tags["waterway"] in {"river", "stream", "riverbank"} else None

    lifecycle = next((f"{p}:highway" for p in LIFECYCLE_PREFIXES if f"{p}:highway" in tags), None)
    highway = tags.get("highway")

    if lifecycle:
        value = tags[lifecycle]
        if value not in ACTIVE_HIGHWAYS and value not in {"unclassified", "service", "trunk", "tertiary"}:
            return None
        return "trail", LIFECYCLE_WEIGHT
    if highway not in ACTIVE_HIGHWAYS:
        return None
    if tags.get("surface") in INVISIBLE_SURFACES:
        return "excluded", 0.0
    if tags.get("bridge") in {"yes", "boardwalk"} or tags.get("tunnel") == "yes":
        return "excluded", 0.0

    vis = tags.get("trail_visibility")
    return "trail", VISIBILITY_WEIGHT.get(vis, DEFAULT_WEIGHT)
