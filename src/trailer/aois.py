"""Area-of-interest registry.

Every AOI here has been checked against the USGS 3DEP entwine index and probed
for ground-return density. Notes record what each tile is *for* -- the set is
chosen to span canopy, elevation, substrate and terrain roughness, not just to
accumulate area.

Roles:
    train    -- goes into the training split
    harvest  -- auto-selected by `trailer harvest`; also trains
    eval     -- held out; scored but never trained on
    control  -- no mapped trails, used to measure the false-positive rate
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Aoi:
    key: str
    name: str
    lat: float
    lon: float
    role: str = "train"
    size_m: int = 1000
    notes: str = ""
    # Set when the tile needs special handling during raster build.
    flags: frozenset[str] = field(default_factory=frozenset)
    #: Why this tile's score is not evidence, or "" if it is.
    #:
    #: A reason rather than a boolean, and carried beside the number rather
    #: than filtering it out, because a held-out tile that quietly disappears
    #: from a report is worse than one that argues for itself: the next reader
    #: re-derives the number, believes it, and puts it in a release note. Every
    #: consumer that prints held-out scores prints this next to them.
    advisory: str = ""

    @property
    def slug(self) -> str:
        return self.key


#: Measured during survey; see docs/survey.md for the full table.
AOIS: tuple[Aoi, ...] = (
    # ---- forested / low elevation -----------------------------------------
    Aoi("giant_forest", "Giant Forest", 36.56523, -118.76903,
        notes="1941 m, 71% canopy, 6.9 ground/m2. Densest OSM trail network "
              "(7.8 km/km2). Generals Highway included as a positive control."),
    Aoi("devils_postpile", "Devils Postpile", 37.63000, -119.08500,
        notes="2304 m, 46% canopy, flat (10 m relief). Volcanic substrate -- "
              "the only non-granite tile. JMT/PCT, 4.6 km/km2."),
    Aoi("mineral_king", "Mineral King", 36.45200, -118.59500,
        notes="2390 m, 34% canopy. Historic mining district: old roads and "
              "tracks are confounders, and unmapped historic trails abound."),

    # ---- subalpine ---------------------------------------------------------
    Aoi("evolution_creek", "Evolution Creek crossing", 37.19606, -118.77891,
        notes="2809 m, 50% canopy, 12 m relief. Tread SNR 4.4."),
    Aoi("blackcap", "Blackcap trail", 37.04877, -118.77430,
        notes="2937 m, 50% canopy. Weakest measured signal (SNR 3.0); keep as "
              "a hard case but do not benchmark on it."),
    Aoi("horseshoe_meadow", "Horseshoe Meadow", 36.45000, -118.17000,
        notes="3056 m, sandy decomposed granite, flat. Best signal in the "
              "survey (SNR 12.5) and best ground density (10.4/m2). The "
              "unambiguous-positive anchor for the training set."),

    # ---- alpine / granite --------------------------------------------------
    Aoi("kearsarge_pass", "Kearsarge Pass", 36.77000, -118.39000,
        notes="3307 m, 18% canopy. Treeline transition, talus switchbacks."),
    Aoi("rae_lakes", "Rae Lakes isthmus", 36.80376, -118.39911,
        flags=frozenset({"water"}),
        notes="3228 m. ~40% open water -- MUST be masked or it trains the "
              "model on interpolated lake surface."),
    Aoi("colby_pass", "Colby Pass", 36.61874, -118.50097,
        notes="3652 m, 4% canopy, 124 m relief. Alpine granite; tread is "
              "present (48 mm) but terrain noise is 310 mm."),
    Aoi("whitney_switchbacks", "Mt Whitney switchbacks", 36.56600, -118.26000,
        notes="3432 m, 243 m relief. Deepest tread measured (101 mm) -- "
              "heavily engineered switchbacks. Strongest positive class."),

    # ---- moraine / confounder ---------------------------------------------
    Aoi("moraine_lake", "Moraine Lake", 36.47131, -118.46124,
        notes="2889 m, 16.7 ground/m2 (densest in the set), 404 m relief. "
              "Lateral moraine crests beside real trail -- the confounder "
              "tile. Moraine Lake Trail measures SNR 13.3, second only to "
              "Horseshoe Meadow. Label-sparse: only 1.5 km of trail in tile. "
              "Big Arroyo Trail (trail_visibility=horrible) is ~1 km away and "
              "NOT in this box; widen size_m to ~2500 to capture it."),

    # ---- abandoned-trail evaluation ---------------------------------------
    Aoi("junction_pass", "Junction Pass old trail", 36.69005, -118.34841,
        role="eval",
        notes="3690 m, 530 m relief. Pre-1932 JMT route. Carries a paired "
              "trail_visibility=intermediate and trail_visibility=horrible way "
              "in identical terrain -- the reference tile for faint-trail "
              "recall. Note the old route is graded by trail_visibility here, "
              "NOT by a lifecycle prefix: this tile has zero lifecycle ways."),
    Aoi("abandoned_south", "Abandoned trail (Kern side)", 36.42955, -118.43660,
        role="eval",
        notes="2679 m, 45% canopy, 20.2 ground/m2 (second densest in the set), "
              "138 m relief. Holds exactly ONE way: 1.31 km of "
              "abandoned:highway=path, w/1498594057, traced from a historical "
              "topo map -- changeset 181381229 declares source='USTopo; USGS "
              "3D Elevation Program'. No active path, despite what this note "
              "used to claim.",
        advisory="nothing along this way is visible to LiDAR in any channel "
                 "measured: a matched filter for the tread notch and the "
                 "bench-and-berm finds no local peak anywhere in +/-40 m, "
                 "where it reads +1.1 flank MAD at zero offset on "
                 "junction_pass active; ground-return density reads +2.7% "
                 "on-line against +22% there; and the one sharp intensity "
                 "spike on the line sits entirely in the drainage the way "
                 "follows (+1.7 flank s.d. over 510 channel sections, -0.2 "
                 "over 765 hillslope ones). Scoring abandoned-trail recall "
                 "here measures the label, not the model"),

    # ---- control -----------------------------------------------------------
    Aoi("north_guard", "North Guard (control)", 36.75154, -118.48835,
        role="control",
        notes="3130 m, 252 m relief, zero mapped ways. Avalanche chutes and "
              "glacial striations present. The only place false-positive rate "
              "can be measured honestly."),
)

#: Auto-selected tiles written by `trailer harvest`. Kept in a generated file
#: rather than appended here: this module is hand-annotated with what each tile
#: is *for*, and hundreds of machine-picked entries would drown that.
HARVEST_REGISTRY = Path("data/harvest.json")


def load_harvest(path: Path | None = None) -> tuple[Aoi, ...]:
    path = path or HARVEST_REGISTRY
    if not path.exists():
        return ()
    return tuple(
        Aoi(**(rec | {"flags": frozenset(rec.get("flags", ()))}))
        for rec in json.loads(path.read_text())
    )


def all_aois(harvest: bool = True) -> tuple[Aoi, ...]:
    return AOIS + (load_harvest() if harvest else ())


BY_KEY: dict[str, Aoi] = {a.key: a for a in AOIS}


def advisory(key: str) -> str:
    """Why ``key``'s score is not evidence, or "" if it is (or is unknown).

    Returns "" for a key that is not in the registry rather than raising: a
    caller holding a directory name should not be the thing that fails when a
    tile is built from a registry it has since dropped out of.
    """
    return BY_KEY[key].advisory if key in BY_KEY else ""


def select(keys: str | None = None, role: str | None = None,
           harvest: bool = True) -> list[Aoi]:
    """Resolve a comma-separated key list and/or a role filter to AOIs."""
    registry = all_aois(harvest)
    by_key = {a.key: a for a in registry}
    out = list(registry)
    if keys and keys != "all":
        wanted = [k.strip() for k in keys.split(",")]
        missing = [k for k in wanted if k not in by_key]
        if missing:
            raise KeyError(f"unknown AOI(s): {', '.join(missing)}. "
                           f"known: {', '.join(by_key)}")
        out = [by_key[k] for k in wanted]
    if role:
        out = [a for a in out if a.role == role]
    return out
