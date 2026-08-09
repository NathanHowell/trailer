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

    # ---- held-out spread ---------------------------------------------------
    #
    # Promoted out of the harvest pool so every visibility class is scored on at
    # least three AOIs the model has never seen. One tile per class was never an
    # eval set: across this corpus the same checkpoint scores per-tile lifecycle
    # F1 anywhere from 0.00 to 0.94, so a single draw carries no information
    # about the model -- which is how a topo-map trace with no measurable tread
    # came to be read as a capability gap. See trailer-360.
    #
    # Three constraints, in order:
    #
    # * **No neighbour inside the 600 m buffer.** The harvest grid puts cells a
    #   kilometre apart, so promoting a tile whose neighbour still trains leaks
    #   context straight across the split -- a training crop sees pixels a
    #   held-out crop also sees. A first pass at this set picked three tiles
    #   that were *adjacent* to training cells, gap ~0 m. Checked in
    #   test_aois.py rather than by eye.
    # * **A floor, not a ranking.** Each carries at least ~1 km of its class at
    #   an effect size (tread over per-transect noise) of 0.15 or better, and
    #   the set then spans the measurable range rather than skimming the
    #   strongest. Choosing the clearest tiles would build an eval set that
    #   flatters the model.
    # * **Spread of terrain**: 1430-3009 m, 0-92% canopy, 42-527 m relief.
    #
    # Numbers are from `trailer qa`: berm-to-centre incision against
    # per-transect terrain noise, and `eff` is their ratio.
    Aoi("h_381905_s1193237", "Held-out active/lifecycle, 38.19N 119.32W",
        38.19046, -119.32371, role="eval",
        notes="2136 m, 8% canopy, 7.8 ground/m2, 53 m relief. "
              "active 4.60 km, 88 mm tread on 125 mm noise (SNR 33.7, eff 0.71); "
              "lifecycle 2.04 km, 22 mm tread on 113 mm noise (SNR 6.2, eff 0.20)."),
    Aoi("h_373473_s1185298", "Held-out active/lifecycle, 37.35N 118.53W",
        37.34732, -118.52982, role="eval",
        notes="1681 m, 4% canopy, 6.7 ground/m2, 246 m relief. The low-signal "
              "end of the active range, deliberately. "
              "active 4.49 km, 41 mm tread on 200 mm noise (SNR 9.5, eff 0.20); "
              "lifecycle 4.70 km, 25 mm tread on 199 mm noise (SNR 6.0, eff 0.13)."),
    Aoi("h_365429_s1186925", "Held-out active/lifecycle, forested, 36.54N 118.69W",
        36.54291, -118.69255, role="eval",
        notes="1430 m, 92% canopy, 5.1 ground/m2, 336 m relief. The densest "
              "canopy in the held-out set, and the thinnest ground returns. "
              "active 2.43 km, 38 mm tread on 181 mm noise (SNR 7.2, eff 0.21); "
              "lifecycle 2.25 km, 47 mm tread on 222 mm noise (SNR 7.0, eff 0.21)."),
    Aoi("h_368724_s1182959", "Held-out faint, 36.87N 118.30W",
        36.87239, -118.29591, role="eval",
        notes="1752 m, 16% canopy, 6.2 ground/m2, 508 m relief. "
              "faint 3.40 km, 62 mm tread on 164 mm noise (SNR 15.5, eff 0.38). "
              "Its 0.41 km of active at eff 0.14 is too thin to read."),
    Aoi("h_370339_s1183661", "Held-out faint, 37.03N 118.37W",
        37.03385, -118.36611, role="eval",
        notes="2259 m, 5% canopy, 10.2 ground/m2, 527 m relief. "
              "faint 2.20 km, 30 mm tread on 144 mm noise (SNR 7.0, eff 0.21). "
              "Its 0.44 km of active at eff 0.17 is too thin to read."),
    Aoi("h_364230_s1181657", "Held-out faint, alpine, 36.42N 118.17W",
        36.42302, -118.16570, role="eval",
        notes="3009 m, 42% canopy, 19.4 ground/m2, 285 m relief. The alpine end "
              "of the faint range. "
              "faint 2.06 km, 27 mm tread on 135 mm noise (SNR 6.4, eff 0.20). "
              "Also 0.40 km active and 0.74 km lifecycle -- both under a "
              "kilometre, so read those two here as thin rather than as "
              "evidence."),
    Aoi("h_376597_s1187516", "Held-out lifecycle, 37.66N 118.75W",
        37.65968, -118.75160, role="eval",
        notes="2067 m, 0% canopy, 7.6 ground/m2, 42 m relief. Flat and quiet, "
              "the cleanest lifecycle case in the held-out set. "
              "lifecycle 2.77 km, 32 mm tread on 72 mm noise (SNR 16.3, eff 0.44)."),
    Aoi("h_362058_s1182515", "Held-out lifecycle, forested, 36.21N 118.25W",
        36.20579, -118.25146, role="eval",
        notes="2651 m, 45% canopy, 10.2 ground/m2, 94 m relief. "
              "lifecycle 2.69 km, 52 mm tread on 146 mm noise (SNR 13.0, eff 0.35)."),

    # ---- control -----------------------------------------------------------
    Aoi("north_guard", "North Guard (control)", 36.75154, -118.48835,
        role="control",
        notes="3130 m, 252 m relief, zero mapped ways. Avalanche chutes and "
              "glacial striations present. The only place false-positive rate "
              "can be measured honestly."),
)

BY_KEY: dict[str, Aoi] = {a.key: a for a in AOIS}


#: Auto-selected tiles written by `trailer harvest`. Kept in a generated file
#: rather than appended here: this module is hand-annotated with what each tile
#: is *for*, and hundreds of machine-picked entries would drown that.
HARVEST_REGISTRY = Path("data/harvest.json")


def load_harvest(path: Path | None = None) -> tuple[Aoi, ...]:
    """Machine-picked tiles, minus any this module already annotates by hand.

    The hand-annotated entry wins, and that precedence is load-bearing rather
    than tidy. ``data/harvest.json`` is gitignored and machine-local, so a tile
    promoted out of the harvest pool into an eval role here is promoted *only
    here*. Without this filter, a workspace whose registry still listed it would
    load the same key twice with two different roles, the harvest copy would win
    by insertion order, and the tile would quietly go back to training -- an
    eval set that silently dissolves when the work moves to another machine.
    """
    path = path or HARVEST_REGISTRY
    if not path.exists():
        return ()
    return tuple(
        Aoi(**(rec | {"flags": frozenset(rec.get("flags", ()))}))
        for rec in json.loads(path.read_text())
        if rec.get("key") not in BY_KEY
    )


def all_aois(harvest: bool = True) -> tuple[Aoi, ...]:
    return AOIS + (load_harvest() if harvest else ())



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
