"""Area-of-interest registry.

Every AOI here has been checked against the USGS 3DEP entwine index and probed
for ground-return density. Notes record what each tile is *for* -- the set is
chosen to span canopy, elevation, substrate and terrain roughness, not just to
accumulate area.

Roles:
    train    -- goes into the training split
    eval     -- held out; scored but never trained on
    control  -- no mapped trails, used to measure the false-positive rate
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
        notes="3690 m, 530 m relief. Pre-1932 JMT route. Active tread 63 mm "
              "vs abandoned 15 mm in identical terrain -- the reference tile "
              "for abandoned-trail recall."),
    Aoi("abandoned_south", "Abandoned trail (Kern side)", 36.42955, -118.43660,
        role="eval",
        notes="Paired abandoned:highway=path and active path."),

    # ---- control -----------------------------------------------------------
    Aoi("north_guard", "North Guard (control)", 36.75154, -118.48835,
        role="control",
        notes="3130 m, 252 m relief, zero mapped ways. Avalanche chutes and "
              "glacial striations present. The only place false-positive rate "
              "can be measured honestly."),
)

BY_KEY: dict[str, Aoi] = {a.key: a for a in AOIS}


def select(keys: str | None = None, role: str | None = None) -> list[Aoi]:
    """Resolve a comma-separated key list and/or a role filter to AOIs."""
    out = list(AOIS)
    if keys and keys != "all":
        wanted = [k.strip() for k in keys.split(",")]
        missing = [k for k in wanted if k not in BY_KEY]
        if missing:
            raise KeyError(f"unknown AOI(s): {', '.join(missing)}. "
                           f"known: {', '.join(BY_KEY)}")
        out = [BY_KEY[k] for k in wanted]
    if role:
        out = [a for a in out if a.role == role]
    return out
