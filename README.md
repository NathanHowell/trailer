# trailer

Detecting hiking trails in LiDAR terrain data, for a JOSM plugin that colours
where trails probably are and lets a human decide.

The output is a probability heatmap, not vectors and not an import. OSM has
strong and well-earned opinions about machine-generated geometry; an assistive
overlay a mapper traces over is a different thing entirely.

## Status

Data pipeline and survey are done. Model training is not started.

## Requirements

PDAL and GDAL are used as command-line tools, not Python packages:

```sh
brew install pdal gdal
uv sync
```

## Usage

```sh
uv run trailer survey                     # 3DEP coverage per AOI, no downloads
uv run trailer build --aoi all            # point clouds -> features -> labels
uv run trailer build --aoi giant_forest --res 0.25
uv run trailer qa                         # measured tread signal per tile
uv run trailer preview --aoi moraine_lake # hillshade + label overlay PNG
```

Each tile lands in `data/tiles/<key>/`:

| file | contents |
| --- | --- |
| `points.laz` | 3DEP point cloud, reprojected to UTM |
| `features.tif` | 6-band model input (see below) |
| `labels.tif` | target / weight / ignore |
| `dtm_clean.tif` | bare-earth DTM, for hillshade review |
| `manifest.json` | provenance, density, label statistics |

## What the survey established

Numbers below are measured, not assumed. See `trailer qa`.

**Work at 0.25–0.5 m, not 1 m.** 3DEP ground-return density across the Sierra
is a consistent 6–13 pts/m² regardless of canopy, because the survey spec
targets ground returns. Trail tread is ~1–1.5 m wide and 15–100 mm deep; at 1 m
per pixel you average it away.

**The signal is spatial, not per-pixel.** Pixelwise AUC for terrain derivatives
(micro-relief, slope, roughness, curvature) is 0.51–0.56 — near chance. Ridge
and vesselness filters built for tubular structures do no better. A CNN with
56 m of context reaches 0.649 *while trained on another continent*. Trails are
recognisable only by integrating along their length, which is why a
large-receptive-field CNN is the mechanism rather than a convenience, and why
thresholding a hand-crafted terrain index will not work.

**Terrain roughness is the binding constraint, not tread depth.** Per-transect
noise runs 65 mm in a sandy meadow to 500 mm in alpine talus, while tread is
15–100 mm throughout. Detectability is the ratio.

**LiDAR sees construction, not traffic.** At Junction Pass, in identical
terrain, an active trail shows 63 mm of berm-to-tread incision and a trail
abandoned since 1932 still shows 15 mm. What survives is the earthwork. The
method will be strongest on old *constructed* trails — stock routes, CCC-era
work, mining tracks — and weakest on unconstructed use trails no matter how
heavily walked.

**Lifecycle tags must be labels.** `abandoned:highway`, `disused:highway` and
friends account for 865 km of mapped-but-faint way in the Sierra. Omitting them
scores correct detections as false positives. `trail_visibility` comes along
for free as a graded difficulty axis — rare and worth exploiting for stratified
evaluation.

**Some trails are physically unlearnable.** Boardwalk and paved paths show a
flat cross-profile: they sit on the ground rather than cutting into it. They
are marked *ignore*, not negative.

## Feature bands

| band | why |
| --- | --- |
| `mrm_2m` | micro-relief at tread scale |
| `mrm_10m` | bench-and-berm cross-section |
| `slope` | trails are locally flatter than the sidehill |
| `roughness` | local terrain noise — the denominator of detectability |
| `chm` | canopy gap over the corridor |
| `vdi` | low/high vegetation ratio; cleared understory |

Absolute elevation is deliberately absent. The network must key on tens of
millimetres of local relief, and 2000 m of regional topography only wastes
capacity.

## Labels

```
<= 2 m from centreline   positive
2-5 m                    ignored (OSM alignment is ~1.4 m median here)
> 5 m                    negative
```

Measured OSM-vs-LiDAR offset in this region is 1.4 m median, 84% within 3 m —
good for OSM, but still 3–6 pixels at 0.5 m. Hard negatives immediately
adjacent to the mapped line would punish the model for finding the real tread.

`labels.tif` band 2 carries a per-pixel loss weight from `trail_visibility`
(excellent 1.0 → no 0.35) with lifecycle-tagged ways at 0.6.

## Areas

Fourteen tiles spanning 1941–3690 m, 1.4–71% canopy, 10–530 m relief, and
granite / volcanic / sand / talus substrates. `north_guard` has no mapped ways
at all and exists solely to measure false-positive rate — everywhere else, an
apparent false positive may be a real unmapped trail.

## Prior art worth reading

- [TrailScan](https://github.com/GISLAB-HAWK/TrailScan-QGIS-Plugin) — skid
  trails from ALS in a QGIS plugin via ONNX. Closest analog; same deployment
  shape. GPL-2.0, weights CC BY 4.0.
- [ADAF](https://github.com/EarthObservation/adaf) — archaeological feature
  detection from ALS, retrainable, Apache-2.0.
- CarcassonNet — detects and traces hollow roads in Dutch LiDAR.
- [MapWithAI](https://github.com/JOSM/MapWithAI) — the JOSM-side template for
  ML-assisted mapping.
