# trailer

Detecting hiking trails in LiDAR terrain data, for a JOSM plugin that colours
where trails probably are and lets a human decide.

The output is a probability heatmap, not vectors and not an import. OSM has
strong and well-earned opinions about machine-generated geometry; an assistive
overlay a mapper traces over is a different thing entirely.

## Status

Data pipeline, survey and training stack are done and run end to end. A 74-tile
build (14 curated + 60 harvested) is in flight; no model has been trained on the
full set yet.

The model takes **bare-earth elevation in metres**, not pre-computed rasters.
Terrain derivatives are torch layers inside the graph, so an exported ONNX file
is self-contained: the plugin feeds it one float32 DEM tile and gets a
probability map back, with the median filters, variance windows and
normalisation constants sealed in alongside the weights.

## Requirements

PDAL and GDAL are used as command-line tools, not Python packages:

```sh
brew install pdal gdal
uv sync                 # data pipeline
uv sync --extra train   # adds torch, segmentation-models-pytorch, onnx
```

Inference in JOSM goes through ONNX Runtime's Java API, so torch is an optional
extra rather than a dependency. Export needs `onnx` only — not `onnxscript`,
since it uses the TorchScript exporter rather than dynamo.

## Usage

```sh
uv run trailer survey                     # 3DEP coverage per AOI, no downloads
uv run trailer build --aoi all            # point clouds -> features -> labels
uv run trailer build --aoi giant_forest --res 0.25
uv run trailer qa                         # measured tread signal per tile
uv run trailer preview --aoi moraine_lake # hillshade + label overlay PNG
uv run trailer harvest --limit 60         # find tiles rich in faint/lifecycle way
uv run trailer vet                        # data-quality gates on harvested tiles
uv run trailer train --epochs 40          # U-Net, BCE + Tversky + clDice
uv run trailer predict --aoi colby_pass --tta
uv run trailer export --variant dem1      # ONNX for the JOSM plugin
```

Each tile lands in `data/tiles/<key>/`:

| file | contents |
| --- | --- |
| `points.laz` | 3DEP point cloud, reprojected to UTM; deleted by `--evict-points` |
| `dtm_clean.tif` | bare-earth DTM in metres — **the model's actual input** |
| `features.tif` | 6-band derived stack; now read only for `chm` and `vdi` |
| `labels.tif` | target / weight / ignore |
| `manifest.json` | provenance, density, label statistics |

A built tile is ~420 MB, of which ~78 MB is worth keeping. Bulk runs assume
`--evict-points`; without it sixty tiles is 25 GB.

## What the survey established

Numbers below are measured, not assumed. See `trailer qa`.

**1 m is not disqualifying — and that reverses an early call.** 3DEP
ground-return density across the Sierra is a consistent 6–13 pts/m² regardless
of canopy, because the survey spec targets ground returns, so 0.5 m is cheap to
produce. The survey originally concluded that 1 m would average the tread away,
reasoning from a 1–1.5 m tread width. Measured, that reasoning had the wrong
feature in mind: the mean cross-section is **4–9 m wide** — the bench-and-berm
earthwork, not the tread notch — and the millimetre figure is its *depth*, which
block-averaging preserves. Over ten tiles, median incision retained at 1 m is
1.10–1.12 for active and faint ways and 0.87 for lifecycle ones.

The exception falls exactly where it hurts: `junction_pass`'s faint way halves,
26.2 → 12.9 mm. So 1 m is usable, and sub-metre earns its cost precisely at the
weak end. Both are trained jointly rather than chosen between — see *Model*.

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
terrain, the active trail shows 45.7 mm of berm-to-tread incision and the
pre-1932 route beside it still shows 9.8 mm. What survives is the earthwork.
The method will be strongest on old *constructed* trails — stock routes, CCC-era
work, mining tracks — and weakest on unconstructed use trails no matter how
heavily walked.

**Faint is much harder than the average tile suggests.** Stratified by
`trail_visibility`, faint ways run SNR 0.8–4.1 against 2.4–15.2 for clear ones.
`blackcap` inverts the order — its faint way scores *better* than its active one
because it lies in quieter terrain (89 mm noise against 159 mm) — which is the
roughness-as-denominator result appearing inside a single tile. Report faint
recall separately from headline recall; a single pooled number is close to
meaningless here.

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

The first four are computed **inside the model** (`preprocess.py`) from raw
elevation, at whatever pixel size that variant carries. Every window is defined
in metres, so a band means the same thing at 0.5 m and at 1 m; the clip bounds
scale with the window actually used, without which the 3 px floor saturated 15%
of the roughness band at 1 m before the network saw it.

`chm` and `vdi` cannot be recovered from a DTM — they need the point cloud — so
bare-earth variants do without them entirely.

Absolute elevation is deliberately absent, and removed inside the graph. Every
derivative is a local difference, so a constant offset cancels algebraically but
not in float32: at 3000 m the representable spacing is 0.24 mm, 1.6% of a 15 mm
tread. Centring first makes the cancellation exact. The same trap caught
roughness, where `E[z²] - E[z]²` on raw elevation drifted by 7% of the band's own
spread — and the drift grew with elevation, so identical terrain scored
differently at 300 m and 3000 m. De-trending over 6 m first fixes it.

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

## Model

U-Net with an ImageNet-pretrained ResNet-34 encoder, one output channel.
Pretraining earns its place even though micro-relief looks nothing like
photographs — the early layers are edge and ridge detectors, and this is far too
little data to learn those from scratch.

### Input variants

What JOSM can reach at runtime is the USGS 3DEP ImageServer: `pixelType F32`,
one band, 1 m, **bare earth**. Real float elevation rather than a rendered
hillshade, which is the good news; the bad news is that going to 1 m and losing
canopy are the same event. Training only on 0.5 m six-band stacks would produce
a model that cannot be deployed at all.

| variant | source | pixel size | bands |
| --- | --- | --- | --- |
| `lidar05` | our own point-cloud stacks | 0.5 m | all six |
| `dem1` | 3DEP ImageServer — the runtime case | 1 m | four, bare earth |
| `lidar1` | as `lidar05`, decimated | 1 m | all six; isolates resolution from band loss |

Each variant gets its own small stem — 32k of 24.6M parameters — feeding one
shared trunk and head. The trunk always runs at 1 m, so its kernels mean one
fixed physical size rather than having to be scale-invariant, and the stems
absorb the difference. A 0.5 m stem is stride-2, which lets it learn a matched
filter across the tread *before* decimating rather than mean-pooling the detail
away. The alternative — one model per source — would split an already small
training set in half.

Variants interleave step by step during training, not epoch by epoch, so the
trunk never spends a stretch seeing one input scale and drifting towards it.
Selection is on the mean relaxed F1 across variants: optimising the 0.5 m path
alone would quietly let the deployable one rot. Held-out tiles are scored per
variant, so the gap between them prices what deploying against a public DEM
costs instead of our own point clouds.

**Output is at 1 m for every variant**, and the 0.5 m path is supervised against
max-pooled 1 m labels. Sub-metre input buys signal *fidelity*, not output
*resolution*. Against a 5 m label tolerance that is not the binding constraint,
but it is a real choice — per-resolution heads would restore 0.5 m output at the
cost of a shared head.

Three loss terms, each covering what the others miss:

| term | covers |
| --- | --- |
| weighted BCE | per-pixel calibration; alone it collapses to all-background at 0.7% positives |
| Tversky (α=0.3, β=0.7) | region overlap, false negatives penalised harder |
| clDice | topology — Dice barely notices a one-pixel gap in a 1 m trail, but that gap is what makes a proposal unusable |

**Imbalance** is handled by sampling and loss, not by the optimizer — AdamW
already rescales the small sparse gradients a rare class produces. Trails are
0.55–2.5% of pixels depending on the tile; positive-biased sampling lifts that
to ~3.1% and, more usefully, equalises it across tiles so a label-sparse tile
still contributes a full share of signal per batch. `pos_weight=8` then brings
the effective ratio to about 3.9:1. The output conv's bias is initialised to the
weighted base rate, so the model starts at the prior instead of predicting
p=0.5 everywhere and spending its opening steps unlearning that — measured, it
cuts step-0 BCE by 43%.

Recall is deliberately bought at the cost of precision. A reviewer in JOSM
dismisses a false positive in a second; a trail never drawn is invisible to
them. clDice is ramped in after a few epochs, since skeletonising random early
predictions produces gradients that fight the region terms.

**Scoring** is relaxed precision/recall at a 5 m tolerance, not strict pixel
overlap — a perfect prediction can miss the OSM centreline by three pixels and
still be exactly what the mapper wants. Pixel AP is reported for comparison with
the survey baselines, but never used for model selection: it rewards fattening
predictions until they cover the label slop, and relaxed F1 does not.

**Splits.** Each training tile reserves a column band for validation, with the
boundary placed by label quantile rather than at a fixed fraction of width —
trails cluster, and a fixed cut can hand validation a strip containing no trail
at all. The eval-role tiles (abandoned trails) and the control tile are scored
once at the end of a run, on full tiles with sliding-window inference. They are
a test set; selecting on them would spend the only honest estimate of
abandoned-trail recall and false-positive rate that exists.

**Augmentation** is limited to transforms needing no interpolation. D4 — the
eight dihedral transforms — is a pure array permutation, and terrain has no
canonical orientation, so it is free diversity. Arbitrary rotation, stretching
and elastic warps are not used: bilinear resampling smooths away exactly the
relief the model has to see. Noise is injected into *elevation, in metres*,
where a sensor actually puts it, and its sigma is swept per crop rather than
fixed — detectability is the tread-to-roughness ratio, and that ratio is the
axis faint trails fail on.

Inference blends 50%-overlapping windows under a 2-D Hann taper, with optional
D4 test-time augmentation — nearly free accuracy at 8× the compute.

## Deployment

`trailer export --variant dem1` freezes one bare-earth variant to ONNX:

```
input   elevation_m         (1, 1, N, N) float32, metres, NaN for nodata
output  trail_probability   (1, 1, N, N) float32, at 1 m
```

The window is fixed rather than dynamic. That is a real constraint of the
architecture, not an export limitation — a ResNet-34 U-Net needs its input
divisible by 32 — and it costs nothing, because the plugin must tile with a
Hann-tapered overlapping window regardless.

Export goes through the TorchScript exporter, not dynamo, which fails on the
stems' BatchNorm. The one op with no ONNX equivalent is the median filter;
`im2col` + `TopK` reproduces `scipy.ndimage.median_filter` bit-for-bit at
k = 3, 4, 10, 20, including the lower-median convention for even k. A round trip
through onnxruntime on real elevation matches torch to 2.7e-6.

## Areas

**Fourteen curated tiles** spanning 1941–3690 m, 1.4–71% canopy, 10–530 m
relief, and granite / volcanic / sand / talus substrates. `north_guard` has no
mapped ways at all and exists solely to measure false-positive rate — everywhere
else, an apparent false positive may be a real unmapped trail.

**Sixty harvested tiles**, because the curated set spans *terrain* well and the
thing the model must find badly: of 35.2 km of labelled trail in it, 34.2 km is
active, 0.98 km faint, and **none at all** lifecycle-tagged. `trailer harvest`
asks Overpass for every faint and lifecycle-tagged way in the High Sierra,
bins them onto a 1 km grid, and ranks cells by how much of that kind of way they
hold — 1202 ways over 1453 cells, of which the top 60 carry 142.6 km, roughly
145× the faint evidence available before.

Held-out tiles are excluded with a 600 m buffer, which has to exceed the model's
context window rather than merely the tile edge, or a training crop can see
pixels a held-out crop also sees.

`trailer vet` gates harvested tiles on ground density, valid fraction and
in-tile trail length. Signal strength is deliberately *not* among the gates:
faint trails are low-SNR by construction, so rejecting weak-signal tiles would
discard precisely the examples harvesting exists to gather. SNR is recorded
instead, for stratified evaluation.

## Prior art worth reading

- [TrailScan](https://github.com/GISLAB-HAWK/TrailScan-QGIS-Plugin) — skid
  trails from ALS in a QGIS plugin via ONNX. Closest analog; same deployment
  shape. GPL-2.0, weights CC BY 4.0.
- [ADAF](https://github.com/EarthObservation/adaf) — archaeological feature
  detection from ALS, retrainable, Apache-2.0.
- CarcassonNet — detects and traces hollow roads in Dutch LiDAR.
- [MapWithAI](https://github.com/JOSM/MapWithAI) — the JOSM-side template for
  ML-assisted mapping.
