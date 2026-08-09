# trailer

Detecting hiking trails in LiDAR terrain data, for a JOSM plugin that colours
where trails probably are and lets a human decide.

The output is a probability heatmap, not vectors and not an import. OSM has
strong and well-earned opinions about machine-generated geometry; an assistive
overlay a mapper traces over is a different thing entirely.

## Status

Data pipeline, survey and training stack are done and run end to end. The
74-tile build (14 curated + 60 harvested) is complete: 233.8 km of labelled
trail, 59 of 60 harvested tiles passing `trailer vet`.

A full-set run has completed and been scored on the eval-role tiles — the
honest estimate, since validation F1 (0.6944, the number that picked this
checkpoint) is not. Held-out **active/faint** generalises reasonably (`dem1`
f1@0.5 0.683, active 0.850 — one held-out tile, `junction_pass`). Held-out
**lifecycle** reads f1@0.5 **0.000** on `abandoned_south`, and that number
should be discarded rather than believed: the tile's single OSM way was traced
from a historical topo map (`source=USTopo`), and it has no tread. Cross-
sections along it, with a quadratic fit to the 2.5–12 m flanks removed so the
hillslope and the drainage it follows come out, give a centreline residual of
**−0.1 ± 0.3 cm** against a +0.8 cm random-line null, where tiles the model
scores 0.4–0.9 on read −2 to −14 cm. There is nothing there to find.

A leave-AOI-out control run settles the question the single tile could not.
Retraining the same recipe with six lifecycle and four faint AOIs withheld from
training entirely scores, on ground the model has never seen, `dem1` lifecycle
f1 median **0.614** (range 0.562–0.724) and `lidar05` **0.753** (0.670–0.798) —
level with the within-tile validation band. Lifecycle transfers. Faint is the
class with a real spread: `dem1` median 0.484 over four unseen AOIs, but 0.133
at the low end. See `trailer-360`.

The control tile with zero trail pixels (`north_guard`) keeps false positives
low for both variants — `fp_rate@0.5` 0.00069 `dem1`, 0.00018 `lidar05` — but
**100%** of `dem1`'s sit in the outer four pixels of the tile, which is a
tiling artefact rather than terrain confusion. See `trailer-c02`.

Harvesting was the point of that build, and it worked. Trainable labels went
from 34.2 km active / 0.98 faint / 0.00 lifecycle to **69.7 / 65.2 / 98.9** — a
class balance of 30/28/42 instead of 97/3/0.

The model takes **bare-earth elevation in metres**, not pre-computed rasters.
Terrain derivatives are torch layers inside the graph, so an exported ONNX file
is self-contained: the plugin feeds it one float32 DEM tile and gets a
probability map back, with the median filters, variance windows and
normalisation constants sealed in alongside the weights.

## Requirements

PDAL and GDAL are used as command-line tools, not Python packages:

```sh
brew install pdal gdal            # macOS
sudo apt install pdal gdal-bin    # Debian/Ubuntu; or conda-forge

uv sync                 # data pipeline
uv sync --extra train   # adds torch, segmentation-models-pytorch, onnx
```

The lock file resolves for both platforms — the CUDA wheels, NCCL and Triton
carry `platform_machine == 'x86_64' and sys_platform == 'linux'` markers — so
the same `uv.lock` gives MPS on Apple silicon and CUDA on a Linux x86_64 box
with no edits. `pick_device` prefers CUDA, then MPS, then CPU.

`data/` and `runs/` are gitignored and large (~6 GB and ~1 GB), so moving a
workspace to another machine means `rsync`, not `git clone`. Training state
travels: `runs/<name>/last.pt` carries the optimiser and LR schedule, and
`trailer train --resume` restores them.

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
uv run trailer dem                        # fetch USGS's published 1 m DEM per tile
uv run trailer relabel                    # rebuild labels from cached OSM
uv run trailer train --epochs 40          # U-Net, BCE + Tversky + clDice
uv run trailer predict --aoi colby_pass --tta
uv run trailer export --variant dem1      # ONNX for the JOSM plugin
```

Each tile lands in `data/tiles/<key>/`:

| file | contents |
| --- | --- |
| `points.laz` | 3DEP point cloud, reprojected to UTM; deleted by `--evict-points` |
| `dtm_clean.tif` | our bare-earth DTM in metres — input to 0.5 m variants |
| `dem1m.tif` | USGS's published 1 m DEM — input to 1 m variants, incl. the deployable one |
| `features.tif` | 6-band derived stack; now read only for `chm` and `vdi` |
| `labels.tif` | target / weight / ignore / class |
| `manifest.json` | provenance, density, label statistics |

The two elevation rasters are not interchangeable. 1 m variants read `dem1m.tif`
because that is what the plugin will be handed; a tile without one is skipped for
those variants rather than silently substituted.

A built tile is ~420 MB, of which ~78 MB is worth keeping. Bulk runs assume
`--evict-points`; without it sixty tiles is 25 GB.

## What the survey established

Numbers below are measured, not assumed. See `trailer qa`.

**1 m costs real signal, and the cost is worst where it hurts most.** 3DEP
ground-return density across the Sierra is a consistent 6–13 pts/m² regardless
of canopy, because the survey spec targets ground returns, so 0.5 m is cheap for
us to produce. The survey originally concluded 1 m would average the tread away,
reasoning from a 1–1.5 m tread width. That reasoning had the wrong feature in
mind — the mean cross-section is **4–9 m wide**, the bench-and-berm earthwork
rather than the tread notch, and the millimetre figure is its *depth*.

But the conclusion still lands close to where it started, for a different
reason. Measured against USGS's **published** 1 m DEM — which is what a JOSM
plugin actually gets — median incision retained is:

| class | retained at 1 m |
| --- | --- |
| active | 0.75 |
| faint | 0.99 |
| lifecycle | **0.54** |

Lifecycle suffers most because those features are shallowest (7–22 mm here), so
smoothing takes proportionally more of them — and lifecycle is both the hardest
target and, after harvesting, the most abundant.

An earlier version of this section reported 1.10–1.12 retained and called 1 m
"not disqualifying". That measurement used a 2×2 block-mean of our own gridding
as a stand-in for the published product, which is the friendly case: same
interpolation, just coarser. The real product is gridded differently, and its
tread-scale band correlates with ours at only r = 0.10–0.18 (see *Input
variants*). Measure against the artefact you will actually deploy on.

This is the argument for the 0.5 m stem rather than against 1 m: the sub-metre
path retains what the published product loses, and both are trained jointly
rather than chosen between — see *Model*.

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
meaningless here. Model selection now scores each visibility class separately for
exactly this reason — see **Selection is stratified** below.

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
(excellent 1.0 → no 0.35) with lifecycle-tagged ways at 0.6. Band 4 codes which
kind of way covers the pixel (active 1, faint 2, lifecycle 3), so metrics and the
crop sampler can stratify instead of pooling a set that used to be 97% active.

**The loss is tolerant at the same radius the metric scores at.** Training
pixel-exact against geometry known to sit ~1.4 m off would spend gradient
teaching the model to reproduce tracing error — measured, a strict loss saturates
by 3 px, punishing a 3 m offset exactly as hard as predicting nothing. The
relaxation is asymmetric like `metrics.relaxed`: the prediction is dilated for
recall, the label for precision.

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

**Selection is stratified**, and the warning above about pooled numbers is now
enforced rather than merely written down. The score a checkpoint is kept on is
the mean over (input variant × visibility class) of relaxed F1, taken at the best
threshold in 0.2–0.8 rather than read at 0.5. Two reasons:

* Pooled recall is weighted by labelled kilometres, so it hands the decision to
  whichever class the corpus holds most of — currently lifecycle, at 98.9 km
  against faint's 65.2. A model that finds every active metre and no faint one
  scores 0.97 pooled on the synthetic case in `tests/test_metrics.py`, and 0.50
  stratified.
* Where the sigmoid sits drifts between runs, because the output bias is set from
  the sampled prior and the loss reweights positives. Searching the threshold
  measures the ranking, which is the model's property; calibration is a
  deployment choice made later.

Precision is deliberately *not* split by class. A predicted pixel carries no
class, and a highlight drawn over a real abandoned trail is correct however the
score is sliced — so each class combines its own recall with the pooled
precision.

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
output  window_taper        (1, 1, N, N) float32, blending weight
```

The taper is a graph output rather than something the plugin computes, and
`--tta` bakes the D4 average into the graph rather than exposing a flag: both
are cases of the same rule, that anything the plugin would otherwise
reimplement lives behind the ONNX boundary. Numbers the plugin needs in order
to tile — stride, step, pad mode — are in the sidecar JSON as numbers, read
rather than re-derived. Prose in that sidecar is how the two once came to
disagree about the window step.

The window is fixed rather than dynamic. That is a real constraint of the
architecture, not an export limitation — a ResNet-34 U-Net needs its input
divisible by 32 — and it costs nothing, because the plugin must tile with a
Hann-tapered overlapping window regardless.

Export goes through the TorchScript exporter, not dynamo, which fails on the
stems' BatchNorm. The one op with no ONNX equivalent is the median filter;
`im2col` + `TopK` reproduces `scipy.ndimage.median_filter` bit-for-bit at
k = 3, 4, 10, 20, including the lower-median convention for even k. A round trip
through onnxruntime on real elevation matches torch to 2.7e-6.

The plugin's whole-raster path — reflect-pad, tile, run, blend, crop — is
checked end to end against `infer.predict` on the same weights, not step by
step: **3.9e-7** across a finished raster, where shifting one window by a single
column moves it by **1.0**. The fixture runs a 2.6 KB stand-in graph rather than
the 99 MB trained one, since what is under test is the tiling and the session
plumbing, and it covers a stride-2 variant as well as the deployable stride-1
one so the body-grid division is exercised by something.

## Licensing

Two artefacts, two licences, on purpose.

| | Licence | |
|---|---|---|
| Source code, JOSM plugin | **MIT** | `LICENSE` |
| Trained weights (`.onnx`, `.pt`, sidecar) | **CC BY-SA 4.0** | `LICENSE-MODEL` |

MIT is GPL-2.0-compatible, so a plugin combined with JOSM is fine: the combined
work is GPL and the plugin's own terms stay MIT.

The weights are share-alike because they are trained on two sources with very
different terms. USGS 3DEP is a US government work and carries no copyright.
OpenStreetMap is ODbL, which *is* share-alike, and OSM geometry is used here as
training labels.

Whether neural network weights are an ODbL "Derivative Database" — which must be
ODbL — or a "Produced Work" — which may be licensed freely with attribution —
is not settled, and the OSM Foundation has issued no guideline that squarely
answers it. This project's position is Produced Work: the model reproduces no
OSM geometry, cannot be queried for OSM features, does not consult OSM at
inference time, and emits a raster rather than data. But that is an argument,
not a ruling, so the weights are released share-alike regardless. If the
Produced Work reading is right, CC BY-SA gives away more than required; if it is
wrong, the obligation has been met anyway. Being wrong in that direction costs
nothing, which is the only asymmetry that matters. Full reasoning, and why
CC BY-SA rather than ODbL itself or plain CC BY, is in `LICENSE-MODEL`.

Attribution travels in the export sidecar rather than living only here, because
what a mapper downloads is the weights. `ModelSpec` **refuses to load a model
whose sidecar has no attribution**, and refuses rather than substituting a
built-in default — a fallback would let a stripped file paint anyway, which is
the one thing the check exists to stop. The plugin shows the notice in the
layer's info panel.

## Using this in OSM

Read this before tracing anything.

**This is an assistive overlay, not a data source.** It colours where a trail
probably is. It does not know whether that trail exists, whether it is public,
whether it is a trail at all rather than an old skid road, a firebreak, a
cattle path or a stream cut. You do.

**It is not an import and must not be used as one.** There is no "convert to
way" button and there will not be one. Every way is drawn by a person who
looked at the evidence and judged it real. That is what keeps ML-assisted
mapping welcome in OSM, and it is a product decision rather than an unfinished
feature.

**Verify against something else.** The model buys recall at the cost of
precision — deliberately, because a faint candidate you reject costs a moment
and one you never see costs the trail. A high-probability blob is a prompt to
go and look at the hillshade, the imagery, your GPS traces or your memory of
walking it. It is not evidence on its own.

**Do not bulk-trace a sweep.** Lowering the threshold until the map lights up
and tracing everything is exactly the behaviour that gets tools like this
banned. If you cannot say why a specific line is a trail, do not draw it.

**Tag honestly.** If you traced from this overlay, `source=…` should say so
alongside the underlying elevation source, e.g.
`source=USGS 3DEP LiDAR;trailer`. A reviewer who finds a mistake needs to know
what to distrust.

**Scope.** The model is trained on the western US mountain terrain listed below
and validated there. Nothing establishes that it transfers to other landscapes,
and its false-positive behaviour off that distribution is simply unmeasured.

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
