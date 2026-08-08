"""Crop sampling over built tiles.

The dataset serves raw bare-earth elevation, not derived bands. Derivation moved
into the model (``preprocess``) so the exported ONNX is self-contained, and that
means training has to feed the same thing the JOSM plugin will: metres of
elevation, ``NaN`` for nodata. Canopy is the exception -- ``chm`` and ``vdi``
cannot be recovered from a DTM, so they are still read from ``features.tif``,
and only canopy-bearing variants get them.

Every variant reads the *same* native window and differs only in how it is
reduced, so a 0.5 m crop and the 1 m crop derived from it cover identical
ground. Comparisons across variants are therefore paired rather than confounded
by which patch of mountain each happened to draw.

Labels always come back at ``BODY_RES``, since that is where the shared head
predicts. Target is max-pooled -- a trail thinner than the coarse pixel must
survive, not dissolve -- and weight is mean-pooled, which keeps the 2-5 m
alignment ring as a soft margin instead of an all-or-nothing one.

Two sampling decisions carry most of the weight here:

* **Positive-biased crops.** Trails cover 0.5-1% of pixels. Uniform random crops
  would be almost entirely empty, so half of each batch is centred on a labelled
  trail pixel and half is drawn uniformly. The uniform half is not padding --
  it is where the model learns that moraine crests and avalanche chutes are not
  trails, which is the failure mode that actually matters.

* **Spatial validation split, not held-out tiles.** A column band of every tile
  is reserved for validation, so the validation terrain distribution matches
  training and val loss is usable for model selection. The eval-role tiles
  (abandoned trails) and the control tile stay entirely untouched -- they are a
  test set, and scoring them during training would burn them.

  The band boundary is placed by *label* quantile rather than at a fixed 20% of
  width. Trails are not spread evenly across a tile -- Moraine Lake's are all in
  the middle -- so a fixed cut can hand validation a strip with no trail in it
  at all, and every recall number computed from it is then noise.

Augmentation is limited to transforms needing no interpolation. D4 -- the eight
dihedral transforms -- is a pure array permutation, and terrain has no canonical
orientation, so it is free diversity. Arbitrary rotation, stretching and elastic
warps are not used: the signal is 15-100 mm of relief on a cross-section a few
pixels wide, and bilinear resampling smooths away exactly what the model has to
see. Noise is now injected into *elevation, in metres*, which is where a sensor
actually puts it -- previously it went into normalised derivative bands, which
models nothing physical.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from . import osm
from . import variants as var_mod

log = logging.getLogger(__name__)

#: Target share of each tile's *labelled trail pixels* held back for validation.
VAL_COL_FRAC = 0.2

#: The boundary is clamped into this fraction-of-width range, so a tile whose
#: trails all sit at one edge cannot hand most of its area to either split.
BOUNDARY_RANGE = (0.55, 0.85)

#: Cap on remembered trail-pixel locations per tile, per class. A 1 km tile at
#: 0.5 m holds ~40k positive pixels; keeping every one buys nothing over a
#: strided sample.
MAX_CENTRES = 20_000

#: Share of positive crops drawn from each visibility class, independent of how
#: many kilometres of each exist. Harvesting fixed the supply -- trainable faint
#: went from 0.98 km to 62.8 km and lifecycle from nothing to 95.5 km -- but
#: supply is not exposure: drawing centres uniformly over trail pixels would
#: still hand the model whatever the length ratio happens to be, and that ratio
#: is an accident of what got harvested rather than a statement about what we
#: want it to learn. Equal thirds says the quiet part directly.
CLASS_MIX = {"active": 1 / 3, "faint": 1 / 3, "lifecycle": 1 / 3}

#: Band of labels.tif carrying the class code (1-indexed for rasterio).
CLASS_BAND = 4

#: USGS's published 1 m DEM, fetched by `trailer dem`. Variants at BODY_RES read
#: this rather than a block-mean of our own gridding, because they are not the
#: same raster: measured across four tiles, slope and mrm_10m agree well
#: (r 0.79-0.92) but the tread-scale mrm_2m band correlates at only r = 0.10-0.18
#: with roughly half the amplitude. USGS grids from ground returns with its own
#: interpolation and hydro-flattening, so the fine-scale content is different
#: content, not a smoothed version of ours. Training on the proxy would leave the
#: deployed model reading a band it had never seen.
DEM_1M = "dem1m.tif"

#: Bands of features.tif holding chm and vdi (1-indexed for rasterio).
CANOPY_BANDS = (5, 6)

#: Correlation length of the band-limited noise, sampled per crop. Chosen to
#: straddle the measured 4-9 m bench-and-berm cross-section, so the noise lands
#: on the same scale as the feature whose SNR it is meant to sweep.
BAND_NOISE_SCALE_M = (4.0, 8.0)


def _boundary(target: np.ndarray, crop: int) -> int:
    """Column separating the train band from the validation band.

    Placed so roughly VAL_COL_FRAC of the tile's trail pixels fall to the right
    of it, then clamped so neither split can be starved of area.

    The clamp also has to leave both bands wider than one crop. A band exactly
    one crop wide admits no crop *centres* at all -- the centre has to sit half
    a crop from either edge -- so the tile silently drops out of validation.
    That bit the two weakest-signal tiles once already, at 384 px; reserving one
    crop was not enough, and doubling the crop to cover 1 m body pixels
    re-triggered it on every tile at once. Reserve half a crop more.
    """
    w = target.shape[1]
    lo, hi = (int(w * f) for f in BOUNDARY_RANGE)
    lo, hi = max(lo, crop), min(hi, w - 3 * crop // 2)
    if lo > hi:  # tile too narrow to split at all
        lo = hi = w // 2
    per_col = (target > 0).sum(axis=0)
    total = per_col.sum()
    if total == 0:
        return int(w * (1 - VAL_COL_FRAC))
    cut = int(np.searchsorted(np.cumsum(per_col), (1 - VAL_COL_FRAC) * total))
    return int(np.clip(cut, lo, hi))


def _fold(a: np.ndarray, k: int) -> np.ndarray:
    """Fold trailing spatial dims into blocks of k for reduction."""
    h, w = (a.shape[-2] // k) * k, (a.shape[-1] // k) * k
    a = a[..., :h, :w]
    return a.reshape(*a.shape[:-2], h // k, k, w // k, k)


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    return a if k == 1 else _fold(a, k).mean(axis=(-3, -1))


def block_max(a: np.ndarray, k: int) -> np.ndarray:
    return a if k == 1 else _fold(a, k).max(axis=(-3, -1))


def block_min_nonzero(a: np.ndarray, k: int) -> np.ndarray:
    """Reduce a class-code plane: the lowest nonzero code in a block wins.

    Not block_max, which would hand every shared body pixel to lifecycle -- the
    highest code and, since the harvest, the most abundant class. This matches
    the precedence labels.py burns the plane with, so a body pixel covering two
    ways is called the same thing whether the overlap happened at label
    resolution or at this reduction. Zero (background) is excluded rather than
    winning by being smallest.
    """
    if k == 1:
        return a
    f = _fold(a, k)
    out = np.where(f > 0, f, np.inf).min(axis=(-3, -1))
    return np.where(np.isfinite(out), out, 0.0).astype("float32")


def band_noise(shape, sigma: float, scale_px: float,
               rng: np.random.Generator) -> np.ndarray:
    """Spatially correlated noise: Gaussian on a coarse grid, bilinearly upsampled.

    White noise is the wrong stimulus for this task. Measured on a junction_pass
    crop, per-pixel noise at sigma=0.05 m moves the tread band mrm_2m by 0.326 of
    its spread but the bench band mrm_10m by only 0.083 -- it sweeps the SNR of
    the wrong feature four times harder than the right one. Noise drawn on a
    4-8 m grid moves mrm_10m just as much (0.074) while barely touching mrm_2m
    (0.049). Real terrain noise -- talus, gullying, tree throw -- is correlated
    anyway; white noise models nothing physical at these scales.

    Interpolation is fine here, unlike everywhere else in this file: the
    no-interpolation rule protects signal from being smoothed away, and this is
    noise being synthesised, not data being resampled.
    """
    from scipy.ndimage import zoom

    k = max(scale_px, 2.0)
    small = rng.normal(0.0, 1.0, (int(shape[0] / k) + 2, int(shape[1] / k) + 2))
    up = zoom(small, k, order=1)[:shape[0], :shape[1]]
    if up.shape != tuple(shape):  # zoom rounds; pad the last row/column
        pad = [(0, s - u) for s, u in zip(shape, up.shape)]
        up = np.pad(up, pad, mode="edge")
    # Bilinear upsampling suppresses variance, so rescale to the sigma asked for.
    return (up * (sigma / max(up.std(), 1e-9))).astype("float32")


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Translate an array, filling exposed edges with zero."""
    if dr == 0 and dc == 0:
        return a
    out = np.zeros_like(a)
    sr0, dr0 = (0, dr) if dr >= 0 else (-dr, 0)
    sc0, dc0 = (0, dc) if dc >= 0 else (-dc, 0)
    h = a.shape[-2] - abs(dr)
    w = a.shape[-1] - abs(dc)
    out[..., dr0:dr0 + h, dc0:dc0 + w] = a[..., sr0:sr0 + h, sc0:sc0 + w]
    return out


def block_nanmean(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over valid cells; NaN only where the whole block is nodata."""
    if k == 1:
        return a
    with np.errstate(invalid="ignore"):
        return np.nanmean(_fold(a, k), axis=(-3, -1))


class TileDataset(Dataset):
    """Random crops from built AOI directories, for one input variant.

    Each item is ``(z, canopy, y, w, cls)``: elevation in metres at the variant's
    pixel size, canopy bands (empty for bare-earth variants), and target, weight
    and visibility class at ``BODY_RES``. Ignored pixels -- the 2-5 m alignment
    ring, boardwalks, water, and raster nodata -- arrive as ``w == 0`` rather
    than as a separate mask, so every loss gets the masking for free by
    multiplying.

    ``cls`` carries ``osm.CLASS_CODE`` values on trail pixels and 0 elsewhere.
    No loss reads it -- the model is not asked to tell an active trail from an
    abandoned one -- but scoring does, so that a checkpoint is chosen on how it
    does across visibility classes rather than on a pooled number the longest
    class wins by default.
    """

    def __init__(self, dirs: list[Path], variant: var_mod.Variant,
                 body_crop: int = 256, split: str = "train",
                 samples: int = 2000, positive_frac: float = 0.5,
                 augment: bool = True, noise_m: float = 0.05,
                 noise_band_m: float = 0.05,
                 canopy_dropout: float = 0.15, jitter_m: float = 2.0,
                 seed: int = 1234):
        self.variant = variant
        self.body_crop = body_crop
        self.split = split
        self.samples = samples
        self.positive_frac = positive_frac
        self.augment = augment and split == "train"
        self.noise_m = noise_m
        self.noise_band_m = noise_band_m
        self.canopy_dropout = canopy_dropout
        self.jitter_m = jitter_m
        self._open: dict[str, tuple] = {}
        self._rng: np.random.Generator | None = None

        self.native_res = self._native_res(dirs)
        # Every variant reads the same native window, so crops across variants
        # cover identical ground and can be compared pairwise.
        self.label_scale = int(round(var_mod.BODY_RES / self.native_res))
        self.z_scale = int(round(variant.res / self.native_res))
        self.crop = body_crop * self.label_scale

        # A variant at body resolution reads USGS's published DEM directly; one
        # below it derives from our own point cloud. Different rasters, so this
        # is a source choice rather than a resampling detail.
        self.published = abs(variant.res - var_mod.BODY_RES) < 1e-9

        self.tiles = []
        missing_dem = []
        for d in dirs:
            feats, lbls = d / "features.tif", d / "labels.tif"
            dtm = d / "dtm_clean.tif"
            if not (feats.exists() and lbls.exists() and dtm.exists()):
                log.warning("%s not built, skipping", d.name)
                continue
            dem = d / DEM_1M
            if self.published and not dem.exists():
                # Falling back to the block-mean proxy would quietly train this
                # tile on a distribution the deployed model never meets. Two
                # tiles out of seventy-four is a cheap price for not doing that;
                # `trailer dem` recovers them.
                missing_dem.append(d.name)
                continue
            info = self._index(d, feats, lbls, dtm)
            if info is not None:
                info["dem"] = str(dem) if self.published else None
                self.tiles.append(info)
        if missing_dem:
            log.warning("%s: skipped %d tiles with no %s (%s); run `trailer dem`",
                        variant.key, len(missing_dem), DEM_1M,
                        ", ".join(missing_dem[:4]))
        if not self.tiles:
            raise ValueError(f"no usable tiles for split={split!r} in {dirs}")

        # Which tiles can supply each class, and the mix actually achievable.
        self._tiles_with: dict[str, list[int]] = {}
        for i, t in enumerate(self.tiles):
            for name in t["by_class"]:
                self._tiles_with.setdefault(name, []).append(i)
        avail = {k: v for k, v in CLASS_MIX.items() if self._tiles_with.get(k)}
        total = sum(avail.values())
        # Renormalise over what exists. The validation band of a tile may hold
        # no lifecycle way at all, and silently drawing zero of a class is worse
        # than drawing more of the others.
        self._mix = {k: v / total for k, v in avail.items()} if total else {}
        self._mix_names = list(self._mix)
        self._mix_p = np.array([self._mix[k] for k in self._mix_names])

        # Validation draws a fixed crop plan once, so every epoch is scored on
        # exactly the same pixels. Re-rolling them would add sampling noise to
        # the number that decides which checkpoint is kept. The seed is shared
        # across variants, so they are scored on the same ground too.
        self._plan = None
        if split != "train":
            rng = np.random.default_rng(seed)
            self._plan = [self._draw(rng) for _ in range(samples)]

        n_pos = sum(t["n_centres"] for t in self.tiles)
        counts = {k: sum(len(self.tiles[i]["by_class"][k])
                         for i in self._tiles_with.get(k, []))
                  for k in osm.CLASS_CODE}
        log.info("%s/%s split: %d tiles, %d centres %s, %d px crops -> %d px body",
                 variant.key, split, len(self.tiles), n_pos,
                 {k: v for k, v in counts.items() if v}, self.crop // self.z_scale,
                 body_crop)
        missing = [k for k in CLASS_MIX if not self._tiles_with.get(k)]
        if missing:
            log.warning("%s split has no %s centres; mix renormalised to %s",
                        split, "/".join(missing),
                        {k: round(v, 2) for k, v in self._mix.items()})
        if n_pos == 0:
            log.warning("%s split has no trail pixels -- recall from it is "
                        "meaningless; widen the tiles or check the labels", split)

    @staticmethod
    def _native_res(dirs: list[Path]) -> float:
        for d in dirs:
            m = d / "manifest.json"
            if m.exists():
                rec = json.loads(m.read_text())
                if "res" in rec:
                    return float(rec["res"])
        raise ValueError("no manifest with a res field among the given tiles")

    def _index(self, d: Path, feats: Path, lbls: Path, dtm: Path) -> dict | None:
        """Record the crop-legal column range and the trail pixels inside it."""
        with rasterio.open(lbls) as s:
            h, w = s.height, s.width
            target = s.read(1)

        boundary = _boundary(target, self.crop)
        col0, col1 = (0, boundary) if self.split == "train" else (boundary, w)
        # Crop origins must leave a full crop inside the split band.
        if col1 - col0 < self.crop or h < self.crop:
            log.warning("%s too small for %dpx crops in %s split (%dx%d)",
                        d.name, self.crop, self.split, col1 - col0, h)
            return None

        with rasterio.open(lbls) as s:
            klass = s.read(CLASS_BAND) if s.count >= CLASS_BAND else None
        if klass is None:
            log.warning("%s has no class band; run `trailer relabel`", d.name)
            klass = (target > 0).astype("float32")  # everything reads as active

        half = self.crop // 2
        rows, cols = np.nonzero(target[:, col0:col1] > 0)
        cols = cols + col0
        # Only centres whose crop fits entirely within the split band.
        keep = ((rows >= half) & (rows < h - half) &
                (cols >= col0 + half) & (cols < col1 - half))
        rows, cols = rows[keep], cols[keep]

        # Split the centre pool by class and cap each independently, so a class
        # holding a few hundred pixels is not strided away alongside one holding
        # forty thousand.
        codes = klass[rows, cols]
        by_class: dict[str, np.ndarray] = {}
        for name, code in osm.CLASS_CODE.items():
            m = codes == code
            r, c = rows[m], cols[m]
            if len(r) > MAX_CENTRES:
                step = len(r) // MAX_CENTRES + 1
                r, c = r[::step], c[::step]
            if len(r):
                by_class[name] = np.stack([r, c], axis=1)

        return {"name": d.name, "feats": str(feats), "labels": str(lbls),
                "dtm": str(dtm), "h": h, "w": w, "col0": col0, "col1": col1,
                "by_class": by_class,
                "n_centres": sum(len(v) for v in by_class.values())}

    def __getstate__(self) -> dict:
        """Drop unpicklable per-process state before DataLoader ships us out.

        Rasterio dataset handles wrap a C pointer and cannot be pickled. Anything
        that touches the dataset in the parent -- estimating the label prior, for
        instance -- populates the handle cache, and the first worker launch then
        dies trying to serialise it. Workers reopen lazily anyway.
        """
        return self.__dict__ | {"_open": {}, "_rng": None}

    def __len__(self) -> int:
        return self.samples

    def _handles(self, tile: dict):
        """Open datasets lazily and per worker; rasterio handles do not fork."""
        key = tile["dem"] or tile["dtm"]
        if key not in self._open:
            self._open[key] = (rasterio.open(key),
                               rasterio.open(tile["labels"]),
                               rasterio.open(tile["feats"]))
        return self._open[key]

    def _generator(self) -> np.random.Generator:
        if self._rng is None:
            # torch seeds each worker distinctly; inherit that so workers do not
            # draw identical crops.
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._rng

    def _draw(self, rng: np.random.Generator) -> tuple[int, int, int]:
        """Pick (tile index, row, col) for one crop."""
        half = self.crop // 2

        # Class first, then a tile holding it, then a centre. Picking the tile
        # first would waste every draw that landed on a tile without the class,
        # and would re-impose the length imbalance the mix exists to remove.
        centre = None
        if self._mix_names and rng.random() < self.positive_frac:
            name = self._mix_names[int(rng.choice(len(self._mix_names),
                                                  p=self._mix_p))]
            pool = self._tiles_with[name]
            ti = int(pool[int(rng.integers(len(pool)))])
            centres = self.tiles[ti]["by_class"][name]
            centre = centres[int(rng.integers(len(centres)))]
        else:
            ti = int(rng.integers(len(self.tiles)))
        tile = self.tiles[ti]

        if centre is not None:
            r, c = centre
            # Jitter so the trail is not always dead centre, which would let the
            # model cheat on position rather than learn appearance.
            r += rng.integers(-half // 2, half // 2 + 1)
            c += rng.integers(-half // 2, half // 2 + 1)
            row = int(np.clip(r - half, 0, tile["h"] - self.crop))
            col = int(np.clip(c - half, tile["col0"], tile["col1"] - self.crop))
        else:
            row = int(rng.integers(0, tile["h"] - self.crop + 1))
            col = int(rng.integers(tile["col0"], tile["col1"] - self.crop + 1))
        # Snap so a crop covers whole body pixels, keeping labels aligned.
        k = self.label_scale
        return ti, (row // k) * k, (col // k) * k

    def __getitem__(self, idx: int):
        rng = self._generator()
        if self._plan is not None:
            ti, row, col = self._plan[idx % len(self._plan)]
        else:
            ti, row, col = self._draw(rng)
        tile = self.tiles[ti]

        lk = self.label_scale
        win = Window(col, row, self.crop, self.crop)
        dsrc, lsrc, fsrc = self._handles(tile)
        if tile["dem"]:
            # Already at body resolution, so its window is in body pixels.
            z = dsrc.read(1, window=Window(col // lk, row // lk,
                                           self.body_crop, self.body_crop))
            z = z.astype("float32")
            valid = np.isfinite(z) & (z != 0.0)
        else:
            z = dsrc.read(1, window=win).astype("float32")
            # build.py writes 0 for nodata; the model's contract is NaN, and its
            # centring step then excludes those pixels from the tile mean.
            valid = z != 0.0
        lab = lsrc.read(window=win).astype("float32")
        canopy = (fsrc.read(CANOPY_BANDS, window=win).astype("float32")
                  if self.variant.canopy else
                  np.zeros((0, self.body_crop, self.body_crop), dtype="float32"))
        y, w = lab[0:1], lab[1:2]
        # Tiles labelled before the class band existed read as all-active rather
        # than as all-background, so a stale tile degrades the stratification
        # instead of emptying it. _index already warns about them.
        cls = (lab[CLASS_BAND - 1:CLASS_BAND] if lab.shape[0] >= CLASS_BAND
               else (y > 0).astype("float32"))

        if self.augment:
            k = int(rng.integers(4))
            if k:
                z, valid, canopy, y, w, cls = (
                    np.rot90(a, k, axes=(-2, -1))
                    for a in (z, valid, canopy, y, w, cls))
            if rng.random() < 0.5:
                z, valid, canopy, y, w, cls = (
                    a[..., ::-1] for a in (z, valid, canopy, y, w, cls))
            # Sweep the signal-to-noise ratio, not just add a fixed jitter.
            # Terrain noise runs 65 mm in sandy meadow to 500 mm in alpine talus
            # while tread stays 15-100 mm, so detectability is the ratio -- and
            # it is the axis the faint trails fail on. A fixed sigma trains one
            # point on that range; a sampled one covers it.
            #
            # Two components, because one does not reach both features. White
            # noise dominates the tread band and barely touches the 4-9 m
            # bench-and-berm cross-section the model actually keys on;
            # band-limited noise does the reverse. Mixing sweeps both.
            if self.noise_m:
                sigma = rng.uniform(0.0, self.noise_m)
                z = z + rng.normal(0, sigma, z.shape).astype("float32")
            if self.noise_band_m:
                sigma = rng.uniform(0.0, self.noise_band_m)
                lo, hi = BAND_NOISE_SCALE_M
                scale_px = rng.uniform(lo, hi) / self.native_res
                z = z + band_noise(z.shape, sigma, scale_px, rng)
            if self.variant.canopy and self.canopy_dropout \
                    and rng.random() < self.canopy_dropout:
                # Canopy structure varies with forest type and region, and the
                # tread signature does not depend on it. Withholding chm/vdi
                # some of the time stops the model leaning on them -- and the
                # bare-earth variant has to work with none at all.
                canopy = np.zeros_like(canopy)

        # Reduce to each consumer's resolution: elevation to the variant's,
        # labels and weight to the body's. A published DEM is already at body
        # resolution, so its elevation and validity need no reduction.
        zk = 1 if tile["dem"] else self.z_scale
        vk = 1 if tile["dem"] else lk
        z = np.where(valid, z, np.nan)
        z = block_nanmean(z, zk)[None] if zk > 1 else z[None]
        ck = self.z_scale
        canopy = block_mean(canopy, ck) if ck > 1 and canopy.size else canopy

        y = block_max(y, lk)
        w = block_mean(w, lk)
        cls = block_min_nonzero(cls, lk)
        # A body pixel is trainable only if it was fully covered by real ground.
        w = w * (block_mean(valid.astype("float32"), vk) > 0.999)

        if self.augment and self.jitter_m:
            # Shift the labels bodily against the terrain. Most Sierra trail
            # geometry in OSM was traced from satellite imagery, so its error is
            # a rigid per-way offset -- a way was digitised in one sitting off
            # one image -- rather than per-vertex noise. Jittering the whole
            # crop's labels models that mechanism; jittering vertices would not.
            # Exposed pixels become weight 0: after a shift their true label is
            # genuinely unknown, and inventing one would teach the model an edge.
            r = self.jitter_m / var_mod.BODY_RES
            dr = int(round(rng.uniform(-r, r)))
            dc = int(round(rng.uniform(-r, r)))
            y, w, cls = (_shift(a, dr, dc) for a in (y, w, cls))

        return tuple(torch.from_numpy(np.ascontiguousarray(a))
                     for a in (z, canopy, y, w, cls))


def full_tile(d: Path, variant: var_mod.Variant,
              native_res: float | None = None):
    """Read a whole tile for sliding-window evaluation, at a variant's scale."""
    if native_res is None:
        native_res = float(json.loads((d / "manifest.json").read_text())["res"])
    zk = int(round(variant.res / native_res))
    lk = int(round(var_mod.BODY_RES / native_res))

    published = (abs(variant.res - var_mod.BODY_RES) < 1e-9
                 and (d / DEM_1M).exists())
    if published:
        with rasterio.open(d / DEM_1M) as s:
            z = s.read(1).astype("float32")
        valid = np.isfinite(z) & (z != 0.0)
        z = np.where(valid, z, np.nan)[None]
        vk, zk = 1, 1
    else:
        with rasterio.open(d / "dtm_clean.tif") as s:
            z = s.read(1).astype("float32")
        valid = z != 0.0
        z = np.where(valid, z, np.nan)
        z = block_nanmean(z, zk)[None] if zk > 1 else z[None]
        vk = lk

    ck = int(round(variant.res / native_res))
    if variant.canopy:
        with rasterio.open(d / "features.tif") as s:
            canopy = s.read(CANOPY_BANDS).astype("float32")
        canopy = block_mean(canopy, ck) if ck > 1 else canopy
    else:
        canopy = np.zeros((0,) + z.shape[-2:], dtype="float32")

    with rasterio.open(d / "labels.tif") as s:
        lab = s.read().astype("float32")
    y = block_max(lab[0:1], lk)
    w = block_mean(lab[1:2], lk)
    cls = block_min_nonzero(
        lab[CLASS_BAND - 1:CLASS_BAND] if lab.shape[0] >= CLASS_BAND
        else (lab[0:1] > 0).astype("float32"), lk)
    w = w * (block_mean(valid.astype("float32"), vk) > 0.999)

    # Trim to a common footprint, counted in BODY pixels. Elevation is at the
    # variant's pixel size and the labels are at the body's, so a plain
    # min(z.shape, y.shape) is not comparing like with like: at 0.5 m it cut z to
    # the label count and handed inference the top-left QUARTER of the tile while
    # scoring it against labels for all of it. Only a quarter of every held-out
    # tile was evaluated for sub-metre variants, and the 0.5-vs-1 m comparison --
    # the number that says what our own point clouds are worth over the public
    # DEM -- was measured over different ground for each.
    k = int(round(var_mod.BODY_RES / variant.res))
    n0 = min(z.shape[-2] // k, y.shape[-2])
    n1 = min(z.shape[-1] // k, y.shape[-1])
    return (z[..., :n0 * k, :n1 * k], canopy[..., :n0 * k, :n1 * k],
            y[..., :n0, :n1], w[..., :n0, :n1], cls[..., :n0, :n1])
