"""Crop sampling over built tiles.

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
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

#: Target share of each tile's *labelled trail pixels* held back for validation.
VAL_COL_FRAC = 0.2

#: The boundary is clamped into this fraction-of-width range, so a tile whose
#: trails all sit at one edge cannot hand most of its area to either split.
BOUNDARY_RANGE = (0.55, 0.85)

#: Cap on remembered trail-pixel locations per tile. A 1 km tile at 0.5 m holds
#: ~40k positive pixels; keeping every one buys nothing over a strided sample.
MAX_CENTRES = 20_000


def _boundary(target: np.ndarray, crop: int) -> int:
    """Column separating the train band from the validation band.

    Placed so roughly VAL_COL_FRAC of the tile's trail pixels fall to the right
    of it, then clamped so neither split can be starved of area.

    The clamp also has to leave both bands at least one crop wide. Without that,
    a tile whose trails sit far to the right pushes the boundary out until the
    validation band is narrower than a crop and the tile silently drops out of
    validation -- which is exactly what happened to the two weakest-signal tiles
    in the set, quietly biasing validation optimistic.
    """
    w = target.shape[1]
    lo, hi = (int(w * f) for f in BOUNDARY_RANGE)
    lo, hi = max(lo, crop), min(hi, w - crop)
    if lo > hi:  # tile too narrow to split at all
        lo = hi = w // 2
    per_col = (target > 0).sum(axis=0)
    total = per_col.sum()
    if total == 0:
        return int(w * (1 - VAL_COL_FRAC))
    cut = int(np.searchsorted(np.cumsum(per_col), (1 - VAL_COL_FRAC) * total))
    return int(np.clip(cut, lo, hi))


class TileDataset(Dataset):
    """Random crops from built AOI directories.

    Each item is ``(x, y, w)`` where ``x`` is the 6-band feature stack, ``y`` the
    binary target, and ``w`` the per-pixel loss weight. Ignored pixels -- the
    2-5 m alignment ring, boardwalks, water, and raster nodata -- arrive as
    ``w == 0`` rather than as a separate mask, so every loss gets the masking
    for free by multiplying.
    """

    def __init__(self, dirs: list[Path], crop: int = 384, split: str = "train",
                 samples: int = 2000, positive_frac: float = 0.5,
                 augment: bool = True, noise: float = 0.02, seed: int = 1234):
        self.crop = crop
        self.split = split
        self.samples = samples
        self.positive_frac = positive_frac
        self.augment = augment and split == "train"
        self.noise = noise
        self._open: dict[str, tuple] = {}
        self._rng: np.random.Generator | None = None

        self.tiles = []
        for d in dirs:
            feats, lbls = d / "features.tif", d / "labels.tif"
            if not (feats.exists() and lbls.exists()):
                log.warning("%s not built, skipping", d.name)
                continue
            info = self._index(d, feats, lbls)
            if info is not None:
                self.tiles.append(info)
        if not self.tiles:
            raise ValueError(f"no usable tiles for split={split!r} in {dirs}")

        # Validation draws a fixed crop plan once, so every epoch is scored on
        # exactly the same pixels. Re-rolling them would add sampling noise to
        # the number that decides which checkpoint is kept.
        self._plan = None
        if split != "train":
            rng = np.random.default_rng(seed)
            self._plan = [self._draw(rng) for _ in range(samples)]

        n_pos = sum(len(t["centres"]) for t in self.tiles)
        log.info("%s split: %d tiles, %d candidate trail centres",
                 split, len(self.tiles), n_pos)
        if n_pos == 0:
            log.warning("%s split has no trail pixels -- recall from it is "
                        "meaningless; widen the tiles or check the labels", split)

    def _index(self, d: Path, feats: Path, lbls: Path) -> dict | None:
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

        half = self.crop // 2
        rows, cols = np.nonzero(target[:, col0:col1] > 0)
        cols = cols + col0
        # Only centres whose crop fits entirely within the split band.
        keep = ((rows >= half) & (rows < h - half) &
                (cols >= col0 + half) & (cols < col1 - half))
        rows, cols = rows[keep], cols[keep]
        if len(rows) > MAX_CENTRES:
            step = len(rows) // MAX_CENTRES + 1
            rows, cols = rows[::step], cols[::step]

        return {"name": d.name, "feats": str(feats), "labels": str(lbls),
                "h": h, "w": w, "col0": col0, "col1": col1,
                "centres": np.stack([rows, cols], axis=1) if len(rows)
                           else np.zeros((0, 2), dtype=int)}

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
        key = tile["feats"]
        if key not in self._open:
            self._open[key] = (rasterio.open(tile["feats"]),
                               rasterio.open(tile["labels"]))
        return self._open[key]

    def _generator(self) -> np.random.Generator:
        if self._rng is None:
            # torch seeds each worker distinctly; inherit that so workers do not
            # draw identical crops.
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._rng

    def _draw(self, rng: np.random.Generator) -> tuple[int, int, int]:
        """Pick (tile index, row, col) for one crop."""
        ti = int(rng.integers(len(self.tiles)))
        tile = self.tiles[ti]
        half = self.crop // 2

        centres = tile["centres"]
        if len(centres) and rng.random() < self.positive_frac:
            r, c = centres[int(rng.integers(len(centres)))]
            # Jitter so the trail is not always dead centre, which would let the
            # model cheat on position rather than learn appearance.
            r += rng.integers(-half // 2, half // 2 + 1)
            c += rng.integers(-half // 2, half // 2 + 1)
            row = int(np.clip(r - half, 0, tile["h"] - self.crop))
            col = int(np.clip(c - half, tile["col0"], tile["col1"] - self.crop))
        else:
            row = int(rng.integers(0, tile["h"] - self.crop + 1))
            col = int(rng.integers(tile["col0"], tile["col1"] - self.crop + 1))
        return ti, row, col

    def __getitem__(self, idx: int):
        rng = self._generator()
        if self._plan is not None:
            ti, row, col = self._plan[idx % len(self._plan)]
        else:
            ti, row, col = self._draw(rng)
        tile = self.tiles[ti]

        win = Window(col, row, self.crop, self.crop)
        fsrc, lsrc = self._handles(tile)
        x = fsrc.read(window=win).astype("float32")
        lab = lsrc.read(window=win).astype("float32")
        y, w = lab[0:1], lab[1:2]

        # Raster nodata: build.py zeroes every band outside the valid DTM hull.
        # Those pixels are neither trail nor terrain, so they must not train.
        w = w * (np.abs(x).sum(axis=0, keepdims=True) > 0)

        if self.augment:
            k = int(rng.integers(4))
            if k:
                x, y, w = (np.rot90(a, k, axes=(-2, -1)) for a in (x, y, w))
            if rng.random() < 0.5:
                x, y, w = (a[..., ::-1] for a in (x, y, w))
            if self.noise:
                # Terrain roughness varies 65-500 mm across the survey; a little
                # band noise stops the model keying on one substrate's texture.
                x = x + rng.normal(0, self.noise, x.shape).astype("float32")

        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(w)))


def full_tile(d: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a whole tile for sliding-window evaluation."""
    with rasterio.open(d / "features.tif") as s:
        x = s.read().astype("float32")
    with rasterio.open(d / "labels.tif") as s:
        lab = s.read().astype("float32")
    w = lab[1:2] * (np.abs(x).sum(axis=0, keepdims=True) > 0)
    return x, lab[0:1], w
