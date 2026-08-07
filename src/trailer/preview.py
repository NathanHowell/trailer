"""Hillshade + label overlay renders for eyeballing a built tile."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402


def render(aoi_dir: Path, out_png: Path) -> Path:
    with rasterio.open(aoi_dir / "dtm_clean.tif") as s:
        dtm = s.read(1)
        res = s.res[0]
    with rasterio.open(aoi_dir / "features.tif") as s:
        mrm2 = s.read(1)
    with rasterio.open(aoi_dir / "labels.tif") as s:
        target, _weight, ignore = s.read()

    valid = dtm != 0
    filled = np.where(valid, dtm, np.nanmedian(dtm[valid]) if valid.any() else 0)
    ls_nw = LightSource(azdeg=315, altdeg=38)
    ls_ne = LightSource(azdeg=45, altdeg=38)
    hs = (ls_nw.hillshade(filled, vert_exag=2.4, dx=res, dy=res)
          + ls_ne.hillshade(filled, vert_exag=2.4, dx=res, dy=res)) / 2

    fig, axs = plt.subplots(1, 3, figsize=(19, 6.4))
    axs[0].imshow(hs, cmap="gray")
    axs[0].set_title(f"{aoi_dir.name} — multi-directional hillshade ({res} m)")
    axs[1].imshow(mrm2, cmap="Greys_r", vmin=-0.6, vmax=0.6)
    axs[1].set_title("micro-relief, 2 m scale")

    ax = axs[2]
    ax.imshow(hs, cmap="gray")
    over = np.zeros(hs.shape + (4,))
    over[..., 2] = 1.0
    over[..., 3] = (target > 0) * 0.75
    ax.imshow(over)
    over = np.zeros(hs.shape + (4,))
    over[..., 0] = 1.0
    over[..., 1] = 0.75
    over[..., 3] = (ignore > 0) * 0.30
    ax.imshow(over)
    ax.set_title("labels: positive (blue), ignore (yellow)")

    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_png
