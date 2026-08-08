"""Sliding-window inference over whole tiles.

Tiled inference on curvilinear features leaves visible seams: a trail crossing a
window boundary gets half its context on each side and the model hedges. Two
mitigations, both of which the JOSM plugin will need to reproduce:

* 50% overlap with a 2-D Hann taper, so every pixel's final value is dominated
  by the window that had it nearest the centre;
* optional D4 test-time augmentation. Terrain has no canonical orientation, so
  averaging the eight dihedral transforms is nearly free accuracy -- at 8x the
  compute, which is why it is off by default.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hann2d(size: int, device, dtype=torch.float32) -> torch.Tensor:
    w = torch.hann_window(size + 2, periodic=False, device=device, dtype=dtype)[1:-1]
    return (w[:, None] * w[None, :]).clamp_min(1e-3)


def _d4(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, k, dims=(-2, -1))


def _d4_inv(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, -k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


@torch.no_grad()
def predict(model, x: np.ndarray, tile: int = 384, overlap: float = 0.5,
            device=None, batch: int = 8, tta: bool = False) -> np.ndarray:
    """Run a model over a (C, H, W) feature stack. Returns (1, H, W) probs."""
    device = device or next(model.parameters()).device
    model.eval()

    c, h, w = x.shape
    # Reflect-pad so windows tile exactly and edges keep real context.
    pad_h = max(tile - h, 0) + (-(h - tile) % max(int(tile * (1 - overlap)), 1)
                                if h > tile else 0)
    pad_w = max(tile - w, 0) + (-(w - tile) % max(int(tile * (1 - overlap)), 1)
                                if w > tile else 0)
    t = torch.from_numpy(x).unsqueeze(0)
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    t = t.to(device)
    _, _, H, W = t.shape

    step = max(int(tile * (1 - overlap)), 1)
    taper = hann2d(tile, device)
    acc = torch.zeros((1, 1, H, W), device=device)
    den = torch.zeros((1, 1, H, W), device=device)

    origins = [(r, c)
               for r in range(0, H - tile + 1, step)
               for c in range(0, W - tile + 1, step)]

    variants = [(k, f) for f in (False, True) for k in range(4)] if tta else [(0, False)]

    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        crops = torch.cat([t[:, :, r:r + tile, c:c + tile] for r, c in chunk])
        prob = torch.zeros((len(chunk), 1, tile, tile), device=device)
        for k, flip in variants:
            out = model(_d4(crops, k, flip))
            prob += torch.sigmoid(_d4_inv(out, k, flip))
        prob /= len(variants)
        for j, (r, c) in enumerate(chunk):
            acc[0, :, r:r + tile, c:c + tile] += prob[j] * taper
            den[0, :, r:r + tile, c:c + tile] += taper

    out = (acc / den.clamp_min(1e-6))[0, :, :h, :w]
    return out.cpu().numpy()
