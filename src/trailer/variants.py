"""Input variants: one model, several kinds of source raster.

A variant is a pixel size plus whether canopy bands exist. Those two axes are
not independent in practice -- the 1 m source we can actually reach at runtime
is USGS 3DEP, which is bare earth, so going to 1 m and losing canopy are the
same event.

Each variant gets its own small stem; the trunk and head are shared and always
run at ``BODY_RES``. That way the trunk's kernels mean one fixed physical size
instead of having to be scale-invariant, and the stems absorb the difference. A
0.5 m stem is stride-2, so it can learn a matched filter over the tread before
decimating rather than throwing sub-metre detail away to a mean.

The consequence, stated plainly: output is at ``BODY_RES`` for every variant.
Sub-metre input buys signal fidelity, not output resolution. Against a 5 m label
tolerance that is not the binding constraint.

Measured cost of the coarser grid, over ten tiles: median tread incision
retained 1.10-1.12 (active, faint) and 0.87 (lifecycle). It survives because the
feature is the bench-and-berm cross-section -- 4-9 m wide, so still 4-9 px at
1 m -- and the millimetre figure is its depth, which block-averaging preserves.
The exception is junction_pass faint, 26.2 -> 12.9 mm: the weakest signal in the
set is also the one that most needs the 0.5 m stem.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Pixel size the shared trunk and head operate at, in metres.
BODY_RES = 1.0


@dataclass(frozen=True)
class Variant:
    key: str
    res: float
    canopy: bool
    note: str

    @property
    def stride(self) -> int:
        """Stem stride that brings this variant to ``BODY_RES``."""
        s = BODY_RES / self.res
        if abs(s - round(s)) > 1e-6 or round(s) < 1:
            raise ValueError(f"{self.key}: {self.res} m does not divide "
                             f"{BODY_RES} m into an integer stride")
        return int(round(s))

    @property
    def fine_channels(self) -> int:
        """Bands entering the stride convolution: mrm_2m, slope, roughness."""
        return 3 + (2 if self.canopy else 0)

    def crop_px(self, body_crop: int) -> int:
        """Pixels this variant must read to cover ``body_crop`` trunk pixels."""
        return body_crop * self.stride


VARIANTS: dict[str, Variant] = {v.key: v for v in (
    Variant("lidar05", 0.5, True,
            "our own 0.5 m stacks from 3DEP points, with canopy"),
    Variant("dem1", 1.0, False,
            "USGS 3DEP 1 m bare-earth DEM -- the JOSM runtime case"),
    Variant("lidar1", 1.0, True,
            "1 m with canopy; isolates resolution loss from band loss"),
)}

DEFAULT_VARIANTS = ("lidar05", "dem1")


def get(key: str) -> Variant:
    if key not in VARIANTS:
        raise ValueError(f"unknown variant {key!r}; known: {', '.join(VARIANTS)}")
    return VARIANTS[key]


def parse(spec: str | None) -> list[Variant]:
    keys = [k.strip() for k in (spec or ",".join(DEFAULT_VARIANTS)).split(",")]
    return [get(k) for k in keys if k]
