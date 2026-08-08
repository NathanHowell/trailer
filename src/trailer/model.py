"""Network definition.

U-Net with an ImageNet-pretrained ResNet-34 encoder. The pretraining is worth
having even though bare-earth micro-relief looks nothing like photographs: the
early layers are edge and ridge detectors, and a 14 km2 training set is far too
small to learn those from scratch.

The encoder's first convolution is adapted from 3 to 6 input channels by
``segmentation_models_pytorch``, which tiles and rescales the pretrained kernel
so the learned filters survive the change.

Receptive field is the parameter that matters most. Per-pixel terrain indices
score AUC 0.51-0.56 on this data -- essentially chance -- while TrailScan with
56 m of context reaches 0.649 zero-shot. A ResNet-34 U-Net at 0.5 m sees a few
hundred metres, which is the point of using it.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

import torch.nn.functional as F

from . import variants as var_mod
from .preprocess import Background, FineDerivatives, centre

log = logging.getLogger(__name__)

DEFAULT_ENCODER = "resnet34"

#: Licence of the *weights*, which is not the licence of this code. See
#: LICENSE-MODEL for the reasoning: the code is MIT, but the weights are trained
#: on ODbL trail geometry and are released share-alike whether or not ODbL
#: strictly compels it.
MODEL_LICENSE = "CC-BY-SA-4.0"

#: Attribution the licence requires anyone showing the weights' output to
#: display. It travels in the export sidecar rather than living only in a
#: repository the mapper never sees, and ``ModelSpec`` refuses to load a model
#: without it -- showing this is a condition of the licence, not a nicety, so a
#: model file that has lost it is one the plugin has no right to paint.
MODEL_ATTRIBUTION = (
    "Trail probability model (c) 2026 Nathan Howell, CC BY-SA 4.0. "
    "Trained on trail geometry from OpenStreetMap (c) OpenStreetMap "
    "contributors, available under the Open Database Licence (ODbL). "
    "Elevation from the USGS 3D Elevation Program (3DEP), a work of the "
    "United States government and in the public domain."
)

#: Channels a stem hands the shared trunk. Wide enough that a stride-2 stem can
#: encode what it saw at 0.5 m before decimating, which is the whole reason the
#: sub-metre path exists.
STEM_CHANNELS = 32


class Stem(nn.Module):
    """One input variant's adapter: raw elevation in, trunk features out.

    Takes bare-earth elevation in metres (``NaN`` = nodata) at the variant's own
    pixel size, derives the terrain bands there, and brings them to
    ``BODY_RES``. For a 0.5 m variant the stride-2 convolution is doing the real
    work: it can learn a matched filter across a 2-3 px tread and encode the
    result, rather than mean-pooling the detail away before the trunk ever sees
    it.

    The 10 m band is computed after decimation, at body resolution, so it means
    exactly the same thing whichever stem produced it.
    """

    def __init__(self, variant: var_mod.Variant, out_ch: int = STEM_CHANNELS,
                 body_res: float = var_mod.BODY_RES):
        super().__init__()
        self.variant = variant
        self.stride = variant.stride
        self.fine = FineDerivatives(variant.res)
        self.background = Background(body_res)

        self.project = nn.Sequential(
            nn.Conv2d(variant.fine_channels, out_ch, 7,
                      stride=self.stride, padding=3, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + 1, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, z, canopy=None):
        z, mask = centre(z)
        bands = self.fine(z) * mask.to(z.dtype)
        if self.variant.canopy:
            if canopy is None:
                raise ValueError(f"variant {self.variant.key!r} needs canopy "
                                 f"bands (chm, vdi)")
            bands = torch.cat([bands, canopy], dim=1)
        elif canopy is not None:
            raise ValueError(f"variant {self.variant.key!r} takes no canopy "
                             f"bands; it models a bare-earth DEM source")

        fine = self.project(bands)
        if self.stride > 1:
            z = F.avg_pool2d(z, self.stride, self.stride)
        coarse = self.background(z)
        # Guard against an odd crop leaving the two paths a pixel apart.
        if coarse.shape[-2:] != fine.shape[-2:]:
            coarse = F.interpolate(coarse, size=fine.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([fine, coarse], dim=1))


class MultiStemNet(nn.Module):
    """Per-variant stems feeding one shared trunk and head.

    Binding several input resolutions to a single trunk is what makes one
    checkpoint useful both in JOSM, against 1 m 3DEP, and against the 0.5 m
    stacks we render ourselves. The alternative -- one model per source -- would
    split an already small training set.
    """

    def __init__(self, body: nn.Module, stems: dict[str, Stem]):
        super().__init__()
        self.body = body
        self.stems = nn.ModuleDict(stems)

    @property
    def segmentation_head(self):
        # So set_output_prior finds the trunk's output conv rather than a stem's.
        return self.body.segmentation_head

    def forward(self, z, canopy=None, variant: str | None = None):
        if variant is None:
            if len(self.stems) != 1:
                raise ValueError("variant is required when the model has more "
                                 f"than one stem ({', '.join(self.stems)})")
            variant = next(iter(self.stems))
        return self.body(self.stems[variant](z, canopy))


def build_model(arch: str = "unet", encoder: str = DEFAULT_ENCODER,
                variants: list[var_mod.Variant] | None = None,
                pretrained: bool = True) -> MultiStemNet:
    import segmentation_models_pytorch as smp

    factory = {
        "unet": smp.Unet,
        "unetpp": smp.UnetPlusPlus,
        # Dilated bottleneck; wider context per parameter, worth comparing once
        # there is a baseline to compare against.
        "deeplabv3p": smp.DeepLabV3Plus,
    }
    if arch not in factory:
        raise ValueError(f"unknown arch {arch!r}; known: {', '.join(factory)}")

    variants = variants or [var_mod.get(k) for k in var_mod.DEFAULT_VARIANTS]
    body = factory[arch](
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=STEM_CHANNELS,
        classes=1,
    )
    model = MultiStemNet(body, {v.key: Stem(v) for v in variants})
    n = sum(p.numel() for p in model.parameters())
    stem_n = sum(p.numel() for s in model.stems.values() for p in s.parameters())
    log.info("%s/%s: %.1fM params (%.0fk in %d stems: %s)", arch, encoder,
             n / 1e6, stem_n / 1e3, len(variants),
             ", ".join(f"{v.key}@{v.res}m" for v in variants))
    return model


def set_output_prior(model: nn.Module, prior: float,
                     pos_weight: float = 1.0) -> float:
    """Bias the output layer towards the base rate (Lin et al. 2017, sec. 3.3).

    A zero bias makes the untrained model predict p=0.5 everywhere -- "half of
    this mountain is trail" -- and the opening steps are spent driving that down
    against gradients dominated by background. Starting at the base rate skips
    that entirely.

    The target is the *weighted* prior, since ``pos_weight`` shifts where the
    BCE optimum sits. Minimising ``-[w*pi*log p + (1-pi)*log(1-p)]`` over a
    constant p gives ``p* = w*pi / (w*pi + 1 - pi)``.

    Returns the bias applied, for logging.
    """
    prior = min(max(prior, 1e-6), 1 - 1e-6)
    p = pos_weight * prior / (pos_weight * prior + 1 - prior)
    bias = math.log(p / (1 - p))

    head = getattr(model, "segmentation_head", None)
    convs = [m for m in (head.modules() if head is not None else model.modules())
             if isinstance(m, nn.Conv2d) and m.bias is not None]
    if not convs:
        raise ValueError("no biased output conv found; cannot set prior")
    nn.init.constant_(convs[-1].bias, bias)

    log.info("output prior: %.3f%% positive, pos_weight %.1f -> p0 %.3f "
             "(bias %.2f)", 100 * prior, pos_weight, p, bias)
    return bias


class Deployable(nn.Module):
    """One variant, frozen, elevation in and probability out.

    This is the whole point of deriving terrain in the graph: what crosses the
    language boundary into the JOSM plugin is a float32 DEM tile and a
    probability map, with the median filters, variance windows and clip
    constants sealed inside the file. Nothing to reimplement in Java, nothing to
    drift out of sync with the weights.

    Two more things are sealed in here for the same reason.

    **The D4 average.** Terrain has no canonical orientation, so averaging the
    eight dihedral transforms is nearly free accuracy at eight times the compute.
    Done in the graph, it is never written in Kotlin at all -- and the piece most
    likely to be written wrong is the *inverse* transform, whose failure mode is
    a subtly blurred output that looks like an unlucky model rather than a bug.

    **The window taper.** Emitted as a second output rather than recomputed by
    the caller. Its definition has a detail that exists precisely because it is
    not obvious -- ``hann_window(size + 2)`` with the ends trimmed, because a
    plain Hann is exactly zero at both ends and the outermost row and column of
    every tile would contribute nothing. A reimplementation that misses it is
    wrong only along tile edges, which is where seams live and where nobody
    looks.
    """

    def __init__(self, net: MultiStemNet, variant: str, tta: bool = False):
        super().__init__()
        if variant not in net.stems:
            raise ValueError(f"checkpoint has no {variant!r} stem; it has "
                             f"{', '.join(net.stems)}")
        self.stem = net.stems[variant]
        self.body = net.body
        self.tta = tta

    def _once(self, z):
        return torch.sigmoid(self.body(self.stem(z)))

    def forward(self, z):
        from . import infer

        if self.tta:
            acc = None
            for flip in (False, True):
                for k in range(4):
                    out = infer._d4_inv(self._once(infer._d4(z, k, flip)), k, flip)
                    acc = out if acc is None else acc + out
            p = acc / 8.0
        else:
            p = self._once(z)

        # Sized from the output, not from a constant, so it stays correct if the
        # export window changes. Built here rather than as a buffer so it lands
        # in the graph as an initializer of the right shape.
        taper = infer.hann2d(p.shape[-1], p.device, p.dtype)
        # Broadcast against p so the exporter cannot prune an output that does
        # not depend on the input.
        return p, taper.expand_as(p)


def export_onnx(net: MultiStemNet, variant: str, path, size: int = 256,
                overlap: float = 0.5, tta: bool = False) -> dict:
    """Freeze one variant to ONNX. Canopy variants are not deployable this way.

    The window is fixed rather than dynamic. That is a real constraint of the
    architecture, not an export limitation: a ResNet-34 U-Net needs its input
    divisible by 32, and ``torch.export`` correctly refuses to promise arbitrary
    H and W. It costs nothing, because the plugin has to tile with a Hann-
    tapered overlapping window regardless -- the same thing ``infer.predict``
    does -- so a fixed window is what it wants anyway.

    The returned dict is written beside the ``.onnx`` and is a *contract*, not
    documentation. Every number the caller needs in order to tile correctly is
    here as a number, computed by the same code that ``infer.predict`` runs, so
    a second implementation reads them instead of re-deriving them. Prose in this
    dict was how the plugin came to disagree with Python about the window step.
    """
    from . import infer
    v = var_mod.get(variant)
    if v.canopy:
        raise ValueError(f"{variant!r} needs canopy bands, which no raster "
                         "elevation service provides; export a bare-earth "
                         "variant such as 'dem1'")
    if size % 32:
        raise ValueError(f"body window {size} must be divisible by 32")
    model = Deployable(net, variant, tta=tta).eval()
    n = v.crop_px(size)
    # The TorchScript exporter, not dynamo: dynamo currently fails to translate
    # the stems' BatchNorm (_native_batch_norm_legit_no_training). It handles
    # the im2col median fine, which was the op in doubt.
    torch.onnx.export(model, (torch.zeros(1, 1, n, n),), str(path),
                      input_names=["elevation_m"],
                      output_names=["trail_probability", "window_taper"],
                      opset_version=17)
    return {
        "variant": variant,
        "res_m": v.res,
        "out_res_m": var_mod.BODY_RES,
        "input_px": n,
        "output_px": size,
        # Everything below this line is what a tiler needs, as numbers.
        "stride": v.stride,
        "overlap": overlap,
        "step_px": infer.window_step(n, overlap, v.stride),
        "pad_mode": "reflect",
        # Baked into the graph at export time, not switchable at runtime: ONNX
        # cannot branch on it, and shipping both graphs would double what a
        # mapper downloads for an 8x-compute option most will leave alone.
        "tta": tta,
        "outputs": ["trail_probability", "window_taper"],
        # Travels with the weights, because the weights travel without the
        # repository. See LICENSE-MODEL.
        "license": MODEL_LICENSE,
        "attribution": MODEL_ATTRIBUTION,
        "input": "single-band float32 bare-earth elevation in metres, "
                 "NaN for nodata",
        "output": f"trail probability at {var_mod.BODY_RES:g} m",
        "taper": "window_taper is the blending weight for this window; "
                 "accumulate probability*taper and taper, then divide",
    }


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save(model: nn.Module, path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load(path, device: torch.device | None = None) -> tuple[MultiStemNet, dict]:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["arch"], meta["encoder"],
                        variants=[var_mod.get(k) for k in meta["variants"]],
                        pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    if device:
        model.to(device)
    return model, meta
