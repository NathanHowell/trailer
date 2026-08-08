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

import torch
import torch.nn as nn

from .rasters import BAND_NAMES

log = logging.getLogger(__name__)

DEFAULT_ENCODER = "resnet34"


def build_model(arch: str = "unet", encoder: str = DEFAULT_ENCODER,
                in_channels: int = len(BAND_NAMES),
                pretrained: bool = True) -> nn.Module:
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

    model = factory[arch](
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=1,
    )
    n = sum(p.numel() for p in model.parameters())
    log.info("%s/%s: %.1fM params, %d input bands", arch, encoder, n / 1e6,
             in_channels)
    return model


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


def load(path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["arch"], meta["encoder"],
                        in_channels=meta["in_channels"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    if device:
        model.to(device)
    return model, meta
