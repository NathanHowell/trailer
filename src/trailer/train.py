"""Training loop.

Model selection is on relaxed F1 at the 5 m tolerance, not on loss and not on
pixel AP. AP rewards fattening predictions until they cover the label slop;
relaxed F1 does not, and it is closer to the question the JOSM reviewer asks.

The eval-role tiles (abandoned trails) and the control tile are scored only at
the end of a run, on full tiles with sliding-window inference. They are a test
set: selecting on them would spend the only honest estimate of abandoned-trail
recall and false-positive rate that exists here.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import infer, metrics, model as model_mod
from .data import TileDataset, full_tile
from .losses import TrailLoss

log = logging.getLogger(__name__)


def _loaders(dirs: list[Path], cfg) -> tuple[DataLoader, DataLoader]:
    train = TileDataset(dirs, crop=cfg.crop, split="train",
                        samples=cfg.samples, augment=True)
    val = TileDataset(dirs, crop=cfg.crop, split="val",
                      samples=max(cfg.samples // 8, 64), augment=False)
    common = dict(batch_size=cfg.batch, num_workers=cfg.workers,
                  pin_memory=False, persistent_workers=cfg.workers > 0)
    return (DataLoader(train, shuffle=False, drop_last=True, **common),
            DataLoader(val, shuffle=False, **common))


def _param_groups(net, lr: float):
    """Pretrained encoder learns slower than the randomly-initialised decoder."""
    enc, dec = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (enc if name.startswith("encoder.") else dec).append(p)
    return [{"params": enc, "lr": lr * 0.1}, {"params": dec, "lr": lr}]


def _ramp(epoch: int, warmup: int, span: int = 3) -> float:
    """clDice weight schedule: off, then linear in."""
    if epoch < warmup:
        return 0.0
    return min((epoch - warmup + 1) / span, 1.0)


@torch.no_grad()
def validate(net, loader, criterion, res: float, device) -> dict:
    net.eval()
    agg: dict[str, list] = {}
    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        logits = net(x)
        loss, parts = criterion(logits, y, w, ramp=1.0)
        prob = torch.sigmoid(logits)
        row = {"loss": loss.item(), **parts,
               **metrics.sweep(prob, y, w, res)}
        for k, v in row.items():
            agg.setdefault(k, []).append(v)
    return {k: round(float(np.nanmean(v)), 4) for k, v in agg.items()}


def evaluate_tiles(net, dirs: list[Path], res: float, device, cfg) -> dict:
    """Full-tile sliding-window scoring. Used for the held-out set."""
    out = {}
    for d in dirs:
        if not (d / "features.tif").exists():
            continue
        x, y, w = full_tile(d)
        prob = infer.predict(net, x, tile=cfg.crop, device=device,
                             batch=cfg.batch, tta=cfg.tta)
        pt = torch.from_numpy(prob).unsqueeze(0)
        yt = torch.from_numpy(y).unsqueeze(0)
        wt = torch.from_numpy(w).unsqueeze(0)
        rec = metrics.sweep(pt, yt, wt, res)
        rec["ap"] = round(metrics.average_precision(prob, y, w), 4)
        rec["fp_rate@0.5"] = round(metrics.false_positive_rate(prob, w), 5)
        rec["positive_frac"] = round(float((y > 0).mean()), 5)
        out[d.name] = rec
        log.info("%-22s f1@0.5=%.3f ap=%.3f fp=%.4f", d.name,
                 rec["f1@0.5"], rec["ap"], rec["fp_rate@0.5"])
    return out


def run(train_dirs: list[Path], test_dirs: list[Path], cfg) -> dict:
    device = model_mod.pick_device(cfg.device)
    log.info("device=%s", device)

    loader, val_loader = _loaders(train_dirs, cfg)
    net = model_mod.build_model(cfg.arch, cfg.encoder,
                                pretrained=not cfg.no_pretrained).to(device)
    criterion = TrailLoss(cldice_w=cfg.cldice, pos_weight=cfg.pos_weight)
    opt = torch.optim.AdamW(_param_groups(net, cfg.lr), weight_decay=1e-4)

    steps = max(len(loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=cfg.epochs * steps, pct_start=0.15)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"arch": cfg.arch, "encoder": cfg.encoder,
            "in_channels": 6, "crop": cfg.crop, "res": cfg.res,
            "tiles": [d.name for d in train_dirs]}

    best = -1.0
    history = []
    for epoch in range(cfg.epochs):
        net.train()
        ramp = _ramp(epoch, cfg.cldice_warmup)
        t0 = time.time()
        running: dict[str, float] = {}
        for i, (x, y, w) in enumerate(loader):
            x, y, w = x.to(device), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = net(x)
                loss, parts = criterion(logits, y, w, ramp=ramp)
            if not math.isfinite(loss.item()):
                log.warning("non-finite loss at step %d, skipping", i)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in ({"loss": loss.item()} | parts).items():
                running[k] = running.get(k, 0.0) + v

        train_stats = {k: round(v / steps, 4) for k, v in running.items()}
        val_stats = validate(net, val_loader, criterion, cfg.res, device)
        score = val_stats.get("f1@0.5", 0.0)
        history.append({"epoch": epoch, "ramp": round(ramp, 2),
                        "train": train_stats, "val": val_stats,
                        "seconds": round(time.time() - t0, 1)})
        log.info("epoch %2d/%d  train %.4f  val %.4f  "
                 "relaxed p/r/f1 @0.5 %.3f/%.3f/%.3f  %.0fs",
                 epoch + 1, cfg.epochs, train_stats["loss"], val_stats["loss"],
                 val_stats["p@0.5"], val_stats["r@0.5"], score,
                 history[-1]["seconds"])

        if score > best:
            best = score
            model_mod.save(net, outdir / "best.pt", meta | {"val": val_stats,
                                                            "epoch": epoch})
            log.info("  new best (relaxed f1 %.4f) -> %s", best,
                     outdir / "best.pt")

    model_mod.save(net, outdir / "last.pt", meta | {"epoch": cfg.epochs - 1})

    # Held-out set, scored once, with the best checkpoint.
    net, _ = model_mod.load(outdir / "best.pt", device)
    held = evaluate_tiles(net, test_dirs, cfg.res, device, cfg)

    report = {"config": {k: v for k, v in vars(cfg).items() if k != "fn"},
              "best_val_f1": round(best, 4), "history": history,
              "held_out": held}
    (outdir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log.info("wrote %s", outdir / "report.json")
    return report
