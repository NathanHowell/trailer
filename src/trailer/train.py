"""Training loop.

Model selection is on relaxed F1 at the 5 m tolerance, not on loss and not on
pixel AP. AP rewards fattening predictions until they cover the label slop;
relaxed F1 does not, and it is closer to the question the JOSM reviewer asks.

That F1 is stratified: the score is the mean over (variant x visibility class)
of the best-threshold relaxed F1, never a pooled number. Pooled recall is
weighted by labelled kilometres, so it hands the checkpoint decision to whichever
class the corpus happens to hold most of -- and a corpus is a thing that changes.
See metrics.Stratified.

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

from . import infer, metrics, model as model_mod, osm
from .data import TileDataset, full_tile
from .losses import TrailLoss

log = logging.getLogger(__name__)




def _loaders(dirs: list[Path], variants, cfg) -> dict[str, tuple]:
    """One train/val loader pair per input variant, over the same tiles."""
    out = {}
    common = dict(batch_size=cfg.batch, num_workers=cfg.workers,
                  pin_memory=False, persistent_workers=cfg.workers > 0)
    for v in variants:
        train = TileDataset(dirs, v, body_crop=cfg.crop, split="train",
                            samples=cfg.samples, augment=True,
                            noise_m=cfg.noise_m, noise_band_m=cfg.noise_band_m,
                            canopy_dropout=cfg.canopy_dropout,
                            jitter_m=cfg.jitter_m)
        val = TileDataset(dirs, v, body_crop=cfg.crop, split="val",
                          samples=max(cfg.samples // 8, 64), augment=False)
        out[v.key] = (DataLoader(train, shuffle=False, drop_last=True, **common),
                      DataLoader(val, shuffle=False, **common))
    return out


def _param_groups(net, lr: float):
    """Pretrained encoder learns slower than the randomly-initialised decoder.

    Stems are new and tiny, and each sees only its own variant's batches, so
    they stay at the full rate alongside the decoder.
    """
    enc, dec = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (enc if name.startswith("body.encoder.") else dec).append(p)
    return [{"params": enc, "lr": lr * 0.1}, {"params": dec, "lr": lr}]


def estimate_prior(dataset, n: int = 256) -> float:
    """Positive rate the loss actually sees, over unignored pixels.

    Measured from sampled crops rather than from the tiles: positive-biased
    sampling lifts a 0.55-2.5% tile rate to ~3%, and it is the sampled rate that
    the output bias and pos_weight have to match.
    """
    pos = tot = 0
    for i in range(min(n, len(dataset))):
        _, _, y, w, _ = dataset[i]
        m = w > 0
        pos += int((y[m] > 0).sum())
        tot += int(m.sum())
    return pos / max(tot, 1)


def _ramp(epoch: int, warmup: int, span: int = 3) -> float:
    """clDice weight schedule: off, then linear in."""
    if epoch < warmup:
        return 0.0
    return min((epoch - warmup + 1) / span, 1.0)


def _forward(net, batch, variant, device):
    z, canopy, y, w, cls = (t.to(device) for t in batch)
    canopy = canopy if canopy.shape[1] else None
    return net(z, canopy, variant=variant), y, w, cls


@torch.no_grad()
def validate(net, loader, criterion, variant: str, res: float, device) -> dict:
    net.eval()
    agg: dict[str, list] = {}
    strat = metrics.Stratified(res)
    for batch in loader:
        logits, y, w, cls = _forward(net, batch, variant, device)
        loss, parts = criterion(logits, y, w, ramp=1.0)
        prob = torch.sigmoid(logits)
        strat.update(prob, y, w, cls)
        row = {"loss": loss.item(), **parts, **metrics.sweep(prob, y, w, res)}
        for k, v in row.items():
            agg.setdefault(k, []).append(v)
    # Pooled numbers stay for comparability with earlier runs and with the
    # published baselines; the stratified block is what selection reads.
    out = {k: round(float(np.nanmean(v)), 4) for k, v in agg.items()}
    return out | {"strat": strat.result()}


def evaluate_tiles(net, dirs: list[Path], variants, res: float, device,
                   cfg) -> dict:
    """Full-tile sliding-window scoring, per variant. Used for the held-out set.

    Scored for every variant deliberately. The 1 m bare-earth number is the one
    that predicts what a JOSM reviewer will actually see; the 0.5 m number says
    what the extra fidelity is worth, and the gap between them is the price of
    deploying against a public DEM instead of our own point clouds.
    """
    out: dict[str, dict] = {}
    for v in variants:
        for d in dirs:
            if not (d / "dtm_clean.tif").exists():
                continue
            z, canopy, y, w, cls = full_tile(d, v)
            prob = infer.predict(net, z, canopy, variant=v.key,
                                 body_tile=cfg.crop, device=device,
                                 batch=cfg.batch, tta=cfg.tta)
            n = min(prob.shape[-2], y.shape[-2]), min(prob.shape[-1], y.shape[-1])
            prob, y, w, cls = (a[..., :n[0], :n[1]] for a in (prob, y, w, cls))
            pt, yt, wt, ct = (torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0)
                              for a in (prob, y, w, cls))
            rec = metrics.sweep(pt, yt, wt, res)
            strat = metrics.Stratified(res)
            strat.update(pt, yt, wt, ct)
            rec["strat"] = strat.result()
            rec["ap"] = round(metrics.average_precision(prob, y, w), 4)
            rec["fp_rate@0.5"] = round(metrics.false_positive_rate(prob, w), 5)
            rec["positive_frac"] = round(float((y > 0).mean()), 5)
            out.setdefault(v.key, {})[d.name] = rec
            log.info("%-9s %-22s f1@0.5=%.3f ap=%.3f fp=%.4f", v.key, d.name,
                     rec["f1@0.5"], rec["ap"], rec["fp_rate@0.5"])
    return out


def run(train_dirs: list[Path], test_dirs: list[Path], cfg) -> dict:
    from . import variants as var_mod

    device = model_mod.pick_device(cfg.device)
    variants = var_mod.parse(cfg.variants)
    res = var_mod.BODY_RES
    log.info("device=%s, body res=%g m, %d m of context per crop",
             device, res, cfg.crop * res)

    loaders = _loaders(train_dirs, variants, cfg)
    net = model_mod.build_model(cfg.arch, cfg.encoder, variants=variants,
                                pretrained=not cfg.no_pretrained).to(device)
    prior = estimate_prior(loaders[variants[0].key][0].dataset)
    model_mod.set_output_prior(net, prior, cfg.pos_weight)
    log.info("effective neg:pos after sampling and pos_weight: %.1f:1",
             (1 - prior) / max(prior * cfg.pos_weight, 1e-9))
    criterion = TrailLoss(cldice_w=cfg.cldice, pos_weight=cfg.pos_weight,
                          tolerance_m=cfg.tolerance_m, res=res)
    log.info("loss tolerance %.1f m (%d px); label jitter +/-%.1f m",
             cfg.tolerance_m, criterion.radius, cfg.jitter_m)
    opt = torch.optim.AdamW(_param_groups(net, cfg.lr), weight_decay=1e-4)

    # Every variant contributes an optimiser step per batch index, so the
    # schedule has to count them all.
    per_epoch = max(min(len(t) for t, _ in loaders.values()), 1) * len(variants)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=cfg.epochs * per_epoch, pct_start=0.15)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"arch": cfg.arch, "encoder": cfg.encoder,
            "variants": [v.key for v in variants], "crop": cfg.crop,
            "body_res": res, "prior": round(prior, 5),
            "tolerance_m": cfg.tolerance_m, "jitter_m": cfg.jitter_m,
            "tiles": [d.name for d in train_dirs]}

    best = -1.0
    history = []
    for epoch in range(cfg.epochs):
        net.train()
        ramp = _ramp(epoch, cfg.cldice_warmup)
        t0 = time.time()
        running: dict[str, dict[str, float]] = {v.key: {} for v in variants}
        counts = {v.key: 0 for v in variants}

        # Interleave variants step by step rather than epoch by epoch, so the
        # shared trunk never spends a stretch seeing only one input scale and
        # drifting towards it.
        for batches in zip(*(t for t, _ in loaders.values())):
            for v, batch in zip(variants, batches):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, y, w, _ = _forward(net, batch, v.key, device)
                    loss, parts = criterion(logits, y, w, ramp=ramp)
                if not math.isfinite(loss.item()):
                    log.warning("non-finite loss on %s, skipping", v.key)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                counts[v.key] += 1
                for k, val in ({"loss": loss.item()} | parts).items():
                    running[v.key][k] = running[v.key].get(k, 0.0) + val

        train_stats = {vk: {k: round(x / max(counts[vk], 1), 4)
                            for k, x in r.items()}
                       for vk, r in running.items()}
        val_stats = {v.key: validate(net, loaders[v.key][1], criterion,
                                     v.key, res, device) for v in variants}
        # Mean over (variant x visibility class) of best-threshold relaxed F1.
        #
        # Across variants, because optimising for the 0.5 m path alone would
        # quietly let the deployable 1 m one rot, and it is the 1 m path a JOSM
        # reviewer actually meets. Across classes, because pooled recall is
        # length-weighted and would let faint trail -- the thing this project is
        # for -- be traded away for more of what we already detect.
        score = float(np.mean([s["strat"]["score"] for s in val_stats.values()]))
        if epoch == 0:
            for k, s in val_stats.items():
                gone = [c for c in osm.CLASS_CODE if c not in s["strat"]["classes"]]
                if gone:
                    log.warning("%s validation band holds no %s pixels; those "
                                "classes are absent from the selection score",
                                k, "/".join(gone))
        history.append({"epoch": epoch, "ramp": round(ramp, 2),
                        "train": train_stats, "val": val_stats,
                        "score": round(score, 4),
                        "seconds": round(time.time() - t0, 1)})
        log.info("epoch %2d/%d  stratified f1 %.4f  [%s]  %.0fs",
                 epoch + 1, cfg.epochs, score,
                 "  ".join(f"{k} {s['loss']:.3f}/" + "/".join(
                     f"{c[:4]}{s['strat']['by_class'][c]['f1']:.2f}"
                     for c in s["strat"]["classes"])
                     for k, s in val_stats.items()),
                 history[-1]["seconds"])

        if score > best:
            best = score
            model_mod.save(net, outdir / "best.pt",
                           meta | {"val": val_stats, "epoch": epoch})
            log.info("  new best (stratified f1 %.4f) -> %s", best,
                     outdir / "best.pt")

    model_mod.save(net, outdir / "last.pt", meta | {"epoch": cfg.epochs - 1})

    # Held-out set, scored once, with the best checkpoint.
    net, _ = model_mod.load(outdir / "best.pt", device)
    held = evaluate_tiles(net, test_dirs, variants, res, device, cfg)

    report = {"config": {k: v for k, v in vars(cfg).items() if k != "fn"},
              "best_val_stratified_f1": round(best, 4), "history": history,
              "held_out": held}
    (outdir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log.info("wrote %s", outdir / "report.json")
    return report
