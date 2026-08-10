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

import contextlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import aois as aoi_mod
from . import infer, metrics, model as model_mod, osm
from .data import TileDataset, full_tile
from .losses import TrailLoss

log = logging.getLogger(__name__)




def _loaders(dirs: list[Path], variants, cfg) -> dict[str, tuple]:
    """One train/val loader pair per input variant, over the same tiles.

    Workers must be spawned, not forked. The default start method is
    platform-dependent -- 'spawn' on macOS, 'fork' on Linux -- and this run
    has already touched CUDA and GDAL by the time these loaders are built.
    Forking a process with a live CUDA context and background driver threads
    is unsupported: the child does not use CUDA itself, but the corrupted
    heap/library state it can inherit shows up anywhere, and here it showed
    up as GDAL 'ZIPDecode: Decoding error' on the very first batch, reading
    dtm_clean.tif files that are not actually corrupt. Pin this explicitly
    rather than rely on whatever the OS defaults to.
    """
    out = {}
    ctx = "spawn" if cfg.workers > 0 else None
    common = dict(batch_size=cfg.batch, num_workers=cfg.workers,
                  pin_memory=False, persistent_workers=cfg.workers > 0,
                  multiprocessing_context=ctx)
    for i, v in enumerate(variants):
        train = TileDataset(dirs, v, body_crop=cfg.crop, split="train",
                            samples=cfg.samples, augment=True,
                            noise_m=cfg.noise_m, noise_band_m=cfg.noise_band_m,
                            canopy_dropout=cfg.canopy_dropout,
                            jitter_m=cfg.jitter_m)
        val = TileDataset(dirs, v, body_crop=cfg.crop, split="val",
                          samples=max(cfg.samples // 8, 64), augment=False)
        # A generator per (variant, split), not one shared across them. The
        # DataLoader draws each worker's torch seed from this generator, and
        # TileDataset._generator derives from torch.initial_seed(), so seeding
        # here reaches crop sampling and augmentation in every worker without a
        # worker_init_fn. Distinct streams keep the two variants off identical
        # crop sequences and stop val mirroring train.
        gt, gv = torch.Generator(), torch.Generator()
        if cfg.seed is not None:
            gt.manual_seed(cfg.seed + 1000 * i)
            gv.manual_seed(cfg.seed + 1000 * i + 500)
        out[v.key] = (DataLoader(train, shuffle=False, drop_last=True,
                                 generator=gt, **common),
                      DataLoader(val, shuffle=False, generator=gv, **common))
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


def _checkpoint(path, net, opt, sched, scaler, meta, best, history) -> None:
    """Everything needed to carry on, in a file the plain loader still reads.

    ``state_dict`` and ``meta`` stay at the top level and mean what they always
    did, so ``model.load`` -- and therefore export, parity and inference --
    treat this like any other checkpoint and ignore the rest.
    """
    torch.save({"state_dict": net.state_dict(), "meta": meta,
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "best": best, "history": history}, path)


def _resume(path, net, opt, sched, scaler, device, total_steps, epochs):
    """Restore a run. Returns (start_epoch, best, history).

    A checkpoint written before this existed carries weights and nothing else.
    That is a *warm start*, not a resume: the optimiser moments and the position
    in the LR schedule are gone, so training restarts the whole OneCycle from
    those weights. It is a legitimate thing to do and a different thing from
    what the caller asked for, so it says so loudly rather than quietly
    producing a run whose schedule nobody can reconstruct later.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"])

    if "opt" not in ck:
        log.warning("%s holds weights only: no optimiser or scheduler state. "
                    "Warm start -- AdamW moments reset and the LR schedule "
                    "restarts from step 0 over %d epochs.", path, epochs)
        return 0, -1.0, []

    saved = ck["sched"].get("total_steps")
    if saved != total_steps:
        log.warning("scheduler was built for %s steps, this run plans %d; "
                    "the LR curve will not line up. Match --epochs to the "
                    "original run to resume faithfully.", saved, total_steps)
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"])
    start = ck["meta"]["epoch"] + 1
    log.info("resumed %s at epoch %d, best stratified f1 %.4f",
             path, start + 1, ck["best"])
    return start, ck["best"], ck["history"]


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


def selection_score(val_stats: dict) -> tuple[float, float, list[float]]:
    """Which epoch to keep: the deployable variants' stratified F1.

    A canopy-bearing variant cannot be exported -- ``export_onnx`` refuses it,
    because no raster elevation service supplies the bands -- so letting it vote
    picks the epoch that was best for a model nobody will ever run. It still
    trains, and its gradients still shape the shared trunk; it just does not get
    a say in which checkpoint survives.

    Measured on runs/full-b before the change: the old all-variant rule cost
    0.0004 dem1 F1 at the chosen epoch and never once promoted an epoch where
    dem1 had got worse. This is cleanup, not a fix. See trailer-440.24.

    Returns ``(score, score_all, deployable)``. ``score_all`` is the old
    quantity, kept so runs either side of this change stay comparable; it is
    also the fallback when nothing here is deployable, which is a real
    configuration -- a lidar05-only run has to select on something.
    """
    from . import variants as var_mod

    score_all = float(np.mean([s["strat"]["score"] for s in val_stats.values()]))
    deployable = [s["strat"]["score"] for v, s in val_stats.items()
                  if not var_mod.get(v).canopy]
    return (float(np.mean(deployable)) if deployable else score_all,
            score_all, deployable)


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
            # Travels beside the number, into report.json, so a reader who
            # never opens aois.py still cannot mistake this tile's score for
            # evidence. See Aoi.advisory.
            rec["advisory"] = aoi_mod.advisory(d.name)
            out.setdefault(v.key, {})[d.name] = rec
            log.info("%-9s %-22s f1@0.5=%.3f ap=%.3f fp=%.4f%s", v.key, d.name,
                     rec["f1@0.5"], rec["ap"], rec["fp_rate@0.5"],
                     "  ADVISORY, not evidence" if rec["advisory"] else "")
    return out


def run(train_dirs: list[Path], test_dirs: list[Path], cfg) -> dict:
    from . import variants as var_mod

    device = model_mod.pick_device(cfg.device)
    variants = var_mod.parse(cfg.variants)
    res = var_mod.BODY_RES
    log.info("device=%s, body res=%g m, %d m of context per crop",
             device, res, cfg.crop * res)

    # Before build_model and estimate_prior, which both draw from the main
    # process rng -- weight init for the stems and encoder head, and the 256
    # crops the prior is estimated from.
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        log.info("seed %d: weight init and crop sampling are repeatable; "
                 "kernel non-determinism is not addressed", cfg.seed)

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

    # Mixed precision, on whichever accelerator this is. Gating it on "cuda"
    # meant MPS -- the only accelerator this project actually has -- trained
    # entirely in fp32.
    #
    # bf16 is the default on MPS rather than fp16 because it carries fp32's
    # exponent range: the input bands use NaN as the nodata sentinel and the
    # derivative maths clips against absolute bounds, and fp16's 65504 ceiling
    # is close enough to those to be worth not thinking about. Only fp16 needs
    # loss scaling, so the scaler is enabled for it alone.
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg.amp)
    use_amp = amp_dtype is not None and device.type in ("cuda", "mps")
    if cfg.amp != "off" and not use_amp:
        log.warning("--amp %s ignored: no autocast on %s", cfg.amp, device.type)
    autocast = ((lambda: torch.amp.autocast(device.type, dtype=amp_dtype))
                if use_amp else contextlib.nullcontext)
    scaler = torch.amp.GradScaler(
        device.type, enabled=use_amp and amp_dtype is torch.float16)
    log.info("precision: %s", f"autocast {cfg.amp}" if use_amp else "fp32")

    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"arch": cfg.arch, "encoder": cfg.encoder,
            "variants": [v.key for v in variants], "crop": cfg.crop,
            "body_res": res, "prior": round(prior, 5),
            "tolerance_m": cfg.tolerance_m, "jitter_m": cfg.jitter_m,
            "tiles": [d.name for d in train_dirs]}

    best, history = -1.0, []
    start_epoch = 0
    if cfg.resume:
        start_epoch, best, history = _resume(
            Path(cfg.resume), net, opt, sched, scaler, device,
            cfg.epochs * per_epoch, cfg.epochs)

    for epoch in range(start_epoch, cfg.epochs):
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
                with autocast():
                    logits, y, w, _ = _forward(net, batch, v.key, device)
                    loss, parts = criterion(logits, y, w, ramp=ramp)
                # Yes, this is a device-to-host copy in the inner loop, and no,
                # sampling it every Nth step is not available. A single NaN
                # gradient makes clip_grad_norm_ return NaN, which poisons every
                # gradient, then the weights and AdamW's moments -- unrecoverable
                # rather than merely detected late. Under --amp bf16 the
                # GradScaler is disabled, so this is the only guard there is.
                # Measured cost on CUDA: syncing once per step is 3.6%, and
                # syncing five times costs the same as once, because the price is
                # the drain and not the call. Removing the other .item() calls
                # while this one stays therefore buys exactly nothing -- measured
                # at 1.00x, see trailer-440.27. The loop is launch-bound, not
                # sync-bound; trailer-440.28 has where the time actually goes.
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
        # Mean over visibility class of best-threshold relaxed F1, on the
        # variants that can actually deploy.
        #
        # Across classes, because pooled recall is length-weighted and would let
        # faint trail -- the thing this project is for -- be traded away for
        # more of what we already detect.
        score, score_all, deployable = selection_score(val_stats)
        if epoch == 0:
            if not deployable:
                log.warning("no deployable variant among %s: selecting on the "
                            "all-variant mean, which picks a checkpoint for a "
                            "model export_onnx will refuse",
                            "/".join(val_stats))
            for k, s in val_stats.items():
                gone = [c for c in osm.CLASS_CODE if c not in s["strat"]["classes"]]
                if gone:
                    log.warning("%s validation band holds no %s pixels; those "
                                "classes are absent from the selection score",
                                k, "/".join(gone))
        history.append({"epoch": epoch, "ramp": round(ramp, 2),
                        "train": train_stats, "val": val_stats,
                        "score": round(score, 4),
                        "score_all_variants": round(score_all, 4),
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

        # Every epoch, not just at the end: a 22-hour run that is interrupted
        # at hour 20 should cost minutes, not the run. This is what --resume
        # reads.
        _checkpoint(outdir / "last.pt", net, opt, sched, scaler,
                    meta | {"epoch": epoch}, best, history)

    # Held-out set, scored once, with the best checkpoint.
    net, _ = model_mod.load(outdir / "best.pt", device)
    held = evaluate_tiles(net, test_dirs, variants, res, device, cfg)

    report = {"config": {k: v for k, v in vars(cfg).items() if k != "fn"},
              "best_val_stratified_f1": round(best, 4), "history": history,
              "held_out": held,
              # The number a deployment claim should quote, and it is a spread.
              "held_out_spread": metrics.held_out_spread(held)}
    (outdir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log.info("wrote %s", outdir / "report.json")
    return report
