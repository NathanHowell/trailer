"""Command-line driver.

    uv run trailer survey                    # coverage + density, no downloads
    uv run trailer build --aoi all           # full data build
    uv run trailer build --aoi giant_forest,colby_pass --res 0.25
    uv run trailer qa                        # tread signal per tile
    uv run trailer preview --aoi moraine_lake
    uv run trailer train --epochs 40         # U-Net + BCE/Tversky/clDice
    uv run trailer predict --aoi colby_pass --tta
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import build as build_mod
from . import coverage, qa
from .aois import AOIS, select

DEFAULT_ROOT = Path("data/tiles")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_survey(args) -> int:
    cache = Path(args.root) / ".cache"
    aois = select(args.aoi, args.role)
    print(f'{"key":22s} {"role":8s} {"lat,lon":22s} {"3DEP project":24s} notes')
    print("-" * 120)
    for aoi in aois:
        try:
            project = coverage.find_project(aoi.lat, aoi.lon, cache)
        except LookupError as exc:
            project = f"!! {exc.args[0][:30]}"
        print(f"{aoi.key:22s} {aoi.role:8s} "
              f"{aoi.lat:.5f},{aoi.lon:.5f}  {project:24s} {aoi.notes[:60]}")
    return 0


def cmd_harvest(args) -> int:
    from . import harvest as harvest_mod

    root = Path(args.root)
    cache = root / ".cache"
    bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox \
        else harvest_mod.SIERRA_BBOX
    elements = harvest_mod.fetch(bbox, cache_dir=cache, refresh=args.refresh)
    logging.info("%d faint/lifecycle ways in bbox", len(elements))

    cells = harvest_mod.score_cells(elements, size_m=args.size)
    if not cells:
        print("no candidate cells found", file=sys.stderr)
        return 1
    total_km = sum(c["total_m"] for c in cells) / 1000
    print(f"{len(cells)} candidate cells, {total_km:.0f} km of faint/lifecycle "
          f"way outside the curated tiles")

    chosen = harvest_mod.select_cells(cells, args.limit, cache, args.min_m)
    if not chosen:
        print("no covered cells above the threshold", file=sys.stderr)
        return 1

    path = harvest_mod.write_registry(chosen, Path(args.registry))
    picked = sum(c["total_m"] for c in cells[:len(chosen)]) / 1000
    print(f"wrote {len(chosen)} AOIs to {path} (~{picked:.1f} km of "
          f"faint/lifecycle way)")
    print(f"disk: {0.078 * len(chosen):.1f} GB evicting point clouds, "
          f"{0.42 * len(chosen):.1f} GB keeping them; "
          f"{build_mod.free_gb(root):.1f} GB free now")
    print("\nnext:  uv run trailer build --role harvest --evict-points")
    return 0


def cmd_vet(args) -> int:
    import shutil

    from . import harvest as harvest_mod
    from .aois import load_harvest

    root = Path(args.root)
    registry = Path(args.registry)
    entries = load_harvest(registry)
    if not entries:
        print(f"no harvest registry at {registry}", file=sys.stderr)
        return 1

    dirs = [root / a.slug for a in entries]
    verdicts = harvest_mod.vet(dirs)
    built = [v for v in verdicts if "not built" not in v["reasons"]]
    ok = [v for v in verdicts if v["accepted"]]
    bad = [v for v in built if not v["accepted"]]

    print(f'{"tile":26s} {"grnd/m2":>8s} {"valid":>6s} {"km":>6s} '
          f'{"SNR":>18s}  verdict')
    print("-" * 88)
    for v in sorted(verdicts, key=lambda r: (r["accepted"], r["key"])):
        if "not built" in v["reasons"]:
            continue
        snr = " ".join(f'{c[:4]}={s}' for c, s in (v.get("snr") or {}).items())
        gd = v.get("ground_density")
        vf = v.get("valid_frac")
        print(f'{v["key"]:26s} {gd if gd is None else f"{gd:8.1f}":>8} '
              f'{vf if vf is None else f"{vf:6.2f}":>6} {v.get("trail_km", 0):6.2f} '
              f'{snr:>18s}  '
              f'{"ok" if v["accepted"] else "; ".join(v["reasons"])}')

    print(f"\n{len(built)} built, {len(ok)} accepted, {len(bad)} rejected")
    (root / "vet.json").write_text(json.dumps(verdicts, indent=1))

    if args.apply:
        keep = {v["key"] for v in ok}
        harvest_mod.write_registry([a for a in entries if a.key in keep], registry)
        print(f"rewrote {registry} with {len(keep)} tiles")
        if args.prune:
            freed = 0
            for v in bad:
                d = root / v["key"]
                if d.exists():
                    freed += sum(f.stat().st_size for f in d.rglob("*"))
                    shutil.rmtree(d)
            print(f"pruned {len(bad)} rejected tiles, freed {freed / 1e9:.2f} GB")
    elif bad:
        print("re-run with --apply to drop them from the registry "
              "(add --prune to delete their files)")
    return 0


def cmd_build(args) -> int:
    root = Path(args.root)
    aois = select(args.aoi, args.role)
    logging.info("building %d AOI(s) at %.2f m into %s (%.1f GB free)",
                 len(aois), args.res, root, build_mod.free_gb(Path(".")))
    results = build_mod.build_all(aois, root, res=args.res, force=args.force,
                                  evict_points=args.evict_points,
                                  min_free_gb=args.min_free_gb)
    failed = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    if ok:
        total_km = sum(r["labels"]["trail_km"] for r in ok)
        print(f"\nbuilt {len(ok)} tiles, {total_km:.1f} km of labelled trail")
    for r in failed:
        print(f"FAILED {r['aoi']['key']}: {r['error']}", file=sys.stderr)
    return 1 if failed else 0


def cmd_qa(args) -> int:
    root = Path(args.root)
    aois = select(args.aoi, args.role)
    results = []
    for aoi in aois:
        d = root / aoi.slug
        if not (d / "manifest.json").exists():
            logging.warning("%s not built, skipping", aoi.key)
            continue
        try:
            results.append(qa.analyse(d))
        except Exception as exc:
            logging.error("%s qa failed: %s", aoi.key, exc)
    if not results:
        print("nothing to report -- run `trailer build` first", file=sys.stderr)
        return 1
    print(qa.summarise(results))
    (root / "qa.json").write_text(json.dumps(results, indent=1))
    return 0


def cmd_preview(args) -> int:
    from .preview import render
    root = Path(args.root)
    for aoi in select(args.aoi, args.role):
        d = root / aoi.slug
        if not (d / "manifest.json").exists():
            logging.warning("%s not built, skipping", aoi.key)
            continue
        out = render(d, d / "preview.png")
        print(f"wrote {out}")
    return 0


def _built(root: Path, *roles: str) -> list[Path]:
    dirs = [root / a.slug for r in roles for a in select("all", r)]
    return [d for d in dirs if (d / "manifest.json").exists()]


def _tile_res(dirs: list[Path], fallback: float) -> float:
    """Take pixel size from what was actually built, not from a flag."""
    for d in dirs:
        try:
            return float(json.loads((d / "manifest.json").read_text())["res"])
        except (OSError, KeyError, ValueError):
            continue
    return fallback


def cmd_train(args) -> int:
    from . import train as train_mod

    root = Path(args.root)
    train_dirs = _built(root, "train", "harvest")
    if not train_dirs:
        print("no built training tiles -- run `trailer build` first", file=sys.stderr)
        return 1
    test_dirs = _built(root, "eval", "control")
    args.res = _tile_res(train_dirs, args.res)
    logging.info("train on %d tiles, hold out %d, %.2f m source pixels",
                 len(train_dirs), len(test_dirs), args.res)
    report = train_mod.run(train_dirs, test_dirs, args)
    print("\nbest stratified relaxed F1 (val, mean over variant x class) "
          f"{report['best_val_stratified_f1']:.4f}")
    for variant, tiles in report["held_out"].items():
        print(f"  {variant}")
        for name, rec in tiles.items():
            by = rec["strat"]["by_class"]
            # Pooled f1 alongside the per-class split, so the two can be
            # compared on the held-out tiles rather than taken on trust.
            print(f"    {name:22s} f1@0.5 {rec['f1@0.5']:.3f}  "
                  f"fp {rec['fp_rate@0.5']:.5f}  " +
                  " ".join(f"{c[:4]} {by[c]['f1']:.3f}" for c in
                           rec["strat"]["classes"]))
            if rec.get("advisory"):
                print(f"      ^ NOT EVIDENCE: {rec['advisory']}")
    return 0


def cmd_relabel(args) -> int:
    """Rebuild labels.tif from cached OSM without touching point clouds."""
    from . import labels as labels_mod
    from . import rasters

    root = Path(args.root)
    done = skipped = 0
    totals = {"active": 0.0, "faint": 0.0, "lifecycle": 0.0}
    for aoi in select(args.aoi, args.role):
        d = root / aoi.slug
        # A manifest means the tile finished. Relabelling one still being built
        # would race the builder writing the same file.
        if not ((d / "manifest.json").exists() and (d / "features.tif").exists()
                and (d / "osm.json").exists()):
            skipped += 1
            continue
        m = json.loads((d / "manifest.json").read_text())
        if "error" in m:
            skipped += 1
            continue
        epsg = m.get("epsg") or rasters.utm_epsg(aoi.lat, aoi.lon)
        elements = json.loads((d / "osm.json").read_text())["elements"]
        rec = labels_mod.build(elements, d / "features.tif",
                               d / "labels.tif", epsg)
        if "water" in aoi.flags and (d / "dtm_clean.tif").exists():
            frac = labels_mod.mask_water_from_dtm(d / "dtm_clean.tif",
                                                  d / "labels.tif")
            rec["water_masked_frac"] = round(frac, 4)
        m["labels"] = rec
        (d / "manifest.json").write_text(json.dumps(m, indent=1))
        for k, v in rec["trail_km_by_class"].items():
            totals[k] += v
        done += 1
        logging.info("%-22s %s", aoi.key, rec["trail_km_by_class"])

    print(f"relabelled {done}, skipped {skipped}")
    print("  trail km by class: " +
          "  ".join(f"{k} {v:.2f}" for k, v in totals.items()))
    return 0


def cmd_dem(args) -> int:
    from . import dem as dem_mod

    root = Path(args.root)
    dirs = [root / a.slug for a in select(args.aoi, args.role)]
    dirs = [d for d in dirs if (d / "dtm_clean.tif").exists()]
    if not dirs:
        print("no built tiles with a DTM yet", file=sys.stderr)
        return 1
    logging.info("fetching published 1 m DEM for %d tiles", len(dirs))
    recs = dem_mod.build_all(dirs, force=args.force)
    (root / "dem.json").write_text(json.dumps(recs, indent=1))

    bad = [r for r in recs if "error" in r]
    ok = [r for r in recs if "error" not in r and not r.get("skipped")]
    skipped = [r for r in recs if r.get("skipped")]
    print(f"fetched {len(ok)}, skipped {len(skipped)}, failed {len(bad)}")
    if ok:
        import statistics
        c = [r["slope_corr"] for r in ok]
        print(f"  slope correlation vs our DTM: min {min(c):.3f} "
              f"median {statistics.median(c):.3f} max {max(c):.3f}")
    for r in bad:
        print(f"  FAIL {r['key']}: {r['error']}", file=sys.stderr)
    return 1 if bad else 0


def cmd_golden(args) -> int:
    from . import golden

    out = golden.write(Path(args.out))
    print(f"wrote {out}")
    return 0


def cmd_parity(args) -> int:
    from . import golden

    m = golden.real_parity(Path(args.checkpoint), Path(args.tile),
                           Path(args.out), variant=args.variant,
                           window=args.window, max_px=args.max_px,
                           overlap=args.overlap, tta=args.tta)
    print(f"wrote {args.out}")
    for k, v in m.items():
        print(f"  {k}: {v}")
    print("\nCheck the plugin against it with:\n"
          f"  mvn -f plugin/pom.xml test -Dtest=RealModelParityTest "
          f"-Dtrailer.parity={Path(args.out).resolve()}")
    return 0


def cmd_export(args) -> int:
    import json as _json
    from . import model as model_mod

    net, meta = model_mod.load(args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = model_mod.export_onnx(net, args.variant, out, size=args.window,
                                 overlap=args.overlap, tta=args.tta)
    info |= {"checkpoint": str(args.checkpoint), "trained_variants": meta["variants"]}
    (out.with_suffix(".json")).write_text(_json.dumps(info, indent=1))
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB)")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


def cmd_predict(args) -> int:
    import rasterio
    from . import infer
    from . import model as model_mod

    from . import variants as var_mod
    from .data import full_tile

    root = Path(args.root)
    net, meta = model_mod.load(args.checkpoint, model_mod.pick_device(args.device))
    variant = var_mod.get(args.variant or meta["variants"][0])
    for aoi in select(args.aoi, args.role):
        d = root / aoi.slug
        if not (d / "dtm_clean.tif").exists():
            logging.warning("%s not built, skipping", aoi.key)
            continue
        z, canopy, *_ = full_tile(d, variant)
        with rasterio.open(d / "dtm_clean.tif") as s:
            profile = s.profile
        prob = infer.predict(net, z, canopy, variant=variant.key,
                             body_tile=meta.get("crop", 256),
                             device=next(net.parameters()).device,
                             batch=args.batch, tta=args.tta)
        # Predictions land on the body grid, so the transform coarsens with them.
        k = var_mod.BODY_RES / _tile_res([d], 0.5)
        profile.update(count=1, dtype="float32", compress="deflate",
                       predictor=2, height=prob.shape[-2], width=prob.shape[-1],
                       transform=profile["transform"] * rasterio.Affine.scale(k, k))
        out = d / "proposal.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(prob.astype("float32"))
            dst.descriptions = ("trail_probability",)
        print(f"wrote {out}  (mean {prob.mean():.4f}, "
              f"{100 * (prob > 0.5).mean():.2f}% above 0.5)")
    return 0


def main(argv=None) -> int:
    # Shared flags live on a parent parser so they are accepted either side of
    # the subcommand -- `trailer build --aoi x` and `trailer --aoi x build`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=str(DEFAULT_ROOT), help="output directory")
    common.add_argument("--aoi", default="all",
                        help="comma-separated AOI keys, or 'all'")
    common.add_argument("--role", default=None,
                        choices=["train", "harvest", "eval", "control"])
    common.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(prog="trailer", parents=[common],
                                description="LiDAR trail-detection data pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("survey", parents=[common],
                   help="show coverage without downloading").set_defaults(fn=cmd_survey)

    b = sub.add_parser("build", parents=[common],
                       help="extract point clouds, rasters and labels")
    b.add_argument("--res", type=float, default=0.5,
                   help="pixel size in metres (default 0.5; 0.25 is supported "
                        "by ground density but ~4x slower)")
    b.add_argument("--force", action="store_true", help="rebuild cached tiles")
    b.add_argument("--evict-points", action="store_true",
                   help="delete points.laz once features are built "
                        "(~420 MB -> ~78 MB per tile; rebuild needs re-download)")
    b.add_argument("--min-free-gb", type=float, default=5.0,
                   help="stop the run when free disk drops below this")
    b.set_defaults(fn=cmd_build)

    h = sub.add_parser("harvest", parents=[common],
                       help="auto-select tiles by faint/abandoned trail density")
    h.add_argument("--limit", type=int, default=60,
                   help="maximum tiles to select")
    h.add_argument("--min-m", type=float, default=400.0,
                   help="minimum metres of faint/lifecycle way per tile")
    h.add_argument("--size", type=int, default=1000, help="grid cell size in m")
    h.add_argument("--bbox", default=None,
                   help="south,west,north,east (default: High Sierra)")
    h.add_argument("--registry", default="data/harvest.json")
    h.add_argument("--refresh", action="store_true", help="re-query Overpass")
    h.set_defaults(fn=cmd_harvest)

    v = sub.add_parser("vet", parents=[common],
                       help="check harvested tiles for data quality")
    v.add_argument("--registry", default="data/harvest.json")
    v.add_argument("--apply", action="store_true",
                   help="drop rejected tiles from the registry")
    v.add_argument("--prune", action="store_true",
                   help="with --apply, also delete their files")
    v.set_defaults(fn=cmd_vet)

    sub.add_parser("qa", parents=[common],
                   help="measure tread signal per tile").set_defaults(fn=cmd_qa)
    sub.add_parser("preview", parents=[common],
                   help="render hillshade + label overlay").set_defaults(fn=cmd_preview)

    t = sub.add_parser("train", parents=[common],
                       help="train the segmentation model")
    t.add_argument("--out", default="runs/latest", help="checkpoint directory")
    t.add_argument("--arch", default="unet",
                   choices=["unet", "unetpp", "deeplabv3p"])
    t.add_argument("--encoder", default="resnet34")
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--crop", type=int, default=256,
                   help="crop size in body pixels (256 @ 1 m = 256 m of "
                        "context); a 0.5 m variant reads twice this many")
    t.add_argument("--variants", default=None,
                   help="comma-separated input variants to train jointly "
                        "(default lidar05,dem1)")
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--amp", default="bf16", choices=["off", "fp16", "bf16"],
                   help="mixed precision on cuda or mps; bf16 keeps fp32's "
                        "exponent range, which the NaN nodata sentinel and the "
                        "clipped derivative maths both care about")
    t.add_argument("--resume", default=None, metavar="CKPT",
                   help="continue from a checkpoint. last.pt restores the "
                        "optimiser and LR schedule too; a weights-only file "
                        "(best.pt from an older run) is a warm start and says "
                        "so")
    t.add_argument("--samples", type=int, default=2000,
                   help="crops drawn per epoch")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--pos-weight", type=float, default=8.0,
                   # ArgumentDefaultsHelpFormatter %-formats help strings, so a
                   # literal percent sign here crashes `train --help`.
                   help="BCE positive class weight; trails are ~0.7%% of pixels")
    t.add_argument("--cldice", type=float, default=0.5,
                   help="clDice loss weight (0 disables the topology term)")
    t.add_argument("--cldice-warmup", type=int, default=3,
                   help="epochs before clDice is ramped in")
    t.add_argument("--noise-m", type=float, default=0.05,
                   help="max sigma of injected elevation noise in METRES; "
                        "sampled per crop to sweep the tread-to-roughness "
                        "ratio (0 disables)")
    t.add_argument("--noise-band-m", type=float, default=0.05,
                   help="max sigma of spatially correlated elevation noise on a "
                        "4-8 m grid; white noise alone barely reaches the "
                        "bench-scale band the model keys on (0 disables)")
    t.add_argument("--tolerance-m", type=float, default=5.0,
                   help="relax the region losses at this radius, matching the "
                        "scoring tolerance; 0 trains pixel-exact")
    t.add_argument("--jitter-m", type=float, default=2.0,
                   help="random rigid label offset per crop, modelling the "
                        "imagery-derived misalignment in OSM geometry")
    t.add_argument("--canopy-dropout", type=float, default=0.15,
                   help="probability of withholding the chm/vdi canopy bands "
                        "from a canopy-bearing variant")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--device", default=None, help="cuda / mps / cpu")
    t.add_argument("--tta", action="store_true",
                   help="D4 test-time augmentation for held-out scoring")
    t.add_argument("--res", type=float, default=0.5,
                   help="fallback pixel size if manifests are unreadable")
    t.set_defaults(fn=cmd_train)

    pr = sub.add_parser("predict", parents=[common],
                        help="write a probability raster per tile")
    pr.add_argument("--checkpoint", default="runs/latest/best.pt")
    pr.add_argument("--variant", default=None,
                    help="input variant to run (default: the checkpoint's first)")
    pr.add_argument("--batch", type=int, default=8)
    pr.add_argument("--device", default=None)
    pr.add_argument("--tta", action="store_true")
    pr.set_defaults(fn=cmd_predict)

    ex = sub.add_parser("export", parents=[common],
                        help="freeze one variant to ONNX for the JOSM plugin")
    ex.add_argument("--checkpoint", default="runs/latest/best.pt")
    ex.add_argument("--variant", default="dem1",
                    help="must be a bare-earth variant; no elevation service "
                         "provides canopy")
    ex.add_argument("--out", default="runs/latest/trailer.onnx")
    ex.add_argument("--window", type=int, default=256,
                    help="fixed body window in pixels; must be divisible by 32")
    ex.add_argument("--overlap", type=float, default=0.5,
                    help="window overlap the sidecar will tell the plugin to use")
    ex.add_argument("--tta", action="store_true",
                   help="bake the 8-fold D4 average into the graph: better, at "
                        "8x inference cost, and not switchable afterwards")
    ex.set_defaults(fn=cmd_export)

    pa = sub.add_parser("parity", parents=[common],
                        help="full-scale parity fixture: the real model on real "
                             "elevation, for the plugin's opt-in test")
    pa.add_argument("--checkpoint", default="runs/latest/best.pt")
    pa.add_argument("--tile", required=True,
                    help="a built tile directory, e.g. data/tiles/abandoned_south")
    pa.add_argument("--out", required=True,
                    help="output directory; keep it outside the repo, it holds "
                         "a ~99 MB .onnx")
    pa.add_argument("--variant", default="dem1")
    pa.add_argument("--window", type=int, default=256)
    pa.add_argument("--max-px", type=int, default=1024,
                    help="crop the tile to this square, so the check stays "
                         "minutes rather than hours")
    pa.add_argument("--overlap", type=float, default=0.5)
    pa.add_argument("--tta", action="store_true")
    pa.set_defaults(fn=cmd_parity)

    g = sub.add_parser("golden", parents=[common],
                       help="regenerate the JOSM plugin's tiler test fixtures")
    g.add_argument("--out", default=str(Path("plugin/src/test/resources/golden.json")),
                   help="where to write the fixture JSON")
    g.set_defaults(fn=cmd_golden)

    dm = sub.add_parser("dem", parents=[common],
                        help="fetch the published USGS 3DEP 1 m DEM per tile")
    dm.add_argument("--force", action="store_true",
                    help="refetch tiles that already have dem1m.tif")
    dm.set_defaults(fn=cmd_dem)

    rl = sub.add_parser("relabel", parents=[common],
                        help="rebuild labels.tif from cached OSM (no downloads)")
    rl.set_defaults(fn=cmd_relabel)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
