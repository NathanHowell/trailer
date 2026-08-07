"""Command-line driver.

    uv run trailer survey                    # coverage + density, no downloads
    uv run trailer build --aoi all           # full data build
    uv run trailer build --aoi giant_forest,colby_pass --res 0.25
    uv run trailer qa                        # tread signal per tile
    uv run trailer preview --aoi moraine_lake
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


def cmd_build(args) -> int:
    root = Path(args.root)
    aois = select(args.aoi, args.role)
    logging.info("building %d AOI(s) at %.2f m into %s", len(aois), args.res, root)
    results = build_mod.build_all(aois, root, res=args.res, force=args.force)
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


def main(argv=None) -> int:
    # Shared flags live on a parent parser so they are accepted either side of
    # the subcommand -- `trailer build --aoi x` and `trailer --aoi x build`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=str(DEFAULT_ROOT), help="output directory")
    common.add_argument("--aoi", default="all",
                        help="comma-separated AOI keys, or 'all'")
    common.add_argument("--role", default=None,
                        choices=["train", "eval", "control"])
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
    b.set_defaults(fn=cmd_build)

    sub.add_parser("qa", parents=[common],
                   help="measure tread signal per tile").set_defaults(fn=cmd_qa)
    sub.add_parser("preview", parents=[common],
                   help="render hillshade + label overlay").set_defaults(fn=cmd_preview)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
