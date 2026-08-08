# trailer -- LiDAR trail detection for OpenStreetMap
#
# Recipes are thin wrappers around `trailer <subcommand>`. They exist to pin the
# arguments a run should use, so a training run is reproducible from its name
# rather than from someone's shell history.

root := "data/tiles"
run  := "runs/full"

# Training run shape. Overridable: `just epochs=10 retrain`.
epochs   := "40"
samples  := "2000"
crop     := "256"
batch    := "8"
workers  := "4"
variants := "lidar05,dem1"

_default:
    @just --list

# ---------------------------------------------------------------- tests

# Python and Kotlin suites.
test: test-py test-plugin

# Python tests.
test-py:
    uv run --extra train pytest -q

# Regenerates the tiler fixtures from Python before running, which is the point
# of the step -- it is the only guard against the Kotlin tiler and trailer.infer
# drifting apart.

# Kotlin plugin tests, including tiler parity against Python.
test-plugin:
    cd plugin && mvn -q test

# ---------------------------------------------------------------- training

# Class-balanced labels, 1 m variants reading the published USGS DEM, stratified
# selection. Checks the corpus first, because the two cheapest ways to waste this
# run are visible before the first step: a vet-rejected tile still in the
# training set, or a variant whose validation band is missing a whole class.
#
# COST: measured at ~44 min/epoch on MPS (M-series, both variants, crop 256,
# batch 8), so the 40-epoch default is roughly 29 HOURS plus a TTA held-out pass.
# Halving `samples` to 1000 is usually the better trade than halving `epochs`:
# same total crops, but twice as many checkpoints for selection to choose from,
# and selection is now stratified enough to be worth giving choices to.
#     just samples=1000 retrain

# Full retrain on the built corpus. ~29 h at defaults -- read the cost note.
retrain: check-corpus
    uv run --extra train trailer train \
        --root {{root}} --out {{run}} \
        --epochs {{epochs}} --samples {{samples}} --crop {{crop}} \
        --batch {{batch}} --workers {{workers}} --variants {{variants}} \
        --tta

# Two short epochs through the whole loop, to check a code change end to end.
retrain-smoke:
    uv run --extra train trailer train \
        --root {{root}} --out runs/smoke \
        --epochs 2 --samples 128 --crop {{crop}} \
        --batch {{batch}} --workers {{workers}} --variants {{variants}}

# What the last run selected, per class and per variant.
report path=(run / "report.json"):
    #!/usr/bin/env bash
    set -euo pipefail
    uv run python - "{{path}}" <<'PY'
    import json, sys
    r = json.load(open(sys.argv[1]))
    print("best stratified F1:", r["best_val_stratified_f1"])
    h = r["history"][-1]
    print(f"epochs run: {len(r['history'])}  last epoch score: {h['score']}")
    for v, s in h["val"].items():
        by = s["strat"]["by_class"]
        print("  ", v, " ".join(f"{c}={by[c]['f1']:.3f}@t{by[c]['t']}"
                                for c in s["strat"]["classes"]))
    print("held out:")
    for v, tiles in r["held_out"].items():
        for name, rec in tiles.items():
            by = rec["strat"]["by_class"]
            print(f"  {v:8s} {name:18s} fp={rec['fp_rate@0.5']:.5f} " +
                  " ".join(f"{c}={by[c]['f1']:.3f}"
                           for c in rec["strat"]["classes"]))
    PY

# ---------------------------------------------------------------- corpus

# Refuse to start a long run on a corpus with a known defect.
check-corpus:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run python - <<'PY'
    import json, logging, sys
    from pathlib import Path
    logging.disable(logging.WARNING)
    from trailer import variants as var_mod, osm
    from trailer.cli import _built
    from trailer.data import TileDataset

    root = Path("{{root}}")
    problems = []

    # A tile vet rejected is still on disk until `just vet-apply` runs, and the
    # training set reads the registry, not the verdicts.
    vet = root / "vet.json"
    if vet.exists():
        built = {p.parent.name for p in root.glob("*/labels.tif")}
        bad = [r["key"] for r in json.loads(vet.read_text())
               if not r.get("accepted") and r["key"] in built]
        if bad:
            problems.append(f"vet-rejected tiles still in the corpus: {bad} "
                            "-- run `just vet-apply`")

    train_dirs = _built(root, "train", "harvest")
    if not train_dirs:
        problems.append("no built training tiles")
    for key in "{{variants}}".split(","):
        v = var_mod.get(key)
        for split in ("train", "val"):
            try:
                ds = TileDataset(train_dirs, v, body_crop={{crop}}, split=split,
                                 samples=8, augment=False)
            except ValueError as exc:
                problems.append(f"{key}/{split}: {exc}")
                continue
            missing = [c for c in osm.CLASS_CODE if c not in ds._tiles_with]
            if missing:
                # Selection drops absent classes rather than scoring them zero,
                # so this does not crash -- it silently narrows what the run is
                # optimised for, which is worse.
                problems.append(f"{key}/{split} has no {'/'.join(missing)} "
                                "centres; that class would vanish from selection")
    if problems:
        print("corpus check failed:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"corpus ok: {len(train_dirs)} training tiles, all classes present")
    PY

# Destructive, so it is never a dependency of another recipe. Add --prune by hand
# to delete the tiles' files as well as their registry entries.

# Drop vet-rejected tiles from the harvest registry.
vet-apply:
    uv run trailer vet --root {{root}} --apply

# Fetch any missing published 1 m DEMs. Safe to re-run; existing files are kept.
dem:
    uv run trailer dem --root {{root}}

# Rebuild labels.tif from cached OSM. No network, no point-cloud work.
relabel:
    uv run --extra train trailer relabel --root {{root}}

# What is on disk, per artifact.
status:
    #!/usr/bin/env bash
    set -euo pipefail
    n=$(ls -d {{root}}/*/ 2>/dev/null | wc -l | tr -d ' ')
    echo "$n tile directories in {{root}}"
    for f in dtm_clean.tif features.tif labels.tif dem1m.tif; do
        have=$(ls {{root}}/*/$f 2>/dev/null | wc -l | tr -d ' ')
        printf '  %-16s %s/%s\n' "$f" "$have" "$n"
    done
    du -sh {{root}} 2>/dev/null || true

# ---------------------------------------------------------------- shipping

# Freeze a variant to ONNX for the plugin. dem1 is what a JOSM user gets.
export variant="dem1" checkpoint=(run / "best.pt"):
    uv run --extra train trailer export --checkpoint {{checkpoint}} --variant {{variant}}

# Probability rasters for the held-out tiles, to look at before shipping.
predict checkpoint=(run / "best.pt"):
    uv run --extra train trailer predict --root {{root}} \
        --checkpoint {{checkpoint}} --role eval

# Build the shaded JOSM plugin jar.
plugin:
    cd plugin && mvn -q package
