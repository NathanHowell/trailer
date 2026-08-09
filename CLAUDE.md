# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Build & Test

PDAL and GDAL are command-line tools, not Python packages — install them with
the system package manager (`brew install pdal gdal`, or `apt install pdal
gdal-bin` / conda-forge on Linux).

```bash
uv sync                        # data pipeline only
uv sync --extra train          # adds torch, smp, onnx
uv run pytest tests/ -q        # Python tests
mvn -f plugin/pom.xml test     # Kotlin plugin tests
mvn -f plugin/pom.xml package  # shaded jar for JOSM

just retrain                   # full training run
just report                    # what the last run selected, per class and variant
```

`uv.lock` resolves for both platforms: the CUDA wheels, NCCL and Triton carry
`platform_machine == 'x86_64' and sys_platform == 'linux'` markers, so the same
lock gives MPS on an Apple machine and CUDA on a Linux box with no edits.
`model.pick_device` prefers CUDA, then MPS, then CPU.

Training state moves between machines: `runs/<name>/last.pt` is written every
epoch with the optimiser, scheduler and scaler alongside the weights, and
`trailer train --resume <ckpt>` restores all of it. A checkpoint holding weights
only is a *warm start* — moments reset, the LR schedule restarts — and the log
says so rather than pretending it resumed. `data/` and `runs/` are gitignored
and large (6 GB and 1 GB), so moving a workspace means `rsync`, not `git clone`.

## Architecture Overview

Detect hiking trails in bare-earth LiDAR terrain and show a **probability
heatmap** a human traces over in JOSM. Never vectors, never an import — OSM has
well-earned opinions about machine-generated geometry.

- `src/trailer/` — data pipeline (3DEP fetch, PDAL rasterise, OSM labels),
  training, and ONNX export. `cli.py` is the entry point for every subcommand.
- `plugin/` — the JOSM plugin in Kotlin, running the exported graph through
  onnxruntime's Java API.

The model is a ResNet-34 U-Net with **per-variant stems feeding one shared
trunk** at `BODY_RES = 1.0` m. A variant is a pixel size plus whether canopy
bands exist; a 0.5 m stem is stride-2 so it can encode the tread before
decimating. Only `dem1` (1 m, bare earth) exports, because 3DEP cannot supply
canopy — so **`dem1`'s per-class F1 is the go/no-go signal, not the blend**.

Terrain derivatives are torch layers *inside* the graph, so an exported `.onnx`
is self-contained: the plugin feeds it one float32 DEM tile and gets
probabilities back.

See `README.md` for the survey findings, feature bands, label taxonomy and
deployment detail.

## Conventions & Patterns

**The Python/Kotlin boundary.** Anything the plugin would otherwise reimplement
goes behind the ONNX boundary one of two ways. *Algorithms go into the graph* —
terrain derivatives, the D4 test-time average, the Hann taper (emitted as a
second output). *Numbers go into the export sidecar as numbers, never prose* —
stride, overlap, `step_px`, `pad_mode`. `ModelSpec.kt` **validates rather than
defaults**: a missing field is a version mismatch, not a reason to guess.

**Model selection is stratified, never pooled.** The score is the mean over
(variant × visibility class) of best-threshold relaxed F1 at the 5 m tolerance.
Pooled recall is length-weighted and would let faint trail — the thing this
project exists for — disappear into an average.

**Writes that could be interrupted go through `atomic.staged`** — temp file in
the destination directory, then `os.replace`. A partial download must never
become a trusted cache.

**Watch a test fail before trusting it.** Mutate the code it covers and confirm
it bites; several tests here previously asserted their own arithmetic and agreed
with themselves.

**Never hand-write beads IDs.** Create with `bd create ... --silent`, capture
the returned ID, use that. See `bd prime` for the rest.

**Commits do not carry `Co-Authored-By` lines.** Signing stays on — if the agent
refuses, stop and ask, never `--no-gpg-sign`.
