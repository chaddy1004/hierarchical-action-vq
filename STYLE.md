# STYLE.md — how I start and build research projects (with an LLM)

A project-agnostic reference. Hand this to the LLM at the start of a new
project: it describes how I scaffold a repo, how code is written, and how we
work together. `<pkg>` = the project's package name. Living document.

---

## 1. Bootstrapping a new project

- `git init` immediately. Commit early and often — before any restructuring,
  snapshot first. Never rely on the working tree as the only copy of anything.
- Package manager: `uv`. `pyproject.toml` at root (hatchling,
  `packages = ["<pkg>"]`) so the package is installable from day one.
- Write `PLAN.md` at the root before any code: the research goal in a few
  sentences, plus **only the immediate 1–2 experiments** with a decision gate
  for each ("if X beats baseline → proceed to …; if not → …"). Defer the long
  lever list to a separate doc or the plan's "explicitly NOT now" section —
  planning too far ahead is how you get lost in the sauce.
- `.gitignore` experiment artifacts (features, checkpoints, results, media)
  from the start.

## 2. Repo layout

```
<pkg>/                  # the single installable package
  configs/config.yaml   # canonical config (all hyperparameters)
  data/
    preprocessing/      # raw data -> cached arrays
    dataset/            # Dataset classes / loaders
  model/
  trainer/
  utils/                # config loader, shared helpers
run_<stage>.py          # thin entry scripts at repo root
main.py                 # full-pipeline entry (when one exists)
PLAN.md
archive/                # dead ends live here, not in the trash
```

- Every package dir has an `__init__.py`.
- Entry points are thin `run_*.py` scripts at the repo root: argparse +
  `logging.basicConfig` + one call into the package. Library modules are
  import-only — no `__main__` blocks or argparse inside `<pkg>/`.
- Files are named by what they do; directories are named by stage.

## 3. Configuration

- **Every hyperparameter lives in `<pkg>/configs/config.yaml`.** No hardcoded
  paths, sizes, thresholds, or model ids in code.
- Configs are plain nested dicts via `yaml.safe_load`. **No Hydra, no
  OmegaConf, no dataclass config objects.** Access is explicit:
  `cfg["features"]["stride"]`.
- CLI flags are only for: the config path, behavioral switches
  (`--overwrite`), and one-off overrides of a config value. A flag is never
  the only home of a setting.
- `device` is a top-level config key.

## 4. Code conventions

- `os.path`, **never** `pathlib`.
- `from __future__ import annotations` + type hints on public functions.
- Module-level `logger = logging.getLogger(__name__)`; `logging.basicConfig`
  only in entry scripts. Progress in tight loops may use `print(..., end="\r")`.
- Classes that do work are config-driven: `__init__(self, config: dict)`.
  Heavy resources (models) load lazily on first use so construction is free.
- Every module opens with a docstring stating: what the stage does, what it
  reads, and the **exact output paths and formats** it writes.
- Reusing code from a previous project: **copy-paste and adapt, never import
  across repos.** Note the provenance in the docstring. Each repo stands alone.
- Minimal dependencies; no frameworks without a stated reason.

## 5. Data & pipeline conventions

- Every expensive stage caches its output to disk and is **idempotent**: skip
  outputs that already exist unless `overwrite=True`, so any stage can be
  re-run or resumed safely.
- Cached arrays get a sidecar `<id>_meta.json` recording shapes, rates, and
  the settings that produced them. Outputs whose content depends on settings
  go in a subdirectory named by those settings (e.g. `clip10_stride10/`).
- One canonical id per data item (e.g. file basename without extension), used
  consistently across every stage.
- **Never load a whole media file into memory** — stream/decode incrementally
  with a bounded buffer.
- Arrays have declared conventions and stick to them (e.g. frames
  `T x H x W x C uint8`, embeddings `(N, D) float32`, normalize before
  similarity). State the convention once, in the docstring of the producer.
- Ground truth stays at annotation resolution; only predictions get quantized
  to the model's grid.

## 6. Experiments

- Experiments are driven by config values (or explicit run-script overrides),
  never by editing library code.
- **A raw metric means nothing without a matched yardstick.** Every headline
  number ships with at least one same-budget baseline (random with matched
  count, copy/no-change prediction, the naive cue the method claims to beat).
- Write the decision gate down before running: what result kills the idea,
  what result advances it.
- Failed directions are archived, not deleted: snapshot commit, then move to
  `archive/<name>/` with the results/writeup that explains why it died.

## 7. Working with the LLM

- Build **one file at a time**; I review each before the next. No multi-file
  code drops unless I explicitly grant "automode" for a push.
- Verify before handing over: at minimum an import test plus a cheap
  functional check of the core logic (no full GPU runs needed for review).
- The LLM asks before: deleting anything, long/expensive runs, and scope
  changes. Everything else it just does.
- When the project feels tangled, the reset move is: archive everything,
  rewrite `PLAN.md` down to the goal + next 1–2 experiments, start from the
  scaffold again.
