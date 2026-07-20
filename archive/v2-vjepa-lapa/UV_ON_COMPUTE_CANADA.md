# Using `uv` on Compute Canada (Alliance)

How to run a `uv`-managed project on an Alliance cluster (Rorqual, Fir, …). This
is the `uv` alternative to the CC-native `pip install --no-index` + wheelhouse
route. Use it when you want `uv.lock` reproducibility, or a package that isn't in
CC's wheelhouse.

The one fact that shapes everything: **compute nodes have no internet; login
nodes do.** `$HOME`, `/project`, and `/scratch` are shared network filesystems
visible from both. So you **build the environment once on a login node** and the
job only *activates* it — nothing downloads at runtime.

---

## TL;DR

```bash
# ---- on a LOGIN node (has internet) ----
module load StdEnv/2023 python/3.11
curl -LsSf https://astral.sh/uv/install.sh | sh        # once, if uv not installed
export PATH="$HOME/.local/bin:$PATH"                   # put this in ~/.bashrc

cd ~/myproject
uv venv --python "$(which python)" .venv               # CC's python, uv-managed venv
uv sync                                                # from pyproject + uv.lock
#   (or ad-hoc:  uv pip install -e .)

# pre-fetch anything the job would otherwise download at runtime (HF weights,
# datasets) into $HOME or /project, and point your config at the local paths.

# ---- in the SLURM script (compute node, NO internet) ----
source ~/myproject/.venv/bin/activate                  # do NOT call `uv` in the job
export HF_HUB_OFFLINE=1
python run.py ...
```

---

## Step 1 — get `uv` (login node, once)

`uv` is **not** a CC module; install it yourself into `$HOME`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # -> ~/.local/bin/uv
# or:  module load python && pip install --user uv
export PATH="$HOME/.local/bin:$PATH"                   # add to ~/.bashrc
```

## Step 2 — create the venv (login node)

Two choices for the interpreter:

- **CC's module Python (recommended)** — stays on the cluster's supported Python;
  uv only manages the venv and packages:
  ```bash
  module load python/3.11
  uv venv --python "$(which python)" .venv
  ```
- **uv's own standalone Python** — `uv venv .venv`. uv downloads a CPython into
  `~/.local/share/uv/`. Works (that path is on shared `$HOME`, so the compute node
  can run it), but you're fully off the module system.

Keep the venv in `$HOME` or `/project`, **not `/scratch`** (scratch is purged;
files untouched for ~60 days are deleted).

## Step 3 — install dependencies (login node, uses internet)

- **Reproducible (preferred):** with a `[project]` table in `pyproject.toml`,
  ```bash
  uv lock          # resolve -> writes uv.lock
  uv sync          # install EXACTLY what's in uv.lock
  ```
- **Ad-hoc:** `uv pip install -e .` / `uv pip install torch transformers av ...`

Because uv pulls from PyPI, "not in CC's wheelhouse" is simply not a category that
exists here — you just get the package.

## Step 4 — pre-stage anything the job would download (login node)

The job is offline, so fetch runtime downloads **now**:

```bash
# example: a HuggingFace model
huggingface-cli download facebook/vjepa2-vitg-fpc64-384 \
    --local-dir /project/$USER/models/vjepa2-vitg-fpc64-384
```

Point your config at the local dir, and set the offline switches in the job so no
library tries to phone home:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## Step 5 — the SLURM script

Just source the venv and run. **Do not invoke `uv` inside the job** — `uv run` /
`uv sync` may try to reach the network, and there isn't any. Sourcing the venv
sidesteps that entirely.

```bash
#!/bin/bash
#SBATCH --account=def-xxxxx
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

module load StdEnv/2023 cuda            # cuda only if you compile against system CUDA
source ~/myproject/.venv/bin/activate   # the venv built on the login node
export HF_HUB_OFFLINE=1
python run.py --config configs/config.yaml
```

---

## GPU / CUDA note

- PyPI `torch` **bundles its own CUDA runtime**, so a GPU job usually runs without
  `module load cuda`. You only need the CUDA module if you compile CUDA extensions
  yourself — then match the versions.
- That bundled runtime must be compatible with the node's NVIDIA driver. Normally
  fine (driver forward-compat); occasionally a surprise on older drivers.

## Caveats (why CC steers you to their wheelhouse)

- **Filesystem load.** A venv plus uv's cache are tens of thousands of small files
  in `$HOME`; the Lustre metadata server dislikes that. Don't spray dozens of
  venvs around. `UV_CACHE_DIR` defaults to `~/.cache/uv` (fine — same filesystem
  as the venv, so uv hardlinks instead of copying).
- **Not CC-optimized.** PyPI wheels aren't tuned to the cluster's CPU/CUDA/MPI.
  Fine for torch/transformers/av; matters more for numpy/scipy/mpi4py-heavy or
  multi-node MPI work — prefer CC's wheelhouse there.
- **Off the supported path.** CC support's first answer is "use
  `pip install --no-index`." Native-library breakage is yours to debug.

## When to use CC's wheelhouse instead

- Multi-node MPI, or heavy numpy/scipy where their MKL/OpenBLAS build matters.
- You don't need `uv.lock` reproducibility and every dep shows up in
  `avail_wheels <name>`.
- Pattern:
  ```bash
  module load StdEnv/2023 python/3.11
  virtualenv --no-download ~/env && source ~/env/bin/activate
  pip install --no-index torch transformers av numpy ...
  pip install -e . --no-deps
  ```

## Hybrid fallback

If one package won't build or run from PyPI, pull just that one from the
wheelhouse into the same venv: `pip install --no-index <that-pkg>`. Mixing works,
though it muddies the `uv.lock` reproducibility story.
