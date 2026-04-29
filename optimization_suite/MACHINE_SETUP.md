# Machine Setup Guide

This guide explains how to configure the optimization suite for a new machine — whether a SLURM cluster or a local Ubuntu workstation. You only ever need to edit **two files**.

---

## File layout (relevant files only)

```
optimization_suite/
├── suite_paths.env        ← EDIT THIS: one variable per machine
├── start_py_311.sh        ← EDIT THIS: Python path for SLURM only
├── launch_local.sh        ← run experiments locally (no SLURM)
├── launch_all.sh          ← run experiments on SLURM
├── MACHINE_SETUP.md       ← this file
└── experiments/
    └── <exp>/
        ├── config.env     ← per-experiment settings (paths auto-resolved)
        └── run_experiment.sh
```

---

## Step 1 — Edit `suite_paths.env`

This is the **only place** with machine-specific paths. Everything else derives from it.

```bash
# suite_paths.env

SCRATCH_BASE="/scratch/nas/4/nasser"    # cluster example
# SCRATCH_BASE="/home/myuser/work"      # local Ubuntu example

STG_TMP_BASE="${SCRATCH_BASE}/tmp"      # cluster: scratch tmp (NFS-safe)
# STG_TMP_BASE="/tmp"                   # local: fast local tmp
```

| Variable | Cluster | Local Ubuntu |
|---|---|---|
| `SCRATCH_BASE` | Root of your scratch space | Any writable directory |
| `STG_TMP_BASE` | `${SCRATCH_BASE}/tmp` | `/tmp` |

---

## Step 2 — Configure `start_py_311.sh` (SLURM only)

`start_py_311.sh` is **sourced automatically** by every experiment. It detects whether it is running under SLURM or locally and behaves accordingly.

### SLURM cluster

The script expects a Python 3.11 installation at:

```
${SCRATCH_BASE}/python311/python/bin/python
```

If your cluster uses a different layout, update **only the SLURM block** inside `start_py_311.sh`:

```bash
# ── cluster-specific block (inside start_py_311.sh) ─────────────────────────
_PY311_BIN="${SCRATCH_BASE}/python311/python/bin"   # ← change this path
export PATH="${_PY311_BIN}:${PATH}"
export ASTRA_SIM_PYTHON="${_PY311_BIN}/python"
```

Common alternatives:

```bash
# conda environment on cluster
_PY311_BIN="${SCRATCH_BASE}/miniconda3/envs/astra/bin"

# module system (add before the block)
module load python/3.11

# pyenv on cluster
_PY311_BIN="${HOME}/.pyenv/versions/3.11.11/bin"
```

### Local Ubuntu machine

**No changes needed.** When `SLURM_JOB_ID` is not set, the script:
- Skips the fake HOME, custom PATH, and SSH agent setup
- Uses whatever Python is already active in your shell (venv, conda, pyenv, etc.)
- Only sets `ASTRA_SIM_PYTHON` if it isn't already exported

To use a specific Python on local:

```bash
# Option A — activate your venv before running (recommended)
source /path/to/astraenv/bin/activate
bash launch_local.sh

# Option B — export explicitly before running
export ASTRA_SIM_PYTHON=/path/to/python
bash launch_local.sh
```

---

## Step 3 — Run experiments

### Local Ubuntu

```bash
cd optimos/optimization_suite

# preview what would run
bash launch_local.sh --dry-run

# run selected experiments sequentially
bash launch_local.sh

# run up to 4 in parallel, with 4 workers each
bash launch_local.sh --parallel 4 --workers 4
```

Edit the `ACTIVE_EXPERIMENTS` array at the top of `launch_local.sh` to select which experiments to run.

### SLURM cluster

```bash
cd optimos/optimization_suite

# preview submissions (no sbatch calls)
bash launch_all.sh --dry-run

# submit all active experiments
bash launch_all.sh
```

Edit the `ACTIVE_EXPERIMENTS` array at the top of `launch_all.sh`.

---

## Porting checklist

When moving to a new machine:

- [ ] Edit `suite_paths.env`: set `SCRATCH_BASE` and `STG_TMP_BASE`
- [ ] **SLURM only**: verify/update the `_PY311_BIN` path in the SLURM block of `start_py_311.sh`
- [ ] **SLURM only**: copy or build the astra-sim binaries under `${SCRATCH_BASE}/astra-sim/build/`
- [ ] **Local only**: activate your Python venv before running `launch_local.sh`
- [ ] Run `bash launch_local.sh --dry-run` (or `launch_all.sh --dry-run`) to verify paths resolve correctly

---

## How `start_py_311.sh` is found

Every experiment's `run_experiment.sh` looks for `start_py_311.sh` via:

1. `START_ENV_SCRIPT` in the experiment's `config.env` (relative: `../../start_py_311.sh`)
2. Falls back to `<suite_root>/start_py_311.sh` (auto-resolved from experiment directory)

If neither exists the experiment continues with a warning, using the current shell's Python environment. This means local runs work out of the box even without the file.
