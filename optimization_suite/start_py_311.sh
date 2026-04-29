#!/usr/bin/env bash
# =============================================================================
# start_py_311.sh — Portable environment setup for running optimization experiments.
#
# Sourced by run_experiment.sh before launching the sweep.
# Works in two modes detected automatically:
#
#   SLURM mode  (SLURM_JOB_ID is set)
#     • Adds the cluster Python 3.11 installation to PATH
#     • Sets a fake HOME to avoid NFS permission issues on compute nodes
#     • Sets PIP_CACHE_DIR on scratch
#
#   Local mode  (no SLURM_JOB_ID)
#     • Assumes the correct venv/conda is already active in the shell
#     • Only fills in variables that are not already set
#     • Skips cluster-specific setup (fake HOME, SSH agent, etc.)
#
# To configure for a new machine, edit suite_paths.env next to this file.
# See MACHINE_SETUP.md for full instructions.
# =============================================================================

# Locate suite_paths.env next to this script so every path variable is
# available regardless of the caller's working directory.
_START_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_START_SCRIPT_DIR/suite_paths.env" ]]; then
  # shellcheck disable=SC1091
  source "$_START_SCRIPT_DIR/suite_paths.env"
else
  echo "Warning: suite_paths.env not found at $_START_SCRIPT_DIR — SCRATCH_BASE may be unset." >&2
fi

# ---------------------------------------------------------------------------
# Temp directories (both modes)
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" && -w "${SLURM_TMPDIR}" ]]; then
  _LOCAL_TMP_BASE="${SLURM_TMPDIR}"
elif [[ -d "/tmp" && -w "/tmp" ]]; then
  _LOCAL_TMP_BASE="/tmp/${USER:-user}"
else
  _LOCAL_TMP_BASE="${SCRATCH_BASE}/tmp"
fi

export TMPDIR="${_LOCAL_TMP_BASE}/astra_tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export GIT_TMPDIR="$TMPDIR"
export JOBLIB_TEMP_FOLDER="$TMPDIR"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# AstraSim paths (both modes)
# Honour any values already exported by the calling shell (e.g. venv activation).
# ---------------------------------------------------------------------------
if [[ -z "${ASTRA_SIM_ROOT:-}" ]]; then
  export ASTRA_SIM_ROOT="${SCRATCH_BASE}/astra-sim"
fi

export ASTRA_SIM="${ASTRA_SIM_ROOT}"
export ASTRA_SIM_BIN_AWARE="${ASTRA_SIM_ROOT}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
export ASTRA_SIM_BIN_UNAWARE="${ASTRA_SIM_ROOT}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
export G2_SIM_BIN="${ASTRA_SIM_ROOT}/build/astra_g2/build/bin/AstraSim_G2_congestion"

# ---------------------------------------------------------------------------
# SLURM-specific setup
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then

  # Fake home: avoids NFS permission / dotfile errors on compute nodes.
  export HOME="${SCRATCH_BASE}/fake_home"
  mkdir -p "$HOME"

  # Cluster Python 3.11 — prepend to PATH.
  _PY311_BIN="${SCRATCH_BASE}/python311/python/bin"
  export PATH="${_PY311_BIN}:${PATH}"
  export ASTRA_SIM_PYTHON="${_PY311_BIN}/python"
  if [[ ! -x "$ASTRA_SIM_PYTHON" ]]; then
    echo "Error: cluster Python not found: $ASTRA_SIM_PYTHON" >&2
    echo "       Check SCRATCH_BASE in suite_paths.env and that python311 is installed." >&2
    exit 1
  fi

  # pip cache on scratch (avoids filling up the home quota).
  export PIP_CACHE_DIR="${SCRATCH_BASE}/pip_cache"
  mkdir -p "$PIP_CACHE_DIR"

  # SSH agent for git access on compute nodes (soft-fail if key absent).
  if command -v ssh-agent >/dev/null 2>&1; then
    eval "$(ssh-agent -s)" 2>/dev/null || true
    ssh-add "${HOME}/../.ssh/iid_ed25519" 2>/dev/null || true
    export GIT_SSH_COMMAND="ssh -i ${HOME}/../.ssh/iid_ed25519"
  fi

# ---------------------------------------------------------------------------
# Local setup
# ---------------------------------------------------------------------------
else

  # On a local machine the correct venv/conda should already be active.
  # We only set ASTRA_SIM_PYTHON if it was not already exported.
  if [[ -z "${ASTRA_SIM_PYTHON:-}" ]]; then
    _FOUND_PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    if [[ -n "$_FOUND_PY" ]]; then
      export ASTRA_SIM_PYTHON="$_FOUND_PY"
    else
      echo "Warning: no Python found in PATH for local run." >&2
      echo "         Activate your venv before running, or set ASTRA_SIM_PYTHON manually." >&2
    fi
  fi

fi
