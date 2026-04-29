#!/usr/bin/env bash
# =============================================================================
# run_sweep_combinations.sh
#
# Runs every combination of tracker / patience / workers settings by calling
# launch_local.sh (or launch_all.sh for SLURM) once per combination.
#
# ── Quick examples ────────────────────────────────────────────────────────────
#
# Run all combinations locally (sequential):
#   bash run_sweep_combinations.sh
#
# Run all combinations locally, 2 experiments in parallel per combination:
#   bash run_sweep_combinations.sh --parallel 2
#
# Submit all combinations to SLURM:
#   bash run_sweep_combinations.sh --slurm
#
# Preview all combinations without running anything:
#   bash run_sweep_combinations.sh --dry-run
#   bash run_sweep_combinations.sh --slurm --dry-run
#
# ── Run a single configuration directly (bypass this script) ─────────────────
#
# Local — tracker ON, patience 60, 6 workers, 2 experiments in parallel:
#   bash launch_local.sh --tracker true --patience 60 --workers 6 --parallel 2
#
# Local — tracker OFF, no patience override, 1 worker:
#   bash launch_local.sh --tracker false --workers 1
#
# SLURM — tracker ON, patience 60, 6 CPUs per job:
#   bash launch_all.sh --tracker true --patience 60 --workers 6
#
# SLURM — dry-run preview of what would be submitted:
#   bash launch_all.sh --tracker true --patience 60 --workers 6 --dry-run
#
# =============================================================================
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# COMBINATION SPACE — edit these arrays to define the sweep
# =============================================================================

# Tracker: enable or disable DeepHyper's early-termination tracker.
TRACKER_VALUES=(
  "true"
  "false"
)

# Patience: early-stopping patience value passed to the optimizer.
#   Positive integer → use that value.
#   -1               → keep each experiment's own setting (no override).
PATIENCE_VALUES=(
  -1     # use experiment config
  60
)

# Workers: number of parallel evaluation workers / CPUs per SLURM job.
# No empty string here — always specify an explicit value.
WORKERS_VALUES=(
  1
  6
)

# =============================================================================
# OPTIONS
# =============================================================================

USE_SLURM=0
DRY_RUN=0
PARALLEL=1   # parallel experiments per launch call (local only)

usage() {
  cat <<USAGE
Usage: bash run_sweep_combinations.sh [--slurm] [--dry-run] [--parallel N]

Options:
  --slurm        Use launch_all.sh (SLURM) instead of launch_local.sh.
  --dry-run      Pass --dry-run to the launcher (preview only, no execution).
  --parallel N   Parallel experiments per combination run (local mode only, default: 1).

Edit the TRACKER_VALUES / PATIENCE_VALUES / WORKERS_VALUES arrays at the top
of this script to define the combination space.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slurm)    USE_SLURM=1;              shift ;;
    --dry-run)  DRY_RUN=1;               shift ;;
    --parallel) PARALLEL="${2:-1}";      shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$USE_SLURM" -eq 1 ]]; then
  LAUNCHER="$SUITE_DIR/launch_all.sh"
else
  LAUNCHER="$SUITE_DIR/launch_local.sh"
fi

# Build base flags
BASE_FLAGS=()
[[ "$DRY_RUN"  -eq 1  ]] && BASE_FLAGS+=(--dry-run)
[[ "$USE_SLURM" -eq 0 ]] && BASE_FLAGS+=(--parallel "$PARALLEL")

# Count total combinations
total=$(( ${#TRACKER_VALUES[@]} * ${#PATIENCE_VALUES[@]} * ${#WORKERS_VALUES[@]} ))
current=0

echo "============================================================"
echo "Sweep Combinations Runner"
echo "============================================================"
echo "Launcher  : $LAUNCHER"
echo "Tracker   : ${TRACKER_VALUES[*]}"
echo "Patience  : ${PATIENCE_VALUES[*]}"
echo "Workers   : ${WORKERS_VALUES[*]}"
echo "Total runs: $total"
[[ "$DRY_RUN" -eq 1 ]] && echo "[DRY-RUN MODE]"
echo

for tracker in "${TRACKER_VALUES[@]}"; do
  for patience in "${PATIENCE_VALUES[@]}"; do
    for workers in "${WORKERS_VALUES[@]}"; do
      current=$((current + 1))

      # Build per-combination flags
      combo_flags=("${BASE_FLAGS[@]}")
      combo_flags+=(--tracker "$tracker")
      combo_flags+=(--patience "$patience")
      combo_flags+=(--workers "$workers")

      label="tracker=$tracker patience=$patience workers=$workers"
      echo "------------------------------------------------------------"
      echo "[$current/$total] $label"
      echo "  Command: bash $(basename "$LAUNCHER") ${combo_flags[*]}"
      echo "------------------------------------------------------------"

      bash "$LAUNCHER" "${combo_flags[@]}"

      echo "[$current/$total] DONE: $label"
      echo
    done
  done
done

echo "============================================================"
echo "All $total combinations completed."
echo "============================================================"
