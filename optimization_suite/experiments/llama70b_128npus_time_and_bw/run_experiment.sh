#!/usr/bin/env bash
ulimit -c 0
#SBATCH -q large
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/config.env" ]]; then
  EXP_DIR="${SLURM_SUBMIT_DIR}"
else
  EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$EXP_DIR/config.env"

# Load machine-specific paths (edit ../../suite_paths.env to port to another machine)
_SUITE_ENV="$EXP_DIR/../../suite_paths.env"
[[ -f "$_SUITE_ENV" ]] && source "$_SUITE_ENV"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/outputs"

# Resolve core paths robustly so the suite remains valid after moving directories.
DEFAULT_SWEEP_SCRIPT="$(cd "$EXP_DIR/../../../Optimization/examples" && pwd)/example_deephyper_opt_sweep.py"
DEFAULT_START_ENV_SCRIPT="$(cd "$EXP_DIR/../../" && pwd)/start_py_311.sh"
DEFAULT_SEARCH_SPACE_PATH="$EXP_DIR/inputs/search_space.json"
DEFAULT_RESULT_FOLDER_PREFIX="$EXP_DIR/outputs"

if [[ -z "${SWEEP_SCRIPT:-}" || ! -f "$SWEEP_SCRIPT" ]]; then
  SWEEP_SCRIPT="$DEFAULT_SWEEP_SCRIPT"
fi
if [[ -z "${START_ENV_SCRIPT:-}" || ! -f "$START_ENV_SCRIPT" ]]; then
  START_ENV_SCRIPT="$DEFAULT_START_ENV_SCRIPT"
fi
if [[ -z "${SEARCH_SPACE_PATH:-}" || ! -f "$SEARCH_SPACE_PATH" ]]; then
  SEARCH_SPACE_PATH="$DEFAULT_SEARCH_SPACE_PATH"
fi
if [[ -z "${RESULT_FOLDER_PREFIX:-}" ]]; then
  RESULT_FOLDER_PREFIX="$DEFAULT_RESULT_FOLDER_PREFIX"
fi

# Expand relative paths from config.env against EXP_DIR.
if [[ "$SWEEP_SCRIPT" != /* ]]; then
  SWEEP_SCRIPT="$EXP_DIR/$SWEEP_SCRIPT"
fi
if [[ "$START_ENV_SCRIPT" != /* ]]; then
  START_ENV_SCRIPT="$EXP_DIR/$START_ENV_SCRIPT"
fi
if [[ "$SEARCH_SPACE_PATH" != /* ]]; then
  SEARCH_SPACE_PATH="$EXP_DIR/$SEARCH_SPACE_PATH"
fi
if [[ "$RESULT_FOLDER_PREFIX" != /* ]]; then
  RESULT_FOLDER_PREFIX="$EXP_DIR/$RESULT_FOLDER_PREFIX"
fi

if [[ -f "$START_ENV_SCRIPT" ]]; then
  # shellcheck disable=SC1090
  source "$START_ENV_SCRIPT"
else
  echo "Warning: start_py_311.sh not found ($START_ENV_SCRIPT). Using current shell Python environment." >&2
fi
if [[ ! -f "$SWEEP_SCRIPT" ]]; then
  echo "Sweep script not found: $SWEEP_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SEARCH_SPACE_PATH" ]]; then
  echo "Search space file not found: $SEARCH_SPACE_PATH" >&2
  exit 1
fi


if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

# Apply launch-level overrides (exported by launch scripts).
# ENABLE_TRACKER_OVERRIDE : "True" or "False" — overrides experiment config.
# EARLY_STOPPING_PATIENCE_OVERRIDE : integer to override (-1 = Python default); empty = no override.
[[ -n "${ENABLE_TRACKER_OVERRIDE:-}" ]] && ENABLE_TRACKER="$ENABLE_TRACKER_OVERRIDE"
if [[ "${EARLY_STOPPING_PATIENCE_OVERRIDE:-}" =~ ^-?[0-9]+$ ]]; then
  EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE_OVERRIDE"
fi

N_WORKERS_EFFECTIVE="${N_WORKERS_OVERRIDE:-${SLURM_CPUS_PER_TASK:-$N_WORKERS}}"
# --enable-tracker is action="store_true" in Python — only pass the flag when True.
_TRACKER_FLAG=""
[[ "${ENABLE_TRACKER,,}" == "true" ]] && _TRACKER_FLAG="--enable-tracker"

EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-$((N_WORKERS_EFFECTIVE))}"
EARLY_STOPPING_MIN_EVALUATIONS="${EARLY_STOPPING_MIN_EVALUATIONS:-${INIT_SAMPLES}}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_PREFIX="$RESULT_FOLDER_PREFIX/run_${TS}"
mkdir -p "$RUN_PREFIX"

# Keep optimizer artifacts under absolute outputs/, but pass a simulation-safe
# relative prefix to avoid embedding absolute paths into workload/output roots.
UPC_ROOT="${OPTIMOS_ROOT}"
if [[ "$EXP_DIR" == "$UPC_ROOT"/* ]]; then
  EXP_DIR_REL="${EXP_DIR#"$UPC_ROOT"/}"
else
  # Fallback when experiment directory is outside OPTIMOS root.
  EXP_DIR_REL="$(basename "$EXP_DIR")"
fi
SIM_FOLDER_PREFIX="$EXP_DIR_REL/run_${TS}"

LOG_FILE="$EXP_DIR/logs/optimization_${TS}.log"

echo "Starting experiment: $EXP_NAME" | tee -a "$LOG_FILE"
echo "SLURM job: ${SLURM_JOB_ID:-N/A}" | tee -a "$LOG_FILE"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}" | tee -a "$LOG_FILE"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}" | tee -a "$LOG_FILE"
echo "ASTRA_SIM_PYTHON: ${ASTRA_SIM_PYTHON:-N/A}" | tee -a "$LOG_FILE"
echo "which python: $(command -v python || echo N/A)" | tee -a "$LOG_FILE"
echo "python --version: $(python --version 2>&1 || echo N/A)" | tee -a "$LOG_FILE"

# Use node-local temp storage for Python multiprocessing artifacts.
# This avoids NFS .nfs* cleanup races at interpreter shutdown.
TMP_BASE="${SLURM_TMPDIR:-/tmp}"
JOB_TMP_DIR="${TMP_BASE%/}/astra_tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP_DIR"
export TMPDIR="$JOB_TMP_DIR"
export TMP="$JOB_TMP_DIR"
export TEMP="$JOB_TMP_DIR"
export STG_TMP_DIR="${STG_TMP_BASE}/stg_${SLURM_JOB_ID:-$$}"
mkdir -p "$STG_TMP_DIR"
echo "TMPDIR: $TMPDIR" | tee -a "$LOG_FILE"
echo "STG_TMP_DIR: $STG_TMP_DIR" | tee -a "$LOG_FILE"

_EXP_START=$(date +%s)
time python "$SWEEP_SCRIPT" \
  --objective "$OBJECTIVE_KEY" \
  --model-num "$MODEL_NUM" \
  --model-name "$MODEL_NAME" \
  --num-npus "$NUM_NPUS" \
  --network-name "FoldedClos" \
  --budget "$BUDGET" \
  --init-samples "$INIT_SAMPLES" \
  --n-workers "$N_WORKERS_EFFECTIVE" \
  --top-k "$TOP_K" \
  --cleanup-batch-size "$CLEANUP_BATCH_SIZE" \
  --folder-prefix "$SIM_FOLDER_PREFIX" \
  --search-space-path "$SEARCH_SPACE_PATH" \
  --compress-and-clean \
  --include-categories "$INCLUDE_CATEGORIES" \
  ${_TRACKER_FLAG:+$_TRACKER_FLAG} \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --early-stopping-min-evaluations "$EARLY_STOPPING_MIN_EVALUATIONS" \
  2>&1 | tee -a "$LOG_FILE"

_EXP_END=$(date +%s)
_ELAPSED=$(( _EXP_END - _EXP_START ))
printf "Experiment wall time: %02dh %02dm %02ds (%ds total)\n" \
  $(( _ELAPSED/3600 )) $(( (_ELAPSED%3600)/60 )) $(( _ELAPSED%60 )) "$_ELAPSED" \
  | tee -a "$LOG_FILE"

echo "Completed experiment: $EXP_NAME" | tee -a "$LOG_FILE"
