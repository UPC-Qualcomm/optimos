#!/usr/bin/env bash
# launch_local.sh — Run experiments on a plain Ubuntu machine (no SLURM).
#
# Each experiment's run_experiment.sh already handles all SLURM env-var
# fallbacks, so no patching is needed — this script just drives them locally.
#
# Usage:
#   bash launch_local.sh [--dry-run] [--parallel N] [--max-jobs N] [--workers W]
#
# Options:
#   --dry-run         Print what would run without actually running anything.
#   --parallel N      Run up to N experiments concurrently (default: 1).
#   --max-jobs N      Process at most the first N experiments in the manifest.
#   --workers W       Override N_WORKERS inside each experiment (default: from config.env).

set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SUITE_DIR/suite_paths.env"

# For local runs, experiments live next to this script — not on SCRATCH_BASE.
EXPERIMENTS_DIR="$SUITE_DIR/experiments"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="$SUITE_DIR/launch_logs/$TIMESTAMP"
mkdir -p "$LAUNCH_LOG_DIR"

#################################################################################
# EXPERIMENT MANIFEST — Comment out experiments you DON'T want to run
# (Uncommented experiments will be executed locally)
#################################################################################
ACTIVE_EXPERIMENTS=(
  "llama70b_128npus_time"
  "llama70b_128npus_latency_network"
  "gpt60b_128npus_time"
  "gpt60b_128npus_latency_network"
  "llama8b_32npus_time"
  "llama8b_32npus_latency_network"
  #"llama70b_128npus_energy"
  #"llama70b_128npus_edp"
  #"llama70b_128npus_edp_and_bw"
  #"llama70b_128npus_energy_and_time"
  #"llama70b_128npus_memory_and_time"
  #"llama70b_128npus_time_and_throughput_per_energy"
  #"gpt60b_128npus_energy"
  #"gpt60b_128npus_edp"
  #"gpt60b_128npus_edp_and_bw"
  #"gpt60b_128npus_energy_and_time"
  #"gpt60b_128npus_memory_and_time"
  #"gpt60b_128npus_time_and_throughput_per_energy"
  #"llama8b_32npus_energy"
  #"llama8b_32npus_edp"
  #"llama8b_32npus_edp_and_bw"
  #"llama8b_32npus_energy_and_time"
  #"llama8b_32npus_memory_and_time"
  #"llama8b_32npus_time_and_throughput_per_energy"
)

usage() {
  cat <<USAGE
Usage: bash launch_local.sh [OPTIONS]

Options:
  --dry-run           Print what would run without actually running anything.
  --parallel N        Run up to N experiments concurrently (default: 1).
  --max-jobs N        Process at most the first N experiments in the manifest.
  --workers W         Override N_WORKERS for every experiment (default: from config.env).
  --tracker true|false  Override ENABLE_TRACKER for every experiment.
  --patience N        Override EARLY_STOPPING_PATIENCE (positive int).
                      Use -1 or omit to keep each experiment's own setting.

Experiment Selection:
  Edit the ACTIVE_EXPERIMENTS array above to choose which experiments to run.
  Comment out (#) experiments you don't want to run.

Tip — to port to this machine, edit suite_paths.env:
  SCRATCH_BASE="..."     (where astra-sim outputs go)
  STG_TMP_BASE="/tmp"    (local temp, avoids NFS races)
USAGE
}

DRY_RUN=0
PARALLEL=1
MAX_JOBS=0
WORKERS_OVERRIDE=""
TRACKER_OVERRIDE=""
PATIENCE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN=1;                    shift ;;
    --parallel)  PARALLEL="${2:-1}";           shift 2 ;;
    --max-jobs)  MAX_JOBS="${2:-0}";           shift 2 ;;
    --workers)   WORKERS_OVERRIDE="$2";        shift 2 ;;
    --tracker)   TRACKER_OVERRIDE="$2";        shift 2 ;;
    --patience)  PATIENCE_OVERRIDE="$2";       shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# Normalise tracker value to "True" / "False" / "" (empty = no override)
case "${TRACKER_OVERRIDE,,}" in
  true|1|yes)  TRACKER_OVERRIDE="True"  ;;
  false|0|no)  TRACKER_OVERRIDE="False" ;;
  "")          TRACKER_OVERRIDE=""      ;;
  *) echo "Invalid --tracker value: '$TRACKER_OVERRIDE' (use true or false)" >&2; exit 1 ;;
esac

# Empty PATIENCE_OVERRIDE = no override (use experiment config).
# Any integer including -1 is passed through; -1 tells Python to use its default patience.

# Resolve experiment paths
EXPERIMENT_PATHS=()
for exp_name in "${ACTIVE_EXPERIMENTS[@]}"; do
  exp_path="$EXPERIMENTS_DIR/$exp_name"
  if [[ -d "$exp_path" ]]; then
    EXPERIMENT_PATHS+=("$exp_path")
  else
    echo "Warning: experiment directory not found: $exp_path" >&2
  fi
done

if [[ ${#EXPERIMENT_PATHS[@]} -eq 0 ]]; then
  echo "No active experiments found in $EXPERIMENTS_DIR" >&2
  exit 1
fi

if [[ "$MAX_JOBS" -gt 0 ]] && [[ "$MAX_JOBS" -lt "${#EXPERIMENT_PATHS[@]}" ]]; then
  EXPERIMENT_PATHS=("${EXPERIMENT_PATHS[@]:0:$MAX_JOBS}")
fi

ASSIGNMENT_CSV="$LAUNCH_LOG_DIR/assignments.csv"
echo "experiment,status,pid,log_file" > "$ASSIGNMENT_CSV"

echo "Local Launch Summary"
echo "===================="
[[ "$DRY_RUN" -eq 1 ]] && echo "[DRY-RUN MODE]"
echo "Parallel jobs : $PARALLEL"
echo "Experiments   : ${#EXPERIMENT_PATHS[@]}"
echo "Workers/exp   : ${WORKERS_OVERRIDE:-from config.env}"
echo "Tracker       : ${TRACKER_OVERRIDE:-from config.env}"
echo "Patience      : ${PATIENCE_OVERRIDE:-from config.env}"
echo "Log dir       : $LAUNCH_LOG_DIR"
echo

submitted=0
skipped=0
active_jobs=0  # number of currently running background jobs

for exp_dir in "${EXPERIMENT_PATHS[@]}"; do
  exp_name="$(basename "$exp_dir")"
  run_script="$exp_dir/run_experiment.sh"
  cfg="$exp_dir/config.env"

  if [[ ! -f "$run_script" || ! -f "$cfg" ]]; then
    echo "SKIP $exp_name (missing run_experiment.sh or config.env)" >&2
    [[ ! -f "$run_script" ]] && echo "  Missing: $run_script" >&2
    [[ ! -f "$cfg" ]]        && echo "  Missing: $cfg" >&2
    echo "$exp_name,SKIPPED,," >> "$ASSIGNMENT_CSV"
    skipped=$((skipped + 1))
    continue
  fi

  mkdir -p "$exp_dir/logs"
  log_file="$exp_dir/logs/local_${TIMESTAMP}.out"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] Would run: bash $run_script"
    echo "          Log      : $log_file"
    echo "$exp_name,DRY_RUN,$log_file" >> "$ASSIGNMENT_CSV"
    submitted=$((submitted + 1))
    continue
  fi

  # If we have reached the parallelism limit, wait for one job to finish first.
  if (( active_jobs >= PARALLEL )); then
    wait -n 2>/dev/null || true
    active_jobs=$((active_jobs - 1))
  fi

  echo "Launching: $exp_name"
  echo "  Log: $log_file"

  (
    # cd into the experiment dir so Python's relative save_dir ("./experiments/...")
    # resolves inside this experiment folder, matching SLURM's --chdir behaviour.
    cd "$exp_dir"
    [[ -n "$WORKERS_OVERRIDE"  ]] && export N_WORKERS_OVERRIDE="$WORKERS_OVERRIDE"
    [[ -n "$TRACKER_OVERRIDE"  ]] && export ENABLE_TRACKER_OVERRIDE="$TRACKER_OVERRIDE"
    [[ -n "$PATIENCE_OVERRIDE" ]] && export EARLY_STOPPING_PATIENCE_OVERRIDE="$PATIENCE_OVERRIDE"
    bash "$run_script" >> "$log_file" 2>&1
    echo "  Done: $exp_name"
  ) &

  echo "$exp_name,LAUNCHED,$log_file" >> "$ASSIGNMENT_CSV"
  submitted=$((submitted + 1))
  active_jobs=$((active_jobs + 1))
done

# Wait for all remaining background jobs to finish.
echo
echo "All experiments launched — waiting for completion..."
wait

echo
echo "Launch statistics"
echo "================="
echo "- Launched  : $submitted"
echo "- Skipped   : $skipped"
echo "- Total     : ${#EXPERIMENT_PATHS[@]}"
echo "- Log dir   : $LAUNCH_LOG_DIR"
echo "- Assignments: $ASSIGNMENT_CSV"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "This was a DRY-RUN. Run without --dry-run to actually execute."
fi
