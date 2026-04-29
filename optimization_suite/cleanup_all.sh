#!/usr/bin/env bash
set -euo pipefail

# Load machine-specific paths (edit suite_paths.env to port to another machine)
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SUITE_DIR/suite_paths.env"

ROOT_DIR="$SCRATCH_BASE/optimos/optimization_suite"
EXPERIMENTS_DIR="$ROOT_DIR/experiments"

# Verify that the scratch directory is accessible
if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Error: Experiments directory not found at: $EXPERIMENTS_DIR" >&2
  echo "Please ensure experiments are synced to $ROOT_DIR" >&2
  exit 1
fi

#################################################################################
# EXPERIMENT MANIFEST - Comment out experiments you want to KEEP
# (Uncommented experiments will have their results CLEANED)
#################################################################################
ACTIVE_EXPERIMENTS=(
  "llama70b_128npus_edp"
  "llama70b_128npus_edp_and_bw"
  "llama70b_128npus_energy_and_time"
  "llama70b_128npus_memory_and_time"
  "llama70b_128npus_time"
  "llama70b_128npus_time_and_bw"
  "llama70b_128npus_latency_network"
  "llama70b_128npus_time_and_throughput_per_energy"
  "gpt60b_128npus_edp"
  "gpt60b_128npus_edp_and_bw"
  "gpt60b_128npus_energy_and_time"
  "gpt60b_128npus_memory_and_time"
  "gpt60b_128npus_time"
  "gpt60b_128npus_time_and_bw"
  "gpt60b_128npus_latency_network"
  "gpt60b_128npus_time_and_throughput_per_energy"
  "llama8b_32npus_edp"
  "llama8b_32npus_edp_and_bw"
  "llama8b_32npus_energy_and_time"
  "llama8b_32npus_memory_and_time"
  "llama8b_32npus_time"
  "llama8b_32npus_time_and_bw"
  "llama8b_32npus_time_and_throughput_per_energy"
  "llama8b_32npus_latency_network"
  "llama8b_32npus_energy"
  "gpt60b_128npus_energy"
  "llama70b_128npus_energy"
  "gpt175b_1024npus_energy"
)

usage() {
  cat <<USAGE
Usage: bash cleanup_all.sh [--dry-run] [--keep-logs] [--keep-outputs]

Options:
  --dry-run       Print what would be deleted without actually deleting.
  --keep-logs     Don't delete logs/ directories.
  --keep-outputs  Don't delete outputs/ directories.
  
Instructions:
  1. Edit the ACTIVE_EXPERIMENTS array above
  2. Comment out (prefix with #) any experiments you want to KEEP
  3. Run: bash cleanup_all.sh [--dry-run]   (to preview)
  4. Run: bash cleanup_all.sh                (to actually clean)
USAGE
}

DRY_RUN=0
KEEP_LOGS=0
KEEP_OUTPUTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --keep-logs)
      KEEP_LOGS=1
      shift
      ;;
    --keep-outputs)
      KEEP_OUTPUTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Validate that experiments directory exists
if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Error: Experiments directory not found: $EXPERIMENTS_DIR" >&2
  exit 1
fi

# Check if there are any experiments to clean
if [[ ${#ACTIVE_EXPERIMENTS[@]} -eq 0 ]]; then
  echo "No experiments to clean (all are commented out in ACTIVE_EXPERIMENTS)" >&2
  exit 0
fi

cleaned_count=0
skipped_count=0

echo "Cleanup Summary"
echo "==============="
[[ "$DRY_RUN" -eq 1 ]] && echo "[DRY-RUN MODE]"
echo "Keep logs: $KEEP_LOGS"
echo "Keep outputs: $KEEP_OUTPUTS"
echo

for exp_name in "${ACTIVE_EXPERIMENTS[@]}"; do
  exp_dir="$EXPERIMENTS_DIR/$exp_name"
  
  if [[ ! -d "$exp_dir" ]]; then
    echo "⚠️  SKIPPED: $exp_name (directory not found)"
    ((skipped_count+=1))
    continue
  fi

  deleted_dirs=()

  # Clean logs directory
  if [[ "$KEEP_LOGS" -eq 0 ]] && [[ -d "$exp_dir/logs" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "   [DRY-RUN] Would delete: $exp_dir/logs/*"
    else
      rm -rf "$exp_dir/logs"/*
      echo "   Deleted: $exp_dir/logs/*"
    fi
    deleted_dirs+=("logs")
  fi

  # Clean outputs directory
  if [[ "$KEEP_OUTPUTS" -eq 0 ]] && [[ -d "$exp_dir/outputs" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "   [DRY-RUN] Would delete: $exp_dir/outputs/*"
    else
      rm -rf "$exp_dir/outputs"/*
      echo "   Deleted: $exp_dir/outputs/*"
    fi
    deleted_dirs+=("outputs")
  fi

  # Clean experiments directory
  if [[ "$KEEP_OUTPUTS" -eq 0 ]] && [[ -d "$exp_dir/outputs" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "   [DRY-RUN] Would delete: $exp_dir/outputs/*"
    else
      rm -rf "$exp_dir/experiments"/*
      echo "   Deleted: $exp_dir/outputs/*"
    fi
    deleted_dirs+=("outputs")
  fi

  if [[ ${#deleted_dirs[@]} -gt 0 ]]; then
    echo "✓ CLEANED: $exp_name (${deleted_dirs[*]})"
    ((cleaned_count+=1))
  else
    echo "✓ SKIPPED: $exp_name (no directories to clean or all kept)"
    ((skipped_count+=1))
  fi
done


rm -r "$ROOT_DIR/launch_logs"/* || true

echo
echo "Cleanup Statistics"
echo "==================="
echo "- Experiments cleaned: $cleaned_count"
echo "- Experiments skipped: $skipped_count"
echo "- Total processed: ${#ACTIVE_EXPERIMENTS[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "This was a DRY-RUN. Run without --dry-run to actually delete files."
fi
