#!/usr/bin/env bash
set -euo pipefail

# Load machine-specific paths (edit suite_paths.env to port to another machine)
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SUITE_DIR/suite_paths.env"

ROOT_DIR="$SCRATCH_BASE/optimos/optimization_suite"
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="$ROOT_DIR/launch_logs/$TIMESTAMP"
mkdir -p "$LAUNCH_LOG_DIR"

# Verify that the scratch directory is accessible
if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Error: Experiments directory not found at: $EXPERIMENTS_DIR" >&2
  echo "Please ensure experiments are synced to $ROOT_DIR" >&2
  exit 1
fi

# Node states considered usable for scheduling.
NODE_STATES="idle,mix"

# Nodes to exclude from scheduling (known unavailable/problematic nodes).
#SKIP_NODES=("sert-2201" "sert-1430" "sert-1419" "sert-1434" "sert-1433" "sert-1431" "sert-1425" "sert-1424" "sert-1906")
SKIP_NODES=()
SLURM_QOS="large"
# Optional SLURM parameters (uncomment and set if needed)
# SLURM_ACCOUNT=""      # e.g., --account myaccount
# SLURM_QOS=""          # e.g., --qos large (if valid for your account/partition)
# SLURM_EXTRA_ARGS=""   # Additional sbatch arguments

# Retry configuration for failed sbatch submissions
MAX_RETRIES_PER_EXPERIMENT=3

# Global defaults (can be overridden per experiment in config.env)
DEFAULT_CORES_PERCENT=32
DEFAULT_MEM_PER_CORE_GB=4
MAX_CPUS_PER_EXPERIMENT=1
MIN_CPUS_PER_EXPERIMENT=1   # Never schedule fewer than this many CPUs; skip node if memory can't fit even this many

#################################################################################
# EXPERIMENT MANIFEST - Comment out experiments you DON'T want to run
# (Uncommented experiments will be submitted to SLURM)
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
Usage: bash launch_all.sh [--dry-run] [--partition PARTITION] [--max-jobs N] [--allow-node-reuse]

Options:
  --dry-run              Print assignments and sbatch commands without submitting.
  --partition PART       Force SLURM partition for all jobs.
  --max-jobs N           Submit at most N experiments (in manifest order).
  --allow-node-reuse     Compatibility flag (scheduler already packs multiple jobs per node).
  --workers W            Override N_WORKERS for every experiment (default: auto from node).
  --tracker true|false   Override ENABLE_TRACKER for every experiment.
  --patience N           Override EARLY_STOPPING_PATIENCE (positive int).
                         Use -1 or omit to keep each experiment's own setting.

Experiment Selection:
  Edit the ACTIVE_EXPERIMENTS array above to choose which experiments to run.
  Comment out (#) experiments you don't want to run.
USAGE
}

DRY_RUN=0
FORCED_PARTITION=""
MAX_JOBS=0
ALLOW_NODE_REUSE=0
WORKERS_OVERRIDE=""
TRACKER_OVERRIDE=""
PATIENCE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --partition)
      FORCED_PARTITION="${2:-}"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="${2:-0}"
      shift 2
      ;;
    --allow-node-reuse)
      ALLOW_NODE_REUSE=1
      shift
      ;;
    --workers)
      WORKERS_OVERRIDE="${2:-}"
      shift 2
      ;;
    --tracker)
      TRACKER_OVERRIDE="${2:-}"
      shift 2
      ;;
    --patience)
      PATIENCE_OVERRIDE="${2:-}"
      shift 2
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

# Normalise tracker value
case "${TRACKER_OVERRIDE,,}" in
  true|1|yes)  TRACKER_OVERRIDE="True"  ;;
  false|0|no)  TRACKER_OVERRIDE="False" ;;
  "")          TRACKER_OVERRIDE=""      ;;
  *) echo "Invalid --tracker value: '$TRACKER_OVERRIDE' (use true or false)" >&2; exit 1 ;;
esac

# Empty PATIENCE_OVERRIDE = no override (use experiment config).
# Any integer including -1 is passed through; -1 tells Python to use its default patience.

# When --workers W is set, pin the SLURM CPU allocation to exactly W so the
# node scheduler neither under-allocates (hits MAX cap) nor over-allocates
# (exceeds MIN floor).  Both limits are set to W so the scheduler has no choice
# but to request exactly W CPUs per job.
if [[ -n "$WORKERS_OVERRIDE" ]]; then
  MAX_CPUS_PER_EXPERIMENT="$WORKERS_OVERRIDE"
  MIN_CPUS_PER_EXPERIMENT="$WORKERS_OVERRIDE"
fi

if ! command -v sinfo >/dev/null 2>&1 || ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sinfo/sbatch not available in PATH. Run on SLURM login node." >&2
  exit 1
fi

# Filter experiments by ACTIVE_EXPERIMENTS list
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

SINFO_OUT="$LAUNCH_LOG_DIR/sinfo_nodes.txt"

# Sort nodes in descending order (higher IDs first = more resources)
sinfo -N -h -t "$NODE_STATES" -o "%N|%P|%t|%c|%m" | sort -rV > "$SINFO_OUT"

if [[ ! -s "$SINFO_OUT" ]]; then
  echo "No available nodes in states: $NODE_STATES" >&2
  exit 1
fi

# Parse node inventory.
NODES=()
PARTITIONS=()
NODE_CORES=()
NODE_MEM_MB=()
while IFS='|' read -r node partition state cores mem_mb; do
  [[ -z "$node" ]] && continue
  skip_node=0
  for excluded in "${SKIP_NODES[@]}"; do
    if [[ "$node" == "$excluded" ]]; then
      skip_node=1
      break
    fi
  done
  [[ $skip_node -eq 1 ]] && continue
  partition="${partition%%\**}"
  NODES+=("$node")
  PARTITIONS+=("$partition")
  NODE_CORES+=("$cores")
  NODE_MEM_MB+=("$mem_mb")
done < "$SINFO_OUT"

if [[ ${#NODES[@]} -eq 0 ]]; then
  echo "No parseable nodes found from sinfo output." >&2
  exit 1
fi

ASSIGNMENT_CSV="$LAUNCH_LOG_DIR/assignments.csv"
cat > "$ASSIGNMENT_CSV" <<CSV
experiment,node,partition,node_cores_total,node_mem_total_mb,node_cores_free,node_mem_free_mb,cpus_per_task,mem_per_cpu_gb,mem_per_cpu_slurm,job_name,submit_status,job_id
CSV

get_node_free_resources() {
  local node_name="$1"
  local node_info
  node_info="$(scontrol show node "$node_name" 2>/dev/null || true)"

  local cpu_tot cpu_alloc mem_tot alloc_mem
  cpu_tot="$(grep -oE 'CPUTot=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  cpu_alloc="$(grep -oE 'CPUAlloc=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  mem_tot="$(grep -oE 'RealMemory=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  alloc_mem="$(grep -oE 'AllocMem=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"

  [[ -z "$cpu_tot" ]] && cpu_tot=0
  [[ -z "$cpu_alloc" ]] && cpu_alloc=0
  [[ -z "$mem_tot" ]] && mem_tot=0
  [[ -z "$alloc_mem" ]] && alloc_mem=0

  local cpu_free mem_free
  cpu_free=$(( cpu_tot - cpu_alloc ))
  mem_free=$(( mem_tot - alloc_mem ))
  (( cpu_free < 1 )) && cpu_free=1
  (( mem_free < 1 )) && mem_free=1

  echo "$cpu_tot|$mem_tot|$cpu_free|$mem_free"
}

declare -i node_idx=0
submitted=0
failed=0

NODE_CORES_TOTAL_RUNTIME=()
NODE_MEM_TOTAL_RUNTIME=()
NODE_FREE_CORES_INITIAL=()
NODE_FREE_MEM_INITIAL_MB=()
NODE_POOL_CPUS=()
NODE_REMAINING_CPUS=()
NODE_REMAINING_MEM_MB=()
NODE_ASSIGNED_JOBS=()
NODE_ASSIGNED_CPUS=()

# Build per-node scheduling pool based on policy percent of TOTAL CPUs,
# then cap by currently free CPUs.
for i in "${!NODES[@]}"; do
  resource_line="$(get_node_free_resources "${NODES[$i]}")"
  IFS='|' read -r node_cores_total node_mem_total_mb node_cores_free node_mem_free_mb <<< "$resource_line"

  if [[ "$node_cores_total" -le 0 ]]; then
    node_cores_total="${NODE_CORES[$i]}"
    node_cores_free="${NODE_CORES[$i]}"
  fi
  if [[ "$node_mem_total_mb" -le 0 ]]; then
    node_mem_total_mb="${NODE_MEM_MB[$i]}"
    node_mem_free_mb="${NODE_MEM_MB[$i]}"
  fi

  policy_pool_cpus=$(( node_cores_total * DEFAULT_CORES_PERCENT / 100 ))
  (( policy_pool_cpus < 1 )) && policy_pool_cpus=1

  # Never schedule beyond what is currently free on the node snapshot.
  pool_cpus="$policy_pool_cpus"
  (( pool_cpus > node_cores_free )) && pool_cpus="$node_cores_free"
  (( pool_cpus < 1 )) && pool_cpus=1

  NODE_CORES_TOTAL_RUNTIME+=("$node_cores_total")
  NODE_MEM_TOTAL_RUNTIME+=("$node_mem_total_mb")
  NODE_FREE_CORES_INITIAL+=("$node_cores_free")
  NODE_FREE_MEM_INITIAL_MB+=("$node_mem_free_mb")
  NODE_POOL_CPUS+=("$pool_cpus")
  NODE_REMAINING_CPUS+=("$pool_cpus")
  NODE_REMAINING_MEM_MB+=("$node_mem_free_mb")
  NODE_ASSIGNED_JOBS+=("0")
  NODE_ASSIGNED_CPUS+=("0")
done

for exp_dir in "${EXPERIMENT_PATHS[@]}"; do
  cfg="$exp_dir/config.env"
  run_script="$exp_dir/run_experiment.sh"

  if [[ ! -f "$cfg" || ! -f "$run_script" ]]; then
    echo "Skipping $exp_dir (missing config.env or missing run_experiment.sh)" >&2
    [[ ! -f "$cfg" ]] && echo "  - Missing: $cfg" >&2
    [[ ! -f "$run_script" ]] && echo "  - Missing: $run_script" >&2
    ((failed+=1))
    continue
  fi

  # shellcheck disable=SC1090
  source "$cfg"

  cpus_override="${CPUS_PER_TASK_OVERRIDE:-}"
  mem_override="${MEM_PER_CPU_GB_OVERRIDE:-}"

  if [[ -n "$mem_override" ]]; then
    mem_per_cpu_gb="$mem_override"
  else
    mem_per_cpu_gb="$DEFAULT_MEM_PER_CORE_GB"
  fi

  # Retry loop for sbatch submission
  submission_successful=0
  retry_count=0
  TRIED_NODES=()

  while [[ $submission_successful -eq 0 ]] && [[ $retry_count -lt $MAX_RETRIES_PER_EXPERIMENT ]]; do

    # Choose a node slot: each node can host multiple experiments,
    # skipping nodes that already failed for this experiment
    selected_node_pos=-1
    selected_cpus=0
    selected_req_mem_mb=0
    is_fallback=0

    for ((try_i=0; try_i<${#NODES[@]}; try_i++)); do
      node_pos=$(( (node_idx + try_i) % ${#NODES[@]} ))
      node="${NODES[$node_pos]}"

      # Skip nodes already tried for this experiment
      skip_node=0
      for tried_node in "${TRIED_NODES[@]}"; do
        if [[ "$tried_node" == "$node" ]]; then
          skip_node=1
          break
        fi
      done
      [[ $skip_node -eq 1 ]] && continue

      remaining_cpus="${NODE_REMAINING_CPUS[$node_pos]}"
      remaining_mem_mb="${NODE_REMAINING_MEM_MB[$node_pos]}"

      # Real-time guard: only consider resources that are still free *now* on the node.
      # This avoids submitting to a node that became busy after the initial snapshot.
      current_resource_line="$(get_node_free_resources "$node")"
      IFS='|' read -r _cpu_tot_now _mem_tot_now cpu_free_now mem_free_now <<< "$current_resource_line"

      effective_available_cpus="$remaining_cpus"
      (( effective_available_cpus > cpu_free_now )) && effective_available_cpus="$cpu_free_now"

      effective_available_mem_mb="$remaining_mem_mb"
      (( effective_available_mem_mb > mem_free_now )) && effective_available_mem_mb="$mem_free_now"

      (( effective_available_cpus < 1 )) && continue
      (( effective_available_mem_mb < 1 )) && continue

      if [[ -n "$cpus_override" ]]; then
        candidate_cpus="$cpus_override"
        (( candidate_cpus > effective_available_cpus )) && candidate_cpus="$effective_available_cpus"
      else
        candidate_cpus="$effective_available_cpus"
        (( candidate_cpus > MAX_CPUS_PER_EXPERIMENT )) && candidate_cpus="$MAX_CPUS_PER_EXPERIMENT"
      fi

      (( candidate_cpus < MIN_CPUS_PER_EXPERIMENT )) && continue
      candidate_req_mem_mb=$(( candidate_cpus * mem_per_cpu_gb * 1024 ))

      if (( candidate_req_mem_mb > effective_available_mem_mb )); then
        max_cpus_by_mem=$(( effective_available_mem_mb / (mem_per_cpu_gb * 1024) ))
        (( max_cpus_by_mem < MIN_CPUS_PER_EXPERIMENT )) && continue
        (( max_cpus_by_mem < candidate_cpus )) && candidate_cpus="$max_cpus_by_mem"
        candidate_req_mem_mb=$(( candidate_cpus * mem_per_cpu_gb * 1024 ))
      fi

      # Final check: enforce minimum after all adjustments
      (( candidate_cpus < MIN_CPUS_PER_EXPERIMENT )) && continue

      selected_node_pos="$node_pos"
      selected_cpus="$candidate_cpus"
      selected_req_mem_mb="$candidate_req_mem_mb"
      break
    done

    # Fallback: no node currently has enough free resources for immediate placement.
    # Select the node with the most free CPUs from the top of the list and submit
    # anyway — SLURM will queue the job (PD) and start it when resources free up.
    if (( selected_node_pos < 0 )); then
      echo "No node has immediate capacity for $(basename "$exp_dir"); queuing on best available node." >&2
      fallback_node_pos=-1
      fallback_best_cpus=-1
      for ((fb_i=0; fb_i<${#NODES[@]}; fb_i++)); do
        fb_node="${NODES[$fb_i]}"
        skip_node=0
        for excluded in "${SKIP_NODES[@]}"; do
          [[ "$fb_node" == "$excluded" ]] && { skip_node=1; break; }
        done
        [[ $skip_node -eq 1 ]] && continue
        skip_node=0
        for tried_node in "${TRIED_NODES[@]}"; do
          [[ "$tried_node" == "$fb_node" ]] && { skip_node=1; break; }
        done
        [[ $skip_node -eq 1 ]] && continue
        fb_cpus_free="${NODE_FREE_CORES_INITIAL[$fb_i]}"
        if (( fb_cpus_free > fallback_best_cpus )); then
          fallback_best_cpus="$fb_cpus_free"
          fallback_node_pos="$fb_i"
        fi
      done

      if (( fallback_node_pos < 0 )); then
        # Tried nodes exhausted — reset and pick from the very top of the list
        fallback_node_pos=-1
        fallback_best_cpus=-1
        for ((fb_i=0; fb_i<${#NODES[@]}; fb_i++)); do
          fb_node="${NODES[$fb_i]}"
          skip_node=0
          for excluded in "${SKIP_NODES[@]}"; do
            [[ "$fb_node" == "$excluded" ]] && { skip_node=1; break; }
          done
          [[ $skip_node -eq 1 ]] && continue
          fb_cpus_free="${NODE_FREE_CORES_INITIAL[$fb_i]}"
          if (( fb_cpus_free >= MIN_CPUS_PER_EXPERIMENT && fb_cpus_free >= MIN_CPUS_PER_EXPERIMENT && fb_cpus_free > fallback_best_cpus )); then
            fallback_best_cpus="$fb_cpus_free"
            fallback_node_pos="$fb_i"
          fi
        done
      fi

      if (( fallback_node_pos < 0 )); then
        echo "FAILED scheduling $(basename "$exp_dir"): no usable nodes in the cluster" >&2
        echo "$(basename "$exp_dir"),N/A,N/A,0,0,0,0,0,$mem_per_cpu_gb,${mem_per_cpu_gb}G,${JOB_NAME:-$(basename "$exp_dir")},FAILED_NO_NODES," >> "$ASSIGNMENT_CSV"
        ((failed+=1))
        break
      fi

      selected_node_pos="$fallback_node_pos"
      fb_node_total_cpus="${NODE_CORES_TOTAL_RUNTIME[$fallback_node_pos]}"
      if [[ -n "$cpus_override" ]]; then
        selected_cpus="$cpus_override"
      else
        policy_cpus=$(( fb_node_total_cpus * DEFAULT_CORES_PERCENT / 100 ))
        (( policy_cpus < 1 )) && policy_cpus=1
        selected_cpus="$policy_cpus"
        (( selected_cpus > MAX_CPUS_PER_EXPERIMENT )) && selected_cpus="$MAX_CPUS_PER_EXPERIMENT"
        # Enforce minimum for fallback allocation
        (( selected_cpus < MIN_CPUS_PER_EXPERIMENT )) && selected_cpus="$MIN_CPUS_PER_EXPERIMENT"
      fi
      selected_req_mem_mb=$(( selected_cpus * mem_per_cpu_gb * 1024 ))
      is_fallback=1
    fi

    node_pos="$selected_node_pos"
    node="${NODES[$node_pos]}"
    partition="${PARTITIONS[$node_pos]}"
    node_cores_total="${NODE_CORES_TOTAL_RUNTIME[$node_pos]}"
    node_mem_total_mb="${NODE_MEM_TOTAL_RUNTIME[$node_pos]}"
    node_cores_free="${NODE_FREE_CORES_INITIAL[$node_pos]}"
    node_mem_free_mb="${NODE_FREE_MEM_INITIAL_MB[$node_pos]}"
    cpus_per_task="$selected_cpus"
    requested_mem_mb="$selected_req_mem_mb"

    NODE_REMAINING_CPUS[$node_pos]=$(( ${NODE_REMAINING_CPUS[$node_pos]} - cpus_per_task ))
    NODE_REMAINING_MEM_MB[$node_pos]=$(( ${NODE_REMAINING_MEM_MB[$node_pos]} - requested_mem_mb ))
    NODE_ASSIGNED_JOBS[$node_pos]=$(( ${NODE_ASSIGNED_JOBS[$node_pos]} + 1 ))
    NODE_ASSIGNED_CPUS[$node_pos]=$(( ${NODE_ASSIGNED_CPUS[$node_pos]} + cpus_per_task ))
    (( NODE_REMAINING_CPUS[$node_pos] < 0 )) && NODE_REMAINING_CPUS[$node_pos]=0
    (( NODE_REMAINING_MEM_MB[$node_pos] < 0 )) && NODE_REMAINING_MEM_MB[$node_pos]=0
    node_idx=$(( node_pos + 1 ))

    mem_per_cpu_slurm="${mem_per_cpu_gb}G"
    job_name="${JOB_NAME:-$(basename "$exp_dir")}"
    logs_dir="$exp_dir/logs"
    mkdir -p "$logs_dir"

    submit_partition="$partition"
    if [[ -n "$FORCED_PARTITION" ]]; then
      submit_partition="$FORCED_PARTITION"
    elif [[ -n "${PARTITION_OVERRIDE:-}" ]]; then
      submit_partition="$PARTITION_OVERRIDE"
    fi

    wrap_cmd="ulimit -c 0; bash \"$run_script\""

    sbatch_cmd=(
      sbatch
      --job-name "$job_name"
      --chdir "$exp_dir"
      #--nodelist "$node"
      --partition "$submit_partition"
      --cpus-per-task "$cpus_per_task"
      --mem-per-cpu "$mem_per_cpu_slurm"
      --export "ALL,N_WORKERS_OVERRIDE=${WORKERS_OVERRIDE:-$cpus_per_task},ENABLE_TRACKER_OVERRIDE=${TRACKER_OVERRIDE},EARLY_STOPPING_PATIENCE_OVERRIDE=${PATIENCE_OVERRIDE}"
      --output "$logs_dir/slurm-%j.out"
      --error "$logs_dir/slurm-%j.err"
      --wrap "$wrap_cmd"
    )

    # Final real-time guard before submission: skip node if current free resources
    # cannot satisfy this job anymore.  In fallback mode we skip this guard —
    # the job is intentionally queued (PD) on the best available node.
    current_resource_line="$(get_node_free_resources "$node")"
    IFS='|' read -r _cpu_tot_now _mem_tot_now cpu_free_now mem_free_now <<< "$current_resource_line"
    if [[ "$is_fallback" -eq 0 ]] && (( cpus_per_task > cpu_free_now || requested_mem_mb > mem_free_now )); then
      echo "Skipping $node for $(basename "$exp_dir"): insufficient current free resources (need cpu=$cpus_per_task mem_mb=$requested_mem_mb, have cpu=$cpu_free_now mem_mb=$mem_free_now)" >&2
      TRIED_NODES+=("$node")

      # Restore reserved pool resources and try another node.
      NODE_REMAINING_CPUS[$node_pos]=$(( ${NODE_REMAINING_CPUS[$node_pos]} + cpus_per_task ))
      NODE_REMAINING_MEM_MB[$node_pos]=$(( ${NODE_REMAINING_MEM_MB[$node_pos]} + requested_mem_mb ))
      NODE_ASSIGNED_JOBS[$node_pos]=$(( ${NODE_ASSIGNED_JOBS[$node_pos]} - 1 ))
      NODE_ASSIGNED_CPUS[$node_pos]=$(( ${NODE_ASSIGNED_CPUS[$node_pos]} - cpus_per_task ))
      (( NODE_ASSIGNED_JOBS[$node_pos] < 0 )) && NODE_ASSIGNED_JOBS[$node_pos]=0
      (( NODE_ASSIGNED_CPUS[$node_pos] < 0 )) && NODE_ASSIGNED_CPUS[$node_pos]=0
      continue
    fi

    # Add optional SLURM parameters if configured
    [[ -n "${SLURM_ACCOUNT:-}" ]] && sbatch_cmd+=(--account "$SLURM_ACCOUNT")
    [[ -n "${SLURM_QOS:-}" ]] && sbatch_cmd+=(--qos "$SLURM_QOS")
    [[ -n "${SLURM_EXTRA_ARGS:-}" ]] && sbatch_cmd+=($SLURM_EXTRA_ARGS)

    if [[ "$DRY_RUN" -eq 1 ]]; then
      local_dry_status="DRY_RUN"
      [[ "$is_fallback" -eq 1 ]] && local_dry_status="DRY_RUN_QUEUED_FALLBACK"
      echo "[DRY-RUN${is_fallback:+/QUEUED-FALLBACK}] ${sbatch_cmd[*]}"
      echo "$(basename "$exp_dir"),$node,$submit_partition,$node_cores_total,$node_mem_total_mb,$node_cores_free,$node_mem_free_mb,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,$local_dry_status," >> "$ASSIGNMENT_CSV"
      ((submitted+=1))
      submission_successful=1
    else
      set +e
      out="$("${sbatch_cmd[@]}" 2>&1)"
      rc=$?
      set -e

      if [[ $rc -eq 0 ]]; then
        job_id="$(awk '{print $NF}' <<< "$out")"
        if [[ "$is_fallback" -eq 1 ]]; then
          echo "Queued $(basename "$exp_dir") on $node (job $job_id, cpus=$cpus_per_task, mem/cpu=$mem_per_cpu_slurm) [QUEUED-fallback, will start when resources free up]"
          echo "$(basename "$exp_dir"),$node,$submit_partition,$node_cores_total,$node_mem_total_mb,$node_cores_free,$node_mem_free_mb,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,QUEUED_FALLBACK,$job_id" >> "$ASSIGNMENT_CSV"
        else
          echo "Submitted $(basename "$exp_dir") to $node (job $job_id, cpus=$cpus_per_task, mem/cpu=$mem_per_cpu_slurm)"
          echo "$(basename "$exp_dir"),$node,$submit_partition,$node_cores_total,$node_mem_total_mb,$node_cores_free,$node_mem_free_mb,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,SUBMITTED,$job_id" >> "$ASSIGNMENT_CSV"
        fi
        ((submitted+=1))
        submission_successful=1
      else
        echo "⚠️  Failed on $node (retry $(($retry_count+1))/$MAX_RETRIES_PER_EXPERIMENT): $out" >&2
        TRIED_NODES+=("$node")
        ((retry_count+=1))
        
        # Restore resources since this node failed
        NODE_REMAINING_CPUS[$node_pos]=$(( ${NODE_REMAINING_CPUS[$node_pos]} + cpus_per_task ))
        NODE_REMAINING_MEM_MB[$node_pos]=$(( ${NODE_REMAINING_MEM_MB[$node_pos]} + requested_mem_mb ))
        NODE_ASSIGNED_JOBS[$node_pos]=$(( ${NODE_ASSIGNED_JOBS[$node_pos]} - 1 ))
        NODE_ASSIGNED_CPUS[$node_pos]=$(( ${NODE_ASSIGNED_CPUS[$node_pos]} - cpus_per_task ))
        (( NODE_ASSIGNED_JOBS[$node_pos] < 0 )) && NODE_ASSIGNED_JOBS[$node_pos]=0
        (( NODE_ASSIGNED_CPUS[$node_pos] < 0 )) && NODE_ASSIGNED_CPUS[$node_pos]=0
      fi
    fi
  done

  if [[ $submission_successful -eq 0 ]]; then
    echo "FAILED submitting $(basename "$exp_dir") after $MAX_RETRIES_PER_EXPERIMENT retries" >&2
    echo "$(basename "$exp_dir"),RETRY_EXHAUSTED,N/A,0,0,0,0,0,$mem_per_cpu_gb,${mem_per_cpu_gb}G,${JOB_NAME:-$(basename "$exp_dir")},FAILED_EXHAUSTED," >> "$ASSIGNMENT_CSV"
    ((failed+=1))
  fi
done

echo
echo "Per-node scheduling summary"
echo "node,partition,jobs_assigned,pool_cpus,cpus_assigned,cpus_remaining,mem_free_initial_mb,mem_used_mb,mem_remaining_mb"
for i in "${!NODES[@]}"; do
  mem_free_initial_mb="${NODE_FREE_MEM_INITIAL_MB[$i]}"
  mem_remaining_mb="${NODE_REMAINING_MEM_MB[$i]}"
  mem_used_mb=$(( mem_free_initial_mb - mem_remaining_mb ))
  (( mem_used_mb < 0 )) && mem_used_mb=0
  echo "${NODES[$i]},${PARTITIONS[$i]},${NODE_ASSIGNED_JOBS[$i]},${NODE_POOL_CPUS[$i]},${NODE_ASSIGNED_CPUS[$i]},${NODE_REMAINING_CPUS[$i]},${mem_free_initial_mb},${mem_used_mb},${mem_remaining_mb}"
done

echo
echo "Launch summary"
echo "- Total considered: ${#EXPERIMENT_PATHS[@]}"
echo "- Submitted: $submitted"
echo "- Failed/skipped: $failed"
echo "- Node snapshot: $SINFO_OUT"
echo "- Assignment file: $ASSIGNMENT_CSV"
