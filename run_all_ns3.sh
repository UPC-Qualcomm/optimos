#!/bin/bash

# AstraSim with NS3 Network Model Runner Script
# This script runs AstraSim with NS3 network backend instead of analytical models

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Override with: PYTHON_EXEC=/path/to/python bash all_in_one_ns3.sh
PYTHON_EXEC="${PYTHON_EXEC:-python}"
NETWORK_CONFIG="${NETWORK_CONFIG:-./configuration/ns3/configs/FoldedClos_16_config3.txt}"

echo "=== AstraSim NS3 Integration Runner ==="
echo "Using Python: $PYTHON_EXEC"

# Model configuration - choose one of the following
folder_name="GPT_3_1300M"
model_num=5


# Configuration paths
workload_configuration="./workload/${folder_name}"
memory_config="./configuration/RemoteMemory.json"
network_log="./network_log/${folder_name}/"
output="./output/${folder_name}/"
result="./results/${folder_name}/"

# NS3 Network Model Configuration
sim_type="ns3"

# Clean up previous runs
rm -rf "$output"
rm -rf "$result"
rm -rf "$workload_configuration"
rm -rf "$network_log"

# Generate workloads
time "$PYTHON_EXEC" generate_workloads.py --model "$model_num" --folder_name "$folder_name"


# You can choose different topologies by commenting/uncommenting the sections below

# 64-Node Ring Topology (Compatible with current workload generation)

time "$PYTHON_EXEC" run_astrasim_ns3.py \
    --workload_dir "$workload_configuration" \
    --system ./configuration/FoldedClos_sys.json \
    --network_config "$NETWORK_CONFIG" \
    --logical_topology ./configuration/ns3/logical_topo.json \
    --memory "$memory_config" \
    --output_dir "${output}" \
    --network_log "${network_log}"

time "$PYTHON_EXEC" gather_all_NPUs_results.py --sim_logfile "${output}" --output_filename "${result}"
