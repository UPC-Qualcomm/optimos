#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_EXEC="${PYTHON_EXEC:-python}"


folder_name="GPT_3_1300M"
model_num=5


workload_configuration="./workload/${folder_name}"
memory_config="./configuration/RemoteMemory.json"
network_log="./network_log/${folder_name}/"
sim_type="analytical_unaware"

output="./output/${folder_name}/"
result="./results/${folder_name}/"

rm -rf "${output}"
rm -rf "${result}"
rm -rf "${workload_configuration}"
rm -rf "${network_log}"


time "${PYTHON_EXEC}" generate_workloads.py --model "${model_num}" --folder_name "${folder_name}"

#FoldedClos
time "${PYTHON_EXEC}" run_astrasim.py \
    --workload_dir "$workload_configuration" \
    --system ./configuration/FoldedClos_sys.json \
    --network ./configuration/FoldedClos.yml \
    --memory "$memory_config"  \
    --output_dir "${output}FoldedClos" \
    --network_log "${network_log}FoldedClos" \
    --sim_type "${sim_type}"

#Dragonfly
time "${PYTHON_EXEC}" run_astrasim.py \
    --workload_dir "$workload_configuration" \
    --system ./configuration/Dragonfly_sys.json \
    --network ./configuration/Dragonfly.yml \
    --memory "$memory_config"  \
    --output_dir "${output}Dragonfly" \
    --network_log "${network_log}Dragonfly" \
    --sim_type "${sim_type}"


# #Collect results
#FoldedClos
time "${PYTHON_EXEC}" gather_all_NPUs_results.py --sim_logfile "${output}FoldedClos" --output_filename "${result}FoldedClos"

#Dragonfly
time "${PYTHON_EXEC}" gather_all_NPUs_results.py --sim_logfile "${output}Dragonfly" --output_filename "${result}Dragonfly"