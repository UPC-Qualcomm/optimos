#!/bin/bash


npus=_32
model="llama8b_32npus"
workload_configuration_file_name=${model}${npus}
folder_name=${model}"analytical_"${npus}
model_num=17


workload_configuration="./workload/"${workload_configuration_file_name}
memory_config="./configuration/RemoteMemory.json"
network_log="./network_log/"${folder_name}"/"
sim_type="analytical_unaware"


output="./output/"${folder_name}"/"
result="./results/"${folder_name}"/"

rm -rf $output
rm -rf $result
rm -rf $workload_configuration
rm -rf $network_log

time python generate_workloads.py --model $model_num --folder_name $workload_configuration_file_name

system="./configuration/FoldedClos_sys.json"

folder_name=${model}"analytical_"${npus}
echo "start: ${folder_name}"

network_log="./network_log/"${folder_name}"/"
output="./output/"${folder_name}"/"
result="./results/"${folder_name}"/"

rm -rf $output
rm -rf $result
rm -rf $network_log

#FoldedClos
time python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system $system \
    --network ./configuration/FoldedClos.yml \
    --memory $memory_config  \
    --output_dir ${output}FoldedClos \
    --network_log ${network_log}FoldedClos \
    --sim_type ${sim_type} #> ${folder_name}_log.txt 2>&1

time python gather_all_NPUs_results.py --sim_logfile ${output}FoldedClos --output_filename ${result}FoldedClos

echo "Finish: ${folder_name}"

analytical_csv=${result}FoldedClos.csv



folder_name=${model}"ns3_"${npus}
sim_type="ns3"
echo "start: ${folder_name}"

network_log="./network_log/"${folder_name}"/"
output="./output/"${folder_name}"/"
result="./results/"${folder_name}"/"

# NS3 Network Model Configuration

# Clean up previous runs
rm -rf $output
rm -rf $result
rm -rf $network_log

mkdir -p ./configuration/ns3/output

time python run_astrasim_ns3.py \
    --workload_dir $workload_configuration \
    --system $system \
    --network_config ./configuration/ns3/configs/FoldedClos_config.txt \
    --logical_topology ./configuration/ns3/logical_topo.json \
    --memory $memory_config \
    --output_dir ${output}FoldedClos \
    --network_log ${network_log}FoldedClos

time python gather_all_NPUs_results.py --sim_logfile ${output}FoldedClos --output_filename ${result}FoldedClos

ns3_csv=${result}FoldedClos.csv

comparison_output_dir="./results/${model}comparison_${npus}/"

time python ./plot_scritps/generate_slope_graph.py \
    --analytical $analytical_csv \
    --ns3 $ns3_csv \
    --output-dir $comparison_output_dir