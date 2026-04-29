#!/usr/bin/python3
import os
import subprocess
import multiprocessing
import argparse
import pandas as pd
from intervaltree import IntervalTree
import time
from functools import partial
import re


def get_timings_df(csv_trace_file, output_file_name):
    """Process timing data from simulation trace files"""
    df = pd.read_csv(csv_trace_file)
    # Filter issues and rename 'tick' to 'issue_tick'
    df_issues = df.query("action == 'issue'").drop(columns="action")

    # Filter callbacks and rename 'tick' to 'callback_tick'
    df_callbacks = (
        df.query("action == 'callback'")
        .drop(columns="action")
        .rename(columns={"issue_tick": "callback_tick"})
    )

    # Merge issues with callbacks on the identifying columns
    merged_df = df_issues.merge(
        df_callbacks[["sys_id", "node_id", "node_name", "node_type", "callback_tick"]],
        on=["sys_id", "node_id", "node_name", "node_type"],
        how="left",
        suffixes=("", ""),
    )

    # Add elapsed_time column
    merged_df["elapsed_time"] = merged_df["callback_tick"] - merged_df["issue_tick"]
    merged_df.fillna(0, inplace=True)

    num_sys = merged_df["sys_id"].max()

    extended_data = []
    for sys_id in range(num_sys+1):
        results_comm = get_exposed(merged_df, sys_id, node_op_type="comm")
        results_comp = get_exposed(merged_df, sys_id, node_op_type="comp")
        results_merged = pd.concat([results_comm, results_comp])
        results_merged["exposed_percent"] = (
            100 * results_merged["exposed"] / results_merged["elapsed_time"]
        )
        results_merged["overlap_percent"] = (
            100 * results_merged["overlap"] / results_merged["elapsed_time"]
        )
        extended_data.append(results_merged)

    extended_data = pd.concat(extended_data, ignore_index=True)
    extended_data.to_csv(output_file_name)
    
    os.remove(csv_trace_file)

def build_compute_interval_tree(nodes):
    tree = IntervalTree()
    for row in nodes.itertuples(index=False):
        tree.addi(row.issue_tick, row.callback_tick, row.node_id)
    tree.merge_overlaps()
    return tree

def compute_overlap(tree, node_id, start, end):
    overlaps = tree.overlap(start, end)
    overlap_duration = 0
    for o in overlaps:
        if o.data != node_id:
            overlap_start = max(start, o.begin)
            overlap_end = min(end, o.end)
            overlap_duration += max(0, overlap_end - overlap_start)
    return overlap_duration

def get_exposed(df, sys_id, node_op_type="comm"):
    group = df[df["sys_id"] == sys_id]

    if node_op_type == "comm":
        main_nodes = group[group["node_type"].isin([5, 6, 7])]
    elif node_op_type == "comp":
        main_nodes = group[group["node_type"] == 4]
    else:
        raise ValueError("node_op_type must be either 'comm' or 'comp'")

    comp_nodes = group[group["node_type"] == 4]
    comm_nodes = group[group["node_type"].isin([5, 6, 7])]

    comp_tree = build_compute_interval_tree(comp_nodes)
    comm_tree = build_compute_interval_tree(comm_nodes)

    results = []

    for row in main_nodes.itertuples(index=False):
        duration = row.elapsed_time

        overlap_comp = compute_overlap(comp_tree, row.node_id, row.issue_tick, row.callback_tick)
        overlap_comm = compute_overlap(comm_tree, row.node_id, row.issue_tick, row.callback_tick)

        total_overlap = min(duration, overlap_comp + overlap_comm)

        result = {
            "sys_id": row.sys_id,
            "node_id": row.node_id,
            "node_name": row.node_name,
            "col_type": row.col_type,
            "node_type": row.node_type,
            "elapsed_time": duration,
            "exposed": duration - total_overlap,
            "overlap": total_overlap,
            "overlap_with_comp": overlap_comp,
            "overlap_with_comm": overlap_comm,
            "num_ops": row.num_ops,
            "tensor_size": row.tensor_size,
            "perf": row.perf,
            "operational_intensity": row.operational_intensity,
            "issue_tick": row.issue_tick,
            "callback_tick": row.callback_tick,
        }

        results.append(result)

    if results:
        return pd.DataFrame.from_records(results)
    else:
        return pd.DataFrame(
            columns=[
                "sys_id",
                "node_id",
                "node_name",
                "col_type",
                "node_type",
                "exposed",
                "elapsed_time",
                "overlap",
                "overlap_with_comp",
                "overlap_with_comm",
                "exposed_with_comp",
                "exposed_with_comm",
                "num_ops",
                "tensor_size",
                "perf",
                "operational_intensity",
                "issue_tick",
                "callback_tick",
            ]
        )

def run_command(command, cwd=None):
    """Execute a shell command with timing information"""
    match = re.search(r'(\d+_\d+_\d+_\d+_\d+)', command)
    identifier = ""
    if match:
        identifier = f" for {match.group(1)}"

    print(command)
    start_time = time.time()
    result = subprocess.run(command, shell=True, cwd=cwd)
    
    end_time = time.time()
    print(f"Total time{identifier}: {end_time - start_time:.2f} seconds")
        
    return result.returncode == 0


def list_workloads(root):
    """List all workload files in the given directory"""
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(".0.et"):
            filtered.append(os.path.join(root, file[:-5]))
    return filtered


def run_astrasim_ns3(workload_path, system, network_config, logical_topology, memory, output_dir, network_log, suffix=None):
    """Run AstraSim with NS3 network backend"""
    
    # Get the NS3 executable path
    # ASTRA_SIM_ROOT points to the astra-sim submodule inside optimos.
    astra_sim_dir = os.environ.get('ASTRA_SIM_ROOT', '')
    if not astra_sim_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        astra_sim_dir = os.path.join(script_dir, 'astra-sim')
    ns3_dir = os.path.join(astra_sim_dir, "extern", "network_backend", "ns-3")
    ns3_executable = os.path.join(ns3_dir, "build", "scratch", "ns3.42-AstraSimNetwork-default")
    
    if not os.path.exists(ns3_executable):
        raise RuntimeError(f"NS3 executable not found at: {ns3_executable}")
    
    # Convert relative paths to absolute paths
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    
    # Make workload path absolute
    if not os.path.isabs(workload_path):
        workload_path = os.path.join(file_dir, workload_path)
    workload_path = os.path.abspath(workload_path)
    
    system = os.path.join(file_dir, system) if not os.path.isabs(system) else system
    network_config = os.path.join(file_dir, network_config) if not os.path.isabs(network_config) else network_config
    logical_topology = os.path.join(file_dir, logical_topology) if not os.path.isabs(logical_topology) else logical_topology
    memory = os.path.join(file_dir, memory) if not os.path.isabs(memory) else memory
    
    # Make all paths absolute
    system = os.path.abspath(system)
    network_config = os.path.abspath(network_config)
    logical_topology = os.path.abspath(logical_topology)
    memory = os.path.abspath(memory)
    
    # Create output directories
    os.makedirs(os.path.join(file_dir, output_dir), exist_ok=True)
    os.makedirs(os.path.join(file_dir, network_log), exist_ok=True)
    
    # Set up log file
    log = os.path.join(file_dir, output_dir, os.path.split(workload_path)[1])
    if suffix is not None:
        log = log + suffix
    
    # Build the NS3 command with absolute paths
    comm_group_config = f"{workload_path}.json"
    
    # Create absolute logging folder path to fix the filesystem error
    log_abs_path = os.path.abspath(log)
    
    cmd = (
        f"cd {ns3_dir}/build/scratch && "
        f"{ns3_executable} "
        f"--workload-configuration={workload_path} "
        f"--system-configuration={system} "
        f"--network-configuration={network_config} "
        f"--logical-topology-configuration={logical_topology} "
        f"--remote-memory-configuration={memory} "
        )

    # Conditionally add comm-group-configuration if the file exists
    comm_group_config_path = f"{workload_path}.json"
    if os.path.exists(comm_group_config_path):
        cmd += f"--comm-group-configuration={comm_group_config_path} "
    # Add logging arguments
    cmd += (
        f"--logging-configuration=empty "
        f"--logging-folder={log_abs_path} "
    )
    
    print("Running NS3 simulation with command:")
    print(cmd)
    
    success = run_command(cmd, cwd=os.path.join(ns3_dir, "build", "scratch"))
    
    if success:
        # Process results if simulation was successful
        trace_file = f"{log}_trace.csv"
        if os.path.exists(trace_file):
            get_timings_df(trace_file, f"{log}_trace_matched_timing.csv")
    
    if not success:
        return cmd
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AstraSim with NS3 network backend")
    parser.add_argument(
        "--workload_dir",
        type=str,
        help="The folder containing the workload files",
        required=True,
    )
    parser.add_argument(
        "--system", 
        type=str, 
        help="The system configuration file", 
        required=True
    )
    parser.add_argument(
        "--network_config", 
        type=str, 
        help="The NS3 network configuration file", 
        required=True
    )
    parser.add_argument(
        "--logical_topology", 
        type=str, 
        help="The logical topology configuration file", 
        required=True
    )
    parser.add_argument(
        "--memory", 
        type=str, 
        help="The memory configuration file", 
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory for simulation results",
        required=True,
    )
    parser.add_argument(
        "--network_log",
        type=str,
        help="The directory for network logs",
        required=True,
    )
    
    args = parser.parse_args()

    # Get list of workloads to simulate
    design_space = list_workloads(str(args.workload_dir))
    
    # Create partial function with fixed arguments
    func = partial(
        run_astrasim_ns3,
        system=args.system,
        network_config=args.network_config,
        logical_topology=args.logical_topology,
        memory=args.memory,
        output_dir=args.output_dir,
        network_log=args.network_log,
    )

    # Run simulations in parallel
    with multiprocessing.Pool(int(2)) as pool:
        failed_cmds = pool.map(func, design_space)
        print("\n\nFailed commands:")
        for cmd in failed_cmds:
            if cmd != "":
                print(cmd)