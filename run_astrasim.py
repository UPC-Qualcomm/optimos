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

    #merged_df["issue_tick"] = merged_df["issue_tick"].astype(int)
    #merged_df["callback_tick"] = merged_df["callback_tick"].astype(int)
    merged_df["elapsed_time"] = merged_df["callback_tick"] - merged_df["issue_tick"]
    merged_df.fillna(0, inplace=True)

    #merged_df.to_csv(output_file_name)
    # TODO: To optimize performance Add the exposed and overlaped amount of cycles to each node 
    # Use best_combinatoin.py
    num_sys = int(merged_df["sys_id"].max())

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

    
    #os.remove(csv_trace_file)

def build_compute_interval_tree(nodes):
    tree = IntervalTree()
    for row in nodes.itertuples(index=False):
        tree.addi(row.issue_tick, row.callback_tick, row.node_id)
    tree.merge_overlaps()  # OPTIMIZED: merge once here
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

def run_command(command, cwd=None, return_pid=False):
    
    # Find the specific number pattern in the command string
    match = re.search(r'(\d+_\d+_\d+_\d+_\d+)', command)
    identifier = ""
    if match:
        identifier = f" for {match.group(1)}"

    print("run simulation command:", command)
    start_time = time.time()
    
    if return_pid:
        # Use Popen to get PID for early termination
        # Create a new process group so we can kill the entire group including children
        # Return process object so caller can manage waiting
        process = subprocess.Popen(
            command, 
            shell=True, 
            cwd=cwd,
            preexec_fn=os.setsid  # Create new process group
        )
        return process, identifier, start_time
    else:
        result = subprocess.run(command, shell=True, cwd=cwd)
        end_time = time.time()
        print(f"Total time{identifier}: {end_time - start_time:.2f} seconds")
        return result.returncode == 0


def list_workloads(root):
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(".0.et"):
            filtered.append(os.path.join(root, file[:-5]))
    return filtered


def post_process_simulation(log_path, keep_trace=False):
    """
    Post-process simulation results after completion.
    
    Args:
        log_path: Base path for simulation output files (without extension)
        keep_trace: If True, keep the raw trace CSV file after processing
    
    Returns:
        True if post-processing succeeded, False otherwise
    """
    err_file = f'{log_path}.err'
    if os.path.exists(err_file) and os.path.getsize(err_file) == 0:
        try:
            get_timings_df(f"{log_path}_trace.csv", f"{log_path}_trace_matched_timing.csv")
            if not keep_trace:
                os.remove(f"{log_path}_trace.csv")
        except Exception as e:
            print(f"    ⚠️  Error processing trace: {e}")
            return False
        try:
            os.remove(err_file)
        except:
            pass
        return True
    return False


def run_astrasim(workload_path, system, network, memory, output_dir, network_log, sim_type, suffix=None, return_pid=False, keep_trace=False):
    #astrasim_root = os.environ.get("ASTRA_SIM")
    #if astrasim_root is None:
    #    raise RuntimeError("ASTRA_SIM is not set.")

    if sim_type == "analytical_unaware":
        astrasim_bin = os.environ.get("ASTRA_SIM_BIN_UNAWARE")
    elif sim_type == "analytical_aware":
        astrasim_bin = os.environ.get("ASTRA_SIM_BIN_AWARE")
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")

    if astrasim_bin is None:
        if sim_type == "analytical_unaware":
            raise RuntimeError("ASTRA_SIM_BIN_UNAWARE is not set.")
        elif sim_type == "analytical_aware":
            raise RuntimeError("ASTRA_SIM_BIN_AWARE is not set.")

    file_dir = os.path.split(os.path.abspath(__file__))[0]

    system = os.path.join(file_dir, system)
    network = os.path.join(file_dir, network)
    memory = os.path.join(file_dir, memory)
    os.makedirs(os.path.join(file_dir, output_dir), exist_ok=True)
    os.makedirs(os.path.join(file_dir, network_log), exist_ok=True)
    log = os.path.join(file_dir, output_dir, os.path.split(workload_path)[1])
    if suffix is not None:
        log = log + suffix
    # with open(log, 'w') as outfile:
    #    pass
    
    # Build network log file path with suffix
    network_log_file = os.path.join(
        file_dir, network_log, os.path.split(workload_path)[1]
    )
    if suffix is not None:
        network_log_file = network_log_file + suffix
    network_log_file = network_log_file + ".csv"
    
    # Base command
    cmd = (
        f"{astrasim_bin} "
        f"--system-configuration={system} "
        f"--workload-configuration={workload_path} "
        f"--network-configuration={network} "
        f"--remote-memory-configuration={memory} "
    )

    # Conditionally add comm-group-configuration if the file exists
    comm_group_config_path = f"{workload_path}.json"
    if os.path.exists(comm_group_config_path):
        cmd += f"--comm-group-configuration={comm_group_config_path} "

    # Add logging arguments
    cmd += (
        f"--logging-folder={log} "
        f"--network-log={network_log_file} "
    )
    
    if return_pid:
        process, identifier, start_time = run_command(cmd, return_pid=True)
        pid = process.pid
        # Return process object and metadata immediately without waiting
        # Caller is responsible for calling process.wait() and post-processing
        return process, pid, log, keep_trace
    else:
        success = run_command(cmd)
        pid = None
    
        if success:
            post_process_simulation(log, keep_trace)
    
        return "" if success else cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload_dir",
        type=str,
        help="The folder containing the workload",
        required=True,
    )
    parser.add_argument(
        "--system", type=str, help="The folder containing the workload", required=True
    )
    parser.add_argument(
        "--network", type=str, help="The folder containing the workload", required=True
    )
    parser.add_argument(
        "--memory", type=str, help="The folder containing the workload", required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The folder containing the workload",
        required=True,
    )
    parser.add_argument(
        "--network_log",
        type=str,
        help="The folder containing the network logs",
        required=True,
    )
    parser.add_argument(
        "--sim_type",
        type=str,
        default="analytical_unaware",
        choices=["analytical_unaware", "analytical_aware"],
        help="The type of simulator to run.",
    )
    args = parser.parse_args()

    design_space = list_workloads(str(args.workload_dir))
    func = partial(
        run_astrasim,
        system=args.system,
        network=args.network,
        memory=args.memory,
        output_dir=args.output_dir,
        network_log=args.network_log,
        sim_type=args.sim_type,
    )

    with multiprocessing.Pool(int(10)) as pool:
        failed_cmds = pool.map(func, design_space)
        print("\n\nrunfails:")
        for cmd in failed_cmds:
            if not cmd == "":
                print(cmd)
