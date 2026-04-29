#!/usr/bin/python3
import re
import argparse
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from intervaltree import IntervalTree
import subprocess

def extract_runtime_results_old(log_path):
    pattern = r"\[(\d+)\] finished, (\d+) cycles, exposed communication (\d+) cycles"
    memory_pattern = r"sys\[(\d+)\] peak memory usage: ([\d.]+) GB"
    oom_pattern = r"sys\[(\d+)\] is OOM: (\d+)"

    sys_ids = []
    exec_cycles = []
    communication_cycles = []
    peak_memory = {}
    is_oom = {}

    with open(log_path, "r") as f:
        log_lines = f.read()
        matches = re.findall(pattern, log_lines)

        if not matches:
            return pd.DataFrame(columns=[
                'sys_id', 'exec_cycles', 'exposed_comm_cycles', 'peak_memory_gb', 'is_oom'
            ])

        for sys_id, exec_cycles, comm_cycles in matches:
            sys_ids.append(sys_id)
            exec_cycles.append(int(exec_cycles))
            communication_cycles.append(int(comm_cycles))
        
        # Extract memory information
        memory_matches = re.findall(memory_pattern, log_lines)
        for sys_id, memory in memory_matches:
            peak_memory[sys_id] = float(memory)
        
        # Extract OOM information
        oom_matches = re.findall(oom_pattern, log_lines)
        for sys_id, oom in oom_matches:
            is_oom[sys_id] = int(oom)

    df = pd.DataFrame({
        'sys_id': sys_ids,
        'exec_cycles': exec_cycles,
        'exposed_comm_cycles': communication_cycles
    })
    
    # Add memory columns
    df['peak_memory_gb'] = df['sys_id'].map(peak_memory)
    df['is_oom'] = df['sys_id'].map(is_oom)

    return df

def extract_runtime_results(file_path, file_identifier="_trace_matched_timing.csv"):
    df = pd.read_csv(file_path)
    num_sys = df["sys_id"].max() 
    
    # Extract memory information from corresponding log file
    log_file_path = file_path.replace(file_identifier, ".log")
    peak_memory = {}
    is_oom = {}
    
    if os.path.exists(log_file_path):
        memory_pattern = r"sys\[(\d+)\] peak memory usage: ([\d.]+) GB"
        oom_pattern = r"sys\[(\d+)\] is OOM: (\d+)"
        
        with open(log_file_path, "r") as f:
            log_lines = f.read()
            
            # Extract memory information
            memory_matches = re.findall(memory_pattern, log_lines)
            for sys_id, memory in memory_matches:
                peak_memory[int(sys_id)] = float(memory)
            
            # Extract OOM information
            oom_matches = re.findall(oom_pattern, log_lines)
            for sys_id, oom in oom_matches:
                is_oom[int(sys_id)] = int(oom)

    out_data = []
    for sys_id in range(num_sys+1):
        sys_df = df[df['sys_id'] == sys_id]

        sys_summary_df = collect_summary(sys_df, sys_id)

        total_cycles = sys_summary_df['total_comm_time'] + sys_summary_df['comp_exposed_to_comm']

        comm_cycles = sys_summary_df['total_comm_time']
        exposed_comm_cycles = sys_summary_df['comm_exposed_to_comp']
        
        comp_cycles = sys_summary_df['total_comp_time']
        exposed_comp_cycles = sys_summary_df['comp_exposed_to_comm']

        out_data.append({
            'sys_id': sys_id,
            'exec_cycles': total_cycles,
            'comm_cycles': comm_cycles,
            'exposed_comm_cycles':exposed_comm_cycles,
            'comp_cycles': comp_cycles,
            'exposed_comp_cycles': exposed_comp_cycles,
            'comm_cycles_percent': comm_cycles * 100 / total_cycles,
            'exposed_comm_cycles_percent':exposed_comm_cycles * 100 / total_cycles,
            'comp_cycles_percent': comp_cycles * 100 / total_cycles,
            'exposed_comp_cycles_percent': exposed_comp_cycles * 100 / total_cycles,
            'peak_memory_gb': peak_memory.get(sys_id, None),
            'is_oom': is_oom.get(sys_id, None)
        })

    return pd.DataFrame(out_data)


def build_interval_tree(nodes):
    tree = IntervalTree()
    for row in nodes.itertuples(index=False):
        tree.addi(row.issue_tick, row.callback_tick)
    tree.merge_overlaps()  
    return tree

def total_active_time(tree):
    return sum(interval.end - interval.begin for interval in tree)

def exposed_time(primary_tree, excluding_tree):
    total_exposed = 0
    for interval in primary_tree:
        overlaps = excluding_tree.overlap(interval.begin, interval.end)
        if not overlaps:
            total_exposed += interval.end - interval.begin
        else:
            sub_intervals = [(interval.begin, interval.end)]
            for o in overlaps:
                new_sub_intervals = []
                for s_start, s_end in sub_intervals:
                    if o.begin >= s_end or o.end <= s_start:
                        new_sub_intervals.append((s_start, s_end))
                    else:
                        if s_start < o.begin:
                            new_sub_intervals.append((s_start, o.begin))
                        if o.end < s_end:
                            new_sub_intervals.append((o.end, s_end))
                sub_intervals = new_sub_intervals
            total_exposed += sum(e - s for s, e in sub_intervals)
    return total_exposed

def collect_summary(df, sys_id):
    comm_nodes = df[df["node_type"].isin([5, 6, 7])]
    comp_nodes = df[df["node_type"] == 4]

    comm_tree = build_interval_tree(comm_nodes)
    comp_tree = build_interval_tree(comp_nodes)

    total_comm_time = total_active_time(comm_tree)
    total_comp_time = total_active_time(comp_tree)

    comm_exposed_to_comp = exposed_time(comm_tree, comp_tree)
    comp_exposed_to_comm = exposed_time(comp_tree, comm_tree)

    return {
        "sys_id": sys_id,
        "total_comm_time": total_comm_time,
        "total_comp_time": total_comp_time,
        "comm_exposed_to_comp": comm_exposed_to_comp,
        "comp_exposed_to_comm": comp_exposed_to_comm,
    }


def extract_runtime_results_dir(log_dir, output_dir, file_identifier):
    df = extract_runtime_results(log_dir, file_identifier)
    file_name = os.path.basename(log_dir).replace(file_identifier,  "_res.csv")
    path = os.path.join(output_dir, file_name)
    if not df.empty:
        df.to_csv(path, index=False)

def list_logs(root, endwith_str):
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(endwith_str):
            filtered.append(os.path.join(root, file))
    return filtered


def extract_slowest_npu(logs, output_filename):
    slowest_rows = []

    for log in logs:
        df = pd.read_csv(log)
        slowest_row = df[df["sys_id"] == df["sys_id"].min()].copy()
        
        # Check if ANY NPU/system has OOM
        any_oom = False
        if 'is_oom' in df.columns:
            any_oom = (df['is_oom'] == 1).any() or (df['is_oom'] > 0).any()
        
        file_base = os.path.splitext(log)[0].split('/')[-1]
        info = file_base.split(".")
        parallelism_str = info[0]  # or file_base[:9] if format is fixed
        seq = info[1].split("_")[1]
        batch = info[2].split("_")[1]
        parallelism_list = parallelism_str.split("_")
        slowest_row["dp_mp_sp_pp_sharded"] = f'{parallelism_list[0]}_{parallelism_list[1]}_{parallelism_list[2]}_{parallelism_list[3]}_{parallelism_list[4]}'
        slowest_row["dp"] = parallelism_list[0]
        slowest_row["mp"] = parallelism_list[1]
        slowest_row["sp"] = parallelism_list[2]
        slowest_row["pp"] = parallelism_list[3]
        slowest_row["sharding"] = parallelism_list[4]
        slowest_row["seq"] = seq
        slowest_row["batch"] = batch
        
        # Override is_oom to reflect if ANY system had OOM
        slowest_row["is_oom"] = any_oom

        slowest_rows.append(slowest_row)

    summary_df = pd.concat(slowest_rows, ignore_index=True)

    desired_order = ["dp_mp_sp_pp_sharded", "dp", "mp", "sp", "pp", "sharding"]
    other_columns = [col for col in summary_df.columns if col not in desired_order]
    summary_df = summary_df[desired_order + other_columns]

    summary_df.to_csv(output_filename, index=False)



def str_to_bool(v):
    # Convert "true" to True and "false" to False
    return v.lower() in ("true", "t", "1", "yes", "y")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim_logfile", type=str, help="The output log file or directory", required=True
    )
    parser.add_argument(
        "--output_filename", type=str, help="The results file name (only if single file mode)"
    )
    parser.add_argument(
        "--include_power", type=str_to_bool, help="Include power results", default=False,
    )

    args = parser.parse_args()

    if os.path.isfile(args.sim_logfile):
        print(args.sim_logfile)
        print(args.output_filename)
        runtimes_df = extract_runtime_results(args.sim_logfile, file_identifier = "_trace_matched_timing.csv")
        if not args.output_filename:
            raise ValueError("You must specify --output_filename when processing a single log file.")
        runtimes_df.to_csv(args.output_filename, index=False)
        #print(f"Results saved to {args.output_filename}")

    elif os.path.isdir(args.sim_logfile):
        os.makedirs(args.output_filename,  exist_ok=True)
        #runtimes_dfs, filenames = extract_runtime_results_dir(args.sim_logfile)
        file_identifier = "_trace_matched_timing.csv"
        logs = list_logs(args.sim_logfile, file_identifier)
        with multiprocessing.Pool() as pool:
            pool.starmap(extract_runtime_results_dir, [(log, args.output_filename, file_identifier) for log in logs])
        
        # Filter the slowest NPU data in all the experiments the belong to a topology.
        print(args.output_filename)
        logs = list_logs(args.output_filename, "_res.csv")
        out_filename = args.output_filename + ".csv" if len(args.output_filename.split("/")) == 4 else args.output_filename + args.output_filename.split("/")[3] + ".csv"
        extract_slowest_npu(logs, out_filename)
        
        if args.include_power:
            command = (
                f"python merge_power_results.py "
                f"--results {out_filename} "
                f"--power-dir {args.sim_logfile} "
                f"--output {str.replace(out_filename, '.csv', '_with_power.csv')} "
            )
            result = subprocess.run(command, shell=True)

    else:
        raise ValueError(f"{args.sim_logfile} is neither a file nor a directory.")


    
    