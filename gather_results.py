#!/usr/bin/python3
import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import csv


def list_logs(root):
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(".log"):
            filtered.append(os.path.join(root, file))
    return filtered


def extract_runtime(log_path):
    log_filename = os.path.split(log_path)[-1]
    dp, mp, sp, pp, sharded = log_filename[:-4].split("_")
    dp, mp, sp, pp, sharded = int(dp), int(mp), int(sp), int(pp), int(sharded)
    exec_cycles = 0
    comm_cycles = 0
    memory = 0
    activation = 0
    gradient = 0
    parameter = 0
    optimizer = 0
    is_oom = False

    pattern = r"(\d+) cycles, exposed communication (\d+) cycles."

    # Lists to store extracted values
    exec_cycles = []
    communication_cycles = []
    memory_ = []
    activation_ = []
    gradient_ = []
    parameter_ = []
    optimizer_ = []

    with open(log_path, "r") as f:
        log_lines = f.read()
        # Extract runtime value for both the compute and exposed communication
        # Extract for all NPUs and average them put

        # Find matches in the log data
        matches = re.findall(pattern, log_lines)

        if len(matches) == 0:
            return (
                dp,
                mp,
                sp,
                pp,
                sharded,
                -1,
                -1,
                ##memory,
                ##activation,
                ##gradient,
                ##parameter,
                ##optimizer,
                ##is_oom,
            )  # Not enough lines in log file

        # Store extracted values in separate lists
        for (
            exec_cycles,
            comm_cycles,
            ##memory,
            ##activation,
            ##gradient,
            ##parameter,
            ##optimizer,
            ##is_oom
        ) in matches:
            exec_cycles.append(int(exec_cycles))
            communication_cycles.append(int(comm_cycles))
            ##memory_.append(int(memory))
            ##activation_.append(int(activation))
            ##gradient_.append(int(gradient))
            ##parameter_.append(int(parameter))
            ##optimizer_.append(int(optimizer))

        # print("Execution Cycles:", exec_cycles)
        # print("Communication Cycles:", communication_cycles)
        exec_cycles = max(exec_cycles) 
        comm_cycles = max(communication_cycles)
        ##memory = sum(memory_) / len(memory_)
        ##activation = sum(activation_) / len(activation_)
        ##gradient = sum(gradient_) / len(gradient_)
        ##parameter = sum(parameter_) / len(parameter_)
        ##optimizer = sum(optimizer_) / len(optimizer_)

    return (
        dp,
        mp,
        sp,
        pp,
        sharded,
        exec_cycles,
        comm_cycles,
        ##memory,
        ##activation,
        ##gradient,
        ##parameter,
        ##optimizer,
        ##is_oom,
    )


def gather_runtimes(root):
    logs = list_logs(root)
    runtimes = None
    with multiprocessing.Pool() as pool:
        runtimes = pool.map(extract_runtime, logs)
    runtimes_dict = dict()
    for (
        dp,
        mp,
        sp,
        pp,
        sharded,
        exec_cycles,
        comm_cycles,
        ##memory,
        ##activation,
        ##gradient,
        ##parameter,
        ##optimizer,
        ##is_oom,
    ) in runtimes:
        if exec_cycles == -1 or comm_cycles == -1:
            continue
        runtimes_dict[(dp, mp, sp, pp, sharded)] = [
            exec_cycles,
            comm_cycles,
            ##memory,
            ##activation,
            ##gradient,
            ##parameter,
            ##optimizer,
            ##is_oom,
        ]
    return runtimes_dict


def get_fails(runtimes):
    fail_cases = list()
    for key in runtimes.keys():
        if runtimes[key] == -1:
            fail_cases.append(key)
        print(key)
    return fail_cases


def visualize1(runtimes, ssp, sharded):
    max_runtimes = max(runtimes.values())
    num = 7
    mat = -1 * np.ones((num, num))
    # vis all data, x=(dp, mp) y=(sp, pp)
    for ddp in range(num):
        x_value = ddp
        for mmp in range(num):
            y_value = mmp
            for ssp in {ssp}:
                ppp = 6 - ddp - mmp - ssp
                rddp, rmmp = int(2**ddp), int(2**mmp)
                rssp, rppp = int(2**ssp), int(2**ppp)
                key = (rddp, rmmp, rssp, rppp, sharded)
                if key not in runtimes:
                    mat[x_value, y_value] = -1
                elif runtimes[key] == -1:
                    mat[x_value, y_value] = -1
                else:
                    mat[x_value, y_value] = runtimes[key] / max_runtimes
    plt.figure(dpi=120)
    sns.heatmap(mat)
    plt.title(f"dp vs mp, x=dp y=mp, sp={ssp}, sharded={sharded}")
    return plt


def visualize2(runtimes, sharded):
    max_runtimes = max(runtimes.values())
    num = 7
    mat = -1 * np.ones((num * num, num))
    # vis all data, x=(dp, mp) y=(sp, pp)
    for ddp in range(num):
        for mmp in range(num):
            x_value = ddp * num + mmp
            for ssp in range(num):
                y_value = ssp
                ppp = 6 - ddp - mmp - ssp
                rddp, rmmp = int(2**ddp), int(2**mmp)
                rssp, rppp = int(2**ssp), int(2**ppp)
                key = (rddp, rmmp, rssp, rppp, sharded)
                if key not in runtimes:
                    mat[x_value, y_value] = -1
                elif runtimes[key] == -1:
                    mat[x_value, y_value] = -1
                else:
                    mat[x_value, y_value] = 1 - runtimes[key] / max_runtimes
    plt.figure(dpi=120)
    sns.heatmap(mat)
    plt.title(f"all results, x=(dp, mp) y=sp, sharded={sharded}")
    return plt


# def serialize_results(runtimes, json_filename):
#    def get_jsonable_dict(dict_):
#        ret = dict()
#        for key in dict_.keys():
#            ret[str(key)] = dict_[key]
#        return ret
#
#    import json
#
#    f = open(json_filename, "w")
#    json.dump(get_jsonable_dict(runtimes), f, indent=4)
#    f.close()


def topk(runtimes, k=10):
    top_k_items = sorted(
        runtimes.items(),
        key=lambda x: -x[1][0] if x[1][0] != -1 else -1e100,
        reverse=True,
    )[:k]
    for key, value in top_k_items:
        print(f"{key}: {value}")
    return top_k_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim_logfiles_dir", type=str, help="The output file path", required=True
    )
    parser.add_argument(
        "--output_filename", type=str, help="The results file name", required=True
    )
    args = parser.parse_args()
    runtimes = gather_runtimes(args.sim_logfiles_dir)
    # hook = 0
    # for sp in range(7):
    #    plt1 = visualize1(runtimes, sp, 0)
    #    plt.savefig(f"dpvsmp_{sp}_ns.png")
    #    plt1 = visualize1(runtimes, sp, 1)
    #    plt.savefig(f"dpvsmp_{sp}_s.png")
    # plt1 = visualize2(runtimes, 0)
    # plt.savefig(f"all_ns.png")
    # plt1 = visualize2(runtimes, 1)
    # plt.savefig(f"all_s.png")
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    os.makedirs(
        os.path.join(file_dir, os.path.dirname(args.output_filename)), exist_ok=True
    )
    # serialize_results(runtimes, args.output_filename)

    with open(args.output_filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(
            [
                "dp_mp_sp_pp_sharded",
                "dp",
                "mp",
                "sp",
                "pp",
                "sharded",
                "exec_cycles",
                "comm_cycles",
                ##"total_memory",
                ##"activation",
                ##"gradient",
                ##"parameter",
                ##"optimizer",
                ##"is_oom",
            ]
        )

        # Write data rows
        for (dp, mp, sp, pp, sharded), (
            exec_cycles,
            comm_cycles,
            ##memory,
            ##activation,
            ##gradient,
            ##parameter,
            ##optimizer,
            ##is_oom,
        ) in runtimes.items():
            writer.writerow(
                [
                    f"{dp}_{mp}_{sp}_{pp}_{sharded}",
                    dp,
                    mp,
                    sp,
                    pp,
                    sharded,
                    exec_cycles,
                    comm_cycles,
                    ##int(memory) / (1024 * 1024 * 1024),
                    ##int(activation) / (1024 * 1024 * 1024),
                    ##int(gradient) / (1024 * 1024 * 1024),
                    ##int(parameter) / (1024 * 1024 * 1024),
                    ##int(optimizer) / (1024 * 1024 * 1024),
                    ##is_oom,
                ]
            )

    print("top20:")
    topk(runtimes, k=20)
    # print("\n\n\nfail cases")
    # get_fails(runtimes)
