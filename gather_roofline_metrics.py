#!/usr/bin/python3
import re
import os
import argparse

def extract_csv_from_log(file_path, output_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    csv_lines = []
    for line in lines:
        match = re.match(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[workload\] \[info\] \[ROOFLINE\] (.*)$', line)
        if match:
            csv_lines.append(match.group(1))
    
    # Ensure the output file exists before writing
    mode = 'a' if os.path.exists(output_path) else 'w'
    with open(output_path, mode) as out_file:
        if mode == 'a':
            out_file.write("\n")
        out_file.write("\n".join(csv_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim_logfilename", type=str, help="The simulation output file path", required=True
    )
    parser.add_argument(
        "--output_filename", type=str, help="The results file name", required=True
    )
    args = parser.parse_args()


    file_dir = os.path.split(os.path.abspath(__file__))[0]
    os.makedirs(
        os.path.join(file_dir, os.path.dirname(args.output_filename)), exist_ok=True
    )

    extract_csv_from_log(args.sim_logfilename, args.output_filename)


