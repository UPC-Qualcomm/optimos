#!/bin/bash

# This script runs the brute-force exploration of all configurations for a given model and NPU count.
# It then generates the Hsitogram graph similar to Figure 1 in the paper.

# Chack the respective scripts for the details of the commands.
time python ./Optimization/examples/example_bruteforce.py


#NOTE:
# If you change the output folder of the previous step, make sure to update the path to the CSV file in the command below.
python ./Optimization/histogram/plot_exec_time_histogram.py ./output/BRUTEFORCE_llama_8b_bruteforce_analytical/FoldedClos/bruteforce_results_llama_8b_bruteforce_analytical.csv --bins 100 --combined-stats --overlay-oom