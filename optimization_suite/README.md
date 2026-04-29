# SLURM Optimization Suite

This folder contains the scripts and experiment definitions used to launch DeepHyper optimization runs on a SLURM cluster or on a local Linux machine.

The experiment bundles cover these model/objective groups:

- llama-8b (32 NPUs): time, latencu_network_bw, .. ,etc
- gpt-60b (model_num=19, 128 NPUs): time, latencu_network_bw, .. ,etc
- llama-70b (128 NPUs): time, latencu_network_bw, .. ,etc

## Setup

1. Open `suite_paths.env` and set the machine-specific paths.
   - `SCRATCH_BASE` should point to the root where the `astra-sim` tree and run outputs live.
   - `STG_TMP_BASE` should point to a writable temp location. Use `/tmp` on a local machine, or a scratch-backed path on SLURM.
2. Make sure the suite is available under `${SCRATCH_BASE}/astra-sim/optimos/optimization_suite` on the machine where you run it.
3. Confirm the required tooling is available:
   - SLURM mode needs `sbatch` and `sinfo` on a login node.
   - Local mode needs a working Python environment and the dependencies required by the experiment scripts.
4. You do not normally run `start_py_311.sh` directly. It is sourced by each experiment to prepare the Python and temp-directory environment.

You can check `MACHINE_SETUP.md` for further details

## Where the experiments live

The actual experiments are in the `experiments/` directory. Each experiment has its own subdirectory, for example:

```text
experiments/llama8b_32npus_time/
experiments/gpt60b_128npus_edp/
experiments/llama70b_128npus_energy_and_time/
```

Inside each experiment directory you will usually find:

- `config.env` for experiment-specific settings and overrides
- `run_experiment.sh` to launch the optimization job
- `inputs/search_space.json` for the DeepHyper search space
- `logs/`, `experiments/` and `outputs/` created at runtime

## Run a single experiment on SLURM

Use `launch_all.sh` to submit the active experiment list to the cluster:

```bash
cd /path/to/astra-sim/optimos/optimization_suite
bash launch_all.sh
```

Useful options:

- `--dry-run` prints the planned submissions without calling `sbatch`
- `--max-jobs N` limits how many experiments are submitted
- `--workers W` overrides the worker count for every experiment
- `--tracker true|false` forces the tracker on or off
- `--patience N` overrides early-stopping patience

To run only one experiment, edit the `ACTIVE_EXPERIMENTS` array in `launch_all.sh` so it contains just the experiment you want, then run the script. The launcher will only submit the directories listed in that array.

## Run a single experiment locally

Use `launch_local.sh` when you want to execute the same experiment logic without SLURM:

```bash
cd /path/to/astra-sim/optimos/optimization_suite
bash launch_local.sh --parallel 1
```

Useful options:

- `--dry-run` prints what would be launched
- `--parallel N` runs up to N experiments concurrently
- `--max-jobs N` limits how many experiment directories are processed
- `--workers W` overrides the number of workers for each experiment
- `--tracker true|false` overrides the tracker setting
- `--patience N` overrides early-stopping patience

To run exactly one experiment locally, comment out every other entry in the `ACTIVE_EXPERIMENTS` array inside `launch_local.sh`.

## Sweep multiple options

Use `run_sweep_combinations.sh` when you want to evaluate how the tracker and early-stopping mechanism affect the optimization runs. The sweep tests combinations where the tracker is enabled or disabled, and where early stopping is kept at the experiment default or overridden with a patience value.

```bash
cd /path/to/astra-sim/optimos/optimization_suite
bash run_sweep_combinations.sh
```

You can also call the launcher scripts directly when you want one specific configuration:

```bash
bash launch_local.sh --tracker true --patience 60 --workers 6
bash launch_local.sh --tracker false --patience -1 --workers 1
bash launch_all.sh --tracker true --patience 60 --workers 6
bash launch_all.sh --tracker false --patience -1 --workers 1
```

The sweep script is just a convenience wrapper that runs all default combinations defined at the top of the file:

- `TRACKER_VALUES`
- `PATIENCE_VALUES`
- `WORKERS_VALUES`

Examples:

```bash
bash run_sweep_combinations.sh --dry-run
bash run_sweep_combinations.sh --parallel 2
bash run_sweep_combinations.sh --slurm
bash run_sweep_combinations.sh --slurm --dry-run
```

The sweep script calls:

- `launch_local.sh` in local mode
- `launch_all.sh` in SLURM mode

## What each file is for

- `README.md`: usage instructions for the suite
- `suite_paths.env`: machine-specific path configuration shared by all scripts
- `start_py_311.sh`: environment bootstrap sourced by each experiment run
- `launch_all.sh`: submits the active experiments to SLURM
- `launch_local.sh`: runs the active experiments locally without SLURM
- `run_sweep_combinations.sh`: sweeps tracker, patience, and worker combinations
- `cleanup_all.sh`: removes generated logs and outputs from selected experiments
- `cleanup_core_files.sh`: finds and removes `core.*` crash dumps
- `MACHINE_SETUP.md`: more detailed machine setup notes
- `experiments/`: experiment definitions and inputs
- `launch_logs/`: timestamped logs created by the launch scripts

## Resource policy defaults

The launchers apply these defaults unless you override them in `config.env` or from the command line:

- CPU cores per job: 32% of the assigned node cores
- Memory per CPU is 4G

Per-experiment overrides belong in each experiment's `config.env` file.
