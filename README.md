# optimos

optimos is an optimization framework built on top of STAGE and AstraSim to find optimal system and mapping configurations for training large language models (LLMs).

**Note:** This repository provides access to the public code related to the optimization process. It is intended to share scripts, presets, and helper tools used to run and analyze optimization experiments.

**How to run**

- **Setup (one-time)**: install required tools, build AstraSim, and create the Python virtualenv. From the `optimos` root run:

```bash
bash install.sh
```

- **Run experiments (examples)**: several convenience scripts are provided in the repository. Run them from the `optimos` root.

	- Brute-force exploration — generate a histogram of execution-time distributions:

	```bash
	bash bruteforce_exploration.sh
	```

	- Compare network models — compare ordering between the ns-3 and analytical models:

	```bash
	bash compare_network.sh
	```

	- Run sweep / optimization suite — run the optimizer with different tracker / patience / workers combinations (analytical model):

	```bash
	bash optimization_suite/run_sweep_combinations.sh --parallel 2
	# For SLURM submission use:
	bash optimization_suite/run_sweep_combinations.sh --slurm
	```

		This script will iterate the configured combinations of `tracker`, `patience`, and `workers` and call the appropriate launcher (local or SLURM). See the `optimization_suite` directory for additional README and configuration details.

- **Preset experiments**: preset experiment configurations and example scripts live under `Optimization/examples` and related folders. Use those example configs as starting points for your own runs.

**Notes on the example scripts**

- `bruteforce_exploration.sh`: runs the brute-force exploration and then generates a histogram (see the script for exact output paths).
- `compare_network.sh`: runs analytical and ns-3 experiments and produces comparison CSVs and plots.
- `optimization_suite/run_sweep_combinations.sh`: runs optimization experiments with combinations of tracker (enable/disable), early-stopping patience, and worker counts. See the script header and the `optimization_suite` folder README (if present) for full usage and examples.

If you need help running a specific experiment or want me to add example commands for a particular model/config, tell me which model and I will add a short walkthrough.
