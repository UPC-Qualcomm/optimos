#!/usr/bin/env python3
"""
Dedicated DeepHyper sweep example.

This variant exists only to run the same optimization setup across multiple
objective functions without modifying `example_deephyper_opt.py`.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Optimization import (
    create_search_space,
    RandomSampler,
    SimulationRunner,
    DeepHyperOptimizer,
    create_objective,
    get_available_objective_types,
)
from Optimization.core.base_optimizer import format_score


# plot_directions: per-objective display direction, matching objective.score_directions.
# DeepHyper negates minimized objectives in the CSV, so 'min' applies sign=-1 for display.
# Derived from objective.score_directions (True=min, False=max) via create_objective().
OBJECTIVE_METADATA = {
    "time":                       {"plot_labels": [],                                                                                          "plot_directions": []},
    "time_and_network_bw":        {"plot_labels": [],                                                                                          "plot_directions": []},
    "power":                      {"plot_labels": [],                                                                                          "plot_directions": []},
    "energy":                     {"plot_labels": [],                                                                                          "plot_directions": []},
    "edp":                        {"plot_labels": [],                                                                                          "plot_directions": []},
    "ed2p":                       {"plot_labels": [],                                                                                          "plot_directions": []},
    "e2d":                        {"plot_labels": [],                                                                                          "plot_directions": []},
    "power_and_time":             {"plot_labels": ["Total Power (W)", "Training Time (s)"],                                                    "plot_directions": ["min", "min"]},
    "energy_and_time":            {"plot_labels": ["Total Energy (J)", "Training Time (s)"],                                                   "plot_directions": ["min", "min"]},
    "latency_total_network":      {"plot_labels": ["Training Time (s)", "Network Total BW (GB/s)"],                                           "plot_directions": ["min", "min"]},
    "latency_network":            {"plot_labels": ["log10(Training Time (s))", "log10(Network Total BW (GB/s))"],                             "plot_directions": ["min", "min"]},
    "latency_memory":             {"plot_labels": ["log10(Training Time (s))", "log10(Total Memory (GB))"],                                   "plot_directions": ["min", "min"]},
    "network_memory":             {"plot_labels": ["log10(Network Total BW (GB/s))", "log10(Total Memory (GB))"],                            "plot_directions": ["min", "min"]},
    "latency_network_raw":        {"plot_labels": ["Training Time (s)", "Network Total BW (GB/s)"],                                          "plot_directions": ["min", "min"]},
    "latency_network_minmax":     {"plot_labels": ["Normalized Training Time", "Normalized Network BW"],                                     "plot_directions": ["min", "min"]},
    "latency_network_sqrt":       {"plot_labels": ["sqrt(Training Time (s))", "sqrt(Network Total BW (GB/s))"],                              "plot_directions": ["min", "min"]},
    "latency_network_power":      {"plot_labels": ["Training Time (s)^p", "Network BW^p"],                                                   "plot_directions": ["min", "min"]},
    "edp_and_network_bw":         {"plot_labels": ["EDP (J * cycles)", "Network Total BW (GB/s)"],                                          "plot_directions": ["min", "min"]},
    "ed2p_and_network_bw":        {"plot_labels": ["ED\u00b2P (J * cycles\u00b2)", "Network Total BW (GB/s)"],                              "plot_directions": ["min", "min"]},
    "e2d_and_network_bw":         {"plot_labels": ["E\u00b2D (J\u00b2 * cycles)", "Network Total BW (GB/s)"],                              "plot_directions": ["min", "min"]},
    # Special cases: mixed minimize/maximize directions
    "time_and_throughput_per_energy": {"plot_labels": ["Training Time (s)", "Throughput/Energy (samples/J)"],                              "plot_directions": ["min", "max"]},
    "memory_and_time":            {"plot_labels": ["Total Memory (GB)", "Training Time (s)"],                                               "plot_directions": ["max", "min"]},
    # 3-objective (auto-plotting skipped — only 2D is supported)
    "latency_network_memory":     {"plot_labels": ["log10(Training Time (s))", "log10(Network Total BW (GB/s))", "log10(Total Memory (GB))"], "plot_directions": ["min", "min", "min"]},
    "energy_cycles_and_network_bw": {"plot_labels": ["Total Energy (J)", "Training Time (s)", "Network Total BW (GB/s)"],                  "plot_directions": ["min", "min", "min"]},
    "power_cycles_network_bw":    {"plot_labels": ["Total Power (W)", "Training Time (s)", "Network Total BW (GB/s)"],                      "plot_directions": ["min", "min", "min"]},
}

DEFAULT_OBJECTIVE = "e2d_and_network_bw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepHyper optimization sweep with configurable model/objective/search-space settings."
    )
    parser.add_argument("objective", nargs="?", default=None, help="Objective key (legacy positional argument support)")
    parser.add_argument("--objective", dest="objective_flag", default=None, help="Objective key")
    parser.add_argument("--list-objectives", action="store_true", help="List all available objective keys and exit")

    parser.add_argument("--model-num", type=int, default=19, help="Model enum/index used by SimulationRunner")
    parser.add_argument("--model-name", default=None, help="Model name for output naming")
    parser.add_argument("--num-npus", type=int, default=64, help="Number of NPUs")
    parser.add_argument("--network-name", default="FoldedClos", help="Network name")
    parser.add_argument("--budget", type=int, default=100, help="Optimization budget")
    parser.add_argument("--init-samples", type=int, default=20, help="Number of random initial samples")
    parser.add_argument("--n-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--top-k", type=int, default=10, help="Keep top-k checkpoints")
    parser.add_argument("--cleanup-batch-size", type=int, default=20, help="Cleanup frequency")
    parser.add_argument("--folder-prefix", default="EXAMPLE_DEEPHYPER", help="Result directory prefix")
    parser.add_argument("--search-space-path", default=None, help="Path to search space JSON")
    parser.add_argument("--compress-and-clean", action="store_true", help="Enable result compression and cleanup")

    parser.add_argument("--include-categories", default=None, help="Comma-separated list of search space categories")
    parser.add_argument("--enable-tracker", action="store_true", help="Enable DeepHyper's built-in tracker for early termination")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=-1,
        help=(
            "Stop the search after this many consecutive non-improving evaluations. "
            "Set to -1 (default) to disable. Recommended: 3-5x --n-workers "
            "(e.g. --early-stopping-patience 50 for 10 workers)."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-evaluations",
        type=int,
        default=0,
        help=(
            "Minimum number of valid (non-failure) evaluations before the early-stopping "
            "patience counter begins. Set to at least --init-samples to protect the "
            "random exploration phase. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--search-type",
        default="cbo",
        choices=["cbo", "random"],
        help="DeepHyper search type: 'cbo' (Bayesian, default) or 'random' (random search).",
    )
    return parser.parse_args()


def get_objective_key(args: argparse.Namespace) -> str:
    available_objectives = get_available_objective_types()
    if args.list_objectives:
        print("\n".join(available_objectives))
        sys.exit(0)

    objective_key = args.objective_flag or args.objective or DEFAULT_OBJECTIVE
    if objective_key not in available_objectives:
        available = ", ".join(sorted(available_objectives))
        raise ValueError(f"Unknown objective '{objective_key}'. Available: {available}")
    return objective_key


def main():
    args = parse_args()
    objective_key = get_objective_key(args)
    objective_meta = OBJECTIVE_METADATA.get(objective_key, {"plot_labels": []})
    
    if args.include_categories:
        include_categories = [x.strip() for x in args.include_categories.split(",")]
    else:
        include_categories = ["parallelism_strategy", "network"]  # default
            
    
    MODEL_NUM = args.model_num
    MODEL_NAME = f'{args.model_name}_{objective_key}' if args.model_name else f"GPT_40B_{objective_key}"
    NETWORK_NAME = args.network_name
    BUDGET = args.budget
    INIT_SAMPLES = args.init_samples
    N_WORKERS = args.n_workers
    TOP_K = args.top_k
    CLEANUP_BATCH_SIZE = args.cleanup_batch_size
    COMPRESS_AND_CLEAN_IS_ENABLED = args.compress_and_clean
    ENABLE_TRACKER = args.enable_tracker
    multiplier = 50
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    EARLY_STOPPING_MIN_EVALUATIONS = args.early_stopping_min_evaluations if args.early_stopping_min_evaluations > 0 else INIT_SAMPLES
    SKIP_SIM = True
    default_search_space_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "search_space",
        "parallelism_strategy_params.json",
    )
    search_space_path = args.search_space_path or default_search_space_path
    search_space_path = os.path.abspath(search_space_path)
    if not os.path.isfile(search_space_path):
        raise FileNotFoundError(f"Search space file not found: {search_space_path}")

    
    print("=" * 70)
    print("EXAMPLE: DeepHyper Bayesian Optimization Sweep")
    print("=" * 70)
    print(f"Objective key: {objective_key}")
    print(f"Model: {MODEL_NAME}")
    print(f"Network: {NETWORK_NAME}")
    print(f"Budget: {BUDGET} evaluations")
    print(f"Workers: {N_WORKERS} (parallel evaluation)")
    print(f"Search space: {search_space_path}")
    print(f"Skip Simulation: {SKIP_SIM}\n")
    if ENABLE_TRACKER:
        print("Tracker: Enabled (kill at 1.5x threshold)\n")
    else:
        print("Tracker: Disabled\n")
    run_tag = f"{MODEL_NAME}_{objective_key}_{EARLY_STOPPING_PATIENCE}_{ENABLE_TRACKER}_{N_WORKERS}_{SKIP_SIM}_sweep"
    print(run_tag)
    print("1. Creating search space...")
    search_space = create_search_space(
        search_space_path,
        include_categories=include_categories,
    )
    

    print("\n2. Creating sampler...")
    sampler = RandomSampler(seed=42)
    print(f"   Using: {sampler}")


    print("\n3. Creating simulation runner...")
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        network_name=NETWORK_NAME,
        folder_prefix=args.folder_prefix,
        verbose=True,
        skip_sim=SKIP_SIM,
    )
    print(f"   Using: {sim_runner}")

    print("\n4. Creating objective function...")
    objective = create_objective(objective_type=objective_key)
    print(f"   Using: {objective.name}")

    plot_labels = objective_meta["plot_labels"]
    n_obj = len(plot_labels)
    moo_weight = [1.0 / n_obj] * n_obj if n_obj > 1 else None

    print("\n5. Creating DeepHyper optimizer...")
    optimizer = DeepHyperOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=BUDGET,
        objective=objective,
        init_samples=INIT_SAMPLES,
        n_workers=N_WORKERS,
        acq_func="UCBd",
        acq_func_kwargs={"kappa": 10.0, "scheduler": {"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.01}},
        surrogate_model="ET",
        surrogate_model_kwargs={"max_features": "sqrt"},
        acq_optimizer="mixedga",
        random_state=42,
        verbose=True,
        keep_top_k=TOP_K,
        profile_time=True,
        evaluator_method="process",
        acq_optimizer_kwargs={"max_total_failures": -1, "acq_optimizer_freq": 2},
        moo_scalarization_strategy="AugChebyshev",
        moo_scalarization_weight=moo_weight,
        enable_tracker=ENABLE_TRACKER,
        tracker_kill_multiplier=1.5,
        tracker_initial_threshold=1e15,
        cleanup_batch_size=CLEANUP_BATCH_SIZE,
        compress_and_clean_is_enabled=COMPRESS_AND_CLEAN_IS_ENABLED,
        # Early stopping: disabled by default (-1). Enable with --early-stopping-patience.
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_evaluations=EARLY_STOPPING_MIN_EVALUATIONS,
        search_type=args.search_type,
        save_dir=os.path.join("./experiments", run_tag),
        results_filename=f"{run_tag}.csv",
    )
    print(f"   Using: {optimizer}")
    if optimizer.tracker:
        print(f"   Tracker: {optimizer.tracker}")

    print("\n" + "=" * 70)
    print("STARTING OPTIMIZATION")
    print("=" * 70)

    best_config, history = optimizer.run()

    if best_config is not None:
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        config_str = ", ".join([f"{k}={v}" for k, v in best_config.items()])
        print("\n🏆 BEST CONFIGURATION:")
        print(f"   {config_str}")
        print(f"   Score: {format_score(optimizer.best_score)}")
        print(f"\n📊 History saved with {len(history)} evaluations")

        if optimizer.tracker:
            print(optimizer.tracker)
            killed_count = history["was_killed"].sum() if "was_killed" in history.columns else 0
            kill_percentage = (killed_count / len(history) * 100) if len(history) > 0 else 0
            print(f"   Simulations killed: {killed_count}/{len(history)} ({kill_percentage:.1f}%)")
            if killed_count > 0:
                print(f"   ✅ Saved time by terminating {killed_count} slow simulation(s) early")

        print("\n💡 TIP: Check the objective-specific deephyper_results_*.csv for detailed DeepHyper output")

        if len(plot_labels) == 2:
            print("\n" + "=" * 70)
            print("GENERATING PLOTS")
            print("=" * 70)

            print("\n1. Plotting Pareto front (with outlier removal)...")
            try:
                from plot_pareto_front import plot_pareto_front

                csv_path = os.path.join(optimizer.save_dir, optimizer.results_filename)
                model_name = getattr(optimizer.simulation_runner, "model_name", "model")
                output_base = os.path.join(optimizer.save_dir, f"./pareto_front_{model_name}")

                plot_dirs = objective_meta.get("plot_directions", ["min", "min"])
                pareto_plots = plot_pareto_front(
                    results_file=csv_path,
                    obj0_name=plot_labels[0],
                    obj1_name=plot_labels[1],
                    output_file=output_base,
                    plot_format="both",
                    show_labels=True,
                    remove_outliers=False,
                    iqr_multiplier=1.5,
                    obj0_direction=plot_dirs[0] if len(plot_dirs) > 0 else "min",
                    obj1_direction=plot_dirs[1] if len(plot_dirs) > 1 else "min",
                )
            except Exception as e:
                print(f"⚠️  Error plotting Pareto front: {e}")
                pareto_plots = None

            print("\n2. Plotting hypervolume indicator...")
            hv_path, hvi = optimizer.plot_hypervolume()

            if pareto_plots or hv_path:
                print("\n" + "=" * 70)
                print("VISUALIZATION COMPLETE")
                print("=" * 70)
                print("\n📈 Generated plots:")
                if pareto_plots:
                    for plot_path in pareto_plots:
                        if plot_path.endswith(".html"):
                            print(f"   - Pareto Front (Interactive): {plot_path}")
                        elif plot_path.endswith(".png"):
                            print(f"   - Pareto Front (Static): {plot_path}")
                if hv_path:
                    final_hvi = hvi[-1] if hasattr(hvi, "__len__") and len(hvi) > 0 else hvi
                    print(f"   - Hypervolume: {hv_path}")
                    print(f"   - Final HVI: {final_hvi:.4f}")
        elif len(plot_labels) > 2:
            print("\nℹ️  Skipping automatic plotting: this sweep example only auto-plots 2-objective runs.")
        else:
            print("\nℹ️  Skipping Pareto/hypervolume plots for single-objective runs.")
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()
