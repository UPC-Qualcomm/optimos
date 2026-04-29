#!/usr/bin/env python3
"""
Example: DeepHyper Bayesian Optimization

Demonstrates how to use the DeepHyper-based optimizer within the modular 
optimization framework. DeepHyper provides a mature, well-tested BO implementation
with advanced features and efficient parallel evaluation.
"""

import sys
import os

# Add grandparent directory to path to find Optimization package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Optimization import (
    create_search_space,
    RandomSampler,
    SimulationRunner,
    DeepHyperOptimizer,
    create_objective,
    CustomObjective
)
from Optimization.core.base_optimizer import format_score

def main():
    """Run DeepHyper Bayesian Optimization example."""
    
    # Configuration
    MODEL_NUM = 5 # GPT_40B (Model enum value)
    MODEL_NAME = "GPT_1300M_test"  # Descriptive name for results folder and plots
    NUM_NPUS = 64
    NETWORK_NAME = "FoldedClos"
    BUDGET = 50
    INIT_SAMPLES = 10
    N_WORKERS = 10
    TOP_K = 10  # Number of top configurations to keep track of
    CLEANUP_BATCH_SIZE = 120  # Batch size for parallel evaluation (if supported by sim runner)
    COMPRESS_AND_CLEAN_IS_ENABLED = True  # Whether to enable artifact compression and cleanup after each batch of evaluations
    Objective_0_Name = "time"
    Objective_1_Name = "samples_per_sec_per_energy"
    

    print("="*70)
    print("EXAMPLE: DeepHyper Bayesian Optimization")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"NPUs: {NUM_NPUS}")
    print(f"Network: {NETWORK_NAME}")
    print(f"Budget: {BUDGET} evaluations")
    print(f"Workers: {N_WORKERS} (parallel evaluation)")
    print(f"Tracker: Enabled (kill at 1.5x threshold)\n")
    
    # 1. Setup search space
    print("1. Creating search space...")
    search_space_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "search_space", 
        "parallelism_strategy_params.json" 
    )
    search_space = create_search_space(
        search_space_path,
        include_categories=['parallelism_strategy', 'network', 'model']
    )
    
    # 2. Choose sampler (for fallback if needed)
    print("\n2. Creating sampler...")
    sampler = RandomSampler(seed=42)
    print(f"   Using: {sampler}")
    

    # 3. Setup simulation runner
    print("\n3. Creating simulation runner...")
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        network_name=NETWORK_NAME,
        folder_prefix="EXAMPLE_DEEPHYPER",
        verbose=True,
    )
    print(f"   Using: {sim_runner}")
    
    # 4. Create objective function
    print("\n4. Creating objective function...")
    
    objective = create_objective(
        objective_type='latency_network'
    )
    #objective = CustomObjective(
    #    obj_latency_network,  # Raw values - let DeepHyper normalize
    #    "MOO_time_network_total_bw",
    #    minimize=True,
    #    is_multi_objective=True
    #)
    print(f"   Using: {objective.name}")

    # 5. Create DeepHyper optimizer
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
        moo_scalarization_weight=[0.5, 0.5],
        # Use DeepHyper's built-in scaler for normalization
        objective_scaler="minmax",  # Options: 'identity', 'minmax', 'minmaxlog', 'log', 'quantile-uniform'
        # Tracker for early termination (enabled by default)
        enable_tracker=False,
        tracker_kill_multiplier=1.5,
        tracker_initial_threshold=1e15,
        cleanup_batch_size=CLEANUP_BATCH_SIZE,
        compress_and_clean_is_enabled=COMPRESS_AND_CLEAN_IS_ENABLED,
        search_type = "cbo",  # "cbo" or "random"
        # --- Early stopping (search-level no-improvement detector) ---
        early_stopping_patience=-1,
        early_stopping_min_evaluations=INIT_SAMPLES,
    )
    print(f"   Using: {optimizer}")
    if optimizer.tracker:
        print(f"   Tracker: {optimizer.tracker}")
    
    # 6. Run optimization
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION")
    print("="*70)
    
    best_config, history = optimizer.run()
    
    # 7. Display results
    if best_config is not None:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        
        # Format configuration dynamically
        config_str = ", ".join([f"{k}={v}" for k, v in best_config.items()])
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   {config_str}")
        print(f"   Score: {format_score(optimizer.best_score)}")
        print(f"\n📊 History saved with {len(history)} evaluations")
        
        # Show tracker statistics
        if optimizer.tracker:
            print(optimizer.tracker)
            killed_count = history['was_killed'].sum() if 'was_killed' in history.columns else 0
            kill_percentage = (killed_count / len(history) * 100) if len(history) > 0 else 0
            print(f"   Simulations killed: {killed_count}/{len(history)} ({kill_percentage:.1f}%)")
            if killed_count > 0:
                print(f"   ✅ Saved time by terminating {killed_count} slow simulation(s) early")
        
        print(f"\n💡 TIP: Check deephyper_results.csv for detailed DeepHyper output")
    
    
          
        # 8. Generate visualization plots
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        # Plot Pareto front using external script
        print("\n1. Plotting Pareto front (with outlier removal)...")
        try:
            from plot_pareto_front import plot_pareto_front
            
            # Get the CSV file path
            csv_path = os.path.join(optimizer.save_dir, optimizer.results_filename)
            model_name = getattr(optimizer.simulation_runner, 'model_name', 'model')
            output_base = os.path.join(optimizer.save_dir, f"./pareto_front_{model_name}")
            
            # Generate both HTML and PNG plots
            pareto_plots = plot_pareto_front(
                results_file=csv_path,
                obj0_name=Objective_0_Name,
                obj1_name=Objective_1_Name,
                output_file=output_base,
                plot_format="both",
                show_labels=True,
                remove_outliers=True,
                iqr_multiplier=1.5
            )
        except Exception as e:
            print(f"⚠️  Error plotting Pareto front: {e}")
            pareto_plots = None
        
        # Plot hypervolume indicator
        print("\n2. Plotting hypervolume indicator...")
        hv_path, hvi = optimizer.plot_hypervolume()
        
        if pareto_plots or hv_path:
            print("\n" + "="*70)
            print("VISUALIZATION COMPLETE")
            print("="*70)
            print("\n📈 Generated plots:")
            if pareto_plots:
                for plot_path in pareto_plots:
                    if plot_path.endswith('.html'):
                        print(f"   - Pareto Front (Interactive): {plot_path}")
                    elif plot_path.endswith('.png'):
                        print(f"   - Pareto Front (Static): {plot_path}")
            if hv_path:
                print(f"   - Hypervolume: {hv_path}")
                print(f"   - Final HVI: {hvi:.4f}")
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()
