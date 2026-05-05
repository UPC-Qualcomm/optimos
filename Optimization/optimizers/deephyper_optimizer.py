"""DeepHyper Bayesian Optimization wrapper for AstraSim."""

import math
import sys
import os
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
import time
import tempfile
import matplotlib.pyplot as plt
import json
import yaml

# Add parent directory to path for imports
sys.path.append(os.environ['OPTIMOS_ROOT'] + '/Optimization')
from ..core import BaseOptimizer, ArtifactCleanupManager
from ..core.base_optimizer import format_score
from ..core.simulation_tracker import SimulationTracker
from ..core.search_early_stopping import AdaptiveSearchEarlyStopping
from ..helper import evaluate_config_worker, workload_generator

try:
    from deephyper.hpo import HpProblem, CBO, RandomSearch
    from deephyper.evaluator import Evaluator
    DEEPHYPER_AVAILABLE = True
except ImportError:
    DEEPHYPER_AVAILABLE = False

# Sentinel returned to DeepHyper for any run that must not influence the
# surrogate model or Pareto front (killed, OOM, or otherwise invalid).
# DeepHyper treats this as the worst possible score in maximisation space.
# Matches the magnitude of the natural-space PENALTY (1e20) but negated,
# since DeepHyper maximises and a very large negative score is "worst".
PENALTY = -1e20

class ConstrainedRandomSearch(RandomSearch):
    """RandomSearch variant that respects HpProblem constraints and sampling_fn."""

    def _ask(self, n: int = 1) -> list[dict[str, Optional[str | int | float]]]:
        # Reuse the same constrained HpProblem sampling path as CBO.
        # This keeps random search and Bayesian optimization aligned on
        # constraint handling and custom sampling_fn behavior.
        return self._problem.sample(size=n)


def _deephyper_evaluate_wrapper(job, optimizer_state):
    """Wrapper for DeepHyper evaluation. Returns objective value or 'F' for failures."""
    from ..helper.config_utils import enrich_config_with_clusters
    
    config = dict(job.parameters)
    simulation_runner = optimizer_state['simulation_runner']
    objective = optimizer_state['objective']
    clusters = optimizer_state.get('clusters')
    
    # Create cache key BEFORE enriching config (so it matches the DataFrame params)
    config_key = tuple(sorted(config.items()))
    # --- Duplicate config cache ---
    # Use optimizer_state['duplicate_eval_cache'] if present, else create it
    duplicate_eval_cache = optimizer_state['duplicate_eval_cache']
    if config_key in duplicate_eval_cache:
        optimizer_state['cache_statistics']['hits'] += 1
        cached_result = duplicate_eval_cache[config_key]
        print(f"    ⚡ Duplicate config detected, returning cached result for {config_key}")
        return cached_result
    else:
        optimizer_state['cache_statistics']['misses'] += 1

    print(f"Cache statistics (cache hits: {optimizer_state['cache_statistics']['hits']}, cache misses: {optimizer_state['cache_statistics']['misses']})")
    if clusters and 'cluster' in config:
        config = enrich_config_with_clusters(config, clusters)

    def _bump_cleanup_counter(name: str, amount: int):
        cleanup_state = optimizer_state.get('periodic_cleanup')
        if cleanup_state is None or amount == 0:
            return
        counters = cleanup_state.get('cleanup_counters')
        if counters is None:
            return
        counters[name] = int(counters.get(name, 0)) + int(amount)
    
    try:
        # ── Tracker warmup: disable killing during initial random phase ──
        # The surrogate needs the full initial sample to train well.
        # During warmup, detach the tracker from the runner so simulations
        # run to completion.  Threshold updates still happen below so the
        # tracker has accurate baselines when it activates for the CBO phase.
        tracker_warmup = optimizer_state.get('tracker_warmup', 0)
        eval_counter = optimizer_state.get('eval_counter')  # Manager ValueProxy
        eval_lock = optimizer_state.get('eval_lock')        # Manager Lock
        in_warmup = False
        if eval_counter is not None and tracker_warmup > 0:
            with eval_lock:
                eval_counter.value += 1
                eval_num = eval_counter.value
            in_warmup = eval_num <= tracker_warmup
            if in_warmup and simulation_runner.tracker is not None:
                # Process-local: forked worker only, does not affect the parent.
                simulation_runner.tracker = None
            if not in_warmup and eval_num == tracker_warmup + 1:
                print(f"\n✓ Tracker warmup complete ({tracker_warmup} evals). "
                      "Simulation killing is now active.")

        returned_config, exec_time, is_oom, file_paths, metadata = evaluate_config_worker(
            config, simulation_runner
        )

        cleanup_reason = ArtifactCleanupManager.get_immediate_cleanup_reason(exec_time, is_oom, metadata)
        was_killed = cleanup_reason == "killed"

        # Failed evaluation with generated files: clean immediately.
        if exec_time is None:
            cleanup_result = ArtifactCleanupManager.cleanup_path_bundle(
                tracked_paths=file_paths,
                verbose=optimizer_state.get('periodic_cleanup', {}).get('verbose', False) if optimizer_state.get('periodic_cleanup') is not None else False,
                print_deleted_files=optimizer_state.get('periodic_cleanup', {}).get('print_deleted_files', False) if optimizer_state.get('periodic_cleanup') is not None else False,
                reason=cleanup_reason,
            )
            if isinstance(file_paths, dict) and file_paths:
                _bump_cleanup_counter('simulations_cleaned', 1)
            if cleanup_result['total_removed'] > 0:
                _bump_cleanup_counter('files_deleted', cleanup_result['total_removed'])
            
            print("    ⚠️  exec_time error")
            duplicate_eval_cache[config_key] = "F"
            return "F"
        
        # Cache exec_time and config files for enrichment
        if 'extra_data_cache' in optimizer_state:
            config_files = {}
            for key in ['system_config', 'network_config', 'memory_config']:
                path = file_paths.get(key)
                if path:
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f) if path.endswith('.json') else yaml.safe_load(f)
                        config_files[key] = json.dumps(data)
                    except Exception as e:
                        print(f"⚠️  Failed to read {key}: {e}")
                        config_files[key] = None
            
            optimizer_state['extra_data_cache'][config_key] = {
                'exec_time': exec_time,
                'config_files': config_files,
                'was_killed': was_killed,
                'is_oom': bool(is_oom),
                'total_power_W': metadata.get('total_power_W') if isinstance(metadata, dict) else None,
                'total_energy_J': metadata.get('total_energy_J') if isinstance(metadata, dict) else None,
                'has_power_metrics': bool(
                    isinstance(metadata, dict)
                    and metadata.get('total_power_W') is not None
                    and metadata.get('total_energy_J') is not None
                ),
            }
            print(f"✓ Cached data for config_key with {len(config_files)} files, exec_time={exec_time}")
        
        score = objective.compute(exec_time, is_oom, metadata, config)
        if score is None:
            
            print("    ⚠️  objective error")
            duplicate_eval_cache[config_key] = "F"
            return "F"

        # Update tracker threshold only on successful completed runs.
        # Do not adapt threshold from OOM, killed, failed, or penalty scores.
        tracker = optimizer_state.get('tracker')
        sim_failed = bool(metadata.get('sim_failed', False)) if isinstance(metadata, dict) else False
        if tracker and exec_time is not None and not bool(is_oom) and not was_killed and not sim_failed:
            # exec_time here is training_time (seconds); use exec_time_ns (nanoseconds,
            # one simulation step) so the tracker compares against trace ticks correctly.
            # Also pass exec_time (seconds) so the tracker can convert mid-run ticks
            # back to training-time-seconds when estimating scores for in-flight sims,
            # fixing the unit mismatch that caused Pareto-efficient configs to be killed.
            raw_exec_time_ns = metadata.get('exec_time_ns') if isinstance(metadata, dict) else None
            tracker.update_threshold(score, raw_exec_time=raw_exec_time_ns,
                                     training_time_s=exec_time)

        cleanup_state = optimizer_state.get('periodic_cleanup')
        if cleanup_state is not None:
            with cleanup_state['lock']:
                cleanup_state['scores'].append(score)
                cleanup_state['file_paths'].append(dict(file_paths))
                current_idx = len(cleanup_state['scores']) - 1

                # Immediate cleanup for killed/OOM runs.
                if cleanup_reason in {"killed", "oom"}:
                    cleanup_result = ArtifactCleanupManager.cleanup_path_bundle(
                        tracked_paths=dict(file_paths),
                        verbose=cleanup_state.get('verbose', False),
                        print_deleted_files=cleanup_state.get('print_deleted_files', False),
                        reason=cleanup_reason,
                    )
                    if isinstance(file_paths, dict) and file_paths:
                        cleanup_state['cleaned_indices'][current_idx] = True
                        _bump_cleanup_counter('simulations_cleaned', 1)
                    if cleanup_result['total_removed'] > 0:
                        _bump_cleanup_counter('files_deleted', cleanup_result['total_removed'])

                ArtifactCleanupManager.run_periodic_cleanup_for_state(
                    cleanup_state=cleanup_state,
                    force=False,
                )
        
        # Killed simulations: return "F" so DeepHyper treats this as a missing
        # observation and does NOT update the surrogate model.
        #
        # Why NOT return PENALTY (-1e20):
        #   Returning PENALTY tells the surrogate "this config has the worst
        #   possible objective on EVERY dimension."  For MOO objectives such as
        #   (log_latency, log_network_bw), the BW dimension is fully determined
        #   by the config (intra/inter-node-bw × npu_count) and is completely
        #   independent of simulation runtime.  A high-BW config that happens to
        #   be slow gets incorrectly labelled as low-BW, training the surrogate
        #   to avoid otherwise-good regions.
        #
        # Why "F" is correct:
        #   We only know exec_time >= kill_threshold — a one-sided bound on one
        #   objective.  We have no information about other objectives.  "F"
        #   (missing data) is the honest representation; the duplicate_eval_cache
        #   entry prevents re-running the same config, so no time is wasted, but
        #   the surrogate landscape is not corrupted.
        if was_killed:
            duplicate_eval_cache[config_key] = "F"
            return "F"
            #if isinstance(score, (tuple, list)):
            #    result = tuple(PENALTY for _ in score)
            #else:
            #    result = PENALTY
            #duplicate_eval_cache[config_key] = result
            #return result

        # Raw → DeepHyper maximization space (per-objective min/max + invalid).
        # float() casts happen inside to_optimizer_score; int64-safe for Pareto/CSV.
        result = objective.to_optimizer_score(score, optimizer_bad_value=PENALTY)
        duplicate_eval_cache[config_key] = result
        return result
        
    except Exception as e:
        print(f"    ⚠️  Evaluation error: {e}")
        duplicate_eval_cache[config_key] = "F"
        return "F"


class DeepHyperOptimizer(BaseOptimizer):
    """Bayesian Optimization using DeepHyper's CBO."""
    
    def __init__(
        self,
        search_space,
        sampler,
        simulation_runner,
        objective,
        budget: int = 30,
        init_samples: int = 20,
        n_workers: int = 1,
        # Search type selection
        search_type: str = "cbo",  # "cbo" or "random"
        # Core CBO parameters
        random_state: Optional[int] = 42,
        log_dir: Optional[str] = None,
        verbose: bool = True,
        stopper=None,
        checkpoint_history_to_csv: bool = True,
        solution_selection: Optional[str] = None,
        checkpoint_restart: bool = False,
        # Surrogate model parameters (CBO only)
        surrogate_model: str = "ET",
        surrogate_model_kwargs: Optional[Dict] = None,
        # Acquisition function parameters (CBO only)
        acq_func: str = "UCB",
        acq_func_kwargs: Optional[Dict] = None ,
        acq_optimizer: str = "mixedga",
        acq_optimizer_kwargs: Optional[Dict] = None,
        # Multi-point strategy (CBO only)
        multi_point_strategy: str = "cl_max",
        # Initial points parameters
        n_initial_points: Optional[int] = None,
        initial_point_generator: str = "random",
        initial_points: Optional[list] = None,
        # Multi-objective optimization parameters
        moo_lower_bounds=None,
        moo_scalarization_strategy: str = "Chebyshev",
        moo_scalarization_weight=None,
        objective_scaler: str = "minmax",
        # Framework parameters
        save_dir: str = "./experiments",
        keep_top_k: int = -1,
        cleanup_batch_size: int = -1,
        profile_time: bool = False,
        evaluator_method: str = "process",
        results_filename: Optional[str] = None,
        # Simulation tracking
        enable_tracker: bool = True,
        tracker_kill_multiplier: float = 1.5,
        tracker_initial_threshold: float = 1e15,
        # Additional kwargs
        problem_kwargs: Optional[Dict] = None,
        cbo_kwargs: Optional[Dict] = None,
        compress_and_clean_is_enabled: bool = True,
        # Early stopping — search-level no-improvement detector
        early_stopping_patience: int = -1,
        early_stopping_min_evaluations: int = 0,
    ):
        """
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance
            simulation_runner: SimulationRunner instance
            objective: Objective instance
            budget: Total number of evaluations
            init_samples: Number of initial random samples (overrides n_initial_points)
            n_workers: Number of parallel workers
            
            # Search type
            search_type: Type of search ("cbo" for Bayesian Optimization, "random" for Random Search)
            
            # Core CBO parameters
            random_state: Random seed for reproducibility
            log_dir: Directory for DeepHyper logs (None = temp dir)
            verbose: Print progress
            stopper: Custom stopper for early termination
            checkpoint_history_to_csv: Save search history to CSV
            solution_selection: How to select best solution ("argmax_obs", "argmax_est")
            checkpoint_restart: Restart from checkpoint
            
            # Surrogate model parameters (CBO only)
            surrogate_model: Surrogate model ("RF", "ET", "GP", "DUMMY")
            surrogate_model_kwargs: Additional surrogate model arguments
            
            # Acquisition function parameters (CBO only)
            acq_func: Acquisition function ("UCB", "EI", "PI", "gp_hedge", "UCBd")
            acq_func_kwargs: Additional acquisition function arguments
            acq_optimizer: Acquisition optimizer ("mixedga", "sampling", "lbfgs", "auto")
            acq_optimizer_kwargs: Acquisition optimizer arguments (e.g., {"max_total_failures": -1})
            
            # Multi-point strategy (CBO only)
            multi_point_strategy: Strategy for parallel evaluations ("cl_max", "cl_min", "cl_mean", "qUCB")
            
            # Initial points parameters
            n_initial_points: Number of random initial samples (None = use init_samples)
            initial_point_generator: Initial point generation strategy ("random", "sobol", "halton", "hammersly", "lhs", "grid")
            initial_points: Pre-specified initial configurations
            
            # Multi-objective optimization parameters
            moo_lower_bounds: Lower bounds for multi-objective optimization
            moo_scalarization_strategy: Scalarization strategy ("Chebyshev", "Linear", "AugChebyshev")
            moo_scalarization_weight: Weights for scalarization
            objective_scaler: Objective scaler ("minmax", "standardize", "identity")
            
            # Framework parameters
            save_dir: Directory to save results
            keep_top_k: Keep top K results files during periodic cleanup (-1 = disable cleanup)
            cleanup_batch_size: Run cleanup every N successful evaluations (-1 = disable periodic cleanup)
            profile_time: Track detailed timing statistics
            evaluator_method: Parallel evaluation method ("process" or "thread")
            
            # Simulation tracking
            enable_tracker: Enable early termination of slow simulations (default True)
            tracker_kill_multiplier: Kill simulations exceeding threshold * multiplier (default 1.5)
            tracker_initial_threshold: Initial threshold in cycles (default 1e15)
            
            # Additional overrides
            problem_kwargs: Additional HpProblem arguments (advanced)
            cbo_kwargs: Additional CBO arguments (advanced, overrides above)

            # Early stopping (search-level no-improvement detector)
            early_stopping_patience: Number of consecutive non-improving evaluations
                before stopping.  Set to -1 (default) to disable.  Penalty results
                (killed simulations) count as non-improving but do not update the best.
                Failure results ("F") are ignored entirely.  For parallel runs set this
                to at least 3-5x n_workers (e.g. 30-50 for n_workers=10).
            early_stopping_min_evaluations: Minimum number of non-failure evaluations
                to collect before the patience counter starts.  Set to n_initial_points
                (or larger) so the random exploration phase is never interrupted.
                Defaults to 0.
        """
        if not DEEPHYPER_AVAILABLE:
            raise ImportError("DeepHyper is not installed. Please install it with 'pip install deephyper'")
        
        super().__init__(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=simulation_runner,
            objective=objective,
            budget=budget,
            init_samples=init_samples,
            verbose=verbose,
            save_dir=save_dir,
            keep_top_k=keep_top_k,
            cleanup_batch_size=cleanup_batch_size,
            profile_time=profile_time
        )
        import random as py_random
        if random_state is not None:
            py_random.seed(random_state)
            np.random.seed(random_state)
        
        # Framework parameters
        self.n_workers = max(1, n_workers)
        self.evaluator_method = evaluator_method

        # Warn when parallel workers are used, because the order in which they
        # complete is non-deterministic (OS scheduling, simulation runtime
        # variance).  Even with a fixed random_state, DeepHyper's surrogate
        # model updates after each completed batch, so different completion
        # orderings produce different subsequent samples.  For fully
        # reproducible runs use n_workers=1.
        if n_workers > 1 and random_state is not None:
            import warnings
            warnings.warn(
                f"DeepHyperOptimizer: random_state={random_state} is set but "
                f"n_workers={n_workers} > 1.  Parallel worker completion order "
                "is non-deterministic, so runs will NOT be exactly reproducible "
                "even with the same seed.  Set n_workers=1 for fully "
                "reproducible results (slower).",
                UserWarning,
                stacklevel=2,
            )
        self.search_type = search_type.lower()
        
        if self.search_type not in ["cbo", "random"]:
            raise ValueError(f"search_type must be 'cbo' or 'random', got '{search_type}'")
        
        # Core CBO parameters
        self.random_state = random_state
        self.stopper = stopper
        self.checkpoint_history_to_csv = checkpoint_history_to_csv
        self.solution_selection = solution_selection
        self.checkpoint_restart = checkpoint_restart
        
        # Surrogate model parameters (CBO only)
        self.surrogate_model = surrogate_model
        self.surrogate_model_kwargs = surrogate_model_kwargs or {}
        
        # Acquisition function parameters (CBO only)
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs or {}
        self.acq_optimizer = acq_optimizer
        self.acq_optimizer_kwargs = acq_optimizer_kwargs or {"max_total_failures": -1}
        
        # Multi-point strategy (CBO only)
        self.multi_point_strategy = multi_point_strategy
        
        # Initial points parameters — DeepHyper handles sampling internally
        self.initial_points = initial_points
        self.n_initial_points = n_initial_points if n_initial_points is not None else init_samples
        self.initial_point_generator = initial_point_generator

        # Multi-objective optimization parameters
        self.moo_lower_bounds = moo_lower_bounds
        self.moo_scalarization_strategy = moo_scalarization_strategy
        self.moo_scalarization_weight = moo_scalarization_weight
        self.objective_scaler = objective_scaler
        
        # Additional kwargs
        self.problem_kwargs = problem_kwargs or {}
        self.cbo_kwargs = cbo_kwargs or {}
        self.compress_and_clean_is_enabled = compress_and_clean_is_enabled
        
        # Results filename - use model name if not specified
        if results_filename is None:
            model_name = getattr(simulation_runner, 'model_name', 'deephyper')
            self.results_filename = f"deephyper_results_{model_name}.csv"
        else:
            self.results_filename = results_filename
        
        # Setup log directory
        if log_dir is None:
            self.log_dir = tempfile.mkdtemp(prefix="deephyper_")
        else:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Simulation tracking
        self.enable_tracker = enable_tracker
        self.tracker_kill_multiplier = tracker_kill_multiplier
        self.tracker_initial_threshold = tracker_initial_threshold

        # DeepHyper components — created during initialize()
        self.hp_problem = None
        self.evaluator = None
        self.search = None
        self.deephyper_results = None

        # Multiprocess-safe structures for parallel worker evaluations
        from multiprocessing import Manager
        self._manager = Manager()
        self.extra_data_cache = self._manager.dict()
        self.duplicate_eval_cache = self._manager.dict()
        self.cache_statistics = self._manager.dict({'hits': 0, 'misses': 0})
        self.periodic_cleanup_state = None
        if self._periodic_cleanup_enabled():
            self.periodic_cleanup_state = {
                'scores': self._manager.list(),
                'file_paths': self._manager.list(),
                'cleaned_indices': self._manager.dict(),
                'control': self._manager.dict({'next_cleanup_at': self.cleanup_batch_size}),
                'cleanup_counters': self._manager.dict({'simulations_cleaned': 0, 'files_deleted': 0}),
                'lock': self._manager.Lock(),
                'keep_top_k': self.keep_top_k,
                'cleanup_batch_size': self.cleanup_batch_size,
                'verbose': self.verbose,
                'print_deleted_files': self.cleanup_print_deleted_files,
                # Per-objective directions so cleanup_records ranks MOO tuples
                # correctly without needing to pass them at every call site.
                'score_directions': objective.score_directions,
            }

        # Simulation tracker for early termination
        # Early stopping — search-level no-improvement detector
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_evaluations = early_stopping_min_evaluations

        if enable_tracker:
            # objective directions from objective.score_directions internally and
            # builds the correct initial threshold vector (positive for minimize,
            # negative for maximize) without the caller needing to pre-flip signs.
            self.tracker = SimulationTracker(
                initial_threshold=tracker_initial_threshold,
                kill_multiplier=tracker_kill_multiplier,
                verbose=verbose,
                objective=objective,
            )
            self.simulation_runner.tracker = self.tracker
            if self.verbose:
                print(f"✓ Simulation tracker enabled "
                      f"(kill at {tracker_kill_multiplier}x threshold, "
                      f"warmup={self.n_initial_points} evals)")
        else:
            self.tracker = None

    def initialize(self) -> bool:
        """Initialize DeepHyper components (HpProblem, Evaluator, Search)."""
        if self.verbose:
            print("\n" + "="*70)
            print("DEEPHYPER BAYESIAN OPTIMIZATION")
            print("="*70)
            print(f"Model: {self.simulation_runner.model_name}")
            print(f"Budget: {self.budget} evaluations")
            print(f"Initial samples: {self.init_samples}")
            print(f"Workers: {self.n_workers}")
            search_label = "CBO" if self.search_type == "cbo" else "Random Search"
            print(f"Search type: {search_label}")
            print(f"Parallelism sampler: {self.sampler}")
            print(f"Search space: {len(self.search_space.parameters)} parameters, "
                  f"{len(self.search_space.constraints)} constraints")
            print(f"Log directory: {self.log_dir}")
            if self.early_stopping_patience > 0:
                print(
                    f"Early stopping: enabled "
                    f"(patience={self.early_stopping_patience}, "
                    f"warmup={self.early_stopping_min_evaluations} evals)"
                )
            else:
                print("Early stopping: disabled")
            print("="*70 + "\n")

        try:
            with self.time_stats.timer("hp_problem_creation"):
                self.hp_problem = self._create_hp_problem()
            if self.verbose:
                print(f"✓ Created HpProblem with {len(self.hp_problem.space)} hyperparameters")

            with self.time_stats.timer("evaluator_creation"):
                self.evaluator = self._create_evaluator()
            if self.verbose:
                print(f"✓ Created evaluator with {self.n_workers} workers ({self.evaluator_method} method)")

            with self.time_stats.timer("search_creation"):
                if self.search_type == "cbo":
                    self.search = self._create_cbo()
                    search_name = "CBO optimizer"
                else:
                    self.search = self._create_random_search()
                    search_name = "RandomSearch optimizer"
            if self.verbose:
                print(f"✓ Created {search_name}\n")

            return True

        except Exception as e:
            self._log(f"Initialization failed: {e}", "error")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False
    
    def _create_hp_problem(self) -> HpProblem:
        """Create HpProblem directly from search space parameters and constraints.

        Parameters are taken from search_space.parameters (dict of name → list of values).
        Constraints are applied via DeepHyper's set_constraint_fn using the callable
        functions already compiled in search_space.constraints.
        """
        problem = HpProblem(**self.problem_kwargs)
        if self.random_state is not None:
            problem.set_seed(self.random_state)

        if not self.search_space.parameters:
            raise ValueError(
                "Search space has no parameters. Call parse_parameters() first."
            )

        # Add every parameter directly from the search space definition
        for param_name, values in self.search_space.parameters.items():
            if values:
                problem.add_hyperparameter(values, param_name, default_value=values[0])

        if self.verbose:
            total_unconstrained = 1
            for v in self.search_space.parameters.values():
                total_unconstrained *= len(v)
            print(f"✓ Added {len(self.search_space.parameters)} parameters to HpProblem")
            print(f"  Unconstrained combinations: {total_unconstrained:,}")

        # Translate search_space.constraints into a single DeepHyper constraint function
        if self.search_space.constraints:
            clusters = getattr(self.search_space, 'clusters', None)
            constraint_fns = list(self.search_space.constraints)

            def constraint_fn(s: pd.DataFrame) -> pd.Series:
                """Validate all constraints from search space configuration."""
                is_valid = np.ones(len(s), dtype=bool)
                for i, (_, row) in enumerate(s.iterrows()):
                    config = dict(row)
                    # Resolve npu_count from cluster so parallelism constraints work.
                    if clusters and 'cluster' in config and isinstance(config['cluster'], str):
                        cluster_name = config['cluster']
                        if cluster_name in clusters:
                            config['npu_count'] = clusters[cluster_name]['npu_count']
                    for fn in constraint_fns:
                        if not fn(config):
                            is_valid[i] = False
                            break
                return pd.Series(is_valid, index=s.index)

            problem.set_constraint_fn(constraint_fn)

            if self.verbose:
                print(f"✓ Added {len(self.search_space.constraints)} constraint(s):")
                for cs in self.search_space.constraint_strings:
                    print(f"  - {cs}")

        # Install smart constraint-aware sampler (first sample valid parallelism,
        # then sample all remaining parameters).
        rng = None
        if self.random_state is not None:
            np.random.seed(self.random_state)
            rng = np.random.default_rng(self.random_state)
        smart_fn = self._make_constrained_sampling_fn(rng)
        if smart_fn is not None:
            problem.set_sampling_fn(smart_fn)
            if self.verbose:
                print("✓ Smart constraint-aware sampling function installed "
                      "(parallelism-first sampling)")

        return problem

    def _make_constrained_sampling_fn(self, rng):
        """Build a parallelism-first sampler from JSON constraints.

        Strategy:
        1. Generate valid (dp, mp, sp, pp) tuples per cluster which satisfy:
           - dp * mp * sp * pp = npu_count
           - dp <= npu_count, mp <= npu_count, sp <= npu_count, pp <= npu_count
        2. Sample one valid tuple.
        3. Sample all remaining parameters independently.
        """
        params = self.search_space.parameters
        clusters = getattr(self.search_space, 'clusters', {})
        constraint_strings = [str(c) for c in getattr(self.search_space, 'constraint_strings', [])]

        required_keys = {'dp', 'mp', 'sp', 'pp'}
        if not required_keys.issubset(params.keys()):
            return None

        # Enable only when the expected JSON constraints are present.
        has_product = any('dp * mp * sp * pp = npu_count' in c for c in constraint_strings)
        has_dp_max = any('dp <= npu_count' in c for c in constraint_strings)
        has_mp_max = any('mp <= npu_count' in c for c in constraint_strings)
        has_sp_max = any('sp <= npu_count' in c for c in constraint_strings)
        has_pp_max = any('pp <= npu_count' in c for c in constraint_strings)
        if not (has_product and has_dp_max and has_mp_max and has_sp_max and has_pp_max):
            return None

        cluster_names = list(params.get('cluster', clusters.keys()))
        if not cluster_names:
            return None

        dp_vals = list(params['dp'])
        mp_vals = list(params['mp'])
        sp_vals = list(params['sp'])
        pp_vals = list(params['pp'])

        # Precompute valid parallelism tuples per cluster (small and cheap; no full-space build).
        # Store as dicts so they can be fed directly into core.sampler strategies.
        valid_parallelism_by_cluster = {}
        for cluster_name in cluster_names:
            if cluster_name not in clusters:
                continue
            npu_count = int(clusters[cluster_name]['npu_count'])
            valid_tuples = []
            for dp in dp_vals:
                if dp > npu_count:
                    continue
                for mp in mp_vals:
                    if mp > npu_count:
                        continue
                    for sp in sp_vals:
                        if sp > npu_count:
                            continue
                        prefix = dp * mp * sp
                        if prefix == 0:
                            continue
                        if npu_count % prefix != 0:
                            continue
                        pp = npu_count // prefix
                        if pp > npu_count:
                            continue
                        if pp in pp_vals:
                            valid_tuples.append({
                                'dp': dp,
                                'mp': mp,
                                'sp': sp,
                                'pp': pp,
                            })
            if valid_tuples:
                valid_parallelism_by_cluster[cluster_name] = valid_tuples

        if not valid_parallelism_by_cluster:
            return None

        active_clusters = sorted(valid_parallelism_by_cluster.keys())

        # Sample all non-parallelism parameters afterwards.
        other_params = {
            k: list(v)
            for k, v in params.items()
            if k not in {'cluster', 'dp', 'mp', 'sp', 'pp'}
        }

        def _sample_parallelism_for_cluster(cluster_name: str, m: int, rng: np.random.Generator) -> list:
            """Sample m valid parallelism tuples for one cluster using self.sampler."""
            space = valid_parallelism_by_cluster[cluster_name]
            if m <= 0:
                return []

            # Stage 1: enforce diversity across (dp, mp) first to avoid
            # collapsing to low-dp/low-mp regions.
            by_dp_mp = {}
            for cfg in space:
                key = (cfg['dp'], cfg['mp'])
                by_dp_mp.setdefault(key, []).append(cfg)

            group_keys = list(by_dp_mp.keys())
            rng.shuffle(group_keys)

            selected = []
            for key in group_keys[:min(m, len(group_keys))]:
                group = by_dp_mp[key]
                selected.append(group[int(rng.integers(0, len(group)))])

            # Stage 2: use configured sampler on a shuffled candidate pool
            # so order-sensitive samplers (e.g., grid/lhs) do not bias to dp=1.
            remaining = m - len(selected)
            if remaining > 0:
                selected_keys = {(c['dp'], c['mp'], c['sp'], c['pp']) for c in selected}
                perm = rng.permutation(len(space))
                shuffled_space = [space[i] for i in perm]
                sampler_pool = [
                    c for c in shuffled_space
                    if (c['dp'], c['mp'], c['sp'], c['pp']) not in selected_keys
                ]
                if sampler_pool:
                    selected.extend(self.sampler.sample(sampler_pool, min(remaining, len(sampler_pool))))

            # Top-up with replacement if still short.
            while len(selected) < m:
                selected.append(space[int(rng.integers(0, len(space)))])

            rng.shuffle(selected)
            return selected[:m]

        # Precompute cluster weights proportional to number of valid tuples,
        # so each individual valid config has equal probability of being sampled
        # regardless of how many valid tuples each cluster exposes.
        _cluster_weights = np.array(
            [len(valid_parallelism_by_cluster[c]) for c in active_clusters], dtype=float
        )
        _cluster_weights /= _cluster_weights.sum()

        def sampling_fn(n: int) -> list:
            samples = []
            # Choose clusters weighted by their valid-tuple count so that every
            # (cluster, dp, mp, sp, pp) combination has equal sampling probability.
            cluster_picks = rng.choice(active_clusters, size=n, p=_cluster_weights).tolist()
            per_cluster_counts = {}
            for c in cluster_picks:
                per_cluster_counts[c] = per_cluster_counts.get(c, 0) + 1

            sampled_parallelism = {}
            for c, m in per_cluster_counts.items():
                sampled_parallelism[c] = _sample_parallelism_for_cluster(c, m, rng)

            consumed = {c: 0 for c in per_cluster_counts.keys()}
            for c in cluster_picks:
                idx = consumed[c]
                consumed[c] += 1
                par = sampled_parallelism[c][idx]

                config = {
                    'cluster': c,
                    'dp': par['dp'],
                    'mp': par['mp'],
                    'sp': par['sp'],
                    'pp': par['pp'],
                }

                # Then sample every remaining parameter independently.
                for param_name, values in other_params.items():
                    if values:
                        config[param_name] = values[int(rng.integers(0, len(values)))]

                samples.append(config)

            return samples

        return sampling_fn
    
    def _create_evaluator(self) -> Evaluator:
        """Create DeepHyper evaluator for parallel execution."""
        from functools import partial
        
        optimizer_state = {
            'simulation_runner': self.simulation_runner,
            'objective': self.objective,
            'clusters': getattr(self.search_space, 'clusters', None),
            'extra_data_cache': self.extra_data_cache,
            'tracker': self.tracker if self.enable_tracker else None,
            'periodic_cleanup': self.periodic_cleanup_state,
            'duplicate_eval_cache': self.duplicate_eval_cache,
            'cache_statistics': self.cache_statistics,
            # Tracker warmup: number of evaluations to run without killing.
            # During this phase the tracker is detached from the runner so
            # every initial sample completes, giving the surrogate a full
            # training set.  The tracker still receives threshold updates so
            # it has accurate baselines when killing activates.
            'tracker_warmup': self.n_initial_points if self.enable_tracker else 0,
            'eval_counter': self._manager.Value('i', 0) if self.enable_tracker else None,
            'eval_lock': self._manager.Lock() if self.enable_tracker else None,
        }
        
        eval_func = partial(_deephyper_evaluate_wrapper, optimizer_state=optimizer_state)

        callbacks = []
        if self.early_stopping_patience > 0:
            early_stopper = AdaptiveSearchEarlyStopping(
                patience_limit=self.early_stopping_patience,
                min_evaluations_before_check=self.early_stopping_min_evaluations,
                verbose=self.verbose,
            )
            callbacks.append(early_stopper)
            if self.verbose:
                print(
                    f"✓ Early stopping enabled "
                    f"(patience={self.early_stopping_patience}, "
                    f"warmup={self.early_stopping_min_evaluations} evals)"
                )

        evaluator = Evaluator.create(
            eval_func,
            method=self.evaluator_method,
            method_kwargs={"num_workers": self.n_workers, "callbacks": callbacks}
        )
        
        return evaluator
    
    def _enrich_results_with_config_files(self):
        """Enrich DeepHyper results dataframe with config file information.
        
        This method reads the system, network, and memory config files for each
        successful evaluation and appends the information as new columns in the
        results dataframe.
        """
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            return
        
        # Create a temporary optimizer state to re-evaluate configs and get file_paths
        param_cols = [col for col in self.deephyper_results.columns if col.startswith('p:')]
        param_names = [col[2:] for col in param_cols]
        
        config_file_data = []
        
        for idx, row in self.deephyper_results.iterrows():
            # Reconstruct config from row
            config = {name: row[f'p:{name}'] for name in param_names}
            
            try:
                # Retrieve exec_time and config files from cache
                config_key = tuple(config.items())
                if self.verbose and idx < 3:
                    print(f"  Looking up config_key for row {idx}: {config_key}")
                    print(f"  Cache has {len(self.extra_data_cache)} entries")
                    if config_key in self.extra_data_cache:
                        print(f"  ✓ Found in cache")
                    else:
                        print(f"  ✗ NOT found in cache")
                        print(f"  Available keys: {list(self.extra_data_cache.keys())[:2]}")
                
                cached_data = self.extra_data_cache.get(config_key, {})
                if cached_data:
                    config_data = dict(cached_data.get('config_files', {}))
                    config_data['exec_time'] = cached_data.get('exec_time')
                    config_data['was_killed'] = bool(cached_data.get('was_killed', False))
                    config_data['is_oom'] = bool(cached_data.get('is_oom', False))
                    config_data['total_power_W'] = cached_data.get('total_power_W')
                    config_data['total_energy_J'] = cached_data.get('total_energy_J')
                    config_data['has_power_metrics'] = bool(cached_data.get('has_power_metrics', False))
                else:
                    config_data = {}
                config_file_data.append(config_data)
            
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️  Warning: Could not enrich row {idx} with config files: {e}")
                config_file_data.append({})
        
        # Append config file data to results dataframe
        if config_file_data:
            # Add columns for config files (system_config, network_config, memory_config)
            config_file_keys = ['system_config', 'network_config', 'memory_config']
            
            for key in config_file_keys:
                self.deephyper_results[key] = [data.get(key, None) for data in config_file_data]
            
            # Add exec_time column
            self.deephyper_results['exec_time'] = [data.get('exec_time', None) for data in config_file_data]
            self.deephyper_results['was_killed'] = [bool(data.get('was_killed', False)) for data in config_file_data]
            self.deephyper_results['is_oom'] = [bool(data.get('is_oom', False)) for data in config_file_data]
            self.deephyper_results['total_power_W'] = [data.get('total_power_W', None) for data in config_file_data]
            self.deephyper_results['total_energy_J'] = [data.get('total_energy_J', None) for data in config_file_data]
            self.deephyper_results['has_power_metrics'] = [bool(data.get('has_power_metrics', False)) for data in config_file_data]
            
            if self.verbose:
                n_enriched = sum(1 for data in config_file_data if data)
                print(f"✓ Enriched {n_enriched}/{len(self.deephyper_results)} rows with config file information")
                print(
                    f"  Added columns: {', '.join(config_file_keys)}, exec_time, was_killed, is_oom, "
                    "total_power_W, total_energy_J, has_power_metrics"
                )
    
    def _collect_results_from_deephyper(self):
        """Collect and process results from DeepHyper's output dataframe.

        For single-objective runs the score is stored as a scalar.
        For multi-objective runs ALL ``objective_N`` columns are collected and
        converted back to natural objective space as a tuple, so that
        ``self.scores`` holds the full MOO vector and ``is_better`` / cleanup
        ranking can use every dimension.
        """
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            return
        
        param_cols = [col for col in self.deephyper_results.columns if col.startswith('p:')]
        param_names = [col[2:] for col in param_cols]
        
        # Discover all objective columns (objective_0, objective_1, … for MOO;
        # or just 'objective' for single-objective).
        moo_obj_cols = sorted(
            c for c in self.deephyper_results.columns if c.startswith('objective_')
        )
        is_multi_objective = len(moo_obj_cols) > 0

        # Anything at or above this magnitude in natural space is a penalty.
        # Must match the PENALTY constant in objective.py (1e20).
        PENALTY_THRESHOLD = 1e20
        n_failed = 0
        n_success = 0
        n_infeasible = 0
        
        for idx, row in self.deephyper_results.iterrows():
            # ── constraint check ────────────────────────────────────────────
            if 'constraint' in row and not row['constraint']:
                n_infeasible += 1
                continue

            # ── read objective column(s) ─────────────────────────────────────
            if is_multi_objective:
                # Collect every objective_N value from the row.
                raw_opt_values = [row.get(c, None) for c in moo_obj_cols]
                if any(v is None for v in raw_opt_values):
                    n_failed += 1
                    continue
                # Failure sentinel: DeepHyper stores 'F' or NaN for failed evals.
                if any(isinstance(v, str) and v.startswith('F') for v in raw_opt_values):
                    n_failed += 1
                    continue
                try:
                    raw_opt_values = [float(v) for v in raw_opt_values]
                except (ValueError, TypeError):
                    n_failed += 1
                    continue
                score = self.objective.from_optimizer_score(
                    tuple(raw_opt_values),
                    optimizer_bad_value=PENALTY,
                )
            else:
                objective_value = row.get('objective', None)
                if objective_value is None:
                    n_failed += 1
                    continue
                if isinstance(objective_value, str) and objective_value.startswith('F'):
                    n_failed += 1
                    continue
                try:
                    objective_value = float(objective_value)
                except (ValueError, TypeError):
                    n_failed += 1
                    continue
                score = self.objective.from_optimizer_score(
                    objective_value,
                    optimizer_bad_value=PENALTY,
                )

            # ── penalty detection ────────────────────────────────────────────
            # abs() guard catches both +PENALTY (natural space) and any stray
            # non-finite values that survived from_optimizer_score.
            if isinstance(score, tuple):
                is_penalty = any(
                    not math.isfinite(s) or abs(s) >= PENALTY_THRESHOLD
                    for s in score
                )
            else:
                is_penalty = not math.isfinite(score) or abs(score) >= PENALTY_THRESHOLD

            # ── store result ─────────────────────────────────────────────────
            config = {name: row[f'p:{name}'] for name in param_names}
            self.configs.append(config)
            self.scores.append(score)
            metadata = self._get_simulation_metadata()
            metadata['was_killed'] = bool(row.get('was_killed', False))
            self.metadata.append(metadata)
            n_success += 1
            
            # ── verbose progress ─────────────────────────────────────────────
            if self.verbose and n_success <= 10:
                config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
                if isinstance(score, tuple):
                    obj_str = ", ".join(f"Obj{i}: {format_score(s)}" for i, s in enumerate(score))
                    print(f"  Iteration {n_success}: {config_str} | {obj_str}")
                else:
                    print(f"  Iteration {n_success}: {config_str} | Score: {format_score(score)}")
            
            # ── best tracking ────────────────────────────────────────────────
            if not is_penalty and self.objective.is_better(score, self.best_score):
                self.best_score = score
                self.best_config = config
                self.best_iteration = len(self.configs) - 1
                if self.verbose:
                    score_str = format_score(score)
                    if n_success > 10:
                        print(f"    NEW BEST (iter {n_success}): {score_str}")
                    else:
                        print(f"    NEW BEST: {score_str}")
        
        if self.verbose:
            print(f"\nCollected {n_success} successful evaluations")
            print(f"Total DeepHyper evaluations: {len(self.deephyper_results)}")
            if n_infeasible > 0:
                print(f"  - Infeasible (constraint violations): {n_infeasible}")
            if n_failed > 0:
                print(f"  - Failed (simulation errors): {n_failed}")
            if is_multi_objective:
                print(f"  - Multi-objective: {len(moo_obj_cols)} objectives "
                      f"({', '.join(moo_obj_cols)})")
    
    def _read_config_files(self, file_paths: Dict) -> Dict:
        """Read config files and return as JSON strings."""
        config_data = {}
        
        for key in ['system_config', 'network_config', 'memory_config']:
            path = file_paths.get(key)
            if path:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f) if path.endswith('.json') else yaml.safe_load(f)
                    config_data[key] = json.dumps(data)
                except Exception as e:
                    if self.verbose:
                        print(f"    ⚠️  Could not read {key}: {e}")
                    config_data[key] = None
        
        return config_data
    
    def _get_simulation_metadata(self) -> Dict:
        """Get simulation metadata from simulation_runner."""
        sr = self.simulation_runner
        try:
            import sys
            sys.path.insert(0, os.environ['OPTIMOS_ROOT'])
            model = workload_generator.Model(sr.model_num)
            din, dout, dmodel, dff, batch, micro_batch, seq, head, num_stacks = model.get_model_params()
            
            metadata = {
                'model_name': sr.model_name,
                'model_num': sr.model_num,
                'vocab_size_in': din,
                'vocab_size_out': dout,
                'hidden_size': dmodel,
                'ffn_hidden_size': dff,
                'batch_size': batch,
                'sequence_length': seq,
                'num_attention_heads': head,
                'num_layers': num_stacks,
                'num_npus': sr.num_npus,
                'sim_type': sr.net_sim_config.get('sim_type', 'analytical_unaware'),
            }
            if hasattr(sr, 'system_metadata'):
                metadata.update(sr.system_metadata)
            return metadata
        except Exception:
            return {
                'model_name': sr.model_name,
                'model_num': sr.model_num,
                'num_npus': sr.num_npus,
                'sim_type': sr.net_sim_config.get('sim_type', 'analytical_unaware'),
            }
    
    def _create_cbo(self) -> CBO:
        """Create DeepHyper CBO instance."""
        cbo_args = {
            # Required
            "problem": self.hp_problem,
            # Core parameters
            "random_state": self.random_state,
            "log_dir": self.log_dir,
            "verbose": 1 if self.verbose else 0,
            "stopper": self.stopper,
            "checkpoint_history_to_csv": self.checkpoint_history_to_csv,
            "solution_selection": self.solution_selection,
            "checkpoint_restart": self.checkpoint_restart,
            # Surrogate model
            "surrogate_model": self.surrogate_model,
            "surrogate_model_kwargs": self.surrogate_model_kwargs,
            # Acquisition function
            "acq_func": self.acq_func,
            "acq_func_kwargs": self.acq_func_kwargs,
            "acq_optimizer": self.acq_optimizer,
            "acq_optimizer_kwargs": self.acq_optimizer_kwargs,
            # Multi-point strategy
            "multi_point_strategy": self.multi_point_strategy,
            # Initial points
            "n_initial_points": self.n_initial_points,
            "initial_point_generator": self.initial_point_generator,
            "initial_points": self.initial_points,
            # Multi-objective optimization
            "moo_lower_bounds": self.moo_lower_bounds,
            "moo_scalarization_strategy": self.moo_scalarization_strategy,
            "moo_scalarization_weight": self.moo_scalarization_weight,
            "objective_scaler": self.objective_scaler,
        }
        
        # Apply additional overrides from cbo_kwargs
        cbo_args.update(self.cbo_kwargs)
        
        return CBO(**cbo_args)
    
    def _create_random_search(self) -> RandomSearch:
        """Create DeepHyper RandomSearch instance.

        DeepHyper's built-in RandomSearch samples directly from ConfigSpace,
        which bypasses HpProblem.constraint_fn.  Use a constrained variant so
        random search respects the same search-space constraints as CBO.
        """
        random_args = {
            "problem": self.hp_problem,
            "random_state": self.random_state,
            "log_dir": self.log_dir,
            "verbose": 1 if self.verbose else 0,
            "stopper": self.stopper,
            "checkpoint_history_to_csv": self.checkpoint_history_to_csv,
            "solution_selection": self.solution_selection,
        }

        # Forward any compatible overrides from cbo_kwargs
        if self.cbo_kwargs:
            valid_params = {
                "random_state", "log_dir", "verbose", "stopper",
                "checkpoint_history_to_csv", "solution_selection",
            }
            random_args.update(
                {k: v for k, v in self.cbo_kwargs.items() if k in valid_params}
            )

        return ConstrainedRandomSearch(**random_args)
    
    def _recompute_pareto_efficient(self) -> None:
        """Recompute the pareto_efficient column from raw objective values.

        DeepHyper stores objectives in maximisation form (negated for minimise
        problems) and computes its internal Pareto column incrementally during
        the search.  That incremental computation can miss true Pareto points
        when the objective range is skewed (e.g. very slow configs coexisting
        with fast ones in no-tracker runs).

        This method recomputes the column post-hoc using the final, complete set
        of evaluated points.
        """
        df = self.deephyper_results
        if df is None or len(df) == 0:
            return

        if self.objective.is_multi_objective:
            obj_cols = [c for c in df.columns if c.startswith("objective_")]
            if len(obj_cols) < 2:
                return
        else:
            obj_cols = ["objective"] if "objective" in df.columns else []
            if not obj_cols:
                return

        # Convert to float, filtering out failure markers ("F", NaN, -1e10).
        objs = df[obj_cols].copy()
        for col in obj_cols:
            objs[col] = pd.to_numeric(objs[col], errors="coerce")

        valid_mask = objs.notna().all(axis=1) & (objs.abs() < 9e9).all(axis=1)

        if valid_mask.sum() == 0:
            df["pareto_efficient"] = False
            return

        raw = objs[valid_mask].values.astype(float)
        # Negate back from DeepHyper's maximisation space to minimisation space.
        minimise_raw = -raw

        n = len(minimise_raw)
        is_eff = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_eff[i]:
                continue
            point = minimise_raw[i]
            other_mask = is_eff.copy()
            other_mask[i] = False
            if not other_mask.any():
                break
            others = minimise_raw[other_mask]
            # i is dominated if any other point is ≤ in all dims and < in at least one.
            if np.any(np.all(others <= point, axis=1) & np.any(others < point, axis=1)):
                is_eff[i] = False

        pareto_col = np.zeros(len(df), dtype=bool)
        pareto_col[df.index[valid_mask]] = is_eff
        df["pareto_efficient"] = pareto_col

        n_pareto = int(pareto_col.sum())
        n_valid = int(valid_mask.sum())
        if self.verbose:
            print(f"   Pareto front recomputed: {n_pareto}/{n_valid} "
                  f"({n_pareto/max(n_valid,1)*100:.1f}%) Pareto-efficient points")

    def _finalize_and_save_results(self, enrichment_verbosity: bool = True) -> bool:
        """Finalize results by enriching with config files and saving to CSV.
        
        Args:
            enrichment_verbosity: Whether to print enrichment progress messages
            
        Returns:
            True if results were saved successfully, False otherwise
        """
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            if self.verbose:
                self._log("No results to save (optimization interrupted too early)", "warning")
            return False
        
        try:
            # Enrich results with config file information
            if enrichment_verbosity and self.verbose:
                print("\n" + "-"*70 + "\nENRICHING RESULTS WITH CONFIG FILES\n" + "-"*70)
            self._enrich_results_with_config_files()
            
            # Recompute Pareto-efficient flags from the final objective values.
            # DeepHyper's internal column silently fails when objectives
            # reach it as int64 (all non-killed runs).  The float() cast in
            # _deephyper_evaluate_wrapper is the primary fix; this
            # recomputation is a safety net.
            #self._recompute_pareto_efficient()

            # Save to CSV
            dh_results_path = os.path.join(self.save_dir, self.results_filename)
            self.deephyper_results.to_csv(dh_results_path, index=False)
            
            if self.verbose:
                print(f"✓ Results saved to: {dh_results_path}")
                print(f"  {len(self.deephyper_results)} evaluations saved")
            
            # Collect results for BaseOptimizer tracking
            if enrichment_verbosity and self.verbose:
                print("\n" + "-"*70 + "\nCOLLECTING RESULTS\n" + "-"*70)
            self._collect_results_from_deephyper()
            
            return True
            
        except Exception as e:
            if self.verbose:
                self._log(f"Warning: Could not save results: {e}", "warning")
            return False
    
    def optimize_step(self) -> Tuple[Optional[Dict], Optional[float]]:
        """Not supported for DeepHyper (uses batch search instead)."""
        self._log("optimize_step() not supported", "warning")
        return None, None
    
    def run(self) -> Tuple[Optional[Dict], pd.DataFrame]:
        """Run optimization using DeepHyper (CBO or RandomSearch)."""
        self.start_time = time.time()
        self.time_stats.start_total()
        
        with self.time_stats.timer("initialization"):
            if not self.initialize():
                return None, pd.DataFrame()
        
        try:
            search_label = "CBO" if self.search_type == "cbo" else "RANDOM SEARCH"
            if self.verbose:
                print("-"*70 + f"\nRUNNING {search_label}\n" + "-"*70 + "\n")

            if self.periodic_cleanup_state is not None:
                self.periodic_cleanup_state['print_deleted_files'] = self.cleanup_print_deleted_files
            
            # Guard against DeepHyper versions that call np.asarray_chkfinite
            # inside compute_pareto_efficiency without first filtering NaN rows
            # (NaNs come from evaluations that returned "F").  Patch the method
            # on the live history object so the search never aborts on valid but
            # failed runs.
            _history_obj = getattr(self.search, 'history', None)
            if _history_obj is not None and hasattr(_history_obj, 'compute_pareto_efficiency'):
                _orig_cpe = _history_obj.compute_pareto_efficiency
                def _safe_cpe(_orig=_orig_cpe):
                    try:
                        _orig()
                    except ValueError as _e:
                        if "infs or NaNs" in str(_e):
                            print(f"⚠️  DeepHyper Pareto post-processing skipped "
                                  f"(NaN from failed evals, version mismatch): {_e}")
                        else:
                            raise
                _history_obj.compute_pareto_efficiency = _safe_cpe

            with self.time_stats.timer("search"):
                try:
                    self.deephyper_results = self.search.search(
                        evaluator=self.evaluator,
                        max_evals=self.budget,
                    )
                except ValueError as _search_err:
                    if "infs or NaNs" not in str(_search_err):
                        raise
                    # Evaluations are all done; only the Pareto post-processing
                    # step failed (history patch missed because history was not
                    # yet initialised before search()).  Recover the raw results
                    # dataframe directly from the history object.
                    print(f"⚠️  DeepHyper search() raised ValueError ('{_search_err}'); "
                          "recovering results from search history …")
                    _h = getattr(self.search, 'history', None)
                    if _h is not None:
                        if hasattr(_h, 'df'):
                            self.deephyper_results = _h.df
                        elif isinstance(_h, pd.DataFrame):
                            self.deephyper_results = _h
                    if self.deephyper_results is None:
                        raise  # Cannot recover — re-raise original error
            
            # Final cleanup pass to ensure only top-K artifacts remain.
            if self.periodic_cleanup_state is not None:
                with self.periodic_cleanup_state['lock']:
                    ArtifactCleanupManager.run_periodic_cleanup_for_state(
                        cleanup_state=self.periodic_cleanup_state,
                        force=True,
                    )

                self.cleanup_manager.sync_counters(self.periodic_cleanup_state.get('cleanup_counters'))

            # Compress top-K artifacts and delete the originals.
            if self.compress_and_clean_is_enabled:
                self.compress_and_clean()

            # Finalize and save results
            with self.time_stats.timer("save_results"):
                self._finalize_and_save_results(enrichment_verbosity=True)
            
            if self.verbose:
                print("\n" + "-"*70 + f"\n{search_label} COMPLETE\n" + "-"*70)
                print(
                    f"Cleanup summary: simulations_cleaned={self.cleanup_stats.get('simulations_cleaned', 0)}, "
                    f"files_deleted={self.cleanup_stats.get('files_deleted', 0)}"
                )
                self.print_summary()
            
            self.time_stats.end_total()
            return self.best_config, self.get_history()
            
        except KeyboardInterrupt:
            self._log("\n\nOptimization interrupted by user", "warning")
            self._log("Saving intermediate results...", "info")

            if self.periodic_cleanup_state is not None:
                with self.periodic_cleanup_state['lock']:
                    ArtifactCleanupManager.run_periodic_cleanup_for_state(
                        cleanup_state=self.periodic_cleanup_state,
                        force=True,
                    )
                self.cleanup_manager.sync_counters(self.periodic_cleanup_state.get('cleanup_counters'))
            
            # Finalize and save any completed results
            self._finalize_and_save_results(enrichment_verbosity=False)
            
            self.time_stats.end_total()
            return self.best_config, self.get_history()
        
        except Exception as e:
            self._log(f"Optimization failed: {e}", "error")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return None, pd.DataFrame()
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, columns: list, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """Remove outliers using Interquartile Range (IQR) method.
        
        Args:
            df: DataFrame to filter
            columns: List of column names to check for outliers
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
            
        Returns:
            DataFrame with outliers removed
        """
        # First, remove failure values (like -10000000000.0 = -1e10)
        mask = pd.Series([True] * len(df), index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Remove failure markers (-1e10) and NaN values
            # Using abs() to catch both positive and negative failure markers
            valid_mask = (df[col].notna()) & (df[col].abs() < 9e9)
            mask = mask & valid_mask
        
        # Now apply IQR filtering on the valid values
        df_valid = df[mask].copy()
        
        if len(df_valid) == 0:
            return df_valid
        
        for col in columns:
            if col not in df_valid.columns:
                continue
            
            values = df_valid[col]
            
            if len(values) < 4:  # Need at least 4 points for IQR
                continue
            
            # Calculate IQR
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:  # All values are the same
                continue
            
            # Define outlier bounds
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            # Update mask to keep only points within bounds
            df_valid = df_valid[(df_valid[col] >= lower_bound) & (df_valid[col] <= upper_bound)]
        
        return df_valid
    
    def plot_hypervolume(self, save_path: Optional[str] = None):
        """Plot hypervolume indicator over evaluations for multi-objective optimization.
        
        The hypervolume indicator measures the volume of objective space dominated by the
        current Pareto front relative to a reference point. It increases as the optimization
        finds better solutions, providing a single metric to track MOO progress.
        
        A higher hypervolume indicates:
        - Better overall solution quality
        - More diverse Pareto front coverage
        - Improved convergence toward optimal trade-offs
        
        Args:
            save_path: Path to save the plot. If None, uses save_dir/hypervolume.png
        """
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            print("⚠️  No results to plot hypervolume")
            return
        
        # Check if we have multi-objective results
        if "objective_0" not in self.deephyper_results.columns or "objective_1" not in self.deephyper_results.columns:
            print("⚠️  Not a multi-objective optimization - no hypervolume to compute")
            return
        
        try:
            from deephyper.analysis._matplotlib import update_matplotlib_rc
            from deephyper.sklearn.moo import MOOScalarBenchmark
            
            # Update matplotlib style for better plots
            update_matplotlib_rc()
            
            # Create benchmark for scoring
            bench = MOOScalarBenchmark(
                moo_lower_bounds=self.moo_lower_bounds,
                scalarization_strategy=self.moo_scalarization_strategy
            )
            
            # Compute hypervolume over time
            results = self.deephyper_results[["objective_0", "objective_1"]].values
            scorer = bench.scorer
            hvi = scorer.hypervolume(results)
            
            # Create plot
            x = list(range(1, len(hvi) + 1))
            fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
            
            _ = ax.plot(x, hvi, linewidth=2, color="#2E86AB", marker="o", markersize=4, markevery=max(1, len(x)//20))
            _ = ax.fill_between(x, hvi, alpha=0.3, color="#2E86AB")
            _ = ax.grid(alpha=0.3, linestyle="--")
            _ = ax.set_xlabel("Number of Evaluations", fontsize=12)
            _ = ax.set_ylabel("Hypervolume Indicator", fontsize=12)
            _ = ax.set_title("Hypervolume Indicator Progress", fontsize=14, fontweight="bold")
            
            # Add annotation for final hypervolume
            final_hv = hvi[-1]
            _ = ax.annotate(
                f"Final HV: {final_hv:.2f}",
                xy=(len(x), final_hv),
                xytext=(-60, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
            )
            
            # Save figure with model_name
            if save_path is None:
                model_name = getattr(self.simulation_runner, 'model_name', 'model')
                save_path = os.path.join(self.save_dir, f"hypervolume_{model_name}.png")
            
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if self.verbose:
                print(f"✓ Hypervolume plot saved to: {save_path}")
                print(f"  Final hypervolume: {final_hv:.4f}")
                print(f"  Initial hypervolume: {hvi[0]:.4f}")
                print(f"  Improvement: {((final_hv - hvi[0]) / hvi[0] * 100):.2f}%")
            
            plt.close(fig)
            return save_path, hvi
            
        except ImportError as e:
            print(f"⚠️  Cannot plot hypervolume: {e}")
            print("   Install deephyper with: pip install deephyper[analytics]")
            return None, None
        except Exception as e:
            print(f"⚠️  Error computing hypervolume: {e}")
            return None, None

    def __repr__(self) -> str:
        """String representation."""
        return (f"DeepHyperOptimizer(budget={self.budget}, "
                f"init_samples={self.init_samples}, "
                f"n_workers={self.n_workers}, "
                f"acq_func='{self.acq_func}', "
                f"evaluated={len(self.configs)})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        search_name = "CBO" if self.search_type == "cbo" else "RandomSearch"
        info = [
            f"DeepHyperOptimizer ({search_name})",
            f"Budget: {self.budget} evaluations",
            f"Workers: {self.n_workers}",
        ]
        
        if self.search_type == "cbo":
            info.extend([
                f"Initialization: {self.init_samples} samples",
                f"Acquisition: {self.acq_func}",
            ])
        
        info.append(f"Evaluated: {len(self.configs)} configs")
        
        if self.best_config:
            info.append(f"Best score: {self.best_score:.2f}")
        
        return "\n".join(info)
