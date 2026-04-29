"""
BaseOptimizer: Abstract base class for all optimizers.

Defines the common interface that all optimizers must implement.
Provides shared functionality for result tracking, logging, and I/O.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any
import math
import pandas as pd
import numpy as np
import time
import os

from .artifact_cleanup import ArtifactCleanupManager
from .time_statistics import TimeStatistics
from .objective import ObjectiveFunction, MinimizeExecutionTime


def _fmt_one(s: float, penalty: float) -> str:
    """Format a single numeric score, switching to scientific notation for
    values whose magnitude is either very small or very large."""
    try:
        f = float(s)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(f) or abs(f) >= penalty:
        return "N/A"
    # Use fixed-point for "human-scale" values, scientific otherwise.
    if f == 0.0 or (1e-3 <= abs(f) < 1e7):
        return f"{f:.4f}"
    return f"{f:.6g}"


def format_score(score) -> str:
    """Format score for display, handling both floats and tuples."""
    _PEN = 1e20  # matches PENALTY in objective.py
    if isinstance(score, tuple):
        return "(" + ", ".join(_fmt_one(s, _PEN) for s in score) + ")"
    elif score is None or (isinstance(score, float) and (not math.isfinite(score) or abs(score) >= _PEN)):
        return "N/A"
    else:
        return _fmt_one(score, _PEN)


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    All optimizers must implement:
    - initialize(): Setup initial samples
    - optimize_step(): Single optimization iteration
    - run(): Main optimization loop
    
    Shared functionality:
    - Result tracking and history
    - Best configuration tracking
    - Results saving to CSV
    - Progress logging
    """
    
    def __init__(
        self,
        search_space,
        sampler,
        simulation_runner,
        budget: int = 30,
        init_samples: int = 5,
        objective: 'ObjectiveFunction' = MinimizeExecutionTime(),
        verbose: bool = True,
        save_dir: str = ".",
        keep_top_k: int = -1,
        cleanup_batch_size: int = -1,
        profile_time: bool = False
    ):
        """
        Initialize base optimizer.
        
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance for initial sampling
            simulation_runner: SimulationRunner instance
            budget: Total number of evaluations
            init_samples: Number of initial random samples
            objective: ObjectiveFunction to optimize (default: MinimizeExecutionTime)
            verbose: Whether to print progress
            save_dir: Directory to save results
            keep_top_k: Keep only top K results' files during cleanup (-1 = disable cleanup, 0 = keep none)
            cleanup_batch_size: Run cleanup every N successful evaluations (-1 = disable periodic cleanup).
                               Must be greater than keep_top_k when cleanup is enabled.
            profile_time: Whether to track and print detailed time statistics
        """
        self.search_space = search_space
        self.sampler = sampler
        self.simulation_runner = simulation_runner
        self.budget = budget
        self.init_samples = init_samples
        self.objective = objective 
        self.verbose = verbose
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.cleanup_batch_size = cleanup_batch_size
        self.profile_time = profile_time

        # Time profiling
        self.time_stats = TimeStatistics(enabled=profile_time)
        
        # Result tracking
        self.configs: List[Dict] = []  # Evaluated configurations (now dicts)
        self.scores: List[float] = []   # Execution times (lower is better)
        self.iteration_times: List[float] = []  # Time per iteration
        self.history: List[Dict] = []  # Detailed history
        self.file_paths: List[Dict[str, str]] = []  # Track workload and output files
        self.metadata: List[Dict] = []  # Track simulation metadata (model, network, hardware params)
        
        # Best tracking — initialised to the natural-space PENALTY (1e20) which
        # is treated as "no best yet" by is_better (abs >= 1e20 → invalid).
        self.best_config: Optional[Dict] = None
        self.best_score: float = 1e20
        self.best_iteration: int = -1
        
        # State
        self.current_iteration = 0
        self.start_time = None
        self.cleanup_manager = ArtifactCleanupManager(
            keep_top_k=self.keep_top_k,
            cleanup_batch_size=self.cleanup_batch_size,
            verbose=self.verbose,
            log_fn=self._log,
        )
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize optimizer with initial samples.
        
        Should generate init_samples configurations and evaluate them.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def optimize_step(self) -> Tuple[Optional[Dict], Optional[float]]:
        """
        Execute one optimization iteration.
        
        Returns:
            (config, score) tuple if successful, (None, None) otherwise
        """
        pass
    
    @abstractmethod
    def run(self) -> Tuple[Optional[Dict], pd.DataFrame]:
        """
        Run full optimization loop.
        
        Returns:
            (best_config, results_dataframe) tuple
        """
        pass
    
    def evaluate_config(self, config: Dict, verbose: bool = False) -> Optional[float]:
        """
        Evaluate a single configuration.
        
        Args:
            config: Configuration dictionary (e.g., {'dp': 2, 'mp': 4, ...})
            verbose: Whether to print evaluation details
        
        Returns:
            Objective score, or None if evaluation failed
        """
        try:
            # Enrich config with cluster info if cluster parameter exists
            if 'cluster' in config and hasattr(self, 'search_space'):
                config = self.search_space.enrich_config_with_cluster_info(config)
            
            # Run simulation and get execution time + file paths + metadata
            result = self.simulation_runner.run_simulation(config, return_paths=True)
            
            if result is not None:
                if isinstance(result, tuple) and len(result) == 4:
                    # From evaluate_config_worker: (exec_time, is_oom, file_paths, metadata)
                    exec_time, is_oom, file_paths, metadata = result
                elif isinstance(result, tuple) and len(result) == 3:
                    exec_time, is_oom, file_paths = result
                    metadata = {}
                else:
                    exec_time, is_oom = result
                    file_paths = {}
                    metadata = {}

                cleanup_reason = self.cleanup_manager.get_immediate_cleanup_reason(exec_time, is_oom, metadata)

                # Failed run that still produced files: clean immediately.
                if exec_time is None:
                    self.cleanup_manager.cleanup_single_simulation_files(file_paths, reason=cleanup_reason or "failed")
                    if verbose:
                        print("    ⚠️  Evaluation failed")
                    return None
                
                # Compute objective score (pass config as well)
                score = self.objective.compute(exec_time, is_oom, metadata, config)
                
                # Record results
                self.configs.append(config)
                self.scores.append(score)
                self.file_paths.append(file_paths)
                self.metadata.append(metadata)
                
                # Update best
                if self.objective.is_better(score, self.best_score):
                    self.best_score = score
                    self.best_config = config
                    self.best_iteration = self.current_iteration
                    
                    if self.verbose and verbose:
                        print(f"    🏆 NEW BEST! Score: {score:.4f} (exec_time: {exec_time:.2f}s)")

                # Immediate cleanup for killed/OOM runs while keeping their scores in history.
                if cleanup_reason in {"killed", "oom"}:
                    if self.cleanup_manager.cleanup_single_simulation_files(file_paths, reason=cleanup_reason):
                        self.cleanup_manager.mark_index_cleaned(len(self.file_paths) - 1)

                self.cleanup_manager.run_periodic_cleanup(
                    scores=self.scores,
                    file_paths=self.file_paths,
                    score_directions=self.objective.score_directions,
                )
                
                return score
            else:
                if verbose:
                    print("    ⚠️  Evaluation failed")
                return None
                
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Error evaluate config: {e}")
            return None
    
    def get_best_config(self) -> Tuple[Optional[Dict], float]:
        """
        Get the best configuration found so far.
        
        Returns:
            (best_config, best_score) tuple
        """
        return self.best_config, self.best_score
    
    def get_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns:
            DataFrame with columns: iteration, config parameters, exec_time, metadata
        """
        if not self.configs:
            return pd.DataFrame()
        
        history = []
        for i, (config, score, metadata) in enumerate(zip(self.configs, self.scores, self.metadata)):
            # Create record with iteration
            record = {'iteration': i + 1}
            
            # Add optimized configuration parameters (parallelism strategy)
            record.update(config)
            
            # Add results immediately after optimization parameters
            record['exec_time_seconds'] = score
            record['best_so_far'] = min(self.scores[:i+1])
            
            # Add simulation metadata (model, network, hardware parameters)
            record.update(metadata)
            
            history.append(record)
        
        return pd.DataFrame(history)
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save optimization results to CSV.
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            optimizer_name = self.__class__.__name__.replace('Optimizer', '').lower()
            filename = (f"{optimizer_name}_results_"
                       f"{self.simulation_runner.model_name}_"
                       f"{self.simulation_runner.num_npus}npus.csv")
        
        filepath = os.path.join(self.save_dir, filename)
        
        df = self.get_history()
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"\n✓ Results saved to: {filepath}")
        
        return filepath
    
    def print_summary(self):
        """Print optimization summary."""
        if not self.scores:
            print("No results to summarize")
            return
        
        print("\n" + "="*70)
        print(f"OPTIMIZATION SUMMARY - {self.__class__.__name__}")
        print("="*70)
        
        # Objective information
        print(f"\n🎯 OBJECTIVE: {self.objective.name}")
        directions = self.objective.score_directions
        if len(directions) == 1:
            print(f"   Direction: {'Minimize' if directions[0] else 'Maximize'}")
        else:
            dir_str = ", ".join("Min" if d else "Max" for d in directions)
            print(f"   Directions: [{dir_str}]  (Obj0 … Obj{len(directions)-1})")
        
        # Statistics
        scores_array = np.array(self.scores)
        print("\n📊 STATISTICS:")
        print(f"   Total evaluations: {len(self.scores)}")
        if isinstance(self.scores[0], tuple):
            # Multi-objective: show statistics for each objective
            n_objectives = len(self.scores[0])
            for i in range(n_objectives):
                obj_scores = [s[i] for s in self.scores]
                print(f"\n   Objective {i+1}:")
                print(f"     Best: {min(obj_scores):.4f}")
                print(f"     Worst: {max(obj_scores):.4f}")
                print(f"     Mean: {np.mean(obj_scores):.4f}")
                print(f"     Std Dev: {np.std(obj_scores):.4f}")
        else:
            # Single objective
            print(f"   Best score: {scores_array.min():.4f}")
            print(f"   Worst score: {scores_array.max():.4f}")
            print(f"   Mean score: {scores_array.mean():.4f}")
            print(f"   Std Dev: {scores_array.std():.4f}")
        
        # Improvement
        if len(self.scores) > 1 and self.init_samples > 0:
            is_moo = isinstance(self.best_score, tuple)
            print("\n📈 IMPROVEMENT:")
            if is_moo:
                # For MOO, "lexicographic best" is misleading: the best single
                # point by lexicographic ordering may never change even when the
                # Pareto front expands significantly.  Show Pareto front stats
                # instead, and note the lexicographic best for reference only.
                directions = self.objective.score_directions   # list of bool (True=min)
                valid_scores = [
                    s for s in self.scores
                    if isinstance(s, tuple)
                    and not any(not math.isfinite(float(v)) or abs(float(v)) >= 1e20
                                for v in s)
                ]
                n_valid = len(valid_scores)

                # Compute Pareto front (non-dominated set)
                def _dominates(a, b):
                    """True if a Pareto-dominates b (better or equal on all, strictly better on one)."""
                    better_on_any = False
                    for v_a, v_b, is_min in zip(a, b, directions):
                        fa, fb = float(v_a), float(v_b)
                        if is_min:
                            if fa > fb: return False
                            if fa < fb: better_on_any = True
                        else:
                            if fa < fb: return False
                            if fa > fb: better_on_any = True
                    return better_on_any

                pareto_front = []
                for candidate in valid_scores:
                    dominated = False
                    pareto_front = [p for p in pareto_front if not _dominates(candidate, p)]
                    for p in pareto_front:
                        if _dominates(p, candidate):
                            dominated = True
                            break
                    if not dominated:
                        pareto_front.append(candidate)

                init_valid = [
                    s for s in self.scores[:self.init_samples]
                    if isinstance(s, tuple)
                    and not any(not math.isfinite(float(v)) or abs(float(v)) >= 1e20
                                for v in s)
                ]

                print(f"   Valid evaluations: {n_valid}/{len(self.scores)}")
                print(f"   Pareto-optimal configurations found: {len(pareto_front)}")
                if init_valid:
                    print(f"   Initial Pareto front size: {len([s for s in init_valid if all(_dominates(s, p) is False for p in init_valid if p is not s) or True])} valid init samples")
                print(f"   Final Pareto front size: {len(pareto_front)}")
                print(f"   Lexicographic best (obj0-first): {format_score(self.best_score)}")
                print(f"   NOTE: For MOO, check the Pareto front plot for full improvement picture.")
            else:
                initial_best = self.objective.get_best_score(self.scores[:self.init_samples])
                print(f"   Initial best: {format_score(initial_best)}")
                print(f"   Final best: {format_score(self.best_score)}")
                improvement_pct = abs((initial_best - self.best_score) / initial_best * 100) if initial_best != 0 else 0.0
                print(f"   Improvement: {improvement_pct:.1f}%")
        
        # Best configuration
        if self.best_config:
            print("\n🏆 BEST CONFIGURATION:")
            # Print all parameters in the config
            config_str = ", ".join([f"{k}={v}" for k, v in self.best_config.items()])
            print(f"   {config_str}")
            print(f"   Score: {format_score(self.best_score)}")
            print(f"   Found at iteration: {self.best_iteration + 1}")
            
            # Configuration profile (only if parallelism params exist)
            if all(k in self.best_config for k in ['dp', 'mp', 'sp', 'pp']):
                dp = self.best_config['dp']
                mp = self.best_config['mp']
                sp = self.best_config['sp']
                pp = self.best_config['pp']
                sharded = self.best_config.get('sharded', False)
                
                total_npus = dp * mp * sp * pp
                print("\n📋 CONFIGURATION PROFILE:")
                print(f"   Total NPUs: {total_npus}/{self.simulation_runner.num_npus}")
                print(f"   DP/MP ratio: {dp/mp:.2f}")
                print(f"   SP enabled: {'Yes' if sp > 1 else 'No'}")
                print(f"   PP enabled: {'Yes' if pp > 1 else 'No'}")
                print(f"   FSDP enabled: {'Yes' if sharded else 'No'}")
        
        # Timing
        if self.start_time:
            elapsed = time.time() - self.start_time
            print("\n⏱️  TIMING:")
            print(f"   Total time: {elapsed:.1f}s")
            print(f"   Time per evaluation: {elapsed/len(self.scores):.1f}s")
        
        # Time profiling statistics
        if self.profile_time:
            self.time_stats.print_summary()
    
    def _log(self, message: str, level: str = "info"):
        """
        Log message if verbose.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        if self.verbose:
            prefix = {
                'info': '',
                'warning': '⚠️  ',
                'error': '❌ '
            }.get(level, '')
            
            print(f"{prefix}{message}")

    def _periodic_cleanup_enabled(self) -> bool:
        """Return True when periodic cleanup is configured."""
        return self.cleanup_manager.periodic_cleanup_enabled()

    def _run_periodic_cleanup(self, force: bool = False):
        """Run periodic cleanup when the configured successful-evaluation threshold is reached."""
        return self.cleanup_manager.run_periodic_cleanup(
            scores=self.scores,
            file_paths=self.file_paths,
            score_directions=self.objective.score_directions,
            force=force,
        )

    @staticmethod
    def run_periodic_cleanup_for_state(cleanup_state, force: bool = False) -> bool:
        """Run shared top-K cleanup against a state dict used by worker-based optimizers."""
        return ArtifactCleanupManager.run_periodic_cleanup_for_state(
            cleanup_state=cleanup_state,
            force=force,
        )

    def sync_cleanup_counters(self, cleanup_counters: Optional[Dict[str, int]] = None):
        """Copy shared cleanup counters back onto this optimizer instance."""
        self.cleanup_manager.sync_counters(cleanup_counters)

    def set_cleanup_debug(self, print_deleted_files: bool = True):
        """Enable/disable detailed printing of deleted files during cleanup."""
        self.cleanup_manager.set_debug(print_deleted_files)

    def get_cleanup_status(self) -> Dict[str, int]:
        """Return cleanup counters for verification/debugging."""
        return self.cleanup_manager.get_status()

    @staticmethod
    def _get_immediate_cleanup_reason(exec_time, is_oom, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Return the cleanup reason for failed, OOM, or killed simulations."""
        return ArtifactCleanupManager.get_immediate_cleanup_reason(exec_time, is_oom, metadata)

    def _cleanup_single_simulation_files(self, file_paths: Optional[Dict[str, str]], reason: str = "failed") -> bool:
        """Immediately clean files generated by a single simulation."""
        return self.cleanup_manager.cleanup_single_simulation_files(file_paths, reason=reason)

    @staticmethod
    def cleanup_path_bundle(
        tracked_paths: Dict[str, str],
        verbose: bool = False,
        log_fn=None,
        print_deleted_files: bool = False,
        reason: str = "cleanup",
    ) -> Dict[str, Any]:
        """Delete all files associated with one simulation record."""
        return ArtifactCleanupManager.cleanup_path_bundle(
            tracked_paths=tracked_paths,
            verbose=verbose,
            log_fn=log_fn,
            print_deleted_files=print_deleted_files,
            reason=reason,
        )

    @staticmethod
    def cleanup_records(
        scores,
        file_paths,
        keep_top_k: int,
        score_directions: Optional[List[bool]] = None,
        cleaned_indices=None,
        verbose: bool = False,
        log_fn=None,
        print_deleted_files: bool = False,
        cleanup_counters: Optional[Dict[str, int]] = None,
    ):
        """
        Remove tracked files for all non-top-K scored records.

        Args:
            scores: Sequence of recorded objective scores (scalars or tuples for MOO).
            file_paths: Sequence of tracked file-path dictionaries aligned with ``scores``.
            keep_top_k: Number of best-scoring records to preserve.
            score_directions: Per-objective directions (True=minimize, False=maximize).
                              Defaults to ``[True]`` (minimize) when omitted.
            cleaned_indices: Mutable set-like or dict-like object used to avoid repeated cleanup.
            verbose: Whether to emit cleanup logs.
            log_fn: Optional logger callable ``log_fn(message, level)``.
        """
        return ArtifactCleanupManager.cleanup_records(
            scores=scores,
            file_paths=file_paths,
            keep_top_k=keep_top_k,
            score_directions=score_directions,
            cleaned_indices=cleaned_indices,
            verbose=verbose,
            log_fn=log_fn,
            print_deleted_files=print_deleted_files,
            cleanup_counters=cleanup_counters,
        )

    @property
    def cleanup_stats(self) -> Dict[str, int]:
        """Backward-compatible access to cleanup counters."""
        return self.cleanup_manager.stats

    @property
    def cleanup_print_deleted_files(self) -> bool:
        """Backward-compatible access to cleanup verbosity."""
        return self.cleanup_manager.print_deleted_files

    @property
    def _cleaned_file_indices(self):
        """Backward-compatible access to cleaned indices."""
        return self.cleanup_manager.cleaned_file_indices

    def compress_and_clean(
        self,
        archive_dir: Optional[str] = None,
        archive_format: str = "tar.gz",
    ) -> Dict[str, Any]:
        """Compress all remaining experiment files into directory-level archives.

        Compresses the simulation output directory and workload directory each into
        a single archive named after that directory, then deletes the originals.
        Call this once after optimization is complete.

        Args:
            archive_dir: Where to write archives. Defaults to the parent of each
                         experiment directory (i.e. next to the directory itself).
            archive_format: ``"tar.gz"`` (default) or ``"zip"``.

        Returns:
            Dict with keys ``archives_created``, ``files_compressed``, ``files_deleted``.
        """
        dirs: List[str] = []
        sr = getattr(self, "simulation_runner", None)
        if sr is not None:
            if getattr(sr, "output_dir", None):
                dirs.append(sr.output_dir)
            if getattr(sr, "workload_dir", None):
                dirs.append(sr.workload_dir)

        return self.cleanup_manager.compress_and_clean(
            experiment_dirs=dirs,
            archive_dir=archive_dir,
            archive_format=archive_format,
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"budget={self.budget}, "
                f"init_samples={self.init_samples}, "
                f"evaluated={len(self.configs)})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        info = [
            f"{self.__class__.__name__}",
            f"Budget: {self.budget} evaluations",
            f"Initialization: {self.init_samples} samples",
            f"Evaluated: {len(self.configs)} configs",
        ]
        
        if self.best_config:
            info.append(f"Best score: {format_score(self.best_score)}")
        
        return "\n".join(info)
