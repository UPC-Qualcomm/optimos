"""
Objective function module for flexible optimization.

This module provides a modular system for defining optimization objectives.
- Use built-in objectives (time, memory, energy, throughput, etc.)
- Create weighted multi-objective optimizations
- Define completely custom objective functions

Example:
    # Default: minimize execution time
    objective = MinimizeExecutionTime()
    
    # Weighted multi-objective
    objective = create_objective(
        "weighted",
        weights={"exec_time": 0.7, "peak_memory_bytes": 0.3}
    )
    
    # Use in optimizer
    optimizer = ScikitBayesianOptimizer(..., objective=objective)
"""
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Union

# Invalid / OOM / missing-metric scores in *natural* (raw) space.
# A large finite value is used so that comparisons such as ``>`` and sorting
# always behave predictably (``float('inf')`` can cause subtle issues with
# ``math.isfinite`` guards being needed everywhere).
#
# Rule: any raw score component with ``abs(value) >= PENALTY`` is treated as
# invalid / worst-in-class regardless of objective direction.
# For **minimize** objectives the sentinel is ``+PENALTY`` (very large).
# For **maximize** objectives the sentinel is also ``+PENALTY`` — it is
# convention, not a real score.  Never use ``PENALTY`` as a genuinely
# favourable maximize score; use a large but strictly smaller finite value.
PENALTY = 1e20


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.
    
    An objective function computes a scalar score from simulation results.
    """
    
    def __init__(self, name: Optional[str] = None, minimize: bool = True):
        """
        Initialize objective function.
        
        Args:
            name: Human-readable name for this objective
            minimize: Whether to minimize (True) or maximize (False) the score
        """
        self.name = name or self.__class__.__name__
        self.minimize = minimize
    
    @abstractmethod
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute objective score from simulation results.
        
        Args:
            exec_time: Execution time in seconds (from simulation)
            metadata: Additional simulation metadata (model params, hardware config, etc.)
            config: Configuration dictionary with hardware/network parameters (e.g., npu_count, local_mem_bw, etc.)
        
        Returns:
            Scalar score to optimize
        """
        pass
    
    @property
    def score_directions(self) -> List[bool]:
        """
        Per-objective optimization directions as a list of booleans.

        True  = minimize this objective (lower raw score is better).
        False = maximize this objective (higher raw score is better).

        For single-objective functions this is ``[self.minimize]``.
        For multi-objective functions it is derived from
        ``self.objective_directions`` (strings "min"/"max" or booleans)
        when that attribute is set, otherwise falls back to
        ``[self.minimize] * n`` where n is inferred at call time.
        """
        directions = getattr(self, 'objective_directions', None)
        if directions is None:
            return [self.minimize]
        result: List[bool] = []
        for d in directions:
            if isinstance(d, str):
                result.append(d.strip().lower() != 'max')
            else:
                result.append(bool(d))
        return result if result else [self.minimize]

    def to_optimizer_score(
        self,
        raw_score,
        *,
        optimizer_bad_value: Optional[float] = None,
    ):
        """
        Convert a raw objective score to the optimizer's maximization space.

        DeepHyper (and the underlying CBO/RandomSearch) always *maximizes*.
        Per ``score_directions``:

        * **Minimize** → negate raw (lower raw becomes higher optimizer value).
        * **Maximize** → keep raw (higher raw stays higher in optimizer space).

        Args:
            raw_score: Scalar or tuple from ``objective.compute()``.
            optimizer_bad_value: If set, any invalid raw component (NaN, ±inf,
                or ``abs(value) >= PENALTY``) is replaced by this value *instead
                of* applying the sign flip.  Use the optimizer's worst-case
                sentinel (e.g. ``-1e20`` for DeepHyper) so invalid runs do not
                pollute the surrogate model or Pareto front.

        Returns:
            Scalar or tuple in optimizer (maximization) space.
        """
        directions = self.score_directions

        def _conv(s: float, is_min: bool) -> float:
            if optimizer_bad_value is not None and (
                not math.isfinite(s) or abs(s) >= PENALTY
            ):
                return float(optimizer_bad_value)
            return float(-s) if is_min else float(s)

        if isinstance(raw_score, (tuple, list)):
            out = []
            for i, s in enumerate(raw_score):
                is_min = directions[i] if i < len(directions) else directions[-1]
                try:
                    sf = float(s)
                except (TypeError, ValueError):
                    if optimizer_bad_value is not None:
                        out.append(float(optimizer_bad_value))
                    else:
                        raise
                else:
                    out.append(_conv(sf, is_min))
            return tuple(out)

        try:
            sf = float(raw_score)
        except (TypeError, ValueError):
            if optimizer_bad_value is not None:
                return float(optimizer_bad_value)
            raise
        return _conv(sf, directions[0])

    def from_optimizer_score(
        self,
        opt_score,
        *,
        optimizer_bad_value: Optional[float] = None,
    ):
        """
        Map optimizer-space values back to natural (raw) objective space.

        Inverts :meth:`to_optimizer_score` for valid values.  Any optimizer
        value that is non-finite *or* has ``abs(value) >= PENALTY`` (including
        exact equality to ``optimizer_bad_value``) maps back to ``PENALTY``
        (``1e20``) in natural space so that it stays clearly invalid for
        ``is_better`` / bookkeeping.

        Args:
            opt_score: Scalar or tuple as stored by the optimizer (CSV).
            optimizer_bad_value: Same sentinel passed to :meth:`to_optimizer_score`
                (e.g. ``-1e20`` for DeepHyper); equality triggers ``PENALTY``.
        """
        directions = self.score_directions

        def _inv(o: float, is_min: bool) -> float:
            try:
                o = float(o)
            except (TypeError, ValueError):
                return PENALTY
            if optimizer_bad_value is not None and o == optimizer_bad_value:
                return PENALTY
            if not math.isfinite(o) or abs(o) >= PENALTY:
                return PENALTY
            return float(-o) if is_min else float(o)

        if isinstance(opt_score, (tuple, list)):
            return tuple(
                _inv(o, directions[i] if i < len(directions) else directions[-1])
                for i, o in enumerate(opt_score)
            )
        return _inv(float(opt_score), directions[0])

    def is_better(self, score1, score2) -> bool:
        """
        Check if score1 is better than score2.

        Args:
            score1: First score (float or tuple for multi-objective)
            score2: Second score (float or tuple for multi-objective)

        Returns:
            True if score1 is better than score2

        Note:
            ``PENALTY`` (``1e20``) always represents an invalid / worst-case
            score **regardless of objective direction**.  For a maximize
            objective, ``+1e20`` is still treated as "worst".  Any component
            with ``abs(value) >= PENALTY`` — or that is non-finite — is
            considered invalid by the ``_invalid`` helper so that penalty
            tuples like ``(1e20, 1e20)`` can never silently overwrite a valid
            best score.

            For multi-objective tuples uses lexicographic comparison with
            per-objective direction from ``score_directions``.
        """
        def _invalid(s) -> bool:
            """True when s is None, non-finite, or abs(value) >= PENALTY."""
            if s is None:
                return True
            try:
                f = float(s)
                return not math.isfinite(f) or abs(f) >= PENALTY
            except (TypeError, ValueError):
                return True

        directions = self.score_directions

        # ── Early-exit for scalar sentinel (initial best_score = float('inf')) ──
        # Handles the mixed-type case where the initial best_score is a scalar
        # and the incoming score is a tuple (common at the start of MOO runs).
        if not isinstance(score2, (tuple, list)):
            if _invalid(score2):
                # score2 is a scalar invalid/penalty — score1 wins if it has
                # at least one finite component (tuple) or is itself finite.
                if isinstance(score1, (tuple, list)):
                    return not all(_invalid(s) for s in score1)
                return not _invalid(score1)
        if not isinstance(score1, (tuple, list)) and _invalid(score1):
            return False  # scalar invalid score1 is never better

        # ── MOO tuple path ────────────────────────────────────────────────────
        if isinstance(score1, (tuple, list)) and isinstance(score2, (tuple, list)):
            s1_all_bad = all(_invalid(s) for s in score1)
            s2_all_bad = all(_invalid(s) for s in score2)
            if s2_all_bad:
                return not s1_all_bad   # any finite score beats all-penalty
            if s1_all_bad:
                return False

            for i, (s1, s2) in enumerate(zip(score1, score2)):
                is_min = directions[i] if i < len(directions) else directions[-1]
                s1_bad = _invalid(s1)
                s2_bad = _invalid(s2)
                if s1_bad and s2_bad:
                    continue            # both invalid on this axis → tie
                if s2_bad:
                    return True         # s2 invalid, s1 finite → s1 wins
                if s1_bad:
                    return False        # s1 invalid → s2 wins
                v1, v2 = float(s1), float(s2)
                if is_min:
                    if v1 < v2: return True
                    if v1 > v2: return False
                else:
                    if v1 > v2: return True
                    if v1 < v2: return False
            return False  # equal on all objectives

        # ── Scalar path ───────────────────────────────────────────────────────
        s1_bad = _invalid(score1)
        s2_bad = _invalid(score2)
        if s2_bad:
            return not s1_bad
        if s1_bad:
            return False
        is_min = directions[0]
        return float(score1) < float(score2) if is_min else float(score1) > float(score2)

    def get_best_score(self, scores: list):
        """
        Get the best score from a list.

        Args:
            scores: List of scores (scalars or tuples).

        Returns:
            Best score according to objective directions.  For MOO tuples the
            comparison is lexicographic with per-objective direction (the same
            key used by cleanup ranking), so mixed min/max objectives are
            handled correctly.
        """
        directions = self.score_directions

        def _sort_key(s):
            if isinstance(s, (tuple, list)):
                key = []
                for i, v in enumerate(s):
                    is_min_i = directions[i] if i < len(directions) else directions[-1]
                    try:
                        fv = float(v)
                        if not math.isfinite(fv) or abs(fv) >= PENALTY:
                            key.append(math.inf)
                        else:
                            key.append(fv if is_min_i else -fv)
                    except (TypeError, ValueError):
                        key.append(math.inf)
                return tuple(key) if key else (math.inf,)
            # Scalar
            is_min_0 = directions[0]
            try:
                fv = float(s)
                if not math.isfinite(fv) or abs(fv) >= PENALTY:
                    return math.inf
                return fv if is_min_0 else -fv
            except (TypeError, ValueError):
                return math.inf

        if not scores:
            is_min = directions[0]
            return PENALTY if is_min else -PENALTY
        return min(scores, key=_sort_key)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return self.name


class MinimizeExecutionTime(ObjectiveFunction):
    """
    Minimize execution time (default objective).
    
    Simply returns the execution time as the objective score.
    This is the standard objective for performance optimization.
    """
    
    def __init__(self):
        super().__init__("Minimize Execution Time")
    
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> float:
        """Return execution time as objective."""
        if is_oom:
            return PENALTY
        return exec_time
    


class MinimizeExecutionTimeAndNetworkBW(ObjectiveFunction):
    """
    Minimize execution time and the total network bandwidth.
    
    """
    
    def __init__(self):
        super().__init__("Minimize Execution Time and Network Bandwidth Perf per BW/NPU")
    
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute objective based on COSMIC paper formula.
        
        Formula: reward = 1 / sqrt(power(sim_time * sum(network_bw) - 1, 2))
        where network_bw includes both intra-node and inter-node bandwidths."""
        if is_oom:
            return PENALTY
        
        if config is None:
            # Fallback to execution time only if no config provided
            return exec_time
        
        # Extract configuration parameters
        npu_count = config.get('npu_count', 1)
        intra_node_bw = config.get('intra-node-bw', 0)  # GB/s
        inter_node_bw = config.get('inter-node-bw', 0)  # GB/s
        npus_per_node = 8  # Default: 8 NPUs per node
        
        # Calculate number of nodes
        num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)  # Ceiling division
        
        # Calculate total network bandwidth
        # Intra-node: bandwidth within each node (connections between NPUs in same node)
        # Inter-node: bandwidth between nodes
        if num_nodes == 1:
            # Single node: only intra-node bandwidth matters
            total_network_bw = intra_node_bw * (npu_count - 1)  # Connections between NPUs
        else:
            # Multiple nodes: both intra-node and inter-node bandwidth
            intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
            inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
            total_network_bw = intra_bw_total + inter_bw_total
        
        # Avoid division by zero or negative values
        if total_network_bw <= 0 or exec_time <= 0:
            return PENALTY
        
        # COSMIC formula: reward = 1 / sqrt((sim_time * sum(network_bw) - 1)^2)
        obj = exec_time * total_network_bw
        
        denominator = math.sqrt((exec_time * total_network_bw - 1 ) ** 2)
                
        reward = 1.0 / denominator
        
        return reward

class MinimizePower(ObjectiveFunction):
    """
    Minimize total power consumption (Mode D: Full LPM, Energy-Proportional).

    Reads ``total_power_W`` injected into metadata by
    :meth:`SimulationRunner._run_power_estimation`.  Only meaningful for g2
    simulations; returns ``PENALTY`` if the metric is absent.
    """

    def __init__(self):
        super().__init__("Minimize Total Power (W) [Mode D]")

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> float:
        if is_oom:
            return PENALTY
        total_power_W = metadata.get('total_power_W')
        if total_power_W is None:
            return PENALTY
        return float(total_power_W)


class MinimizeEnergy(ObjectiveFunction):
    """
    Minimize total energy consumption (Mode D: Full LPM, Energy-Proportional).

    Reads ``total_energy_J`` injected into metadata by
    :meth:`SimulationRunner._run_power_estimation`.  Only meaningful for g2
    simulations; returns ``PENALTY`` if the metric is absent.
    """

    def __init__(self):
        super().__init__("Minimize Total Energy (J) [Mode D]")

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> float:
        if is_oom:
            return PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None:
            return PENALTY
        return float(total_energy_J)


class MinimizePowerAndTime(ObjectiveFunction):
    """
    Multi-objective: jointly minimize total power (W) and execution time.

    Both objectives are log10-transformed so their scales are comparable
    regardless of absolute magnitude.

    Returns:
        tuple: ``(log10(total_power_W), log10(exec_time))`` — both minimized.
    """

    def __init__(self):
        super().__init__("Minimize Power (W) and Execution Time [Mode D, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_power_W = metadata.get('total_power_W')
        if total_power_W is None or exec_time is None:
            return PENALTY, PENALTY
        return math.log10(max(1.0, total_power_W)), math.log10(max(1.0, exec_time))


class MinimizeEnergyAndTime(ObjectiveFunction):
    """
    Multi-objective: jointly minimize total energy (J) and execution time.

    Both objectives are log10-transformed so their scales are comparable
    regardless of absolute magnitude.

    Returns:
        tuple: ``(log10(total_energy_J), log10(exec_time))`` — both minimized.
    """

    def __init__(self):
        super().__init__("Minimize Energy (J) and Execution Time [Mode D, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY
        return math.log10(max(1.0, total_energy_J)), math.log10(max(1.0, exec_time))


def _compute_total_network_bw(config: Dict[str, Any], npus_per_node: int = 8) -> float:
    """Compute aggregate network bandwidth (GB/s) from optimization config."""
    npu_count     = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)
    inter_node_bw = config.get('inter-node-bw', 0)
    num_nodes     = max(1, (npu_count + npus_per_node - 1) // npus_per_node)

    if num_nodes == 1:
        return intra_node_bw * (npu_count - 1)

    return (intra_node_bw * npus_per_node * num_nodes
            + inter_node_bw * (num_nodes - 1))


class MinimizeLatencyAndTotalNetworkBW(ObjectiveFunction):
    """
    Multi-objective: jointly minimize execution latency and total network BW.

    Returns raw values ``(exec_time, total_network_bw_GBps)``.
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Latency and Total Network BW [Raw, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        if is_oom or exec_time is None or config is None:
            return PENALTY, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        return float(exec_time), float(total_network_bw)


class MinimizeLatencyAndNetworkBW(ObjectiveFunction):
    """
    Multi-objective: jointly minimize latency and network BW (log-scaled).

    Returns ``(log10(exec_time), log10(total_network_bw_GBps))``.
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Latency and Network BW [Log, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        if is_oom or exec_time is None or config is None:
            return PENALTY, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_exec_time = math.log10(max(1.0, exec_time))
        log_network_bw = math.log10(max(1.0, total_network_bw))
        return log_exec_time, log_network_bw


class MinimizeLatencyAndMemory(ObjectiveFunction):
    """
    Multi-objective: jointly minimize latency and total memory footprint.

    Returns ``(log10(exec_time), log10(total_memory_GB))``.
    """

    def __init__(self):
        super().__init__("Minimize Latency and Memory [Log, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        if is_oom or exec_time is None or config is None:
            return PENALTY, PENALTY

        npu_count = config.get('npu_count', 1)
        local_mem_size = config.get('local-mem-size', 0)
        total_memory_usage = local_mem_size * npu_count

        log_exec_time = math.log10(max(1.0, exec_time))
        log_memory_usage = math.log10(max(1.0, total_memory_usage))
        return log_exec_time, log_memory_usage


class MinimizeNetworkBWAndMemory(ObjectiveFunction):
    """
    Multi-objective: jointly minimize network BW and total memory footprint.

    Returns ``(log10(total_network_bw_GBps), log10(total_memory_GB))``.
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Network BW and Memory [Log, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        if is_oom or config is None:
            return PENALTY, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        npu_count = config.get('npu_count', 1)
        local_mem_size = config.get('local-mem-size', 0)
        total_memory_usage = local_mem_size * npu_count

        log_network_bw = math.log10(max(1.0, total_network_bw))
        log_memory_usage = math.log10(max(1.0, total_memory_usage))
        return log_network_bw, log_memory_usage


class MinimizeLatencyNetworkBWAndMemory(ObjectiveFunction):
    """
    Three-objective optimization on latency, network BW, and memory.

    Returns ``(log10(exec_time), log10(total_network_bw_GBps), log10(total_memory_GB))``.
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Latency, Network BW, and Memory [Log, 3-MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        if is_oom or exec_time is None or config is None:
            return PENALTY, PENALTY, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        npu_count = config.get('npu_count', 1)
        local_mem_size = config.get('local-mem-size', 0)
        total_memory_usage = local_mem_size * npu_count

        log_exec_time = math.log10(max(1.0, exec_time))
        log_network_bw = math.log10(max(1.0, total_network_bw))
        log_memory_usage = math.log10(max(1.0, total_memory_usage))
        return log_exec_time, log_network_bw, log_memory_usage


class MinimizeLatencyAndNetworkBWRaw(MinimizeLatencyAndTotalNetworkBW):
    """Compatibility alias for raw latency-network BW objective."""

    def __init__(self, npus_per_node: int = 8):
        super().__init__(npus_per_node=npus_per_node)
        self.name = "Minimize Latency and Network BW [Raw, MOO]"


class MinimizeLatencyAndNetworkBWMinMax(ObjectiveFunction):
    """
    Multi-objective with min-max normalization for latency and network BW.

    Returns ``(norm_latency, norm_network_bw)`` in ``[0, 1]`` (clipped).
    """

    def __init__(
        self,
        npus_per_node: int = 8,
        time_min: float = 1.0,
        time_max: float = 100.0,
        bw_min: float = 100.0,
        bw_max: float = 2000.0,
        exec_time_scale: float = 1e9,
    ):
        super().__init__("Minimize Latency and Network BW [MinMax, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node
        self.time_min = time_min
        self.time_max = time_max
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.exec_time_scale = exec_time_scale

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        if is_oom or exec_time is None or config is None:
            return PENALTY, PENALTY

        #scaled_exec_time = exec_time / self.exec_time_scale if self.exec_time_scale else exec_time
        scaled_exec_time = exec_time
        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)

        time_span = self.time_max - self.time_min
        bw_span = self.bw_max - self.bw_min
        if time_span <= 0 or bw_span <= 0:
            return PENALTY, PENALTY

        norm_time = (scaled_exec_time - self.time_min) / time_span
        norm_network = (total_network_bw - self.bw_min) / bw_span
        norm_time = min(1.0, max(0.0, norm_time))
        norm_network = min(1.0, max(0.0, norm_network))
        return norm_time, norm_network


class MinimizeLatencyAndNetworkBWSqrt(ObjectiveFunction):
    """
    Multi-objective with square-root compression for latency and network BW.

    Returns ``(sqrt(scaled_latency), sqrt(network_bw))``.
    """

    def __init__(self, npus_per_node: int = 8, exec_time_scale: float = 1e9):
        super().__init__("Minimize Latency and Network BW [Sqrt, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node
        self.exec_time_scale = exec_time_scale

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        if is_oom or exec_time is None or config is None:
            return PENALTY, PENALTY

        #scaled_exec_time = exec_time / self.exec_time_scale if self.exec_time_scale else exec_time
        scaled_exec_time = exec_time
        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)

        sqrt_time = math.sqrt(max(0.0, scaled_exec_time))
        sqrt_network = math.sqrt(max(0.0, total_network_bw))
        return sqrt_time, sqrt_network


class MinimizeLatencyAndNetworkBWPower(ObjectiveFunction):
    """
    Multi-objective with configurable power transform for latency/network BW.

    Returns ``(scaled_latency ** power, network_bw ** power)``.
    """

    def __init__(self, npus_per_node: int = 8, power: float = 0.5, exec_time_scale: float = 1e9):
        super().__init__(f"Minimize Latency and Network BW [Power={power}, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node
        self.power = power
        self.exec_time_scale = exec_time_scale

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        if is_oom or exec_time is None or config is None:
            return PENALTY, PENALTY

        #scaled_exec_time = exec_time / self.exec_time_scale if self.exec_time_scale else exec_time
        scaled_exec_time = exec_time
        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)

        power_time = math.pow(max(0.0, scaled_exec_time), self.power)
        power_network = math.pow(max(0.0, total_network_bw), self.power)
        return power_time, power_network


class MinimizeEDPAndNetworkBW(ObjectiveFunction):
    """
    Multi-objective: minimize EDP **and** total network bandwidth jointly.

    Objective 0: ``log10(total_energy_J × exec_cycles)``  — minimize EDP.
    Objective 1: ``log10(total_network_bw_GBps)``          — minimize network cost.

    **Why add network BW as a second objective?**

    EDP alone can be improved by throwing unlimited bandwidth at a workload
    (more links → faster execution → lower EDP even if energy rises).  Making
    total network BW an explicit second objective prevents that shortcut and
    forces the optimizer to find configurations that are simultaneously
    energy-delay efficient *and* network-frugal — important when provisioning
    cost or physical link count is a concern.

    Total network BW is derived from the configuration parameters:

    * Single-node cluster: ``intra_node_bw × (npu_count − 1)``
    * Multi-node cluster:  ``intra_node_bw × npus_per_node × num_nodes
                             + inter_node_bw × (num_nodes − 1)``

    Both objectives are log10-transformed for comparable scale.

    **Requires** ``estimate_power=1`` in ``net_sim_config``.
    Returns ``(PENALTY, PENALTY)`` if EDP metrics are absent or OOM.
    """

    def __init__(self, npus_per_node: int = 8):
        """
        Args:
            npus_per_node: Number of NPUs per physical node (default: 8).
                           Used to split ``npu_count`` into intra / inter
                           bandwidth contributions.
        """
        super().__init__("Minimize EDP and Network BW [Mode D, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY

        # EDP objective
        edp = total_energy_J * exec_time
        log_edp = math.log10(max(1.0, edp))

        # Network BW objective (derived from config)
        if config is None:
            return log_edp, PENALTY
        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)

        log_bw = math.log10(max(1.0, total_network_bw))
        return log_edp, log_bw


class MinimizeED2PAndNetworkBW(ObjectiveFunction):
    """
    Multi-objective: minimize ED²P (delay-sensitive) and network bandwidth.

    Objective 0: ``log10(E) + 2*log10(D)``  (ED²P)
    Objective 1: ``log10(total_network_bw_GBps)``

    This objective is useful when delay sensitivity is important while still
    controlling network over-provisioning.
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize ED²P and Network BW [Mode D, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY

        log_ed2p = (math.log10(max(1.0, total_energy_J))
                    + 2.0 * math.log10(max(1.0, exec_time)))

        if config is None:
            return log_ed2p, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_bw = math.log10(max(1.0, total_network_bw))
        return log_ed2p, log_bw

class MinimizeE2DAndNetworkBW(ObjectiveFunction):
    """
    Multi-objective: minimize E²D (energy-sensitive) and network bandwidth.

    Objective 0: ``2*log10(E) + log10(D)``  (E²D)
    Objective 1: ``log10(total_network_bw_GBps)``

    This objective is useful when energy sensitivity is important while still
    controlling network over-provisioning.
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize E²D and Network BW [Mode D, MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY

        log_e2d = (2.0 * math.log10(max(1.0, total_energy_J))
                   + math.log10(max(1.0, exec_time)))

        if config is None:
            return log_e2d, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_bw = math.log10(max(1.0, total_network_bw))
        return log_e2d, log_bw


class MinimizeEnergyCyclesAndNetworkBW(ObjectiveFunction):
    """
    Three-objective optimization: minimize energy, cycles, and network BW.

    Returns:
        tuple: ``(log10(E), log10(D), log10(BW))``
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Energy, Cycles, and Network BW [Mode D, 3-MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY, PENALTY

        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY, PENALTY, PENALTY

        log_energy = math.log10(max(1.0, total_energy_J))
        log_cycles = math.log10(max(1.0, exec_time))

        if config is None:
            return log_energy, log_cycles, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_bw = math.log10(max(1.0, total_network_bw))
        return log_energy, log_cycles, log_bw


class MinimizePowerCyclesAndNetworkBW(ObjectiveFunction):
    """
    Three-objective optimization: minimize power, cycles, and network BW.

    Returns:
        tuple: ``(log10(P), log10(D), log10(BW))``
    """

    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Power, Cycles, and Network BW [Mode D, 3-MOO]")
        self.is_multi_objective = True
        self.objective_directions = ["min", "min", "min"]
        self.npus_per_node = npus_per_node

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None):
        
        if is_oom:
            return PENALTY, PENALTY, PENALTY

        total_power_W = metadata.get('total_power_W')
        if total_power_W is None or exec_time is None:
            return PENALTY, PENALTY, PENALTY

        log_power = math.log10(max(1.0, total_power_W))
        log_cycles = math.log10(max(1.0, exec_time))

        if config is None:
            return log_power, log_cycles, PENALTY

        total_network_bw = _compute_total_network_bw(config, self.npus_per_node)
        log_bw = math.log10(max(1.0, total_network_bw))
        return log_power, log_cycles, log_bw


class MinimizeWeightedEDP(ObjectiveFunction):
    r"""
    Minimize the generalised Energy-Delay product  E^alpha × D^beta.

    Plain EDP (alpha=beta=1) is **symmetric**: doubling energy and halving
    delay leaves the score unchanged because the two effects cancel exactly.
    This means the optimizer cannot distinguish between:

    * A fast, power-hungry configuration  (low D, high E)
    * A slow, energy-sipping configuration (high D, low E)

    Choosing alpha ≠ beta **breaks the symmetry** and encodes a preference:

    +---------+---------+---------------------------------------------+
    | alpha   | beta    | Interpretation                              |
    +=========+=========+=============================================+
    | 1       | 2       | ED²P — penalises delay twice as much        |
    |         |         | as energy.  Performance-oriented.           |
    |         |         | Doubling D costs 2× what doubling E costs.  |
    +---------+---------+---------------------------------------------+
    | 2       | 1       | E²D — penalises energy twice as much        |
    |         |         | as delay.  Efficiency-oriented.             |
    |         |         | Doubling E costs 2× what doubling D costs.  |
    +---------+---------+---------------------------------------------+
    | 1       | 1       | Classic EDP (symmetric).                    |
    +---------+---------+---------------------------------------------+

    Under log10-transformation the score becomes::

        alpha * log10(E) + beta * log10(D)

    so the optimizer sees a linear combination of log-energy and log-delay
    with the chosen weights — no cancellation is possible when alpha ≠ beta.

    **Requires** ``estimate_power=1`` in ``net_sim_config``.
    Returns ``PENALTY`` if energy metric is absent or OOM.

    Args:
        alpha: Exponent on energy   (default 1.0).
        beta:  Exponent on delay     (default 2.0 → ED²P).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        name = f"Minimize E^{alpha}×D^{beta} "
        if alpha == 1 and beta == 2:
            name += "(ED²P, performance-oriented) [Mode D]"
        elif alpha == 2 and beta == 1:
            name += "(E²D, efficiency-oriented) [Mode D]"
        else:
            name += "(Weighted EDP) [Mode D]"
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta

    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> float:
        
        if is_oom:
            return PENALTY
        total_energy_J = metadata.get('total_energy_J')
        if total_energy_J is None or exec_time is None:
            return PENALTY
        # alpha*log(E) + beta*log(D)  ==  log(E^alpha * D^beta)
        return (self.alpha * math.log10(max(1.0, total_energy_J))
                + self.beta  * math.log10(max(1.0, exec_time)))


class WeightedMultiObjective(ObjectiveFunction):
    """
    Weighted combination of multiple objectives.
    
    Combines multiple objectives with configurable weights.
    Useful for multi-objective optimization.
    
    Example:
        # 70% time, 20% memory, 10% energy
        objective = WeightedMultiObjective({
            'exec_time': 0.7,
            'peak_memory_bytes': 0.2,
            'energy_joules': 0.1
        })
    """
    
    def __init__(self, weights: Dict[str, float], normalize: bool = True):
        """
        Initialize weighted multi-objective.
        
        Args:
            weights: Dictionary mapping metric names to weights
                    Special key 'exec_time' uses execution time
                    Other keys should be in metadata
            normalize: Whether to normalize metrics before combining
        """
        super().__init__("Weighted Multi-Objective")
        self.weights = weights
        self.normalize = normalize
        
        # Track metric ranges for normalization
        self.metric_mins: Dict[str, float] = {}
        self.metric_maxs: Dict[str, float] = {}
    
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> float:
        """Return weighted combination of metrics."""
        if is_oom:
            return PENALTY
        # Collect all metrics TODO: Not complete list
        metrics = {'exec_time': exec_time}
        metrics.update(metadata)
        
        # Update normalization ranges
        if self.normalize:
            for metric_name in self.weights.keys():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if metric_name not in self.metric_mins:
                        self.metric_mins[metric_name] = value
                        self.metric_maxs[metric_name] = value
                    else:
                        self.metric_mins[metric_name] = min(self.metric_mins[metric_name], value)
                        self.metric_maxs[metric_name] = max(self.metric_maxs[metric_name], value)
        
        # Compute weighted sum
        weighted_sum = 0.0
        for metric_name, weight in self.weights.items():
            if metric_name not in metrics:
                raise ValueError(f"Metric '{metric_name}' not found in results")
            
            value = metrics[metric_name]
            
            # Normalize if enabled
            if self.normalize:
                min_val = self.metric_mins[metric_name]
                max_val = self.metric_maxs[metric_name]
                if max_val > min_val:
                    value = (value - min_val) / (max_val - min_val)
                else:
                    value = 0.0
            
            weighted_sum += weight * value
        
        return weighted_sum


class MinimizeTimeMaximizeThroughputPerEnergy(ObjectiveFunction):
    """
    Multi-objective for cluster size optimization.
    
    Objective 0: Minimize execution time (shorter is better).
    Objective 1: Maximize throughput per unit energy (samples/sec/MJ).
    
    Returns tuple: (log10(exec_time), log10(throughput_per_energy))
    
    Directions: ["min", "max"]  # minimize time, maximize efficiency
    
    This objective helps find cluster configurations that are both fast
    and energy-efficient. The throughput/energy metric encourages the optimizer
    to find configurations that deliver high performance without excessive
    energy consumption.
    
    Requires: estimate_power=1 in net_sim_config
    """
    
    def __init__(self, npus_per_node: int = 8):
        super().__init__("Minimize Time, Maximize Throughput/Energy")
        self.npus_per_node = npus_per_node
        self.objective_directions = ["min", "max"]  # minimize time, maximize efficiency
        self.is_multi_objective = True
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Compute (exec_time, throughput_per_energy).
        
        throughput_per_energy = samples_per_sec / (total_energy_MJ)
                              = batch_size / (exec_time_sec * total_energy_J * 1e-6)
        """
        if is_oom:
            return PENALTY, PENALTY

        if metadata is None:
            metadata = {}

        if exec_time is None or exec_time <= 0:
            return PENALTY, PENALTY

        # Requirement: consume the reported metric from power estimator.
        samples_per_mj = metadata.get("samples_per_sec_per_mj")
        
        if samples_per_mj is None or samples_per_mj <= 0 or not math.isfinite(samples_per_mj):
            return PENALTY, PENALTY

        return math.log10(exec_time), math.log10(samples_per_mj)
        


class MaximizeMemoryMinimizeTime(ObjectiveFunction):
    """
    Multi-objective for batch size optimization.
    
    Objective 0: Maximize peak memory usage (higher is better, closer to GPU capacity).
    Objective 1: Minimize execution time (shorter is better).
    
    Returns tuple: (log10(peak_memory_GB), log10(exec_time))
    
    Directions: ["max", "min"]  # maximize memory, minimize time
    
    This objective helps find batch sizes that efficiently use GPU memory
    while maintaining reasonable execution times. The optimizer will prefer
    configurations with higher memory utilization and shorter execution time.
    
    The peak memory is extracted from metadata['peak_memory_bytes'] and
    converted to GB for interpretation.
    """
    
    def __init__(self):
        super().__init__("Maximize Memory, Minimize Time")
        self.objective_directions = ["max", "min"]  # maximize memory, minimize time
        self.is_multi_objective = True
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any],
                config: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Compute (peak_memory_GB, exec_time).
        """
        if is_oom:
            return PENALTY, PENALTY

        if metadata is None:
            metadata = {}
        if config is None:
            config = {}

        if exec_time is None or exec_time <= 0:
            return PENALTY, PENALTY

        peak_memory_gb = metadata.get("peak_memory_gb")

        if peak_memory_gb is None or peak_memory_gb <= 0 or not math.isfinite(peak_memory_gb):
            return PENALTY, PENALTY

        return math.log10(peak_memory_gb), math.log10(exec_time)


class CustomObjective(ObjectiveFunction):
    """
    Custom objective function from user-provided callable.
    
    Allows defining arbitrary objective functions.
    
    Example:
        # Minimize time + 0.1 * sqrt(memory_gb)
        def my_objective(exec_time, metadata):
            memory_gb = metadata['peak_memory_bytes'] / (1024**3)
            return exec_time + 0.1 * (memory_gb ** 0.5)
        
        objective = CustomObjective(my_objective, "My Custom Objective")
    """
    
    def __init__(
        self, 
        compute_fn: Callable[[float, Dict[str, Any]], float], 
        name: str = "Custom Objective",
        minimize: bool = True,
        is_multi_objective: bool = False,
        objective_directions: Optional[List[Union[bool, str]]] = None,
    ):
        """
        Initialize custom objective.
        
        Args:
            compute_fn: Callable that takes (exec_time, is_oom, metadata, config) and returns score or tuple of scores
            name: Name for this objective
            minimize: Whether to minimize (True) or maximize (False)
            is_multi_objective: Whether this objective returns multiple values (tuple)
            objective_directions: Optional per-objective directions for MOO.
                                 Use booleans (True=minimize, False=maximize)
                                 or strings ("min"/"max"). If omitted, this
                                 will be inferred from compute_fn.objective_directions
                                 when present.
        """
        super().__init__(name, minimize)
        self.compute_fn = compute_fn
        self.is_multi_objective = is_multi_objective
        inferred_directions = getattr(compute_fn, "objective_directions", None)
        self.objective_directions = objective_directions if objective_directions is not None else inferred_directions
    
    def compute(self, exec_time: float, is_oom: bool, metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Call user-provided compute function with all parameters including config."""
        return self.compute_fn(exec_time, is_oom, metadata, config)


# Convenience factory function
def create_objective(objective_type: str, **kwargs) -> ObjectiveFunction:
    """
    Factory function to create objective functions by name.
    
    Args:
        objective_type: Type of objective ['time', 'time_and_network_bw', 'weighted', 'custom']
        **kwargs: Additional arguments for the objective
    
    Returns:
        ObjectiveFunction instance
    
    Example:
        objective = create_objective('time')
        objective = create_objective('weighted', weights={'exec_time': 0.7, 'peak_memory_bytes': 0.3})
    """
    objectives = {
        'time': MinimizeExecutionTime,
        'time_and_network_bw':  MinimizeExecutionTimeAndNetworkBW,
        'power': MinimizePower,
        'energy': MinimizeEnergy,
        'power_and_time': MinimizePowerAndTime,
        'energy_and_time': MinimizeEnergyAndTime,
        'latency_total_network': MinimizeLatencyAndTotalNetworkBW,
        'latency_network': MinimizeLatencyAndNetworkBW,
        'latency_memory': MinimizeLatencyAndMemory,
        'network_memory': MinimizeNetworkBWAndMemory,
        'latency_network_memory': MinimizeLatencyNetworkBWAndMemory,
        'latency_network_raw': MinimizeLatencyAndNetworkBWRaw,
        'latency_network_minmax': MinimizeLatencyAndNetworkBWMinMax,
        'latency_network_sqrt': MinimizeLatencyAndNetworkBWSqrt,
        'latency_network_power': MinimizeLatencyAndNetworkBWPower,
        'edp_and_network_bw': MinimizeEDPAndNetworkBW,
        'ed2p_and_network_bw': MinimizeED2PAndNetworkBW,
        'e2d_and_network_bw': MinimizeE2DAndNetworkBW,
        'energy_cycles_and_network_bw': MinimizeEnergyCyclesAndNetworkBW,
        'power_cycles_network_bw': MinimizePowerCyclesAndNetworkBW,
        'edp': MinimizeWeightedEDP,          # E^alpha x D^beta, default ED2P
        'ed2p': lambda: MinimizeWeightedEDP(1, 2),    # performance-oriented shortcut
        'e2d':  lambda: MinimizeWeightedEDP(2, 1),    # efficiency-oriented shortcut
        'time_and_throughput_per_energy': MinimizeTimeMaximizeThroughputPerEnergy,
        'memory_and_time': MaximizeMemoryMinimizeTime,
        'weighted': WeightedMultiObjective,
        'custom': CustomObjective
    }
    
    if objective_type not in objectives:
        raise ValueError(f"Unknown objective type: {objective_type}. "
                        f"Available: {list(objectives.keys())}")
    
    return objectives[objective_type](**kwargs)


def get_available_objective_types(include_non_sweepable: bool = False) -> List[str]:
    """
    Return objective types supported by ``create_objective``.

    Args:
        include_non_sweepable: Include objective types that require extra
            mandatory kwargs (currently ``weighted`` and ``custom``).
    """
    objective_types = [
        'time',
        'time_and_network_bw',
        'power',
        'energy',
        'power_and_time',
        'energy_and_time',
        'latency_total_network',
        'latency_network',
        'latency_memory',
        'network_memory',
        'latency_network_memory',
        'latency_network_raw',
        'latency_network_minmax',
        'latency_network_sqrt',
        'latency_network_power',
        'edp_and_network_bw',
        'ed2p_and_network_bw',
        'e2d_and_network_bw',
        'energy_cycles_and_network_bw',
        'power_cycles_network_bw',
        'edp',
        'ed2p',
        'e2d',
        'time_and_throughput_per_energy',
        'memory_and_time',
    ]
    if include_non_sweepable:
        objective_types.extend(['weighted', 'custom'])
    return objective_types
