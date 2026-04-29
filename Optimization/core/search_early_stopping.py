"""Adaptive search-level early stopping callback for DeepHyper.

Implements a true sliding-window patience mechanism:
- Patience counter starts at 0.
- Any result that is STRICTLY BETTER than the current best resets the counter to 0.
- Any result that is NOT better increments the counter.
- Penalty results (killed simulations, abs(objective) >= threshold) increment the
  counter but do NOT update the best, because they represent intentionally terrible
  configurations.
- Failure results ("F", None, non-numeric) are completely ignored — they carry no
  information about the search landscape and should not influence patience.
- The search is stopped when patience_counter >= patience_limit AND at least
  min_evaluations_before_check valid evaluations have been processed.

For single-objective runs the best is tracked as the running maximum scalar.
For multi-objective runs the best is tracked as the running hypervolume indicator
(HVI), which is the standard aggregated quality measure for a Pareto front.
"""

import numpy as np

try:
    from deephyper.evaluator.callback import Callback
    from deephyper.skopt.moo import hypervolume
    _DEEPHYPER_AVAILABLE = True
except ImportError:
    # Provide a no-op base so that the module can be imported even without DeepHyper
    class Callback:  # type: ignore
        def on_done(self, job): pass
        def on_done_other(self, job): pass
    hypervolume = None
    _DEEPHYPER_AVAILABLE = False

PENALTY = 1e20

class AdaptiveSearchEarlyStopping(Callback):
    """Stop the search when no improvement is seen for ``patience_limit`` evaluations.

    The patience counter uses a true sliding-window reset policy:

    - **Better result** → reset counter to 0, update best aggregate.
    - **Worse result** → increment counter.
    - **Penalty result** (killed simulation, ``abs(obj) >= PENALTY``) →
      increment counter, but do **not** update best.
    - **Failure result** (``"F"``, ``None``, non-numeric) → **ignored entirely**,
      counter and best unchanged.
    - **Warmup phase** (``_total_seen <= min_evaluations_before_check``) →
      results are collected into the running best but patience counting does
      not start yet, to allow the surrogate model to initialise properly.

    For multi-objective runs the progress signal is the Hypervolume Indicator
    (HVI), the same scalar used by DeepHyper's ``ObjectiveRecorder``.  HVI
    increases when the Pareto front improves; stagnation means no new Pareto
    points were discovered by the last ``patience_limit`` evaluations.

    .. important::
        For small search spaces or when using many parallel workers, set
        ``patience_limit`` conservatively (at least 3–5× ``n_workers``) to
        avoid stopping before the surrogate model has seen enough data.

    Args:
        patience_limit (int):
            Number of non-improving evaluations to tolerate before stopping.
            Set to ``-1`` to disable early stopping entirely.
            Defaults to ``50``.
        min_evaluations_before_check (int):
            Minimum number of completed (non-failure) evaluations before the
            patience counter starts.  Should be at least ``n_initial_points``
            so that random exploration finishes before the stopping check begins.
            Defaults to ``0``.
        verbose (bool):
            Print progress messages when the best improves or the search stops.
            Defaults to ``True``.
    """

    def __init__(
        self,
        patience_limit: int = 50,
        min_evaluations_before_check: int = 0,
        verbose: bool = True,
        objective_directions: list = None,
    ):
        """
        Args:
            objective_directions: Optional list of ``"min"`` / ``"max"`` strings,
                one per objective.  When ``None`` (default), all objectives are
                assumed to be in DeepHyper's internal maximisation convention
                (minimised objectives were negated before being returned from the
                evaluation function).  When provided, the values in
                ``job.objective`` are treated as raw (un-negated) values and each
                dimension is converted to minimisation space according to its
                direction (``"min"`` → keep as-is, ``"max"`` → negate).
        """
        self.patience_limit = patience_limit
        self.min_evaluations_before_check = min_evaluations_before_check
        self.verbose = verbose
        self.objective_directions = objective_directions  # None or ["min","max",...]

        # State
        self._patience_counter: int = 0
        self._best_aggregate: float = -float("inf")
        self._total_seen: int = 0           # non-failure results seen so far
        self._valid_objectives: list = []   # accumulates valid (non-failure, non-penalty) MOO vectors
        self._is_multi_objective: bool = False
        self.search_stopped: bool = False

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _is_failure(self, obj) -> bool:
        """Return True for results that carry no usable information ("F", None, …)."""
        if obj is None:
            return True
        if isinstance(obj, str):
            return True
        # For MOO vectors: check if any element is a failure string
        try:
            arr = np.asarray(obj, dtype=object)
            if arr.ndim > 0:
                return any(isinstance(v, str) for v in arr.flat)
        except (TypeError, ValueError):
            return True
        return False

    def _is_penalty(self, obj) -> bool:
        """Return True for penalty-magnitude results (killed simulations)."""
        try:
            arr = np.asarray(obj, dtype=float)
            return bool(np.any(np.abs(arr) >= PENALTY))
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Aggregate computation
    # ------------------------------------------------------------------

    def _compute_aggregate(self) -> float:
        """Return the current best-aggregate over all collected valid objectives."""
        if not self._valid_objectives:
            return -float("inf")

        if not self._is_multi_objective:
            # SOO: running maximum (DeepHyper maximises)
            return float(max(self._valid_objectives))

        # MOO: hypervolume indicator
        objectives = np.asarray(self._valid_objectives, dtype=float)

        if self.objective_directions is not None:
            # Raw objectives — convert each dimension to minimisation space.
            # "min" → keep as-is (lower is already better).
            # "max" → negate (flip so lower is better).
            signs = np.array(
                [1.0 if d == "min" else -1.0 for d in self.objective_directions]
            )
            minimise = objectives * signs
        else:
            # DeepHyper's internal convention: all objectives are in maximisation
            # space (minimised objectives were negated before being returned).
            # A single negation converts back to minimisation space for all cases,
            # including mixed min/max when the evaluation function handled the
            # direction itself (e.g. returned -exec_time for a minimised objective
            # and +bandwidth for a maximised one).
            minimise = -objectives

        # Reference point must be strictly dominated by at least some Pareto points.
        # Using np.max(minimise) as the reference makes every Pareto point
        # contribute zero in at least one dimension, producing HVI = 0 regardless
        # of how good the front is.  A small per-dimension margin fixes this.
        #
        # The margin must be large enough to avoid the collapse when going from
        # 1 point (col_range = 0 → margin falls back to 1.0) to 2+ points
        # (col_range > 0 → 1% * range << 1.0), which would make HVI drop sharply
        # even though the Pareto front improved.  Enforcing a minimum of 1.0
        # keeps the reference box stable across the full run (objectives are in
        # log-scale, so 1 unit = one order of magnitude, a natural floor).
        col_range = np.ptp(minimise, axis=0)                                    # range per dim
        margin = np.maximum(col_range * 0.01, 1.0)                             # at least 1.0
        ref = np.max(minimise, axis=0) + margin
        return float(hypervolume(minimise, ref))

    # ------------------------------------------------------------------
    # Core evaluation logic (shared by on_done / on_done_other)
    # ------------------------------------------------------------------

    def _process_job(self, job) -> None:
        if self.search_stopped:
            return

        obj = job.objective

        # 1. Completely ignore failures
        if self._is_failure(obj):
            return

        # 2. Detect MOO from shape (must happen before penalty check for vectors)
        try:
            if np.ndim(obj) > 0:
                self._is_multi_objective = True
        except Exception:
            pass

        # 3. Count this as a real (non-failure) evaluation seen
        self._total_seen += 1
        in_warmup = self._total_seen <= self.min_evaluations_before_check

        # 4. Penalty result: increment counter only (do not update best)
        if self._is_penalty(obj):
            if not in_warmup:
                self._patience_counter += 1
                if self.verbose:
                    print(
                        f"[EarlyStopping] Penalty result detected. "
                        f"Patience: {self._patience_counter}/{self.patience_limit}"
                    )
                self._check_and_stop()
            return

        # 5. Valid result: collect and compute aggregate
        if self._is_multi_objective:
            try:
                self._valid_objectives.append(list(np.asarray(obj, dtype=float)))
            except (TypeError, ValueError):
                return
        else:
            try:
                self._valid_objectives.append(float(obj))
            except (TypeError, ValueError):
                return

        if in_warmup:
            # During warmup we still track the best so the first post-warmup
            # evaluation has a meaningful baseline to compare against.
            new_aggregate = self._compute_aggregate()
            if new_aggregate > self._best_aggregate:
                self._best_aggregate = new_aggregate
            return

        new_aggregate = self._compute_aggregate()

        if new_aggregate > self._best_aggregate:
            if self.verbose:
                metric = "HVI" if self._is_multi_objective else "Objective"
                print(
                    f"[EarlyStopping] {metric} improved: "
                    f"{self._best_aggregate:.6g} → {new_aggregate:.6g}. "
                    f"Patience reset to 0."
                )
            self._best_aggregate = new_aggregate
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            if self.verbose:
                metric = "HVI" if self._is_multi_objective else "Objective"
                print(
                    f"[EarlyStopping] No improvement (best {metric}={self._best_aggregate:.6g}). "
                    f"Patience: {self._patience_counter}/{self.patience_limit}"
                )
            self._check_and_stop()

    def _check_and_stop(self) -> None:
        if self.patience_limit <= 0:
            # patience_limit <= 0 means disabled; -1 is the canonical sentinel.
            return
        if self._patience_counter >= self.patience_limit:
            if self.verbose:
                metric = "HVI" if self._is_multi_objective else "Objective"
                print(
                    f"\n[EarlyStopping] Stopping search — no improvement for "
                    f"{self.patience_limit} consecutive evaluations "
                    f"(best {metric}={self._best_aggregate:.6g}, "
                    f"valid evals={self._total_seen})."
                )
            self.search_stopped = True

    # ------------------------------------------------------------------
    # DeepHyper Callback interface
    # ------------------------------------------------------------------

    def on_done(self, job) -> None:
        """Called when a local job has been gathered by the Evaluator."""
        self._process_job(job)

    def on_done_other(self, job) -> None:
        """Called for remote jobs gathered alongside local ones."""
        self._process_job(job)
