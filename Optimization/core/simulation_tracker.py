"""
SimulationTracker: Monitor simulation progress and kill slow runs early.

This module provides real-time tracking of simulation execution by monitoring
issue ticks in trace/log files. It implements an adaptive threshold mechanism
to kill simulations that exceed 1.5x the current best execution time.

Features:
- Real-time log file monitoring
- Adaptive threshold management
- Early termination of slow simulations
- Shared state across parallel evaluations
"""

import os
import math
from typing import Optional, Dict, Any, List
import tempfile
import json
import uuid


class SimulationTracker:
    """
    Tracks simulation progress and terminates slow runs early.
    
    The tracker monitors issue ticks in simulation log files and compares
    them against a threshold. If a simulation's ticks exceed 1.5x the
    threshold, it's terminated early to save time.
    
    Uses a small JSON file in /tmp for shared threshold and counters
    across worker processes.
    
    Usage:
        tracker = SimulationTracker(initial_threshold=1e15)
        
        # During simulation
        if tracker.should_kill_simulation(log_file):
            # Kill simulation
            
        # After successful completion
        tracker.update_threshold(exec_time)
    """
    
    _DEFAULT_STATE = {
        'threshold': 1e15,
        'best_raw_exec_time': None,
        'best_training_time_s': None,  # training time in seconds of the best completed run
        'latest_tick': None,
        'total_checked': 0,
        'total_killed': 0,
        'pareto_front': [],  # list of complete score vectors from finished simulations
    }

    def __init__(
        self,
        initial_threshold: float = 1e15,
        kill_multiplier: float = 1.5,
        verbose: bool = False,
        tracker_id: Optional[str] = None,
        minimize: bool = True,
        objective: Optional[Any] = None,
    ):
        """
        Initialize simulation tracker.

        Args:
            initial_threshold: Magnitude of the worst-case score threshold.
                The tracker converts this to the correct sign per objective
                direction automatically (positive for minimize, negative for
                maximize).  Pass a positive value; direction is handled here.
            kill_multiplier: Kill simulations exceeding threshold × multiplier
                (minimize) or below threshold / multiplier (maximize).
            verbose: Print tracking information.
            tracker_id: Optional shared id for the state file across worker
                processes.  A unique id is generated per run when omitted.
            minimize: Fallback primary direction used only when no objective
                is provided.  When ``objective`` is given, direction is derived
                from ``objective.score_directions`` and this parameter is ignored.
            objective: Optional objective function.  When provided, per-objective
                directions are taken from ``objective.score_directions`` so that
                mixed-direction MOO (e.g. minimize time + maximize throughput)
                is handled correctly.
        """
        self.kill_multiplier = kill_multiplier
        self.verbose = verbose
        self.objective = objective

        # ── Derive per-objective directions ──────────────────────────────────
        # Always read from the objective when available so that mixed-direction
        # MOO objectives are handled without the caller needing to pass anything.
        if objective is not None and hasattr(objective, 'score_directions'):
            directions = objective.score_directions
        else:
            directions = [bool(minimize)]

        # Primary direction (used as scalar fallback throughout the class).
        self.minimize = directions[0]

        # ── Initial threshold per objective direction ────────────────────────
        # For minimize: worst case is +inf  → start at  +initial_threshold.
        # For maximize: worst case is -inf  → start at  -initial_threshold.
        # The kill formula is always score × multiplier (minimize) or
        # score / multiplier (maximize), applied uniformly to the objective
        # output space whether it is raw, log10, or any other scale.
        magnitude = abs(float(initial_threshold))
        self._initial_threshold = magnitude
        if len(directions) > 1:
            threshold_init = [magnitude if d else -magnitude for d in directions]
        else:
            threshold_init = magnitude if directions[0] else -magnitude

        # ── File-based shared state (across worker processes) ─────────────
        self._tracker_id = tracker_id or f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._threshold_file = os.path.join(
            tempfile.gettempdir(),
            f'astrasim_tracker_threshold_{self._tracker_id}.json'
        )
        self._initialize_state(threshold_init)

        if self.verbose:
            if len(directions) == 1:
                dir_str = "minimize" if directions[0] else "maximize"
            else:
                dir_str = "[" + ", ".join("min" if d else "max" for d in directions) + "]"
            print(
                f"[Tracker] Initialized with threshold={self._format_threshold(threshold_init)}, "
                f"kill_multiplier={kill_multiplier}, directions={dir_str}"
            )

    def _default_state(self, threshold: Optional[float] = None) -> Dict:
        """Return a default tracker state dict."""
        state = self._DEFAULT_STATE.copy()
        if threshold is not None:
            state['threshold'] = threshold
        return state

    @staticmethod
    def _format_threshold(value: Any) -> str:
        """Format scalar/list threshold for logs."""
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(f"{float(v):.2e}" for v in value) + "]"
        return f"{float(value):.2e}"

    def _write_state(self, data: Dict):
        """Write full tracker state to shared file (atomic)."""
        dir_name = os.path.dirname(self._threshold_file)
        with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp') as tmp_f:
            tmp_path = tmp_f.name
            json.dump(data, tmp_f)
        os.replace(tmp_path, self._threshold_file)

    def _read_state(self) -> Dict:
        """Read current tracker state from shared file."""
        if not os.path.exists(self._threshold_file):
            return self._default_state(self._initial_threshold)
        return self._safe_read_json(self._threshold_file, self._default_state(self._initial_threshold))

    def _initialize_state(self, initial_threshold: float):
        """Create a fresh shared state file for this tracker run."""
        data = self._default_state(initial_threshold)
        try:
            self._write_state(data)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Warning: Could not initialize tracker state: {e}")
    
    def _safe_read_json(self, path: str, default: dict) -> dict:
        """Read a JSON file, returning 'default' if the file is empty or corrupt."""
        try:
            with open(path, 'r') as f:
                content = f.read()
            if not content.strip():
                return default
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return default
        except Exception:
            return default

    @property
    def threshold(self):
        """Get current threshold value from shared file."""
        try:
            return self._read_state().get('threshold', 1e15)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Warning: Could not read threshold: {e}")
            return 1e15
    
    @threshold.setter
    def threshold(self, value):
        """Set threshold value in shared file."""
        try:
            data = self._read_state()
            data['threshold'] = value
            self._write_state(data)
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Warning: Could not write threshold: {e}")
    
    def _get_counter(self, counter_name: str) -> int:
        """Get a counter value from the shared file."""
        try:
            data = self._read_state()
            return int(data.get(counter_name, 0))
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Error reading {counter_name}: {e}")
            return 0
    
    def update_threshold(self, score: Any, raw_exec_time: Optional[float] = None,
                         training_time_s: Optional[float] = None):
        """
        Update per-component best observed scores.

        Call after each successful (non-OOM, non-killed) simulation with the
        full objective score returned by ``objective.compute``.  The stored
        per-component bests are used to compute kill thresholds for future
        in-flight simulations.

        Args:
            score: Scalar or tuple from ``objective.compute``.
            raw_exec_time: Raw execution time in native simulation units (same
                units as the trace ticks read by ``_get_latest_issue_tick``).
                Used for tick-space kill comparison so that simulations exceeding
                ``best_exec × kill_multiplier`` ticks are terminated promptly
                without any log-space confusion.  Pass the value returned by
                ``evaluate_config_worker`` directly (typically nanoseconds from
                the AstraSim trace CSV).
            training_time_s: Training time in seconds as passed to
                ``objective.compute(exec_time=...)``.  Stored alongside
                ``raw_exec_time`` so mid-run tick estimates can be scaled back
                to the correct unit (seconds) before calling objective.compute,
                avoiding the ~9-order-of-magnitude unit mismatch that would
                otherwise make the Pareto dominance guard fire on Pareto-
                efficient configurations.
        """
        # Track best raw exec time and best training time separately from the
        # objective-space score.  Keeping both allows mid-run tick estimates to
        # be scaled back to training-time-seconds (the unit objective.compute
        # expects) via:  estimate_s = best_training_time_s * (tick / best_raw_ns)
        if raw_exec_time is not None:
            try:
                raw = float(raw_exec_time)
                if math.isfinite(raw) and raw > 0:
                    state = self._read_state()
                    current_best = state.get('best_raw_exec_time')
                    if current_best is None or raw < float(current_best):
                        state['best_raw_exec_time'] = raw
                        # Also store the paired training time if available so
                        # _tick_to_training_time_estimate can convert correctly.
                        if training_time_s is not None:
                            try:
                                ts = float(training_time_s)
                                if math.isfinite(ts) and ts > 0:
                                    state['best_training_time_s'] = ts
                            except (TypeError, ValueError):
                                pass
                        self._write_state(state)
            except (TypeError, ValueError):
                pass

        score_vector = self._reduce_score(score)
        if score_vector is None:
            return

        current_threshold = self.threshold
        threshold_vector = self._to_vector(current_threshold, len(score_vector))
        directions = self._objective_directions(len(score_vector))

        old_threshold = list(threshold_vector)
        updated = False
        for idx, (value, old_value) in enumerate(zip(score_vector, threshold_vector)):
            if directions[idx]:          # minimize → lower is better
                if value < old_value:
                    threshold_vector[idx] = value
                    updated = True
            else:                        # maximize → higher is better
                if value > old_value:
                    threshold_vector[idx] = value
                    updated = True

        if updated:
            self.threshold = threshold_vector[0] if len(threshold_vector) == 1 else threshold_vector
            if self.verbose:
                print(
                    f"[Tracker] Threshold updated: {self._format_threshold(old_threshold)} "
                    f"→ {self._format_threshold(threshold_vector)}"
                )
                print(
                    f"          Kill threshold: "
                    f"{self._format_threshold(self.get_kill_score_threshold())}"
                )

        # Maintain Pareto front for dominance-based kill checks (always, regardless
        # of whether the per-dim threshold changed).
        pf_state = self._read_state()
        current_front = pf_state.get('pareto_front', [])
        new_front = self._add_to_pareto_front(current_front, score_vector, directions)
        if new_front is not current_front:  # front was modified
            pf_state['pareto_front'] = new_front
            self._write_state(pf_state)
            if self.verbose:
                print(
                    f"[Tracker] Pareto front updated: "
                    f"{len(current_front)} → {len(new_front)} points"
                )
    
    def get_kill_score_threshold(self) -> Any:
        """
        Get kill threshold(s) in objective-score space.

        The kill formula is applied uniformly regardless of whether the
        objective output is raw, log10-scaled, or any other transform::

            minimize dim i:  kill_i = best_i × kill_multiplier
            maximize dim i:  kill_i = best_i / kill_multiplier

        This is consistent and comparable across runs — the multiplier adds
        a proportional safety margin in whatever space the objective uses.

        Returns a list for MOO objectives, a scalar for SOO.
        """
        threshold = self.threshold
        if isinstance(threshold, (list, tuple)):
            directions = self._objective_directions(len(threshold))
            return [
                float(v) * self.kill_multiplier if is_min else float(v) / self.kill_multiplier
                for v, is_min in zip(threshold, directions)
            ]
        is_min = self._objective_directions(1)[0]
        if is_min:
            return float(threshold) * self.kill_multiplier
        return float(threshold) / self.kill_multiplier

    def get_kill_threshold(self) -> float:
        """
        Backward-compatible alias for score-space kill threshold.
        """
        return self.get_kill_score_threshold()

    def get_latest_tick_threshold(self) -> float:
        """
        Return last observed trace tick used by the tracker.

        Falls back to current score threshold for compatibility.
        """
        state = self._read_state()
        latest_tick = state.get('latest_tick')
        if latest_tick is None:
            threshold = self.threshold
            if isinstance(threshold, (list, tuple)):
                return float(threshold[0]) if threshold else 0.0
            return float(threshold)
        try:
            return float(latest_tick)
        except (TypeError, ValueError):
            threshold = self.threshold
            if isinstance(threshold, (list, tuple)):
                return float(threshold[0]) if threshold else 0.0
            return float(threshold)

    def _to_vector(self, value: Any, target_len: int) -> List[float]:
        """Normalize scalar/list value to a float vector of target length."""
        if isinstance(value, (tuple, list)):
            vec = [float(v) for v in value]
            if len(vec) == target_len:
                return vec
            if len(vec) == 1:
                return vec * target_len
            if len(vec) > target_len:
                return vec[:target_len]
            # If shorter than target_len, pad by repeating the last value.
            return vec + [vec[-1]] * (target_len - len(vec))
        return [float(value)] * target_len

    def _objective_directions(self, n_objectives: int) -> List[bool]:
        """
        Return per-objective optimization direction.

        True means minimize, False means maximize.

        Delegates to ``objective.score_directions`` when an objective is
        available so that all direction logic stays in one place.
        """
        if self.objective is None:
            return [self.minimize] * n_objectives

        # Use the canonical score_directions property when available.
        if hasattr(self.objective, 'score_directions'):
            directions = self.objective.score_directions
        else:
            raw = getattr(self.objective, "objective_directions", None)
            if raw is None:
                return [self.minimize] * n_objectives
            directions = []
            for d in raw:
                if isinstance(d, str):
                    directions.append(d.strip().lower() != "max")
                else:
                    directions.append(bool(d))

        if len(directions) != n_objectives:
            if self.verbose:
                print(
                    f"[Tracker] Warning: objective_directions length {len(directions)} "
                    f"!= {n_objectives}, using global minimize={self.minimize}"
                )
            return [self.minimize] * n_objectives
        return directions

    # Magnitude threshold matching objective.py PENALTY (1e20).
    # Any score component with abs(value) >= this is treated as invalid.
    _PENALTY_MAGNITUDE = 1e20

    def _reduce_score(self, score: Any) -> Optional[List[float]]:
        """Convert scalar/tuple score into finite, non-penalty float vector (strict, for completed runs)."""
        if score is None:
            return None
        raw_values = list(score) if isinstance(score, (tuple, list)) else [score]
        if not raw_values:
            return None

        values: List[float] = []
        for item in raw_values:
            try:
                value = float(item)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value) or abs(value) >= self._PENALTY_MAGNITUDE:
                return None
            values.append(value)
        return values

    @staticmethod
    def _tracker_partial_estimate_vector(raw: Any) -> Optional[List[float]]:
        """
        Parse ``objective.compute`` output for mid-run tracking.

        Components that cannot be evaluated yet (non-finite: ``inf``, ``nan``,
        or unparseable) become ``nan`` placeholders.  At least one finite
        component is required; otherwise returns ``None``.
        """
        if raw is None:
            return None
        _pen = SimulationTracker._PENALTY_MAGNITUDE

        if isinstance(raw, (tuple, list)):
            out: List[float] = []
            for item in raw:
                try:
                    v = float(item)
                except (TypeError, ValueError):
                    out.append(float("nan"))
                    continue
                # Non-finite or penalty-magnitude values are unavailable mid-run.
                out.append(v if (math.isfinite(v) and abs(v) < _pen) else float("nan"))
            if not out or not any(math.isfinite(x) for x in out):
                return None
            return out
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v) or abs(v) >= _pen:
            return None
        return [v]

    @staticmethod
    def _dominates(
        a: List[float], b: List[float], directions: List[bool]
    ) -> bool:
        """
        Return True if vector *a* Pareto-dominates vector *b*.

        *a* dominates *b* when *a* is at least as good as *b* in every
        dimension and strictly better in at least one.  Direction convention:
        ``True`` = minimize (smaller is better), ``False`` = maximize.
        """
        at_least_as_good = True
        strictly_better = False
        for ai, bi, minimize in zip(a, b, directions):
            if minimize:
                if ai > bi:
                    at_least_as_good = False
                    break
                if ai < bi:
                    strictly_better = True
            else:
                if ai < bi:
                    at_least_as_good = False
                    break
                if ai > bi:
                    strictly_better = True
        return at_least_as_good and strictly_better

    @staticmethod
    def _add_to_pareto_front(
        front: List[List[float]],
        new_vec: List[float],
        directions: List[bool],
    ) -> List[List[float]]:
        """
        Return an updated Pareto front after adding *new_vec*.

        If *new_vec* is dominated by any existing point the front is returned
        unchanged (same object).  Otherwise all points dominated by *new_vec*
        are removed and *new_vec* is appended.
        """
        for pt in front:
            if SimulationTracker._dominates(pt, new_vec, directions):
                return front  # new_vec is dominated — no change
        new_front = [
            pt for pt in front
            if not SimulationTracker._dominates(new_vec, pt, directions)
        ]
        new_front.append(new_vec)
        return new_front

    def _tick_to_training_time_estimate(
        self, latest_tick: float, num_steps: Optional[float] = None,
    ) -> float:
        """
        Convert a raw simulation tick (nanoseconds, per-step) to an estimated
        training time in **seconds** — the unit that ``objective.compute``
        receives for completed runs.

        When *num_steps* for the **current** config is available (passed via
        metadata from the simulation runner), the conversion is exact::

            estimate_s = num_steps × (latest_tick / 1e9)

        Otherwise falls back to proportional scaling from the best completed
        run::

            estimate_s = best_training_time_s × (latest_tick / best_raw_exec_time_ns)

        The proportional fallback assumes num_steps is similar between the
        reference (best) run and the in-flight run.  Different ``dp`` values
        cause a proportional error, so the direct formula is preferred.

        Falls back to ``latest_tick`` unchanged when no reference run exists yet.
        """
        # Direct conversion when the current config's num_steps is known.
        if num_steps is not None:
            try:
                ns = float(num_steps)
                if ns > 0 and math.isfinite(ns):
                    return ns * (float(latest_tick) / 1e9)
            except (TypeError, ValueError):
                pass

        # Proportional fallback from best completed run.
        try:
            state = self._read_state()
            best_raw = state.get('best_raw_exec_time')
            best_training = state.get('best_training_time_s')
            if best_raw is None or best_training is None:
                return latest_tick  # no reference yet — fallback
            best_raw_f = float(best_raw)
            best_training_f = float(best_training)
            if best_raw_f <= 0 or not math.isfinite(best_training_f) or best_training_f <= 0:
                return latest_tick
            return best_training_f * (float(latest_tick) / best_raw_f)
        except Exception:
            return latest_tick

    def _estimate_running_score(
        self,
        latest_tick: float,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[float]]:
        """
        Estimate objective score for an in-flight simulation.

        Uses ``objective.compute`` with an estimated training-time (seconds) so
        that the estimate is in the same unit/scale as the completed-run scores
        stored in the Pareto front.  The tick is converted to seconds via
        ``_tick_to_training_time_estimate``, preferring the current config's
        ``num_steps`` (from metadata) for an exact conversion::

            estimate_s = num_steps × (latest_tick / 1e9)

        When ``num_steps`` is unavailable, falls back to proportional scaling
        from the best known run::

            estimate_s = best_training_time_s × (latest_tick / best_raw_ns)

        Because ``latest_tick ≤ final_exec_time_ns``, the estimate is an
        **optimistic lower bound** for the final training time (assuming similar
        num_steps between runs).  If dominance still holds against this
        optimistic estimate the actual final score is dominated too — safe to
        kill.

        For multi-objective functions, metrics unavailable mid-run (e.g. peak
        memory, power) are marked ``nan`` and excluded from the dominance check.
        """
        if self.objective is not None:
            try:
                # Convert raw NS tick → training-time seconds so that the
                # estimate is in the same unit as scores in the Pareto front.
                num_steps = metadata.get('total_training_steps') if metadata else None
                exec_time_estimate = self._tick_to_training_time_estimate(
                    latest_tick, num_steps=num_steps,
                )
                estimated = self.objective.compute(
                    exec_time=exec_time_estimate,
                    is_oom=False,
                    metadata=metadata or {},
                    config=config,
                )
                return self._tracker_partial_estimate_vector(estimated)
            except Exception as e:
                if self.verbose:
                    print(f"[Tracker] Warning: objective-based score estimate failed: {e}")
                return None

        # Fallback: use raw tick as score proxy.
        return self._reduce_score(latest_tick)
    
    def should_kill_simulation(
        self,
        trace_file: str,
        workload_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if a running simulation should be killed based on its tick progress.

        Kill logic:

        1. **Training-time check** — convert ``latest_tick`` to estimated training
           time (seconds) using the current config's ``num_steps`` and compare
           against ``best_training_time_s × kill_multiplier``.  This operates in
           the unit we optimise and correctly accounts for different ``dp`` values
           that change ``num_steps``.  If training time has *not* been exceeded,
           never kill.

        2. **MOO config-derived dims** — for objectives where non-time dims can
           be estimated mid-run (e.g. network BW or memory from the config), the
           run is only killed when *every* available dim is also past its kill
           threshold.  This prevents killing a slow config that might be
           Pareto-optimal because of a better secondary objective value.

           Kill threshold for score dim *i* (additive formulation, correct for
           log10-transformed objectives)::

               minimize:  kill_score_i = best_score_i + log10(kill_multiplier)
               maximize:  kill_score_i = best_score_i − log10(kill_multiplier)

        3. For power/memory objectives that are unavailable mid-run, only the
           time check (step 1) is applied.
        """
        _ = workload_file  # Reserved for future context-specific logic.

        if not os.path.exists(trace_file):
            return False

        try:
            latest_tick = self._get_latest_issue_tick(trace_file)
            if latest_tick is None:
                return False

            state = self._read_state()
            state['total_checked'] = state.get('total_checked', 0) + 1
            state['latest_tick'] = latest_tick

            # ── Step 1: training-time kill check ─────────────────────────────
            # Compare in *training-time seconds* (the unit we optimise) rather
            # than in raw per-step nanosecond ticks.  Different configs may have
            # different num_steps (because dp changes global batch size), so raw
            # tick comparison is incorrect — a high-dp config has fewer steps
            # and thus lower training time even with a higher per-step tick.
            #
            # estimated_training = num_steps_current × (latest_tick / 1e9)
            # kill_training     = best_training_time_s × kill_multiplier
            best_training_s = state.get('best_training_time_s')
            best_raw = state.get('best_raw_exec_time')
            if best_training_s is None or best_raw is None:
                # No completed simulation yet — cannot establish a kill baseline.
                self._write_state(state)
                return False
            try:
                best_training_s = float(best_training_s)
                best_raw = float(best_raw)
            except (TypeError, ValueError):
                self._write_state(state)
                return False

            num_steps = metadata.get('total_training_steps') if metadata else None
            estimated_training = self._tick_to_training_time_estimate(
                latest_tick, num_steps=num_steps,
            )
            kill_training = best_training_s * self.kill_multiplier

            if self.verbose and state['total_checked'] % 100 == 0:
                print(
                    f"[Tracker Debug] tick={latest_tick:.2e}, "
                    f"est_training={estimated_training:.2e}, "
                    f"best_training={best_training_s:.2e}, kill_at={kill_training:.2e}"
                )

            if not (math.isfinite(kill_training) and estimated_training > kill_training):
                # Training time not exceeded — never kill regardless of secondary dims.
                self._write_state(state)
                return False

            # ── Step 2: Pareto dominance check ────────────────────────────────
            # Use the stored Pareto front (built by update_threshold) to decide
            # whether this in-flight simulation is already Pareto-dominated and
            # therefore safe to discard.  This is strictly weaker than the old
            # per-dim ideal-point filter — a config is only killed when there
            # exists a completed Pareto point that is at least as good on every
            # objective and strictly better on at least one, so configs that are
            # Pareto-optimal (or potentially so) are never killed prematurely.
            score_threshold = state.get('threshold')
            is_moo = (
                isinstance(score_threshold, list)
                and len(score_threshold) > 1
            )

            if is_moo:
                estimated = self._estimate_running_score(
                    latest_tick=latest_tick, config=config, metadata=metadata
                )
                # objective.compute failed entirely — cannot assess any secondary dim.
                if estimated is None:
                    if self.verbose:
                        print(
                            f"[Tracker] Keeping alive: secondary objectives unavailable mid-run "
                            f"(tick={latest_tick:.2e})"
                        )
                    self._write_state(state)
                    return False

                # If ANY dimension is unavailable (nan) mid-run, we cannot fully
                # assess Pareto dominance.  Objectives such as power or peak memory
                # are only known after the simulation finishes.  Keep alive to avoid
                # discarding a potentially Pareto-optimal config on those dims.
                unavailable = [i for i, v in enumerate(estimated) if not math.isfinite(v)]
                if unavailable:
                    if self.verbose:
                        print(
                            f"[Tracker] Keeping alive: dims {unavailable} "
                            f"not available mid-run (tick={latest_tick:.2e})"
                        )
                    self._write_state(state)
                    return False

                # Pareto dominance check: kill only if some completed Pareto point
                # dominates the estimated score vector on every dimension.
                # NOTE: estimated[0] uses latest_tick (a lower bound for the final
                # execution time), so if dominance holds even with this optimistic
                # time estimate the actual final vector is dominated too — safe to kill.
                pareto_front = state.get('pareto_front', [])
                if not pareto_front:
                    # No reference points yet — cannot confirm dominance.
                    self._write_state(state)
                    return False

                directions = self._objective_directions(len(estimated))
                is_dominated = any(
                    self._dominates(pt, estimated, directions)
                    for pt in pareto_front
                    if len(pt) == len(estimated)
                )
                if not is_dominated:
                    if self.verbose:
                        print(
                            f"[Tracker] Keeping alive: estimated score not dominated "
                            f"by any Pareto point (tick={latest_tick:.2e})"
                        )
                    self._write_state(state)
                    return False

            # ── Kill ──────────────────────────────────────────────────────────
            # Gather score-space info for the kill log.  For SOO, estimated
            # was not computed yet (only happens inside the is_moo block above).
            if not is_moo:
                estimated = self._estimate_running_score(
                    latest_tick=latest_tick, config=config, metadata=metadata
                )

            score_threshold = state.get('threshold')

            # Format score threshold(s) and estimated running score(s).
            if isinstance(score_threshold, (list, tuple)):
                thresh_str = "[" + ", ".join(f"{float(v):.4g}" for v in score_threshold) + "]"
            elif score_threshold is not None:
                thresh_str = f"{float(score_threshold):.4g}"
            else:
                thresh_str = "none"

            if estimated is not None:
                est_str = "[" + ", ".join(
                    f"{v:.4g}" if math.isfinite(v) else "n/a"
                    for v in estimated
                ) + "]"
            else:
                est_str = "n/a"

            print(
                f"[Tracker] Killing simulation: tick={latest_tick:.2e}, "
                f"est_training={estimated_training:.2e} "
                f"> kill_training={kill_training:.2e}  "
                f"(best_training={best_training_s:.2e} × {self.kill_multiplier}  |  "
                f"score_threshold={thresh_str}  |  "
                f"estimated_score={est_str})"
            )
            state['total_killed'] = state.get('total_killed', 0) + 1
            self._write_state(state)
            if self.verbose:
                print(f"    Trace file: {os.path.basename(trace_file)}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Error checking simulation: {e}")
            return False
    
    def _get_latest_issue_tick(self, trace_file: str) -> Optional[float]:
        """
        Extract the latest issue tick from trace CSV file.
        
        Reads the trace file and finds the maximum tick value from issue records.
        The last issue tick represents the current progress of the simulation.
        
        Args:
            trace_file: Path to trace CSV file (e.g., workload_trace.csv)
        
        Returns:
            Latest tick value, or None if not found
        """
        try:
            # Read last chunk for efficiency (avoid reading entire file)
            max_tick = None
            
            # Use tail-like approach: read last chunk
            with open(trace_file, 'r') as f:
                # Seek to end and read backwards
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                
                # Read last chunk (up to 500KB for CSV which can have longer lines)
                chunk_size = min(500000, file_size)
                f.seek(max(0, file_size - chunk_size))
                
                lines = f.readlines()
                
                # Parse tick values from CSV
                # CSV format: [timestamp] [I] <trace>: ,action,sys_id,node_id,node_name,col_type,node_type,num_ops,tensor_size,perf,operational_intensity,issue_tick
                # issue_tick is the LAST column (12 total columns with leading comma)
                for line in lines:
                    # Skip header or empty lines
                    if not line.strip() or 'action,sys_id' in line or line.startswith('#'):
                        continue
                    
                    try:
                        # Remove timestamp prefix if present: [timestamp] [I] <trace>:
                        if '<trace>:' in line:
                            line = line.split('<trace>:', 1)[1]
                        
                        # Split CSV and extract issue_tick (last column)
                        parts = line.strip().split(',')
                        if len(parts) >= 12:
                            tick = float(parts[-1])  # Last column is issue_tick
                            
                            if max_tick is None or tick > max_tick:
                                max_tick = tick
                    except (ValueError, IndexError):
                        continue
            
            return max_tick
            
        except Exception as e:
            if self.verbose:
                print(f"[Tracker] Error reading trace file {trace_file}: {e}")
            return None
    
    def get_status(self) -> Dict:
        """
        Get tracker status information.

        Returns:
            Dictionary with threshold, kill_threshold, per-objective directions,
            and running statistics.
        """
        state = self._read_state()
        threshold = state.get('threshold', 1e15)
        n = len(threshold) if isinstance(threshold, (list, tuple)) else 1
        directions = self._objective_directions(n)
        return {
            'threshold': threshold,
            'kill_threshold': self.get_kill_score_threshold(),
            'kill_multiplier': self.kill_multiplier,
            'score_directions': directions,
            'latest_tick': state.get('latest_tick'),
            'total_checked': int(state.get('total_checked', 0)),
            'total_killed': int(state.get('total_killed', 0))
        }
    
    def reset(self, initial_threshold: Optional[float] = None):
        """
        Reset tracker to initial state.
        
        Args:
            initial_threshold: New initial threshold (uses current if None)
        """
        threshold = self.threshold if initial_threshold is None else initial_threshold
        self._initialize_state(threshold)

        if self.verbose:
            print(f"[Tracker] Reset with threshold={self._format_threshold(threshold)}")
    
    def __repr__(self) -> str:
        """String representation."""
        status = self.get_status()
        return (
            f"SimulationTracker(threshold={self._format_threshold(status['threshold'])}, "
            f"kill_at={self._format_threshold(status['kill_threshold'])})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        status = self.get_status()
        directions = status['score_directions']
        if len(directions) == 1:
            dir_str = "minimize" if directions[0] else "maximize"
        else:
            dir_str = "[" + ", ".join("min" if d else "max" for d in directions) + "]"
        return (f"SimulationTracker\n"
                f"  Threshold: {self._format_threshold(status['threshold'])}\n"
                f"  Kill at: {self._format_threshold(status['kill_threshold'])}\n"
                f"  Directions: {dir_str}\n"
                f"  Multiplier: {status['kill_multiplier']}x\n"
                f"  Total checked: {status['total_checked']}\n"
                f"  Total killed: {status['total_killed']}")
