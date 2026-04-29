"""Shared cleanup helpers for optimizer-generated simulation artifacts."""

from typing import Any, Dict, List, Optional
import glob
import math
import os
import tarfile
import zipfile


class ArtifactCleanupManager:
    """Coordinate immediate, periodic, and final artifact cleanup."""

    def __init__(
        self,
        keep_top_k: int,
        cleanup_batch_size: int,
        verbose: bool = False,
        log_fn=None,
        print_deleted_files: bool = False,
    ):
        if keep_top_k >= 0 and cleanup_batch_size > 0 and cleanup_batch_size <= keep_top_k:
            raise ValueError(
                f"cleanup_batch_size ({cleanup_batch_size}) must be greater than keep_top_k ({keep_top_k})"
            )

        self.keep_top_k = keep_top_k
        self.cleanup_batch_size = cleanup_batch_size
        self.verbose = verbose
        self.log_fn = log_fn
        self.print_deleted_files = bool(print_deleted_files)
        self.next_cleanup_at = cleanup_batch_size if self.periodic_cleanup_enabled() else None
        self.cleaned_file_indices = set()
        self.stats: Dict[str, int] = {
            "simulations_cleaned": 0,
            "files_deleted": 0,
        }

    def periodic_cleanup_enabled(self) -> bool:
        """Return True when periodic cleanup is configured."""
        return self.keep_top_k >= 0 and self.cleanup_batch_size > 0 and self.cleanup_batch_size > self.keep_top_k

    def set_debug(self, print_deleted_files: bool = True):
        """Enable or disable detailed logging of deleted files."""
        self.print_deleted_files = bool(print_deleted_files)

    def get_status(self) -> Dict[str, int]:
        """Return cleanup counters for verification/debugging."""
        return {
            "simulations_cleaned": int(self.stats.get("simulations_cleaned", 0)),
            "files_deleted": int(self.stats.get("files_deleted", 0)),
        }

    def mark_index_cleaned(self, idx: int):
        """Mark a tracked result index as already cleaned."""
        self.cleaned_file_indices.add(idx)

    @staticmethod
    def _emit(verbose: bool, log_fn, message: str, level: str = "info"):
        """Emit cleanup logs through provided logger or stdout."""
        if not verbose:
            return
        if log_fn is not None:
            log_fn(message, level)
            return
        prefix = {
            "info": "",
            "warning": "⚠️  ",
            "error": "❌ ",
        }.get(level, "")
        print(f"{prefix}{message}")

    @staticmethod
    def _can_run_periodic_cleanup(
        record_count: int,
        keep_top_k: int,
        cleanup_batch_size: int,
        next_cleanup_at: Optional[int],
        force: bool,
    ) -> bool:
        """Return whether periodic cleanup should run for current state."""
        if keep_top_k < 0 or cleanup_batch_size == 0:
            return False
        if record_count <= keep_top_k:
            return False
        if force:
            return True
        if cleanup_batch_size < 0:
            return False
        if next_cleanup_at is None:
            return False
        return record_count >= next_cleanup_at

    @staticmethod
    def _is_cleaned(cleaned_indices, idx: int) -> bool:
        if cleaned_indices is None:
            return False
        if hasattr(cleaned_indices, "get"):
            return bool(cleaned_indices.get(idx, False))
        return idx in cleaned_indices

    @staticmethod
    def _mark_cleaned(cleaned_indices, idx: int):
        if cleaned_indices is None:
            return
        if hasattr(cleaned_indices, "__setitem__"):
            cleaned_indices[idx] = True
        else:
            cleaned_indices.add(idx)

    @staticmethod
    def _bump_counter(cleanup_counters: Optional[Dict[str, int]], name: str, amount: int):
        if cleanup_counters is None or amount == 0:
            return
        if hasattr(cleanup_counters, "get") and hasattr(cleanup_counters, "__setitem__"):
            cleanup_counters[name] = int(cleanup_counters.get(name, 0)) + int(amount)

    # Magnitude threshold for detecting penalty values in natural objective space.
    # Must match PENALTY in objective.py (1e20).
    _PENALTY_MAGNITUDE = 1e20

    @staticmethod
    def _score_sort_key(score, score_directions: List[bool]):
        """Convert a raw score (scalar or tuple) to a sort key where lower = better rank.

        Rules applied per objective element:
        - **Minimize** direction: value kept as-is  (lower value → lower key → better rank).
        - **Maximize** direction: value negated      (higher value → lower negated key → better rank).
        - **Penalty / NaN / non-finite / abs >= 1e20**: always maps to ``float('inf')``
          so it sorts last regardless of direction.

        For multi-objective tuples the key is itself a tuple, giving per-direction
        lexicographic comparison: objectives are compared left-to-right, and the
        first position where the values differ determines the winner.  This
        implements the tiebreak-on-objective-0 behaviour requested for MOO cleanup.

        Args:
            score: A scalar float or a tuple/list of floats in natural objective space.
            score_directions: Per-objective directions (True = minimize, False = maximize).
                              If shorter than the score tuple, the last element is repeated.
        """
        _WORST = float('inf')
        _PEN = ArtifactCleanupManager._PENALTY_MAGNITUDE

        def _element_key(s, is_min: bool) -> float:
            try:
                s = float(s)
            except (TypeError, ValueError):
                return _WORST
            if s != s or not math.isfinite(s) or abs(s) >= _PEN:  # NaN, ±inf, or penalty
                return _WORST
            return s if is_min else -s

        if isinstance(score, (tuple, list)):
            key = []
            for i, s in enumerate(score):
                is_min = score_directions[i] if i < len(score_directions) else score_directions[-1]
                key.append(_element_key(s, is_min))
            return tuple(key) if key else (_WORST,)

        # Scalar score
        is_min = score_directions[0] if score_directions else True
        return _element_key(score, is_min)

    def cleanup_single_simulation_files(self, file_paths: Optional[Dict[str, str]], reason: str = "failed") -> bool:
        """Immediately clean files generated by a single simulation."""
        if not file_paths:
            return False

        result = self.cleanup_path_bundle(
            tracked_paths=file_paths,
            verbose=self.verbose,
            log_fn=self.log_fn,
            print_deleted_files=self.print_deleted_files,
            reason=reason,
        )

        has_any_path = isinstance(file_paths, dict) and bool(file_paths)
        if has_any_path:
            self.stats["simulations_cleaned"] += 1
        if result["total_removed"] > 0:
            self.stats["files_deleted"] += result["total_removed"]
        return has_any_path

    def run_periodic_cleanup(
        self,
        scores,
        file_paths,
        score_directions: Optional[List[bool]] = None,
        force: bool = False,
    ) -> bool:
        """Run periodic or final top-K cleanup for in-process optimizers.

        Args:
            scores: List of recorded scores (scalars or tuples for MOO).
            file_paths: Parallel list of file-path dicts for each record.
            score_directions: Per-objective directions (True=minimize, False=maximize).
                              Defaults to ``[True]`` (minimize) when omitted.
            force: Run cleanup immediately regardless of batch threshold.
        """
        if not self.periodic_cleanup_enabled():
            return False

        record_count = len(scores)
        if not self._can_run_periodic_cleanup(
            record_count=record_count,
            keep_top_k=self.keep_top_k,
            cleanup_batch_size=self.cleanup_batch_size,
            next_cleanup_at=self.next_cleanup_at,
            force=force,
        ):
            return False

        self._emit(
            verbose=self.verbose,
            log_fn=self.log_fn,
            message=f"Periodic cleanup: keeping top {self.keep_top_k} out of {record_count} successful evaluations",
            level="info",
        )

        self.cleanup_records(
            scores=scores,
            file_paths=file_paths,
            keep_top_k=self.keep_top_k,
            score_directions=score_directions,
            cleaned_indices=self.cleaned_file_indices,
            verbose=self.verbose,
            log_fn=self.log_fn,
            print_deleted_files=self.print_deleted_files,
            cleanup_counters=self.stats,
        )

        if self.cleanup_batch_size > 0:
            self.next_cleanup_at = ((record_count // self.cleanup_batch_size) + 1) * self.cleanup_batch_size

        return True

    def sync_counters(self, cleanup_counters: Optional[Dict[str, int]] = None):
        """Copy shared cleanup counters back to this manager."""
        if cleanup_counters is None:
            return

        self.stats["simulations_cleaned"] = int(cleanup_counters.get("simulations_cleaned", 0))
        self.stats["files_deleted"] = int(cleanup_counters.get("files_deleted", 0))

    @staticmethod
    def get_immediate_cleanup_reason(exec_time, is_oom, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Return the cleanup reason for failed, OOM, or killed simulations."""
        metadata = metadata or {}

        if bool(metadata.get("was_killed", False)):
            return "killed"
        if bool(is_oom):
            return "oom"
        if exec_time is None or bool(metadata.get("sim_failed", False)):
            return "failed"

        return None

    @staticmethod
    def run_periodic_cleanup_for_state(cleanup_state, force: bool = False) -> bool:
        """Run shared top-K cleanup against a state dict used by worker-based optimizers.

        ``score_directions`` is read from the state dict (stored at optimizer
        init via the ``'score_directions'`` key).  No direction parameter is
        required at the call site.
        """
        if cleanup_state is None:
            return False

        keep_top_k = int(cleanup_state.get("keep_top_k", -1))
        cleanup_batch_size = int(cleanup_state.get("cleanup_batch_size", -1))
        scores = list(cleanup_state.get("scores", []))

        # Per-objective directions stored in the state dict at optimizer init.
        score_directions = cleanup_state.get("score_directions", None)

        next_cleanup_at = cleanup_state.get("control", {}).get("next_cleanup_at", cleanup_batch_size)
        if not ArtifactCleanupManager._can_run_periodic_cleanup(
            record_count=len(scores),
            keep_top_k=keep_top_k,
            cleanup_batch_size=cleanup_batch_size,
            next_cleanup_at=next_cleanup_at,
            force=force,
        ):
            return False

        ArtifactCleanupManager.cleanup_records(
            scores=scores,
            file_paths=list(cleanup_state.get("file_paths", [])),
            keep_top_k=keep_top_k,
            score_directions=score_directions,
            cleaned_indices=cleanup_state.get("cleaned_indices"),
            verbose=cleanup_state.get("verbose", False),
            print_deleted_files=cleanup_state.get("print_deleted_files", False),
            cleanup_counters=cleanup_state.get("cleanup_counters"),
        )

        control = cleanup_state.get("control")
        if control is not None and cleanup_batch_size > 0:
            control["next_cleanup_at"] = ((len(scores) // cleanup_batch_size) + 1) * cleanup_batch_size

        return True

    @staticmethod
    def cleanup_path_bundle(
        tracked_paths: Dict[str, str],
        verbose: bool = False,
        log_fn=None,
        print_deleted_files: bool = False,
        reason: str = "cleanup",
    ) -> Dict[str, Any]:
        """Delete all files associated with one simulation record."""

        deleted_files: List[str] = []
        workload_files_removed = 0
        output_files_removed = 0

        workload_base = tracked_paths.get("workload") if isinstance(tracked_paths, dict) else None
        if workload_base:
            for workload_file in glob.glob(workload_base + ".*"):
                if os.path.exists(workload_file):
                    try:
                        os.remove(workload_file)
                        deleted_files.append(workload_file)
                        workload_files_removed += 1
                    except Exception as exc:
                        ArtifactCleanupManager._emit(verbose, log_fn, f"Warning: Could not remove {workload_file}: {exc}", "warning")

        output_base = tracked_paths.get("output_pattern") if isinstance(tracked_paths, dict) else None
        if output_base:
            for output_file in glob.glob(output_base + "*"):
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                        deleted_files.append(output_file)
                        output_files_removed += 1
                    except Exception as exc:
                        ArtifactCleanupManager._emit(verbose, log_fn, f"Warning: Could not remove {output_file}: {exc}", "warning")

        total_removed = workload_files_removed + output_files_removed
        if total_removed > 0 and verbose:
            ArtifactCleanupManager._emit(verbose, log_fn,
                f"Immediate cleanup ({reason}): removed {total_removed} files "
                f"({workload_files_removed} workload + {output_files_removed} output)",
                "info")
            if print_deleted_files:
                for path in deleted_files:
                    ArtifactCleanupManager._emit(verbose, log_fn, f"  deleted: {path}", "info")

        return {
            "deleted_files": deleted_files,
            "workload_files_removed": workload_files_removed,
            "output_files_removed": output_files_removed,
            "total_removed": total_removed,
        }

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
        """Remove tracked files for all non-top-K scored records.

        Args:
            scores: Recorded scores in natural objective space.  Each element
                    may be a scalar float or a tuple of floats for MOO.
            file_paths: Parallel list of file-path dicts.
            keep_top_k: Number of top-ranked records to retain.
            score_directions: Per-objective directions (True=minimize,
                              False=maximize).  Objectives are compared
                              left-to-right with per-direction sign so
                              tiebreaks fall to objective 0.
                              Defaults to ``[True]`` (minimize) when omitted.
        """

        if keep_top_k < 0 or len(scores) <= keep_top_k:
            return False

        # Default to minimize-everything when no directions are provided.
        effective_directions: List[bool] = score_directions if score_directions else [True]

        # Rank by direction-aware sort key (lower key = better rank for all objectives).
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: ArtifactCleanupManager._score_sort_key(scores[i], effective_directions),
        )
        top_k_set = set(sorted_indices[:keep_top_k])

        files_removed = 0
        output_files_removed = 0
        cleaned_records = 0

        for idx, tracked_paths in enumerate(file_paths):
            if idx in top_k_set or not tracked_paths or ArtifactCleanupManager._is_cleaned(cleaned_indices, idx):
                continue

            result = ArtifactCleanupManager.cleanup_path_bundle(
                tracked_paths=tracked_paths,
                verbose=verbose,
                log_fn=log_fn,
                print_deleted_files=print_deleted_files,
                reason="periodic_topk",
            )

            cleaned_records += 1
            if result["total_removed"] > 0:
                files_removed += result["workload_files_removed"]
                output_files_removed += result["output_files_removed"]

            ArtifactCleanupManager._mark_cleaned(cleaned_indices, idx)

        ArtifactCleanupManager._bump_counter(cleanup_counters, "simulations_cleaned", cleaned_records)
        ArtifactCleanupManager._bump_counter(cleanup_counters, "files_deleted", files_removed + output_files_removed)

        if verbose:
            ArtifactCleanupManager._emit(verbose, log_fn,
                f"Cleanup complete: pruned {cleaned_records} records, removed {files_removed} workload files and {output_files_removed} output files",
                "info")

            if cleanup_counters is not None:
                ArtifactCleanupManager._emit(verbose, log_fn,
                    f"Cleanup totals so far: simulations_cleaned={int(cleanup_counters.get('simulations_cleaned', 0))}, "
                    f"files_deleted={int(cleanup_counters.get('files_deleted', 0))}",
                    "info")

        return True

    @staticmethod
    def compress_path_bundle(
        tracked_paths: Dict[str, str],
        archive_path: str,
        verbose: bool = False,
        log_fn=None,
        print_deleted_files: bool = False,
    ) -> Dict[str, Any]:
        """Compress all files for one simulation record into an archive, then delete the originals."""
        collected_files: List[str] = []

        workload_base = tracked_paths.get("workload") if isinstance(tracked_paths, dict) else None
        if workload_base:
            collected_files.extend(f for f in glob.glob(workload_base + ".*") if os.path.isfile(f))

        output_base = tracked_paths.get("output_pattern") if isinstance(tracked_paths, dict) else None
        if output_base:
            collected_files.extend(f for f in glob.glob(output_base + "*") if os.path.isfile(f))

        if not collected_files:
            return {"archive_path": None, "files_compressed": 0, "files_deleted": 0, "success": False}

        try:
            if archive_path.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for f in collected_files:
                        zf.write(f, arcname=os.path.basename(f))
            else:  # default: tar.gz
                with tarfile.open(archive_path, "w:gz") as tf:
                    for f in collected_files:
                        tf.add(f, arcname=os.path.basename(f))
        except Exception as exc:
            ArtifactCleanupManager._emit(verbose, log_fn, f"Could not create archive {archive_path}: {exc}", "warning")
            return {"archive_path": None, "files_compressed": 0, "files_deleted": 0, "success": False}

        deleted_count = 0
        for f in collected_files:
            try:
                os.remove(f)
                deleted_count += 1
                if print_deleted_files:
                    ArtifactCleanupManager._emit(verbose, log_fn, f"  deleted (compressed): {f}", "info")
            except Exception as exc:
                ArtifactCleanupManager._emit(verbose, log_fn, f"Could not remove {f}: {exc}", "warning")

        if verbose:
            ArtifactCleanupManager._emit(verbose, log_fn,
                f"Compressed {len(collected_files)} files -> {archive_path} (deleted {deleted_count} originals)",
                "info")

        return {
            "archive_path": archive_path,
            "files_compressed": len(collected_files),
            "files_deleted": deleted_count,
            "success": True,
        }

    def compress_and_clean(
        self,
        experiment_dirs: List[str],
        archive_dir: Optional[str] = None,
        archive_format: str = "tar.gz",
    ) -> Dict[str, Any]:
        """Compress all remaining files in experiment directories into archives.

        Should be called once after optimization is complete (and after the final
        periodic cleanup has removed non-top-K artifacts). Each supplied directory
        is packed into a single archive named after that directory, placed in the
        directory's parent (or ``archive_dir`` if given), and its files are deleted.

        Args:
            experiment_dirs: List of directory paths to compress.  Typically the
                             simulation output directory and workload directory.
            archive_dir: Where to write the archives.  Defaults to the parent of
                         each directory being compressed.
            archive_format: ``"tar.gz"`` (default) or ``"zip"``.

        Returns:
            Dict with keys ``archives_created``, ``files_compressed``, ``files_deleted``.
        """
        empty = {"archives_created": 0, "files_compressed": 0, "files_deleted": 0}

        if not experiment_dirs:
            return empty

        ext = ".zip" if archive_format == "zip" else ".tar.gz"
        archives_created = 0
        total_compressed = 0
        total_deleted = 0

        self._emit(self.verbose, self.log_fn,
            f"Compressing {len(experiment_dirs)} experiment directory/directories ({archive_format})...",
            "info")

        for dir_path in experiment_dirs:
            dir_path = os.path.abspath(dir_path)

            if not os.path.isdir(dir_path):
                self._emit(self.verbose, self.log_fn,
                    f"Skipping missing directory: {dir_path}", "warning")
                continue

            dir_name = os.path.basename(dir_path)
            dest = archive_dir if archive_dir is not None else dir_path
            os.makedirs(dest, exist_ok=True)
            archive_path = os.path.join(dest, dir_name + ext)

            all_files = sorted(
                f for f in glob.glob(os.path.join(dir_path, "*"))
                if os.path.isfile(f) and os.path.abspath(f) != os.path.abspath(archive_path)
            )

            if not all_files:
                self._emit(self.verbose, self.log_fn,
                    f"Skipping empty directory: {dir_path}", "info")
                continue

            try:
                if archive_path.endswith(".zip"):
                    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for f in all_files:
                            zf.write(f, arcname=os.path.basename(f))
                else:
                    with tarfile.open(archive_path, "w:gz") as tf:
                        for f in all_files:
                            tf.add(f, arcname=os.path.basename(f))
            except Exception as exc:
                self._emit(self.verbose, self.log_fn,
                    f"Could not create archive {archive_path}: {exc}", "warning")
                continue

            deleted_count = 0
            for f in all_files:
                try:
                    os.remove(f)
                    deleted_count += 1
                    if self.print_deleted_files:
                        self._emit(self.verbose, self.log_fn, f"  deleted (compressed): {f}", "info")
                except Exception as exc:
                    self._emit(self.verbose, self.log_fn, f"Could not remove {f}: {exc}", "warning")

            archives_created += 1
            total_compressed += len(all_files)
            total_deleted += deleted_count

            self._emit(self.verbose, self.log_fn,
                f"  [{dir_name}] {len(all_files)} files -> {archive_path} "
                f"(deleted {deleted_count} originals)",
                "info")

        self.stats["files_deleted"] += total_deleted

        self._emit(self.verbose, self.log_fn,
            f"Compression complete: {archives_created} archives created, "
            f"{total_compressed} files compressed, {total_deleted} originals deleted.",
            "info")

        return {
            "archives_created": archives_created,
            "files_compressed": total_compressed,
            "files_deleted": total_deleted,
        }