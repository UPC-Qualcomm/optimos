#!/usr/bin/env python3
"""
Brute-Force Exhaustive Search

Enumerates every valid configuration in the search space, evaluates each one
using multiple worker processes, and produces histograms of the objective values.
This is useful for characterising the full landscape of a (small-to-medium)
search space.

Usage:
    python example_bruteforce.py                       # run all configs
    python example_bruteforce.py --resume results.csv  # resume from partial CSV
    python example_bruteforce.py --histogram results.csv  # histogram-only (no sims)
    python example_bruteforce.py --workers 8           # use 8 parallel processes
"""

import sys
import os
import argparse
import time
import signal
import multiprocessing as mp
from itertools import product
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — same convention as existing examples
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Optimization import (
    create_search_space,
    SimulationRunner,
    create_objective,
)
from Optimization.core.base_optimizer import format_score
from Optimization.helper.config_utils import evaluate_config_worker

# ---------------------------------------------------------------------------
# Configuration — edit these for your experiment
# ---------------------------------------------------------------------------
MODEL_NUM            = 17        # Model enum (e.g. 17 = GPT_40B)
MODEL_NAME           = "llama_8b_bruteforce_analytical"
NUM_NPUS             = 64
NETWORK_NAME         = "FoldedClos"
FOLDER_PREFIX        = "BRUTEFORCE"
OBJECTIVE_TYPE       = "time"   # any key accepted by create_objective()
N_WORKERS            = 10         # Number of parallel worker processes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def enumerate_design_space(search_space) -> List[Dict[str, Any]]:
    """
    Generate the full Cartesian product of every parameter in the search space
    and return only configurations that satisfy all constraints.
    """
    param_names = list(search_space.parameters.keys())
    param_values = [search_space.parameters[n] for n in param_names]

    valid_configs: List[Dict[str, Any]] = []
    for combo in product(*param_values):
        config = dict(zip(param_names, combo))
        if all(c(config) for c in search_space.constraints):
            valid_configs.append(config)

    return valid_configs


def config_to_key(config: Dict) -> tuple:
    return tuple(sorted(config.items()))


# ---------------------------------------------------------------------------
# Worker function — runs in a child process via mp.Pool
# ---------------------------------------------------------------------------

def _eval_single(args_tuple):
    """
    Evaluate one configuration.  Executed inside a worker process.

    Each worker creates its own SimulationRunner (with a PID-based folder
    prefix) so that parallel workers never collide on the filesystem.
    """
    (config, eval_idx, model_num, model_name, network_name,
     folder_prefix, objective_type, clusters) = args_tuple

    from Optimization.helper.config_utils import (
        evaluate_config_worker as _eval_worker,
        enrich_config_with_clusters,
    )

    # Per-worker SimulationRunner — folder_prefix includes worker PID to
    # guarantee unique output paths.
    worker_prefix = f"{folder_prefix}_w{os.getpid()}"
    sim_runner = SimulationRunner(
        model_num=model_num,
        model_name=model_name,
        network_name=network_name,
        folder_prefix=worker_prefix,
        verbose=False,
    )

    objective = create_objective(objective_type)

    # Enrich config with cluster info
    if clusters and 'cluster' in config:
        enriched = enrich_config_with_clusters(config, clusters)
    else:
        enriched = config.copy()

    # Run simulation
    returned_config, exec_time, is_oom, file_paths, metadata = _eval_worker(
        enriched, sim_runner
    )

    # Compute score
    score = None
    if exec_time is not None:
        score = objective.compute(exec_time, is_oom, metadata, enriched)

    # Build result row
    row: Dict[str, Any] = {"eval_idx": eval_idx}
    row.update(config)  # original (non-enriched) params
    row["exec_time"] = exec_time
    row["is_oom"] = bool(is_oom) if is_oom is not None else None

    if score is not None:
        if isinstance(score, tuple):
            for i, s in enumerate(score):
                row[f"objective_{i}"] = s
        else:
            row["objective_0"] = score

    return row


# ---------------------------------------------------------------------------
# Histogram generation
# ---------------------------------------------------------------------------

def build_histogram(df: pd.DataFrame, column: str, title: str,
                    output_path: str, bins: int = 50, log_scale: bool = False):
    """Build and save a single histogram (PNG + interactive HTML)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = df[column].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        print(f"  ⚠️  No valid data for {column}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.75)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    png_path = output_path + ".png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")

    # Interactive HTML via plotly (optional)
    try:
        import plotly.express as px
        fig_html = px.histogram(df, x=column, nbins=bins, title=title,
                                marginal="rug")
        html_path = output_path + ".html"
        fig_html.write_html(html_path)
        print(f"  Saved: {html_path}")
    except ImportError:
        pass


def generate_histograms(csv_path: str, save_dir: str, model_name: str):
    """Read the results CSV and produce histograms for every numeric column."""
    df = pd.read_csv(csv_path)
    print(f"\nGenerating histograms from {len(df)} rows …")

    # Identify objective / score columns
    score_cols = [c for c in df.columns if c.startswith("objective_") or c == "exec_time"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    histogram_cols = list(dict.fromkeys(score_cols + numeric_cols))  # dedupe, keep order

    for col in histogram_cols:
        out = os.path.join(save_dir, f"hist_{model_name}_{col}")
        title = f"{model_name} — {col} distribution ({len(df)} configs)"
        build_histogram(df, col, title, out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Brute-force exhaustive search")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to partial results CSV to resume from")
    parser.add_argument("--histogram", type=str, default=None,
                        help="Path to completed CSV — only generate histograms (no sims)")
    parser.add_argument("--search-space", type=str, default=None,
                        help="Path to search space JSON (overrides default)")
    parser.add_argument("--model-num", type=int, default=MODEL_NUM)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--num-npus", type=int, default=NUM_NPUS)
    parser.add_argument("--objective", type=str, default=OBJECTIVE_TYPE)
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Number of parallel worker processes (default: {N_WORKERS})")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Evaluate at most N configs (random subset of the full space)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only enumerate and count configs, do not run simulations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --max-configs subset selection")
    args = parser.parse_args()
    
    # ── Histogram-only mode ──────────────────────────────────────────────
    if args.histogram:
        save_dir = os.path.dirname(args.histogram) or "."
        generate_histograms(args.histogram, save_dir, args.model_name)
        return

    # ── Setup ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("BRUTE-FORCE EXHAUSTIVE SEARCH")
    print("=" * 70)
    

    # 1. Search space
    search_space_path = args.search_space or os.path.join(
        os.path.dirname(__file__), "..",
        "search_space", "bruteforce.json"
    )
    search_space = create_search_space(
        search_space_path,
        include_categories=["parallelism_strategy", "network", "model", "system"],
    )
    print(f"Search space: {search_space}")

    # 2. Enumerate all valid configs
    print("\nEnumerating design space …")
    all_configs = enumerate_design_space(search_space)
    print(f"  Total valid configurations: {len(all_configs)}")
    if not all_configs:
        print("ERROR: design space is empty — check your search space JSON / constraints.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: exiting without running simulations.")
        print(f"\nSample config: {all_configs[0]}")
        return

    # Optional: evaluate only a random subset
    if args.max_configs and args.max_configs < len(all_configs):
        import random
        random.seed(args.seed)
        all_configs = random.sample(all_configs, args.max_configs)
        print(f"  Sub-sampled to {len(all_configs)} configs (seed={args.seed})")


    # 3. Objective (printed here; each worker creates its own instance)
    objective = create_objective(args.objective)
    is_moo = getattr(objective, "is_multi_objective", False)
    print(f"  Objective: {objective.name}  (multi-objective={is_moo})")
    print(f"  Workers:   {args.workers}")

    # 4. Prepare output directory
    base_dir = os.environ["OPTIMOS_ROOT"]
    save_dir = os.path.join(
        base_dir, "output",
        f"{FOLDER_PREFIX}_{args.model_name}", NETWORK_NAME,
    )
    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, f"bruteforce_results_{args.model_name}.csv")

    # 5. Resume support
    evaluated_set: set = set()
    rows: List[Dict] = []
    if args.resume and os.path.exists(args.resume):
        prev = pd.read_csv(args.resume)
        rows = prev.to_dict("records")
        param_cols = [c for c in prev.columns if not c.startswith("objective_")
                      and c not in ("exec_time", "is_oom", "eval_idx")]
        for _, row in prev.iterrows():
            key = tuple(sorted((c, row[c]) for c in param_cols if c in row))
            evaluated_set.add(key)
        print(f"  Resumed {len(rows)} evaluations from {args.resume}")

    remaining = [c for c in all_configs if config_to_key(c) not in evaluated_set]
    total = len(all_configs)
    done = len(rows)
    print(f"\nEvaluating {len(remaining)} remaining configs ({done}/{total} done) …\n")

    # ── Build work items for the pool ────────────────────────────────────
    clusters = search_space.get_all_clusters()
    work_items = []
    for idx, config in enumerate(remaining):
        work_items.append((
            config,
            done + idx + 1,              # eval_idx
            args.model_num,
            args.model_name,
            NETWORK_NAME,
            FOLDER_PREFIX,
            args.objective,
            clusters,
        ))

    # ── Parallel evaluation ──────────────────────────────────────────────
    # Ignore SIGINT in the parent before forking so children inherit the
    # ignore disposition.  We restore it right after creating the pool.
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    t_start = time.time()
    new_rows: List[Dict] = []

    try:
        with mp.Pool(processes=args.workers) as pool:
            # Restore SIGINT in the parent so Ctrl+C is caught here.
            signal.signal(signal.SIGINT, original_sigint)

            # imap_unordered gives us results as they finish so we can
            # checkpoint periodically into the CSV.
            checkpoint_interval = max(50, args.workers * 2)
            completed = 0

            try:
                for row in pool.imap_unordered(_eval_single, work_items, chunksize=1):
                    rows.append(row)
                    new_rows.append(row)
                    completed += 1

                    # Progress line
                    score_str = format_score(
                        tuple(row[k] for k in sorted(row) if k.startswith("objective_"))
                        or row.get("objective_0")
                    ) if any(k.startswith("objective_") for k in row) else "FAIL"
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(remaining) - completed) / rate if rate > 0 else 0
                    print(f"[{done + completed}/{total}] score={score_str}  "
                          f"exec_time={row.get('exec_time')}  "
                          f"[{rate:.1f} eval/s, ETA {eta/60:.0f}m]")

                    # Periodic checkpoint
                    if completed % checkpoint_interval == 0:
                        df = pd.DataFrame(rows)
                        if "eval_idx" in df.columns:
                            df = df.sort_values("eval_idx").reset_index(drop=True)
                        df.to_csv(results_csv, index=False)
                        print(f"  💾 Checkpoint saved → {results_csv}")

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted — terminating workers …")
                pool.terminate()
                pool.join()

    except KeyboardInterrupt:
        pass

    # Restore original handler
    signal.signal(signal.SIGINT, original_sigint)

    # ── Save final combined CSV ──────────────────────────────────────────
    df = pd.DataFrame(rows)
    if "eval_idx" in df.columns:
        df = df.sort_values("eval_idx").reset_index(drop=True)
    df.to_csv(results_csv, index=False)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"BRUTE-FORCE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Evaluated: {len(rows)}/{total} configs")
    print(f"  Workers:   {args.workers}")
    print(f"  Elapsed:   {elapsed/60:.1f} min")
    print(f"  Results:   {results_csv}")

    # ── Histograms ───────────────────────────────────────────────────────
    generate_histograms(results_csv, save_dir, args.model_name)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
