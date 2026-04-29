#!/usr/bin/env python3
"""
merge_power_results.py — merge gather_all_NPUs_results CSV with per-experiment
power estimation JSONs into a single flat CSV ready for analysis.

For each row in the performance results CSV (produced by
gather_all_NPUs_results.py), the script looks up the matching
*_power_est.json (produced by estimate_power.py) and appends per-mode
power columns.

Added columns — for each available mode (A / B / C / D):
    {mode}_total_power_W          total system power (GPU + network)
    {mode}_gpu_power_W            GPU-only power
    {mode}_network_power_W        network (links + switches) power
    {mode}_total_energy_J         total energy over the run
    {mode}_gpu_energy_J           GPU energy
    {mode}_network_energy_J       network energy
    {mode}_throughput_samples_per_sec
    {mode}_samples_per_joule
    {mode}_joules_per_sample

JSON lookup key (matches estimate_power.py output naming):
    {dp_mp_sp_pp_sharded}.seq_{seq}.batch_{batch}_power_est.json

Usage
-----
    python merge_power_results.py \\
        --results   ./results/MyExp/FoldedClos_iter2.csv \\
        --power-dir ./output/MyExp/FoldedClos_iter2 \\
        [--output   ./results/MyExp/FoldedClos_iter2_with_power.csv]
"""

import argparse
import json
import os
import sys

import pandas as pd


# Fields extracted from each mode in the power JSON
POWER_FIELDS = [
    "total_power_W",
    "gpu_power_W",
    "network_power_W",
    "total_energy_J",
    "gpu_energy_J",
    "network_energy_J",
    "throughput_samples_per_sec",
    "samples_per_joule",
    "joules_per_sample",
    "samples_per_sec_per_mj",
    "total_execution_time_s",
]


def power_json_path(row, power_dir: str) -> str:
    """Return the expected *_power_est.json path for a results CSV row."""
    stem = f"{row['dp_mp_sp_pp_sharded']}.seq_{row['seq']}.batch_{row['batch']}"
    return os.path.join(power_dir, f"{stem}_power_est.json")


def extract_power_columns(json_path: str) -> dict:
    """Read a *_power_est.json and return a flat {mode_field: value} dict."""
    with open(json_path) as f:
        data = json.load(f)
    flat = {}
    for mode, mode_data in data.items():
        for field in POWER_FIELDS:
            flat[f"{mode}_{field}"] = mode_data.get(field)
    return flat


def main():
    p = argparse.ArgumentParser(
        description="Merge performance results CSV with power estimation JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_power_results.py \\
      --results   ./results/MyExp/FoldedClos_iter2.csv \\
      --power-dir ./output/MyExp/FoldedClos_iter2

  # With explicit output path
  python merge_power_results.py \\
      --results   ./results/MyExp/FoldedClos_iter2.csv \\
      --power-dir ./output/MyExp/FoldedClos_iter2 \\
      --output    ./results/MyExp/FoldedClos_iter2_with_power.csv
        """,
    )
    p.add_argument(
        "--results", required=True,
        help="CSV produced by gather_all_NPUs_results.py.",
    )
    p.add_argument(
        "--power-dir", required=True,
        help="Directory containing *_power_est.json files "
             "(same as --output-dir / --result-dir passed to estimate_power.py).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output CSV path.  Default: <results_stem>_with_power.csv "
             "next to --results.",
    )

    args = p.parse_args()

    if not os.path.isfile(args.results):
        print(f"❌  Results CSV not found: {args.results}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.power_dir):
        print(f"❌  Power directory not found: {args.power_dir}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        stem, _ = os.path.splitext(args.results)
        args.output = stem + "_with_power.csv"

    df = pd.read_csv(args.results)

    # --- build power columns row by row ------------------------------------
    power_rows = []
    missing = []

    for _, row in df.iterrows():
        jpath = power_json_path(row, args.power_dir)
        if not os.path.isfile(jpath):
            missing.append(os.path.basename(jpath))
            power_rows.append({})
        else:
            power_rows.append(extract_power_columns(jpath))

    if missing:
        print(f"⚠️   Power JSON not found for {len(missing)} row(s):", file=sys.stderr)
        for m in missing:
            print(f"      {m}", file=sys.stderr)

    power_df = pd.DataFrame(power_rows)

    # Order columns: A → B → C → D, grouped by metric within each mode
    ordered_cols = [
        f"{mode}_{field}"
        for mode in ("A", "B", "C", "D")
        for field in POWER_FIELDS
        if f"{mode}_{field}" in power_df.columns
    ]

    out = pd.concat([df, power_df[ordered_cols]], axis=1)
    out.to_csv(args.output, index=False)
    print(f"✅  {len(out)} row(s) × {len(out.columns)} column(s)  →  {args.output}")


if __name__ == "__main__":
    main()
