#!/usr/bin/env python3
"""
estimate_power.py — batch power estimation for AstraSim simulation outputs.

Scans an output directory for simulation log files and runs the power model
for each one.  The nodemap path is derived automatically from the
topology_file field in the network config YAML
(topology_file + "_nodemap.json"), so no extra arguments are needed beyond
what run_astrasim.py already uses.

This is the single entry point for power analysis — it handles both single-
experiment directories (one .log file) and batch directories (many .log files)
identically.

Usage
-----
    python estimate_power.py \\
        --output-dir  ./output/MyExp/FoldedClos_iter2 \\
        --network     ./configuration/g2/FoldedClos_16_config_untracked.yml \\
        [--config     ./power_model/a100_config.json]  \\
        [--mode       compare | A | B | C | D]         \\
        [--result-dir ./results/MyExp/FoldedClos_iter2]

Log-file detection
------------------
The C++ spdlog rotating sink writes the active file as  <stem>.log  and
rotated backups as  <stem>.log.1, <stem>.log.2, … .  Only the base (un-
numbered) file is picked up — the others are ignored.

Output (per log file)
---------------------
    <result-dir>/<stem>_power_est.json      — machine-readable breakdown
    <result-dir>/<stem>_power_analysis.txt  — human-readable report
"""

import argparse
import contextlib
import glob
import io
import os
import sys

import yaml  # PyYAML

# Power analysis functions (single source of truth)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_power import analyze_single_mode, compare_modes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_log_files(output_dir: str) -> list:
    """Return sorted list of *.log files (no rotation suffix) in output_dir."""
    # glob '*.log' matches exactly name.log – it will NOT match name.log.1
    return sorted(glob.glob(os.path.join(output_dir, "*.log")))


def extract_topology_file(network_yml: str) -> str:
    """Parse the network YAML and return the raw topology_file string."""
    with open(network_yml, "r") as f:
        cfg = yaml.safe_load(f)
    topo = cfg.get("topology_file", "").strip().strip('"').strip("'")
    if not topo:
        raise ValueError(f"'topology_file' key not found in {network_yml}")
    return topo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here           = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(here, "h100_config.json")

    p = argparse.ArgumentParser(
        description="Batch power estimation for AstraSim simulation outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All 4 modes (default)
  python estimate_power.py \\
      --output-dir ./output/MyExp/FoldedClos_iter2 \\
      --network    ./configuration/g2/FoldedClos_16_config_untracked.yml

  # Single mode, custom config, separate result directory
  python estimate_power.py \\
      --output-dir  ./output/MyExp/FoldedClos_iter2 \\
      --network     ./configuration/g2/FoldedClos_16_config_untracked.yml \\
      --config      ./power_model/a100_config.json \\
      --mode        D \\
      --result-dir  ./results/MyExp/FoldedClos_iter2
        """,
    )

    p.add_argument(
        "--output-dir", required=True,
        help="Simulation output directory (same value as --output_dir in run_astrasim.py).",
    )
    p.add_argument(
        "--network", required=True,
        help="Network config YAML (same value as --network in run_astrasim.py). "
             "The topology_file field is read from this file to locate the nodemap.",
    )
    p.add_argument(
        "--config", default=default_config,
        help=f"Power model config JSON (default: {os.path.basename(default_config)}).",
    )
    p.add_argument(
        "--mode", default="compare",
        choices=["compare", "A", "B", "C", "D"],
        help="compare = run all 4 LPM modes side-by-side (default); "
             "A/B/C/D = run a single mode.",
    )
    p.add_argument(
        "--result-dir", default=None,
        help="Directory to write *_power_est.json files "
             "(default: same as --output-dir).",
    )

    args = p.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    result_dir = os.path.abspath(args.result_dir or args.output_dir)
    os.makedirs(result_dir, exist_ok=True)

    # --- 1. Derive nodemap from topology_file in network YAML ---------------
    topo_file = extract_topology_file(args.network)
    nodemap   = topo_file + "_nodemap.json"
    if not os.path.isfile(nodemap):
        print(
            f"WARNING: nodemap not found at:\n  {nodemap}\n"
            f"  (derived from topology_file in {args.network})\n"
            "  Switch-level power will fall back to link-only estimation.",
            file=sys.stderr,
        )

    # --- 2. Find log files --------------------------------------------------
    log_files = find_log_files(output_dir)
    if not log_files:
        print(f"No .log files found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(log_files)} log file(s) in {output_dir}")
    print(f"Results will be written to: {result_dir}\n")

    # --- 3. Process each log file -------------------------------------------
    skipped = []
    errors  = []

    for log_file in log_files:
        stem         = os.path.splitext(log_file)[0]
        link_traffic = stem + "_link_traffic.csv"
        base         = os.path.basename(stem)
        output_json  = os.path.join(result_dir, base + "_power_est.json")
        output_txt   = os.path.join(result_dir, base + "_power_analysis.txt")

        if not os.path.isfile(link_traffic):
            print(f"  SKIP {os.path.basename(log_file)}: "
                  f"link_traffic CSV not found ({os.path.basename(link_traffic)})")
            skipped.append(log_file)
            continue

        print(f"  Processing: {os.path.basename(log_file)} ", end="", flush=True)

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                if args.mode.upper() == "COMPARE":
                    compare_modes(
                        log_file, link_traffic,
                        output_json=output_json,
                        config_path=args.config,
                        nodemap_file=nodemap,
                    )
                else:
                    analyze_single_mode(
                        log_file, link_traffic,
                        mode=args.mode.upper(),
                        output_json=output_json,
                        config_path=args.config,
                        nodemap_file=nodemap,
                    )
            with open(output_txt, "w") as fh:
                fh.write(buf.getvalue())
            print(f"✅  → {os.path.basename(output_txt)}")
        except Exception as exc:
            print(f"❌  {exc}")
            errors.append(log_file)

    # --- 4. Summary ---------------------------------------------------------
    total     = len(log_files)
    succeeded = total - len(skipped) - len(errors)
    print(f"\n{'='*60}")
    print(f"Power estimation complete: {succeeded}/{total} succeeded"
          + (f", {len(skipped)} skipped, {len(errors)} errors" if skipped or errors else ""))

    if errors:
        print("Failed runs:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
