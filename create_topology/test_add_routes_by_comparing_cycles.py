#!/usr/bin/env python3
"""
Compare exec_cycles, comm_cycles, exposed_comm_cycles, comp_cycles, exposed_comp_cycles
between two result CSV files, matched by (dp, mp, sp, pp, sharding, sys_id).
"""

import argparse
import os
import sys

import pandas as pd

KEY_COLS = ["dp", "mp", "sp", "pp", "sharding", "sys_id"]
METRIC_COLS = [
    "exec_cycles",
    "comm_cycles",
    "exposed_comm_cycles",
    "comp_cycles",
    "exposed_comp_cycles",
]

LABEL_A = "A"
LABEL_B = "B"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selected performance metrics between two Astra-Sim CSV outputs."
    )
    parser.add_argument("--file-a", default=os.getenv("COMPARE_FILE_A"), help="Path to first CSV file")
    parser.add_argument("--file-b", default=os.getenv("COMPARE_FILE_B"), help="Path to second CSV file")
    parser.add_argument("--label-a", default=LABEL_A, help="Label for first file in output")
    parser.add_argument("--label-b", default=LABEL_B, help="Label for second file in output")
    args = parser.parse_args()

    if not args.file_a or not args.file_b:
        parser.error("Both --file-a and --file-b are required (or set COMPARE_FILE_A and COMPARE_FILE_B).")

    return args


def main() -> None:
    args = parse_args()

    file_a = args.file_a
    file_b = args.file_b
    label_a = args.label_a
    label_b = args.label_b

    df_a = load(file_a)
    df_b = load(file_b)

    # Keep only what we need
    df_a = df_a[KEY_COLS + METRIC_COLS].copy()
    df_b = df_b[KEY_COLS + METRIC_COLS].copy()

    merged = df_a.merge(df_b, on=KEY_COLS, suffixes=(f"_{label_a}", f"_{label_b}"))

    if merged.empty:
        print("No matching rows found between the two files.")
        sys.exit(1)

    rows = []
    for _, row in merged.iterrows():
        key = {c: row[c] for c in KEY_COLS}
        for metric in METRIC_COLS:
            val_a = row[f"{metric}_{label_a}"]
            val_b = row[f"{metric}_{label_b}"]
            diff = val_b - val_a
            pct = (diff / val_a * 100) if val_a != 0 else float("nan")
            rows.append(
                {
                    **key,
                    "metric": metric,
                    label_a: val_a,
                    label_b: val_b,
                    "diff (B-A)": diff,
                    "diff_%": pct,
                }
            )

    result = pd.DataFrame(rows)

    key_str = result[KEY_COLS].astype(str).agg("_".join, axis=1)
    result.insert(0, "config", key_str)
    result = result.drop(columns=KEY_COLS)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:,.2f}".format)

    print(f"\n{'='*80}")
    print(f"  FILE A  ({label_a}):")
    print(f"  {file_a}")
    print(f"  FILE B  ({label_b}):")
    print(f"  {file_b}")
    print(f"  Matched rows: {len(merged)}")
    print(f"{'='*80}\n")
    print(result.to_string(index=False))

    print(f"\n{'='*80}")
    print("  PER-METRIC SUMMARY  (mean absolute & percentage difference)")
    print(f"{'='*80}")
    summary = (
        result.groupby("metric")[["diff (B-A)", "diff_%"]]
        .agg(
            mean_diff=("diff (B-A)", "mean"),
            mean_abs_diff=("diff (B-A)", lambda x: x.abs().mean()),
            mean_pct=("diff_%", "mean"),
            mean_abs_pct=("diff_%", lambda x: x.abs().mean()),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
