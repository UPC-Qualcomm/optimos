#!/usr/bin/env python3
"""
plot_two_cols.py — simple two-line, dual-axis plot for AstraSim results.

Each column gets its own y-axis (left / right) with a matching line colour.
Configurations are sorted by the left column ascending before plotting.

Outputs
-------
    <output-dir>/<stem>_<col1>_vs_<col2>.html   — Plotly interactive
    <output-dir>/<stem>_<col1>_vs_<col2>.png    — Matplotlib static

Usage examples
--------------
  # Defaults: exec_cycles (ms) vs D_total_energy_J (J)
  python plot_two_cols.py \\
      --input ../results/MyExp/FoldedClos_iter2_with_power.csv

  # Custom columns
  python plot_two_cols.py \\
      --input  ../results/MyExp/FoldedClos_iter2_with_power.csv \\
      --col1   exec_cycles  --unit1 ms   --div1 1e6 \\
      --col2   A_total_energy_J  --unit2 J  --div2 1 \\
      --top    30

  # samples/(s·MJ) on the right axis
  python plot_two_cols.py \\
      --input  ../results/MyExp/FoldedClos_iter2_with_power.csv \\
      --col1   exec_cycles  --unit1 ms  --div1 1e6 \\
      --col2   D_samples_per_sec_per_mj  --unit2 "samples/(s·MJ)"  --div2 1
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go


# ════════════════════════════════════════════════════════════════════════════
# ── DEFAULTS — change here instead of passing CLI flags every time ───────────
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_COL1  = "exec_cycles"
DEFAULT_UNIT1 = "ms"
DEFAULT_DIV1  = 1e6          # cycles → ms  (1 cycle = 1 ns, 1e6 ns = 1 ms)

DEFAULT_COL2  = "D_total_energy_J"
DEFAULT_UNIT2 = "J"
DEFAULT_DIV2  = 1.0

# Colours — left axis matches col1, right axis matches col2.
COLOR1 = "#1f77b4"   # muted blue  (exec_cycles family)
COLOR2 = "#8c564b"   # chestnut brown (Full LPM family)

# Mode mapping used to build clean legend labels for A/B/C/D prefixed columns
MODE_LABELS = {
    "A": "No LPM (Baseline)",
    "B": "Compute LPM",
    "C": "Comm LPM",
    "D": "Full LPM",
}

METRIC_LABELS = {
    "total_energy_J":             "Energy",
    "gpu_energy_J":               "GPU Energy",
    "network_energy_J":           "Network Energy",
    "total_power_W":              "Power",
    "gpu_power_W":                "GPU Power",
    "network_power_W":            "Network Power",
    "samples_per_sec_per_mj":     "samples / (s·MJ)",
    "samples_per_joule":          "samples / J",
    "joules_per_sample":          "J / sample",
    "throughput_samples_per_sec": "Throughput (samples/s)",
    "exec_cycles":                "Exec Cycles",
    "comm_cycles":                "Comm Cycles",
}

# ════════════════════════════════════════════════════════════════════════════


def _col_label(col: str, unit: str) -> str:
    """Build a clean legend + axis label including the display unit."""
    for mode, desc in MODE_LABELS.items():
        prefix = f"{mode}_"
        if col.startswith(prefix):
            suffix = col[len(prefix):]
            metric = METRIC_LABELS.get(suffix, suffix.replace("_", " "))
            return f"{desc} — {metric} ({unit})"
    name = METRIC_LABELS.get(col, col.replace("_", " "))
    return f"{name} ({unit})"


def load_and_sort(csv_path: str, col1: str, col2: str, top: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("dp_mp_sp_pp_sharded", col1, col2):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.\n"
                             f"Available: {list(df.columns)}")
    df = df.sort_values(col1).reset_index(drop=True)
    if top:
        df = df.head(top)
    return df


# ── Plotly ────────────────────────────────────────────────────────────────────

def make_plotly(df: pd.DataFrame,
                col1: str, label1: str, div1: float,
                col2: str, label2: str, div2: float) -> go.Figure:

    x  = df["dp_mp_sp_pp_sharded"].tolist()
    y1 = (df[col1] / div1).tolist()
    y2 = (df[col2] / div2).tolist()

    fig = go.Figure()

    # Left axis — col1
    fig.add_trace(go.Scatter(
        x      = x,
        y      = y1,
        name   = label1,
        mode   = "lines+markers",
        line   = dict(color=COLOR1, width=2.5),
        marker = dict(size=7),
        yaxis  = "y",
    ))

    # Right axis — col2
    fig.add_trace(go.Scatter(
        x      = x,
        y      = y2,
        name   = label2,
        mode   = "lines+markers",
        line   = dict(color=COLOR2, width=2.5, dash="dot"),
        marker = dict(size=7, symbol="diamond"),
        yaxis  = "y2",
    ))

    fig.update_layout(
        xaxis  = dict(
            title     = "Configuration  (sorted by left column ↑)",
            tickangle = -50,
            tickfont  = dict(size=10),
        ),
        yaxis  = dict(
            title      = dict(text=label1, font=dict(color=COLOR1)),
            tickfont   = dict(color=COLOR1),
        ),
        yaxis2 = dict(
            title      = dict(text=label2, font=dict(color=COLOR2)),
            tickfont   = dict(color=COLOR2),
            overlaying = "y",
            side       = "right",
        ),
        legend    = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1,
            font        = dict(size=11),
        ),
        hovermode = "x unified",
        template  = "plotly_white",
        height    = 560,
        margin    = dict(b=160, t=60, r=80),
    )
    return fig


# ── Matplotlib ────────────────────────────────────────────────────────────────

def make_matplotlib(df: pd.DataFrame,
                    col1: str, label1: str, div1: float,
                    col2: str, label2: str, div2: float) -> plt.Figure:

    n      = len(df)
    x      = list(range(n))
    xlbls  = df["dp_mp_sp_pp_sharded"].tolist()
    y1     = (df[col1] / div1).tolist()
    y2     = (df[col2] / div2).tolist()

    fig, ax1 = plt.subplots(figsize=(max(12, n * 0.55), 5))
    ax2 = ax1.twinx()

    ln1, = ax1.plot(x, y1, color=COLOR1, linewidth=2.2,
                    marker="o", markersize=5, label=label1, zorder=3)
    ln2, = ax2.plot(x, y2, color=COLOR2, linewidth=2.2, linestyle="--",
                    marker="D", markersize=5, label=label2, zorder=2)

    ax1.set_xlabel("Configuration  (sorted by left column ↑)", fontsize=10)
    ax1.set_ylabel(label1, fontsize=10, color=COLOR1)
    ax2.set_ylabel(label2, fontsize=10, color=COLOR2)

    ax1.tick_params(axis="y", colors=COLOR1)
    ax2.tick_params(axis="y", colors=COLOR2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlbls, rotation=50, ha="right", fontsize=8)
    ax1.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax1.set_axisbelow(True)

    handles = [ln1, ln2]
    labels  = [ln1.get_label(), ln2.get_label()]
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=9, framealpha=0.85)

    fig.tight_layout()
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Simple two-line dual-axis plot from a *_with_power.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",      required=True,
                    help="Path to *_with_power.csv.")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory (default: same as --input).")
    ap.add_argument("--top",        default=None, type=int,
                    help="Show only the N lowest-col1 configurations.")

    ap.add_argument("--col1",  default=DEFAULT_COL1,
                    help=f"Left-axis column  (default: {DEFAULT_COL1}).")
    ap.add_argument("--unit1", default=DEFAULT_UNIT1,
                    help=f"Display unit for col1 (default: {DEFAULT_UNIT1}).")
    ap.add_argument("--div1",  default=DEFAULT_DIV1, type=float,
                    help=f"Divide col1 values by this  (default: {DEFAULT_DIV1}).")

    ap.add_argument("--col2",  default=DEFAULT_COL2,
                    help=f"Right-axis column (default: {DEFAULT_COL2}).")
    ap.add_argument("--unit2", default=DEFAULT_UNIT2,
                    help=f"Display unit for col2 (default: {DEFAULT_UNIT2}).")
    ap.add_argument("--div2",  default=DEFAULT_DIV2, type=float,
                    help=f"Divide col2 values by this  (default: {DEFAULT_DIV2}).")

    args = ap.parse_args()

    csv_path   = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(csv_path))
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_sort(csv_path, args.col1, args.col2, args.top)
    print(f"Loaded {len(df)} configurations (sorted by {args.col1} ↑)")

    label1 = _col_label(args.col1, args.unit1)
    label2 = _col_label(args.col2, args.unit2)

    # Build a short filename suffix from the two column names
    slug = f"{args.col1}_vs_{args.col2}"
    stem = os.path.splitext(os.path.basename(csv_path))[0]

    # Interactive HTML
    html_path = os.path.join(output_dir, f"{stem}_{slug}.html")
    fig_html  = make_plotly(df, args.col1, label1, args.div1,
                                args.col2, label2, args.div2)
    fig_html.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive → {html_path}")

    # Static PNG
    png_path = os.path.join(output_dir, f"{stem}_{slug}.png")
    fig_png  = make_matplotlib(df, args.col1, label1, args.div1,
                                   args.col2, label2, args.div2)
    fig_png.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig_png)
    print(f"Saved static PNG  → {png_path}")


if __name__ == "__main__":
    main()
