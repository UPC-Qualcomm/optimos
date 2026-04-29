#!/usr/bin/env python3
"""
plot_cycles_power.py — configurable multi-axis summary plot for AstraSim power results.

Edit the PLOT CONFIGURATION section below to control which columns appear and
on which axes.  Each entry in RIGHT_AXIS_GROUPS creates one independent right
y-axis, so metrics with different units (e.g. Joules and samples/(s·MJ)) can
coexist on the same chart.

Outputs
-------
    <output-dir>/<stem>_interactive.html   — Plotly interactive figure
    <output-dir>/<stem>_static.png         — Matplotlib static figure

Usage
-----
    python plot_cycles_power.py \\
        --input  ./results/MyExp/FoldedClos_iter2_with_power.csv \\
        [--output-dir  ./results/MyExp/] \\
        [--top  40]
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
# ── PLOT CONFIGURATION — edit here to customise output ──────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Left y-axis: cycle-count series.
# Keys are CSV column names.  Set "label" and "color" per series.
LEFT_AXIS_COLS: dict = {
    "exec_cycles": {"label": "Exec Cycles",  "color": "#1f77b4"},   # muted blue
    "comm_cycles": {"label": "Comm Cycles",  "color": "#ff7f0e"},   # safety orange
}

# Right y-axis groups.
# • Each dict defines ONE independent y-axis on the right side of the chart.
# • The first group uses the primary right axis; each additional group gets
#   its own offset axis so metrics with different units can coexist.
#
# Fields per group:
#   "label"   — y-axis title (include units, e.g. "Total Energy (J)")
#   "columns" — list of CSV column names to plot on this axis
#   "colors"  — optional {column: hex} colour overrides.  Any column not
#               listed here is auto-assigned a colour from DEFAULT_PALETTE.
RIGHT_AXIS_GROUPS: list = [
    {
        "label":   "Total Energy (J)",
        "columns": [
            "A_total_energy_J",
            "B_total_energy_J",
            "C_total_energy_J",
            "D_total_energy_J",
        ],
        "colors": {
            "A_total_energy_J": "#2ca02c",   # cooked-asparagus green
            "B_total_energy_J": "#d62728",   # brick red
            "C_total_energy_J": "#9467bd",   # muted purple
            "D_total_energy_J": "#8c564b",   # chestnut brown
        },
    },
    # ── Uncomment to add a second right axis for samples/(s·MJ) ─────────────
    {
        "label":   "Samples / (s·MJ)",
        "columns": [
            "A_samples_per_sec_per_mj",
            "B_samples_per_sec_per_mj",
            "C_samples_per_sec_per_mj",
            "D_samples_per_sec_per_mj",
        ],
    },
]

# Fallback colour palette — used for series not given an explicit colour above.
DEFAULT_PALETTE: list = [
    "#17becf", "#bcbd22", "#e377c2", "#7f7f7f",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
    "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d",
]

# LPM mode A/B/C/D → descriptive labels used in the legend.
MODE_LABELS: dict = {
    "A": "No LPM (Baseline)",
    "B": "Compute LPM",
    "C": "Comm LPM",
    "D": "Full LPM",
}

# Metric suffix (after the mode prefix) → display name.
# Add entries here to customise how any suffix appears in the legend.
METRIC_LABELS: dict = {
    "total_energy_J":             "Energy (J)",
    "gpu_energy_J":               "GPU Energy (J)",
    "network_energy_J":           "Network Energy (J)",
    "total_power_W":              "Power (W)",
    "gpu_power_W":                "GPU Power (W)",
    "network_power_W":            "Network Power (W)",
    "samples_per_sec_per_mj":     "samples / (s·MJ)",
    "samples_per_joule":          "samples / J",
    "joules_per_sample":          "J / sample",
    "throughput_samples_per_sec": "Throughput (samples/s)",
}

# ════════════════════════════════════════════════════════════════════════════


# ── helpers ──────────────────────────────────────────────────────────────────

def _col_color(col: str, group: dict, palette_state: list) -> str:
    """Return colour for *col*; auto-assign from DEFAULT_PALETTE if missing."""
    colors = group.get("colors", {})
    if col in colors:
        return colors[col]
    color = DEFAULT_PALETTE[palette_state[0] % len(DEFAULT_PALETTE)]
    palette_state[0] += 1
    return color


def _col_label(col: str) -> str:
    """Human-friendly legend label for a CSV column name.

    - Mode-prefixed columns (A_*, B_*, C_*, D_*) expand the prefix via
      MODE_LABELS and clean up the metric suffix via METRIC_LABELS.
    - Other columns have underscores replaced with spaces.
    """
    for mode, desc in MODE_LABELS.items():
        prefix = f"{mode}_"
        if col.startswith(prefix):
            suffix = col[len(prefix):]
            metric = METRIC_LABELS.get(suffix, suffix.replace("_", " "))
            return f"{desc} \u2014 {metric}"
    return col.replace("_", " ")


def load_and_sort(csv_path: str) -> pd.DataFrame:
    """Load CSV, validate required columns, sort by exec_cycles ascending."""
    df = pd.read_csv(csv_path)

    required = ["dp_mp_sp_pp_sharded"] + list(LEFT_AXIS_COLS.keys())
    for g in RIGHT_AXIS_GROUPS:
        required.extend(g["columns"])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}\n"
            "Check RIGHT_AXIS_GROUPS / LEFT_AXIS_COLS in the configuration section."
        )

    df = df.sort_values("exec_cycles").reset_index(drop=True)
    return df


# ── Plotly interactive HTML ───────────────────────────────────────────────────

def make_plotly(df: pd.DataFrame, title: str = "") -> go.Figure:
    x        = df["dp_mp_sp_pp_sharded"].tolist()
    n_right  = len(RIGHT_AXIS_GROUPS)
    palette_state = [0]

    # When there are multiple right axes, shrink the x-domain to make room
    # for additional spine labels floating to the right of the plot area.
    SPINE_SPACING = 0.10
    if n_right <= 1:
        x_domain = [0, 1.0]
    else:
        x_end    = max(0.60, 1.0 - (n_right - 1) * SPINE_SPACING)
        x_domain = [0, x_end]

    # --- build layout with named yaxes -----------------------------------
    layout = dict(
        xaxis       = dict(
            title     = "dp_mp_sp_pp_sharding  (sorted by exec_cycles ↑)",
            tickangle = -50,
            tickfont  = dict(size=10),
            domain    = x_domain,
        ),
        yaxis       = dict(title="Cycles", tickformat=".3s"),
        hovermode   = "x unified",
        template    = "plotly_white",
        height      = 640,
        margin      = dict(b=170, t=80, r=max(80, 120 * n_right)),
        legend      = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1,
            font        = dict(size=11),
        ),
    )

    for i, group in enumerate(RIGHT_AXIS_GROUPS):
        ax_key  = f"yaxis{i + 2}"
        ax_dict = dict(
            title     = group["label"],
            overlaying = "y",
            side      = "right",
        )
        if i == 0:
            ax_dict["anchor"] = "x"
        else:
            # Floating axes sit progressively further to the right
            ax_dict["anchor"]   = "free"
            ax_dict["position"] = min(x_domain[1] + i * SPINE_SPACING, 1.0)
        layout[ax_key] = ax_dict

    fig = go.Figure(layout=layout)

    # --- left-axis traces (cycles) ---------------------------------------
    for col, cfg in LEFT_AXIS_COLS.items():
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x      = x,
            y      = df[col],
            name   = cfg["label"],
            mode   = "lines+markers",
            line   = dict(color=cfg["color"], width=2.5),
            marker = dict(size=7),
            yaxis  = "y",
        ))

    # --- right-axis traces -----------------------------------------------
    for i, group in enumerate(RIGHT_AXIS_GROUPS):
        yref = f"y{i + 2}"
        for col in group["columns"]:
            if col not in df.columns:
                print(f"  WARNING: column '{col}' not found in CSV — skipped")
                continue
            color = _col_color(col, group, palette_state)
            fig.add_trace(go.Scatter(
                x      = x,
                y      = df[col],
                name   = _col_label(col),
                mode   = "lines+markers",
                line   = dict(color=color, width=2, dash="dot"),
                marker = dict(size=7, symbol="diamond"),
                yaxis  = yref,
            ))

    return fig


# ── Matplotlib static PNG ─────────────────────────────────────────────────────

def make_matplotlib(df: pd.DataFrame, title: str = "") -> plt.Figure:
    n       = len(df)
    x       = range(n)
    labels  = df["dp_mp_sp_pp_sharded"].tolist()
    n_right = len(RIGHT_AXIS_GROUPS)
    palette_state = [0]

    fig, ax1 = plt.subplots(figsize=(max(14, n * 0.6), 6))

    # One twinx per right-axis group; additional axes have their spines
    # offset outward so labels don't overlap.
    axes_right = []
    for i in range(n_right):
        ax_r = ax1.twinx()
        if i > 0:
            ax_r.spines["right"].set_position(("axes", 1.0 + i * 0.12))
        axes_right.append(ax_r)

    handles = []

    # --- left-axis traces (cycles) ---------------------------------------
    for col, cfg in LEFT_AXIS_COLS.items():
        if col not in df.columns:
            continue
        ln, = ax1.plot(
            x, df[col],
            color=cfg["color"], linewidth=2.2,
            marker="o", markersize=5,
            label=cfg["label"], zorder=3,
        )
        handles.append(ln)

    # --- right-axis traces -----------------------------------------------
    for i, group in enumerate(RIGHT_AXIS_GROUPS):
        ax_r = axes_right[i]
        ax_r.set_ylabel(group["label"], fontsize=10, color="#333333")
        for col in group["columns"]:
            if col not in df.columns:
                print(f"  WARNING: column '{col}' not found in CSV — skipped")
                continue
            color = _col_color(col, group, palette_state)
            ln, = ax_r.plot(
                x, df[col],
                color=color, linewidth=2, linestyle="--",
                marker="D", markersize=5,
                label=_col_label(col), zorder=2,
            )
            handles.append(ln)

    # --- axes formatting -------------------------------------------------
    ax1.set_xlabel(
        "dp_mp_sp_pp_sharding  (sorted by exec_cycles ↑)", fontsize=10,
    )
    ax1.set_ylabel("Cycles", fontsize=10, color="#333333")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v / 1e9:.1f}s")
    )
    ax1.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax1.set_axisbelow(True)

    # Combined legend above the plot
    all_labels = [h.get_label() for h in handles]
    fig.legend(
        handles, all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(4, len(handles)),
        fontsize=9,
        framealpha=0.85,
    )

    fig.tight_layout()
    return fig


# ── Plotly toggleable HTML ────────────────────────────────────────────────────

def make_plotly_toggleable(df: pd.DataFrame) -> go.Figure:
    """Interactive Plotly figure with group-toggle buttons.

    Reuses the same traces as make_plotly and adds a row of buttons at the
    top that show/hide entire axis groups at once.  Individual traces can
    also be toggled by clicking their legend entry (double-click to isolate).
    """
    fig = make_plotly(df)
    n_traces = len(fig.data)

    # Reconstruct which trace indices belong to each group, in the same
    # order that make_plotly adds them (left-axis cols first, then right
    # groups in order).
    trace_idx    = 0
    group_indices: dict = {"cycles": []}

    for col in LEFT_AXIS_COLS:
        if col in df.columns:
            group_indices["cycles"].append(trace_idx)
            trace_idx += 1

    for i, group in enumerate(RIGHT_AXIS_GROUPS):
        key = f"group_{i}"
        group_indices[key] = []
        for col in group["columns"]:
            if col in df.columns:
                group_indices[key].append(trace_idx)
                trace_idx += 1

    def _vis(active: set) -> list:
        return [j in active for j in range(n_traces)]

    # --- toggle buttons --------------------------------------------------
    buttons = [
        dict(
            label  = "Show All",
            method = "update",
            args   = [{"visible": [True] * n_traces}],
        ),
        dict(
            label  = "Cycles",
            method = "update",
            args   = [{"visible": _vis(set(group_indices["cycles"]))}],
        ),
    ]
    for i, group in enumerate(RIGHT_AXIS_GROUPS):
        key   = f"group_{i}"
        label = group["label"]
        buttons.append(dict(
            label  = label,
            method = "update",
            args   = [{"visible": _vis(set(group_indices[key]))}],
        ))

    fig.update_layout(
        updatemenus = [dict(
            type        = "buttons",
            direction   = "right",
            pad         = {"r": 10, "t": 10},
            x           = 0.0,
            xanchor     = "left",
            y           = 1.14,
            yanchor     = "top",
            showactive  = True,
            bgcolor     = "#f4f4f4",
            bordercolor = "#cccccc",
            font        = dict(size=12),
            buttons     = buttons,
        )],
        margin = dict(t=120),
        legend = dict(
            itemclick       = "toggle",
            itemdoubleclick = "toggleothers",
        ),
        annotations = [
            dict(
                text      = "<i>Buttons filter by group · click a legend item to hide/show it · double-click to isolate</i>",
                xref      = "paper", yref = "paper",
                x=0.0, y=-0.22,
                showarrow = False,
                font      = dict(size=10, color="#888888"),
            )
        ],
    )
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot exec/comm cycles (left axis) and configurable right-axis "
            "metrics from a *_with_power.csv file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_cycles_power.py \\
      --input ./results/MyExp/FoldedClos_iter2_with_power.csv

  python plot_cycles_power.py \\
      --input      ./results/MyExp/FoldedClos_iter2_with_power.csv \\
      --output-dir ./results/MyExp/plots/ \\
      --top        30
        """,
    )
    ap.add_argument(
        "--input", required=True,
        help="Path to the *_with_power.csv produced by the power model.",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Directory for output files (default: same directory as --input).",
    )
    ap.add_argument(
        "--top", default=None, type=int,
        help="Show only the N best (lowest exec_cycles) configurations.",
    )
    args = ap.parse_args()

    csv_path   = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(csv_path))
    os.makedirs(output_dir, exist_ok=True)

    stem  = os.path.splitext(os.path.basename(csv_path))[0]
    title = stem.replace("_", " ")

    df = load_and_sort(csv_path)
    if args.top:
        df = df.head(args.top)
    print(f"Loaded {len(df)} configurations (sorted by exec_cycles ↑)")

    # Interactive HTML
    html_path  = os.path.join(output_dir, f"{stem}_interactive.html")
    fig_plotly = make_plotly(df)
    fig_plotly.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive plot    → {html_path}")

    # Toggleable HTML (group-toggle buttons + per-trace legend clicking)
    toggleable_path = os.path.join(output_dir, f"{stem}_toggleable.html")
    fig_toggleable  = make_plotly_toggleable(df)
    fig_toggleable.write_html(toggleable_path, include_plotlyjs="cdn")
    print(f"Saved toggleable plot     → {toggleable_path}")

    # Static PNG
    png_path = os.path.join(output_dir, f"{stem}_static.png")
    fig_mpl  = make_matplotlib(df)
    fig_mpl.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig_mpl)
    print(f"Saved static PNG          → {png_path}")


if __name__ == "__main__":
    main()
