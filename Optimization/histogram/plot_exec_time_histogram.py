#!/usr/bin/env python3
"""
Histogram of exec_time from a brute-force results CSV.

Usage examples:
  python plot_exec_time_histogram.py results.csv
  python plot_exec_time_histogram.py results.csv --filter-oom
  python plot_exec_time_histogram.py results.csv --oom-only
  python plot_exec_time_histogram.py results.csv --bins 30 --output my_hist.png
  python plot_exec_time_histogram.py results.csv --filter-oom --overlay-oom
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

SEQ_LENGTH = 4096
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot exec_time histogram from brute-force CSV."
    )
    parser.add_argument("csv", help="Path to the CSV file")
    parser.add_argument(
        "--filter-oom",
        action="store_true",
        help="Exclude OOM configurations (is_oom == True)",
    )
    parser.add_argument(
        "--oom-only",
        action="store_true",
        help="Show only OOM configurations",
    )
    parser.add_argument(
        "--overlay-oom",
        action="store_true",
        help="Overlay OOM and non-OOM distributions in the same plot",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: exec_time_histogram.png next to CSV)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window, just save the file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="s",
        help="Threshold in s: configs below are 'good', above are 'bad'. "
             "Adds a vertical line and percentage annotations.",
    )
    parser.add_argument(
        "--dataset-size",
        type=float,
        default=300_000_000_000,
        metavar="s",
        help="Dataset size",
    )
    parser.add_argument(
        "--total-training",
        action="store_true",
        help="calculate total training time based on dataset size, dp, and batch_size columns (overrides exec_time if present)",
    )
    parser.add_argument(
        "--zoom-max",
        type=float,
        default=None,
        metavar="s",
        help="Add an inset plot zoomed to exec_time <= this value (e.g. 1200)",
    )
    parser.add_argument(
        "--combined-stats",
        action="store_true",
        help="Show mean/median for the full (combined) data instead of per OOM/non-OOM",
    )
    return parser.parse_args()


def load_and_filter(csv_path, filter_oom, oom_only, dataset_size, total_training):
    df = pd.read_csv(csv_path)
    
    # Convert from seconds to hours for better readability on the plot
    df["exec_time"] = df["exec_time"] / 3600
    
    if total_training:
        if not {"dp", "batch_size"}.issubset(df.columns):
            sys.exit("ERROR: --total_training requires 'dp' and 'batch_size' columns in CSV.")
        print("Calculating total training time based on dataset size, dp, and batch_size...")
        global_batch_size = df['dp'] * df['batch_size']
        tokens_per_step = global_batch_size * SEQ_LENGTH
        total_num_steps = dataset_size / tokens_per_step
        df['exec_time'] = df['exec_time'] * 1e09 / total_num_steps # Scale exec_time to total training time

    #df = df[df["batch_size"].isin([8, 10, 14, 16])]
    df = df[df["inter-node-bw"] == 200]
    df = df[df["intra-node-bw"] == 900]
    df = df[df["exec_time"] <= 1000]
    #df = df[df["batch_size"] == 8]
    #df = df[df["active-chunks-per-dimension"] == 1]
    #df = df[df["collective-optimization"] == "localBWAware"]
    # Normalise column name (case-insensitive lookup)
    col_map = {c.lower(): c for c in df.columns}
    oom_col = col_map.get("is_oom")
    if oom_col is None:
        print("WARNING: 'is_oom' column not found – no OOM filtering applied.")
        return df, None

    # Coerce to bool in case values are stored as strings "True"/"False"
    df[oom_col] = df[oom_col].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )

    if filter_oom and oom_only:
        sys.exit("ERROR: --filter-oom and --oom-only are mutually exclusive.")

    if filter_oom:
        before = len(df)
        df = df[~df[oom_col]]
        print(f"Removed {before - len(df)} OOM rows. Remaining: {len(df)}")
    elif oom_only:
        before = len(df)
        df = df[df[oom_col]]
        print(f"Keeping only OOM rows: {len(df)} / {before}")

    return df, oom_col


def add_kde(ax, data, color, scale_to_counts=True, bins=50, label=None):
    """Overlay a KDE curve scaled to match histogram counts."""
    if len(data) < 2:
        return
    kde = gaussian_kde(data, bw_method="scott")
    x = np.linspace(data.min(), data.max(), 500)
    kde_y = kde(x)
    if scale_to_counts:
        # Scale density to match the y-axis (counts)
        bin_width = (data.max() - data.min()) / bins
        kde_y = kde_y * len(data) * bin_width
    #ax.plot(x, kde_y, color=color, linewidth=1.4, label=label)


MEAN_COLOR   = "#e63946"   # vivid red
MEDIAN_COLOR = "#2a9d8f"   # teal
MEAN_COLOR_OOM   = "#df979d"   # vivid red
MEDIAN_COLOR_OOM = "#8ad3ca"   # teal


def add_mean_median_lines(ax, data, prefix="", oom=False):
    """Draw vertical lines for mean and median with labels."""
    mean_val   = data.mean()
    median_val = data.median()
    mc  = MEAN_COLOR_OOM   if oom else MEAN_COLOR
    mdc = MEDIAN_COLOR_OOM if oom else MEDIAN_COLOR
    ax.axvline(mean_val,   color=mc,  linewidth=1.3, linestyle="--",
               label=f"{prefix}Mean = {mean_val:.0f} s")
    #ax.axvline(median_val, color=mdc, linewidth=1.3, linestyle="-.",
    #           label=f"{prefix}Median = {median_val:.0f} s")


def add_threshold_annotation(ax, data, threshold, prefix=""):
    """Shade good/bad regions and annotate with percentages."""
    n_good = (data < threshold).sum()
    n_bad  = (data >= threshold).sum()
    n_total = len(data)
    pct_good = 100.0 * n_good / n_total if n_total else 0.0
    pct_bad  = 100.0 * n_bad  / n_total if n_total else 0.0

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    # Shade regions
    ax.axvspan(xmin, threshold, alpha=0.08, color="green", zorder=0)
    ax.axvspan(threshold, xmax,  alpha=0.08, color="red",   zorder=0)

    # Threshold line
    tag = f"{prefix}" if prefix else ""
    ax.axvline(threshold, color="#555555", linewidth=1.2, linestyle=":",
               label=f"{tag}Threshold = {threshold:.0f} s")

    # Percentage labels inside each region
    good_x = xmin + (threshold - xmin) * 0.5
    bad_x  = threshold + (xmax - threshold) * 0.5
    label_y = ymax * 0.80

    ax.text(good_x, label_y,
            f"{tag}Good\n{pct_good:.1f}%\n(n={n_good:,})",
            ha="center", va="top", fontsize=6, color="darkgreen",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))
    ax.text(bad_x, label_y,
            f"{tag}Bad\n{pct_bad:.1f}%\n(n={n_bad:,})",
            ha="center", va="top", fontsize=6, color="darkred",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

    print(f"{tag}Threshold {threshold:.0f} s — "
          f"Good: {n_good}/{n_total} ({pct_good:.1f}%)  "
          f"Bad: {n_bad}/{n_total} ({pct_bad:.1f}%)")


def plot_single(ax, data, bins, color, label=None, title=None, threshold=None):
    ax.hist(data, bins=bins, edgecolor="black", linewidth=0.4, color=color, alpha=0.65,
            label=label if label is not None else f"n={len(data):,}")
    add_kde(ax, data, color=color, bins=bins, label="KDE")
    add_mean_median_lines(ax, data, prefix="", oom=False)
    ax.set_xlabel("Training Time (s)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    if title:
        ax.set_title(title, fontsize=14)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=11, loc="upper left")
    if threshold is not None:
        add_threshold_annotation(ax, data, threshold)


def main():
    args = parse_args()

    df, oom_col = load_and_filter(args.csv, args.filter_oom, args.oom_only, args.dataset_size, args.total_training)
    if "exec_time" not in df.columns:
        sys.exit("ERROR: 'exec_time' column not found in CSV.")
    
    out_path = args.output
    if out_path is None:
        import os
        base = os.path.splitext(args.csv)[0]
        suffix = "_oom_only" if args.oom_only else ("_no_oom" if args.filter_oom else "")
        out_path = f"{base}_exec_time_histogram{suffix}.png"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
    })

    if args.overlay_oom and oom_col is not None:
        # Two overlaid distributions
        fig, ax = plt.subplots(figsize=(11, 6))
        non_oom = df[~df[oom_col]]["exec_time"].dropna()
        oom_data = df[df[oom_col]]["exec_time"].dropna()

        bins_range = (df["exec_time"].min(), df["exec_time"].max())
        bins = pd.cut(
            df["exec_time"], bins=args.bins, retbins=True
        )[1]

        ax.hist(non_oom, bins=bins, color="steelblue", alpha=0.50,
                edgecolor="steelblue", linewidth=0.3, label=f"Non-OOM (n={len(non_oom):,})")
        ax.hist(oom_data, bins=bins, color="tomato", alpha=0.50,
                edgecolor="tomato", linewidth=0.3, label=f"OOM (n={len(oom_data):,})")
        if len(non_oom) >= 2:
            add_kde(ax, non_oom, color="steelblue", bins=args.bins, label="KDE Non-OOM")
        if len(oom_data) >= 2:
            add_kde(ax, oom_data, color="tomato", bins=args.bins, label="KDE OOM")
        if args.combined_stats:
            combined = df["exec_time"].dropna()
            if len(combined) >= 1:
                add_mean_median_lines(ax, combined, prefix="All ", oom=False)
        else:
            if len(non_oom) >= 1:
                add_mean_median_lines(ax, non_oom, prefix="Non-OOM ", oom=False)
            if len(oom_data) >= 1:
                add_mean_median_lines(ax, oom_data, prefix="OOM ", oom=True)
        if args.threshold is not None:
            # Show threshold split for the combined data visible in the plot
            combined = df["exec_time"].dropna()
            add_threshold_annotation(ax, combined, args.threshold)

        ax.set_xlabel("Training Time (s)", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title("Training time Distribution", fontsize=14)
        ax.grid(axis="y", alpha=0.35)
        ax.legend(fontsize=9, loc="upper left")

        # ---- Inset (zoomed) plot ----
        if args.zoom_max is not None:
            zoom_limit = args.zoom_max
            non_oom_z = non_oom[non_oom <= zoom_limit]
            oom_z = oom_data[oom_data <= zoom_limit]
            if len(non_oom_z) + len(oom_z) > 0:
                # Draw a rectangle on the main axes to highlight the zoomed region
                ax.axvline(zoom_limit, color="gray", linewidth=1.2, linestyle="--", alpha=0.6)

                # Create inset axes (top-right corner)
                ax_ins = ax.inset_axes([0.52, 0.42, 0.45, 0.52])  # top-right inset
                zoom_bins = np.linspace(
                    min(non_oom_z.min() if len(non_oom_z) else zoom_limit,
                        oom_z.min() if len(oom_z) else zoom_limit),
                    zoom_limit, min(args.bins, 40))
                if len(non_oom_z):
                    ax_ins.hist(non_oom_z, bins=zoom_bins, color="steelblue", alpha=0.55,
                                edgecolor="steelblue", linewidth=0.3)
                    add_kde(ax_ins, non_oom_z, color="steelblue", bins=len(zoom_bins)-1)
                if len(oom_z):
                    ax_ins.hist(oom_z, bins=zoom_bins, color="tomato", alpha=0.55,
                                edgecolor="tomato", linewidth=0.3)
                    add_kde(ax_ins, oom_z, color="tomato", bins=len(zoom_bins)-1)
                ax_ins.set_xlim(zoom_bins[0], zoom_limit)
                #ax_ins.set_title(f"Zoom: exec_time ≤ {zoom_limit:.0f} s", fontsize=9)
                #ax_ins.set_xlabel("exec_time (s)", fontsize=8)
                #ax_ins.set_ylabel("Count", fontsize=8)
                ax_ins.tick_params(labelsize=10)
                ax_ins.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                ax_ins.grid(axis="y", alpha=0.3)

                # Connector lines from main axes to inset
                rect, connectors = ax.indicate_inset_zoom(ax_ins, edgecolor="gray", linewidth=0.8, alpha=0.6)
                rect.set_clip_on(True)
                rect.set_clip_box(ax.bbox)
                connectors[1].set_visible(False)
                connectors[3].set_visible(False)
                connectors[2].set_visible(True)

    else:
        data = df["exec_time"].dropna()

        # Decide title based on filter mode
        if args.filter_oom:
            title = "Training time Distribution (OOM configs excluded)"
            color = "steelblue"
            data_label = f"Non-OOM (n={len(data):,})"
        elif args.oom_only:
            title = "Training time Distribution (OOM configs only)"
            color = "tomato"
            data_label = f"OOM (n={len(data):,})"
        else:
            title = "Training time Distribution (all configs)"
            color = "steelblue"
            data_label = f"All (n={len(data):,})"

        fig, ax = plt.subplots(figsize=(11, 6))
        plot_single(ax, data, bins=args.bins, color=color, label=data_label, title=title, threshold=args.threshold)

        # ---- Inset (zoomed) plot ----
        if args.zoom_max is not None:
            zoom_limit = args.zoom_max
            data_z = data[data <= zoom_limit]
            if len(data_z) > 0:
                ax.axvline(zoom_limit, color="gray", linewidth=1.2, linestyle="--", alpha=0.6)
                ax_ins = ax.inset_axes([0.52, 0.42, 0.45, 0.52])
                zoom_bins = np.linspace(data_z.min(), zoom_limit, min(args.bins, 40))
                ax_ins.hist(data_z, bins=zoom_bins, color=color, alpha=0.55,
                            edgecolor=color, linewidth=0.3)
                add_kde(ax_ins, data_z, color=color, bins=len(zoom_bins)-1)
                ax_ins.set_xlim(zoom_bins[0], zoom_limit)
                ax_ins.tick_params(labelsize=10)
                ax_ins.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                ax_ins.grid(axis="y", alpha=0.3)
                rect, connectors = ax.indicate_inset_zoom(ax_ins, edgecolor="gray", linewidth=0.8, alpha=0.6)
                rect.set_clip_on(True)
                rect.set_clip_box(ax.bbox)
                for c in connectors:
                    c.set_visible(False)

        plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
