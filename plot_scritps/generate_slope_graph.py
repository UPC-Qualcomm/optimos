#!/usr/bin/env python3
"""
Create a slope graph showing order changes between the simulation dataset and analytical results.
Usage: python generate_slope_graph.py [--g2 <g2_csv_path>] --analytical <analytical_csv_path> [--ns3 <ns3_csv_path>] [--title <title>]
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import sys
import os

def _kendall_stats(ranks_a, ranks_b):
    """Return (kendall_corr, kendall_distance, total_pairs, normalized_distance) for two rank arrays."""
    n = min(len(ranks_a), len(ranks_b))
    total_pairs = n * (n - 1) / 2
    print(len(ranks_a), len(ranks_b), n, total_pairs)
    ktau, _ = kendalltau(ranks_a, ranks_b)
    dist = int(total_pairs * (1 - ktau) / 2)
    norm_dist = dist / total_pairs if total_pairs > 0 else 0
    return ktau, dist, int(total_pairs), norm_dist


def generate_slope_graph(g2_file=None, analytical_file=None, title="default title", output_path="order_comparison",
                         output_filename=None, ns3_file=None):
    """Generate slope graph comparing the active simulation dataset and analytical orderings.

    Args:
        g2_file: Optional path to G2 results CSV
        analytical_file: Path to analytical results CSV
        title: Graph title
        output_path: Output directory path (default: current directory)
        output_filename: Custom filename for output (without extension). If None, auto-generates from input paths.
        ns3_file: (optional) Path to NS3 results CSV. When provided a third column is added.
    """
    has_g2 = g2_file is not None
    has_ns3 = ns3_file is not None
    comparison_label = "G2" if has_g2 else "NS3"

    if not has_g2 and not has_ns3:
        raise ValueError("Provide --g2 or --ns3 so the plot has a comparison dataset.")

    # If G2 is omitted, reuse NS3 as the comparison dataset and fall back to a two-column plot.
    if not has_g2 and has_ns3:
        g2_file = ns3_file
        has_ns3 = False

    # ------------------------------------------------------------------ #
    # 1. Read & filter data
    # ------------------------------------------------------------------ #
    print(f"Reading data from:\n  {comparison_label}: {g2_file}\n  Analytical: {analytical_file}")
    if has_ns3:
        print(f"  NS3: {ns3_file}")
    df_analytical = pd.read_csv(analytical_file)
    df_g2 = pd.read_csv(g2_file)
    df_ns3 = pd.read_csv(ns3_file) if has_ns3 else None

    print(f"Analytical data (before OOM filter): {len(df_analytical)} configurations")
    print(f"{comparison_label} data (before OOM filter): {len(df_g2)} configurations")
    if has_ns3:
        print(f"NS3 data (before OOM filter): {len(df_ns3)} configurations")

    if 'is_oom' in df_analytical.columns:
        df_analytical = df_analytical[df_analytical['is_oom'] != True].reset_index(drop=True)
        print(f"Analytical data (after OOM filter): {len(df_analytical)} configurations")

    if 'is_oom' in df_g2.columns:
        df_g2 = df_g2[df_g2['is_oom'] != True].reset_index(drop=True)
        print(f"{comparison_label} data (after OOM filter): {len(df_g2)} configurations")

    if has_ns3 and 'is_oom' in df_ns3.columns:
        df_ns3 = df_ns3[df_ns3['is_oom'] != True].reset_index(drop=True)
        print(f"NS3 data (after OOM filter): {len(df_ns3)} configurations")

    # ------------------------------------------------------------------ #
    # 1b. Keep only configurations common to ALL datasets
    # ------------------------------------------------------------------ #
    if has_ns3:
        common_configs = (set(df_g2['dp_mp_sp_pp_sharded']) &
                          set(df_analytical['dp_mp_sp_pp_sharded']) &
                          set(df_ns3['dp_mp_sp_pp_sharded']))
        print(f"\nCommon configurations (G2 ∩ NS3 ∩ Analytical): {len(common_configs)}")
    else:
        common_configs = (set(df_g2['dp_mp_sp_pp_sharded']) &
                          set(df_analytical['dp_mp_sp_pp_sharded']))
        print(f"\nCommon configurations ({comparison_label} ∩ Analytical): {len(common_configs)}")

    df_g2 = df_g2[df_g2['dp_mp_sp_pp_sharded'].isin(common_configs)].reset_index(drop=True)
    df_analytical = df_analytical[df_analytical['dp_mp_sp_pp_sharded'].isin(common_configs)].reset_index(drop=True)
    if has_ns3:
        df_ns3 = df_ns3[df_ns3['dp_mp_sp_pp_sharded'].isin(common_configs)].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 2. Sort & rank
    # ------------------------------------------------------------------ #
    df_analytical_sorted = df_analytical.sort_values('exec_cycles').reset_index(drop=True)
    df_g2_sorted = df_g2.sort_values('exec_cycles').reset_index(drop=True)
    df_analytical_sorted['analytical_rank'] = range(1, len(df_analytical_sorted) + 1)
    df_g2_sorted['g2_rank'] = range(1, len(df_g2_sorted) + 1)

    if has_ns3:
        df_ns3_sorted = df_ns3.sort_values('exec_cycles').reset_index(drop=True)
        df_ns3_sorted['ns3_rank'] = range(1, len(df_ns3_sorted) + 1)

    # ------------------------------------------------------------------ #
    # 3. Normalize to 0-100
    # ------------------------------------------------------------------ #
    def _normalize(df):
        lo, hi = df['exec_cycles'].min(), df['exec_cycles'].max()
        df = df.copy()
        df['normalized_cycles'] = 100 * (df['exec_cycles'] - lo) / (hi - lo) if hi > lo else 0.0
        return df, lo, hi

    df_g2_sorted, g2_min, g2_max = _normalize(df_g2_sorted)
    df_analytical_sorted, analytical_min, analytical_max = _normalize(df_analytical_sorted)
    g2_range = g2_max - g2_min
    analytical_range = analytical_max - analytical_min
    relative_scale_factor = g2_range / analytical_range if analytical_range > 0 else 1.0

    if has_ns3:
        df_ns3_sorted, ns3_min, ns3_max = _normalize(df_ns3_sorted)
        ns3_range = ns3_max - ns3_min
        ns3_ana_scale_factor = ns3_range / analytical_range if analytical_range > 0 else 1.0
        ns3_g2_scale_factor = ns3_range / g2_range if g2_range > 0 else 1.0

    # ------------------------------------------------------------------ #
    # 4. Pairwise merges & Kendall distances
    # ------------------------------------------------------------------ #
    # Comparison dataset vs Analytical (original)
    merged_g2_ana = pd.merge(
        df_analytical_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'analytical_rank', 'normalized_cycles']],
        df_g2_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'g2_rank', 'normalized_cycles']],
        on='dp_mp_sp_pp_sharded', suffixes=('_analytical', '_g2')
    )
    merged_g2_ana['rank_diff'] = merged_g2_ana['analytical_rank'] - merged_g2_ana['g2_rank']
    merged_g2_ana['abs_rank_diff'] = merged_g2_ana['rank_diff'].abs()
    merged_g2_ana['cycles_diff'] = (merged_g2_ana['exec_cycles_analytical'] - merged_g2_ana['exec_cycles_g2']).abs()
    merged_g2_ana['normalized_diff'] = (merged_g2_ana['normalized_cycles_analytical'] - merged_g2_ana['normalized_cycles_g2']).abs()

    spearman_corr_ana_g2, _ = spearmanr(merged_g2_ana['analytical_rank'], merged_g2_ana['g2_rank'])
    ktau_g2_ana, kdist_g2_ana, kpairs_g2_ana, knorm_g2_ana = _kendall_stats(
        merged_g2_ana['analytical_rank'].values, merged_g2_ana['g2_rank'].values)

    # NS3 pairwise stats (only when ns3_file supplied)
    if has_ns3:
        # NS3 vs G2
        merged_ns3_g2 = pd.merge(
            df_ns3_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'ns3_rank', 'normalized_cycles']],
            df_g2_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'g2_rank', 'normalized_cycles']],
            on='dp_mp_sp_pp_sharded', suffixes=('_ns3', '_g2')
        )
        ktau_ns3_g2, kdist_ns3_g2, kpairs_ns3_g2, knorm_ns3_g2 = _kendall_stats(
            merged_ns3_g2['ns3_rank'].values, merged_ns3_g2['g2_rank'].values)
        spearman_corr_g2_n3, _ = spearmanr(merged_ns3_g2['g2_rank'], merged_ns3_g2['ns3_rank'])

        # NS3 vs Analytical
        merged_ns3_ana = pd.merge(
            df_ns3_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'ns3_rank', 'normalized_cycles']],
            df_analytical_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'analytical_rank', 'normalized_cycles']],
            on='dp_mp_sp_pp_sharded', suffixes=('_ns3', '_analytical')
        )
        ktau_ns3_ana, kdist_ns3_ana, kpairs_ns3_ana, knorm_ns3_ana = _kendall_stats(
            merged_ns3_ana['ns3_rank'].values, merged_ns3_ana['analytical_rank'].values)
        spearman_corr_ana_ns3, _ = spearmanr(merged_ns3_ana['analytical_rank'], merged_ns3_ana['ns3_rank'])

    # ------------------------------------------------------------------ #
    # 5. Print statistics
    # ------------------------------------------------------------------ #
    n_configs = len(merged_g2_ana)
    print(f"Merged {comparison_label} ∩ Analytical configurations: {n_configs}")

    print(f"\n{'='*80}")
    print(f"RANKING CORRELATION METRICS")
    print(f"{'='*80}")
    print(f"\n--- {comparison_label} vs Analytical ({n_configs} common configs) ---")
    print(f"Spearman ρ: {spearman_corr_ana_g2:.6f}")
    print(f"Kendall τ: {ktau_g2_ana:.6f}")
    print(f"Kendall Distance: {kdist_g2_ana} discordant pairs (out of {kpairs_g2_ana})")
    print(f"Normalized Kendall Distance: {knorm_g2_ana:.4f}  ({(1-knorm_g2_ana)*100:.1f}% concordant)")
    print(f"Mean |Δ rank|: {merged_g2_ana['abs_rank_diff'].mean():.2f}")

    if has_ns3:
        print(f"\n--- NS3 vs G2 ({len(merged_ns3_g2)} common configs) ---")
        print(f"Kendall τ: {ktau_ns3_g2:.6f}")
        print(f"Kendall Distance: {kdist_ns3_g2} discordant pairs (out of {kpairs_ns3_g2})")
        print(f"Normalized Kendall Distance: {knorm_ns3_g2:.4f}  ({(1-knorm_ns3_g2)*100:.1f}% concordant)")

        print(f"\n--- NS3 vs Analytical ({len(merged_ns3_ana)} common configs) ---")
        print(f"Kendall τ: {ktau_ns3_ana:.6f}")
        print(f"Kendall Distance: {kdist_ns3_ana} discordant pairs (out of {kpairs_ns3_ana})")
        print(f"Normalized Kendall Distance: {knorm_ns3_ana:.4f}  ({(1-knorm_ns3_ana)*100:.1f}% concordant)")

    print(f"\n{'='*80}")
    print(f"SCALING FACTOR EXPLANATION")
    print(f"{'='*80}")
    print(f"{comparison_label} range: {g2_range:,.0f} cycles (from {g2_min:,.0f} to {g2_max:,.0f})")
    print(f"Analytical range: {analytical_range:,.0f} cycles (from {analytical_min:,.0f} to {analytical_max:,.0f})")
    if has_ns3:
        print(f"NS3 range: {ns3_range:,.0f} cycles (from {ns3_min:,.0f} to {ns3_max:,.0f})")
    print(f"Scaling Factor = {comparison_label} range / Analytical range = {relative_scale_factor:.4f}")
    if has_ns3:
        print(f"Scaling Factor = NS3 range / Analytical range = {ns3_ana_scale_factor:.4f}")
        print(f"Scaling Factor = NS3 range / {comparison_label} range        = {ns3_g2_scale_factor:.4f}")
    print(f"\nInterpretation ({comparison_label} vs Analytical):")
    if relative_scale_factor > 1.05:
        print(f"  → {comparison_label} has {relative_scale_factor:.2f}x MORE variability in results than Analytical")
        print(f"  → {comparison_label} shows a WIDER spread of execution times across configurations")
    elif relative_scale_factor < 0.95:
        print(f"  → {comparison_label} has {1/relative_scale_factor:.2f}x LESS variability than Analytical")
        print(f"  → {comparison_label} shows a NARROWER spread of execution times across configurations")
    else:
        print(f"  → Both methods have approximately the SAME variability in results")
    if has_ns3:
        print(f"\nInterpretation (NS3 vs Analytical):")
        if ns3_ana_scale_factor > 1.05:
            print(f"  → NS3 has {ns3_ana_scale_factor:.2f}x MORE variability than Analytical")
        elif ns3_ana_scale_factor < 0.95:
            print(f"  → NS3 has {1/ns3_ana_scale_factor:.2f}x LESS variability than Analytical")
        else:
            print(f"  → NS3 and Analytical have approximately the SAME variability")
    print(f"\nNote: To compare actual execution times, both datasets are normalized to 0-100 scale")
    print(f"      where 0 = best (lowest exec_cycles) and 100 = worst (highest exec_cycles)")

    # ------------------------------------------------------------------ #
    # 6. Build the slope graph
    # ------------------------------------------------------------------ #
    # Function to detect and adjust overlapping points
    def adjust_overlaps(values, min_distance=1.5):
        """Adjust overlapping values to maintain minimum distance."""
        adjusted = values.copy()
        sorted_indices = np.argsort(adjusted)
        for i in range(1, len(sorted_indices)):
            curr_idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            if abs(adjusted[curr_idx] - adjusted[prev_idx]) < min_distance:
                adjusted[curr_idx] = adjusted[prev_idx] + min_distance
        return adjusted

    # Column x-positions
    if has_ns3:
        g2_x = 0
        ns3_x = 1
        analytical_x = 2
        fig_width = 20
    else:
        g2_x = 0
        analytical_x = 1
        fig_width = 14

    fig_height = max(20, n_configs * 0.15)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Color palette – one unique color per configuration (union of all datasets)
    all_configs_union = (set(df_g2_sorted['dp_mp_sp_pp_sharded']) |
                         set(df_analytical_sorted['dp_mp_sp_pp_sharded']))
    if has_ns3:
        all_configs_union |= set(df_ns3_sorted['dp_mp_sp_pp_sharded'])
    all_configs_union = sorted(all_configs_union)

    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    colors_tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))
    all_colors = np.vstack([colors_tab20, colors_tab20b, colors_tab20c])
    config_to_color = {cfg: all_colors[i % len(all_colors)] for i, cfg in enumerate(all_configs_union)}

    # Adjusted y-positions per column
    g2_norm_adj = adjust_overlaps(df_g2_sorted['normalized_cycles'].values.copy())
    ana_norm_adj = adjust_overlaps(df_analytical_sorted['normalized_cycles'].values.copy())
    g2_config_to_pos = dict(zip(df_g2_sorted['dp_mp_sp_pp_sharded'].values, g2_norm_adj))
    ana_config_to_pos = dict(zip(df_analytical_sorted['dp_mp_sp_pp_sharded'].values, ana_norm_adj))

    if has_ns3:
        ns3_norm_adj = adjust_overlaps(df_ns3_sorted['normalized_cycles'].values.copy())
        ns3_config_to_pos = dict(zip(df_ns3_sorted['dp_mp_sp_pp_sharded'].values, ns3_norm_adj))

    # ---- Draw connecting lines ----
    # G2 → NS3
    if has_ns3:
        common_g2_ns3 = set(df_g2_sorted['dp_mp_sp_pp_sharded']) & set(df_ns3_sorted['dp_mp_sp_pp_sharded'])
        for cfg in common_g2_ns3:
            ax.plot([g2_x, ns3_x], [g2_config_to_pos[cfg], ns3_config_to_pos[cfg]],
                    color=config_to_color[cfg], linewidth=1.5, alpha=0.7, zorder=1)

    # NS3 → Analytical (or G2 → Analytical when no NS3)
    if has_ns3:
        common_ns3_ana = set(df_ns3_sorted['dp_mp_sp_pp_sharded']) & set(df_analytical_sorted['dp_mp_sp_pp_sharded'])
        for cfg in common_ns3_ana:
            ax.plot([ns3_x, analytical_x], [ns3_config_to_pos[cfg], ana_config_to_pos[cfg]],
                    color=config_to_color[cfg], linewidth=1.5, alpha=0.7, zorder=1)
    else:
        common_g2_ana = set(df_g2_sorted['dp_mp_sp_pp_sharded']) & set(df_analytical_sorted['dp_mp_sp_pp_sharded'])
        for cfg in common_g2_ana:
            ax.plot([g2_x, analytical_x], [g2_config_to_pos[cfg], ana_config_to_pos[cfg]],
                    color=config_to_color[cfg], linewidth=1.5, alpha=0.7, zorder=1)

    # ---- Draw dots ----
    dot_size = 50

    def _draw_dots(df_sorted, config_to_pos, x_pos):
        for _, row in df_sorted.iterrows():
            cfg = row['dp_mp_sp_pp_sharded']
            color = config_to_color.get(cfg, 'gray')
            ax.scatter([x_pos], [config_to_pos[cfg]], s=dot_size, color=color,
                       edgecolors='black', linewidth=0.5, zorder=2, alpha=0.9)

    if has_ns3:
        _draw_dots(df_ns3_sorted, ns3_config_to_pos, ns3_x)
    _draw_dots(df_g2_sorted, g2_config_to_pos, g2_x)
    _draw_dots(df_analytical_sorted, ana_config_to_pos, analytical_x)

    # ---- Annotations (all configs) ----
    top_n = len(merged_g2_ana)

    # Left side: G2 (always leftmost)
    left_col_df = df_g2_sorted
    left_col_pos = g2_config_to_pos
    left_x_label = g2_x

    for i, (_, row) in enumerate(left_col_df.head(top_n).iterrows()):
        cfg = row['dp_mp_sp_pp_sharded']
        ax.text(left_x_label - 0.05, left_col_pos[cfg],
                f"{cfg} ({row['exec_cycles']:,.0f})", ha='right', va='center',
                fontsize=7, fontweight='bold' if i < 5 else 'normal')

    # Right side: Analytical
    for i, (_, row) in enumerate(df_analytical_sorted.head(top_n).iterrows()):
        cfg = row['dp_mp_sp_pp_sharded']
        ax.text(analytical_x + 0.05, ana_config_to_pos[cfg],
                f"{cfg} ({row['exec_cycles']:,.0f})", ha='left', va='center',
                fontsize=7, fontweight='bold' if i < 5 else 'normal')

    # Middle (NS3) annotations – right-aligned to the left of the NS3 dot, only when NS3 is present
    if has_ns3:
        for i, (_, row) in enumerate(df_ns3_sorted.head(top_n).iterrows()):
            cfg = row['dp_mp_sp_pp_sharded']
            ax.text(ns3_x - 0.05, ns3_config_to_pos[cfg],
                    f"{cfg} ({row['exec_cycles']:,.0f})", ha='right', va='center',
                    fontsize=7, fontweight='bold' if i < 5 else 'normal')

    # ---- Axes ----
    all_adj_arrays = [g2_norm_adj, ana_norm_adj]
    if has_ns3:
        all_adj_arrays.append(ns3_norm_adj)
    max_adjusted = max(arr.max() for arr in all_adj_arrays)
    y_limit_top = max_adjusted + 5

    x_left_lim = g2_x - 0.9
    x_right_lim = analytical_x + 0.9
    ax.set_xlim([x_left_lim, x_right_lim])
    ax.set_ylim([y_limit_top, -5])

    x_ticks = ([g2_x, ns3_x, analytical_x] if has_ns3 else [g2_x, analytical_x])
    x_labels = (['G2\n(Simulation)', 'NS3\n(Packet Sim)', 'Analytical'] if has_ns3
                else ['G2\n(Simulation)', 'Analytical'])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Execution Cycles (0=Best, 100=Worst)', fontsize=13, fontweight='bold')
    ax.set_yticks(range(0, int(y_limit_top) + 1, 10))
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    ax2 = ax.twinx()
    ax2.set_ylim([y_limit_top, -5])
    ax2.set_yticks([])

    # ---- Range boxes below plot ----
    box_y_position = y_limit_top + 10
    ax.text(g2_x, box_y_position,
            f'{comparison_label} Range:\n{g2_min:,.0f} - {g2_max:,.0f}\ncycles',
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7), family='monospace')
    ax.text(analytical_x, box_y_position,
            f'Analytical Range:\n{analytical_min:,.0f} - {analytical_max:,.0f}\ncycles',
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), family='monospace')
    if has_ns3:
        ax.text(ns3_x, box_y_position,
                f'NS3 Range:\n{ns3_min:,.0f} - {ns3_max:,.0f}\ncycles',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.7), family='monospace')

    # ---- Title ----
    scale_interpretation = ("same range" if 0.95 <= relative_scale_factor <= 1.05 else
                             f"{comparison_label} is {relative_scale_factor:.2f}x wider" if relative_scale_factor > 1.05 else
                             f"{comparison_label} is {1/relative_scale_factor:.2f}x smaller")
    title_text = f'{title}\n'
    title_text += (f'{comparison_label} vs Analytical — Spearman ρ={spearman_corr_ana_g2:.4f} | '
                   f'Kendall τ={ktau_g2_ana:.4f} | '
                   f'Kendall Dist={kdist_g2_ana}/{kpairs_g2_ana} ({knorm_g2_ana:.3f})\n')
    if has_ns3:
        title_text += (f'NS3 vs G2 — 'f"{f'Spearman ρ={spearman_corr_g2_n3:.4f}' if spearman_corr_g2_n3 else ''} | "
                       f'Kendall τ={ktau_ns3_g2:.4f} | '
                       f'Kendall Dist={kdist_ns3_g2}/{kpairs_ns3_g2} ({knorm_ns3_g2:.3f})\n'
                       f'NS3 vs Analytical — 'f"{f'Spearman ρ={spearman_corr_ana_ns3:.4f}' if spearman_corr_ana_ns3 else ''} | "
                       f'Kendall τ={ktau_ns3_ana:.4f} | '
                       f'Kendall Dist={kdist_ns3_ana}/{kpairs_ns3_ana} ({knorm_ns3_ana:.3f})\n')
    title_text += f'Scale Factor {comparison_label}/Analytical: {relative_scale_factor:.4f} ({scale_interpretation})'
    if has_ns3:
        ns3_ana_interp = ("same range" if 0.95 <= ns3_ana_scale_factor <= 1.05 else
                          f"NS3 is {ns3_ana_scale_factor:.2f}x wider" if ns3_ana_scale_factor > 1.05 else
                          f"NS3 is {1/ns3_ana_scale_factor:.2f}x smaller")
        title_text += f'   |   NS3/Analytical: {ns3_ana_scale_factor:.4f} ({ns3_ana_interp})'
    title_text += '\n(Each configuration has a distinct color; overlapping points adjusted for visibility)'
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=25)

    # ---- Legend / stats box ----
    legend_text = 'Statistics:\n'
    legend_text += f'• {comparison_label} ∩ Analytical configs: {n_configs}\n'
    legend_text += f'• {comparison_label} range: {g2_range:,.0f} cycles\n'
    legend_text += f'• Analytical range: {analytical_range:,.0f} cycles\n'
    legend_text += f'• Scale factor ({comparison_label}/Analytical): {relative_scale_factor:.4f}x\n'
    if has_ns3:
        legend_text += f'• Scale factor (NS3/Analytical): {ns3_ana_scale_factor:.4f}x\n'
        legend_text += f'• Scale factor (NS3/{comparison_label}): {ns3_g2_scale_factor:.4f}x\n'
    legend_text += f'─── {comparison_label} vs Analytical ───\n'
    legend_text += f'• Kendall τ: {ktau_g2_ana:.4f} | Dist: {kdist_g2_ana}/{kpairs_g2_ana} ({(1-knorm_g2_ana)*100:.1f}% concordant)\n'
    legend_text += f'• Mean |Δ rank|: {merged_g2_ana["abs_rank_diff"].mean():.1f}\n'
    if has_ns3:
        legend_text += f'─── NS3 vs G2 ({len(merged_ns3_g2)} configs) ───\n'
        legend_text += f'• Kendall τ: {ktau_ns3_g2:.4f} | Dist: {kdist_ns3_g2}/{kpairs_ns3_g2} ({(1-knorm_ns3_g2)*100:.1f}% concordant)\n'
        legend_text += f'─── NS3 vs Analytical ({len(merged_ns3_ana)} configs) ───\n'
        legend_text += f'• Kendall τ: {ktau_ns3_ana:.4f} | Dist: {kdist_ns3_ana}/{kpairs_ns3_ana} ({(1-knorm_ns3_ana)*100:.1f}% concordant)\n'

    legend_x = (g2_x + analytical_x) / 2 if has_ns3 else 0.5
    ax.text(legend_x, y_limit_top + 3, legend_text,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            family='monospace')

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    # 7. Save output
    # ------------------------------------------------------------------ #
    output_dir = output_path if output_path is not None else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if output_filename is None:
        g2_basename = os.path.basename(os.path.dirname(g2_file))
        analytical_basename = os.path.basename(os.path.dirname(analytical_file))
        output_file = os.path.join(output_dir, f'slope_graph_{g2_basename}_vs_{analytical_basename}.png')
    else:
        if not output_filename.endswith('.png'):
            output_filename = output_filename + '.png'
        output_file = os.path.join(output_dir, output_filename)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSlope graph saved to: {output_file}")

    base_filename = os.path.splitext(output_file)[0]
    output_csv = base_filename + '.csv'

    # ------------------------------------------------------------------ #
    # 8. Print top configs & findings
    # ------------------------------------------------------------------ #
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    print("\nANALYTICAL Top 10:")
    for i, row in df_analytical_sorted.head(10).iterrows():
        print(f"  {i+1:2d}. {row['dp_mp_sp_pp_sharded']:20s} - {row['exec_cycles']:,} cycles")
    print(f"\n{comparison_label} Top 10:")
    for i, row in df_g2_sorted.head(10).iterrows():
        print(f"  {i+1:2d}. {row['dp_mp_sp_pp_sharded']:20s} - {row['exec_cycles']:,} cycles")
    if has_ns3:
        print("\nNS3 Top 10:")
        for i, row in df_ns3_sorted.head(10).iterrows():
            print(f"  {i+1:2d}. {row['dp_mp_sp_pp_sharded']:20s} - {row['exec_cycles']:,} cycles")

    print("\n" + "="*80)
    print(f"INTERESTING FINDINGS ({comparison_label} vs Analytical)")
    print("="*80)
    improved_in_g2 = merged_g2_ana.nsmallest(5, 'rank_diff')
    print(f"\nConfigurations that ranked BETTER in {comparison_label} (improved the most):")
    for _, row in improved_in_g2.iterrows():
        print(f"  {row['dp_mp_sp_pp_sharded']:20s}: Rank {row['analytical_rank']:2.0f} → {row['g2_rank']:2.0f} (improved by {-row['rank_diff']:.0f})")
    degraded_in_g2 = merged_g2_ana.nlargest(5, 'rank_diff')
    print(f"\nConfigurations that ranked WORSE in {comparison_label} (degraded the most):")
    for _, row in degraded_in_g2.iterrows():
        print(f"  {row['dp_mp_sp_pp_sharded']:20s}: Rank {row['analytical_rank']:2.0f} → {row['g2_rank']:2.0f} (degraded by {row['rank_diff']:.0f})")
    stable_configs = merged_g2_ana[merged_g2_ana['abs_rank_diff'] <= 1].sort_values('analytical_rank')
    print(f"\nConfigurations with stable ranking (diff ≤ 1 positions): {len(stable_configs)}/{n_configs}")

    merged_g2_ana.sort_values('analytical_rank').to_csv(output_csv, index=False)
    print(f"\nComparison data saved to: {output_csv}")

    print("\n" + "="*80)
    print("FINAL CORRELATION METRICS")
    print("="*80)
    print(f"--- {comparison_label} vs Analytical ---")
    print(f"Spearman Rank Correlation: {spearman_corr_ana_g2:.6f}")
    print(f"Kendall Tau Correlation:   {ktau_g2_ana:.6f}")
    print(f"Kendall Tau Distance:      {kdist_g2_ana} discordant pairs (out of {kpairs_g2_ana})")
    print(f"Normalized Kendall Dist:   {knorm_g2_ana:.4f}")
    if has_ns3:
        print(f"\n--- NS3 vs G2 ---")
        print(f"Spearman Rank Correlation: {spearman_corr_g2_n3:.6f}")
        print(f"Kendall Tau Correlation:   {ktau_ns3_g2:.6f}")
        print(f"Kendall Tau Distance:      {kdist_ns3_g2} discordant pairs (out of {kpairs_ns3_g2})")
        print(f"Normalized Kendall Dist:   {knorm_ns3_g2:.4f}")
        print(f"\n--- NS3 vs Analytical ---")
        print(f"Spearman Rank Correlation: {spearman_corr_ana_ns3:.6f}")
        print(f"Kendall Tau Correlation:   {ktau_ns3_ana:.6f}")
        print(f"Kendall Tau Distance:      {kdist_ns3_ana} discordant pairs (out of {kpairs_ns3_ana})")
        print(f"Normalized Kendall Dist:   {knorm_ns3_ana:.4f}")
    print("="*80)

    return spearman_corr_ana_g2, output_file



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate slope graph from an optional simulation CSV, analytical CSV, and optional NS3 CSV files."
    )
    parser.add_argument("--g2", default=None, help="Optional path to G2 CSV file")
    parser.add_argument("--analytical", required=True, help="Path to analytical CSV file")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--ns3", default=None, help="Optional NS3 CSV file")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for PNG/CSV artifacts (default: current directory)",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Optional output PNG filename stem (without extension)",
    )

    args = parser.parse_args()

    generate_slope_graph(
        g2_file=args.g2,
        analytical_file=args.analytical,
        title=args.title,
        output_path=args.output_dir,
        output_filename=args.output_filename,
        ns3_file=args.ns3,
    )
