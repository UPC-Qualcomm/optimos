#!/usr/bin/env python3
"""
Create a slope graph showing order changes between g2 and analytical results.
Usage: python generate_slope_graph.py <g2_csv_path> <analytical_csv_path> <title>
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
                         output_filename=None, ns3_file=None, cycles_per_ms=1e6):
    """Generate slope graph comparing simulation results.

    Args:
        analytical_file: Path to analytical results CSV (required)
        g2_file: (optional) Path to G2 results CSV
        ns3_file: (optional) Path to NS3 results CSV
        title: Graph title
        output_path: Output directory path (default: current directory)
        output_filename: Custom filename for output (without extension). If None, auto-generates from input paths.
    
    Supports three plotting modes:
        - 2-column: ns3 + analytical (if g2 is None)
        - 2-column: g2 + analytical (if ns3 is None)
        - 3-column: g2 + ns3 + analytical (if both provided)
    """
    has_g2 = g2_file is not None
    has_ns3 = ns3_file is not None
    
    # Validate that we have at least one simulator result
    if not has_g2 and not has_ns3:
        raise ValueError("At least one of --g2 or --ns3 must be provided")
    if analytical_file is None:
        raise ValueError("--analytical is required")

    # ------------------------------------------------------------------ #
    # 1. Read & filter data
    # ------------------------------------------------------------------ #
    print(f"Reading data from:")
    if has_g2:
        print(f"  G2: {g2_file}")
    if has_ns3:
        print(f"  NS3: {ns3_file}")
    print(f"  Analytical: {analytical_file}")
    print(f"Plot mode: {sum([has_g2, has_ns3])}-column")
    
    df_analytical = pd.read_csv(analytical_file)
    df_g2 = pd.read_csv(g2_file) if has_g2 else None
    df_ns3 = pd.read_csv(ns3_file) if has_ns3 else None
    
    if has_g2:
        df_g2 = df_g2.sample(frac=1, random_state=5).reset_index(drop=True)[1:15]
    if has_ns3:
        df_ns3 = df_ns3.sample(frac=1, random_state=5).reset_index(drop=True)[1:15]
    
    print(f"Analytical data (before OOM filter): {len(df_analytical)} configurations")
    if has_g2:
        print(f"G2 data (before OOM filter): {len(df_g2)} configurations")
    if has_ns3:
        print(f"NS3 data (before OOM filter): {len(df_ns3)} configurations")

    if 'is_oom' in df_analytical.columns:
        df_analytical = df_analytical[df_analytical['is_oom'] != True].reset_index(drop=True)
        print(f"Analytical data (after OOM filter): {len(df_analytical)} configurations")

    if has_g2 and 'is_oom' in df_g2.columns:
        df_g2 = df_g2[df_g2['is_oom'] != True].reset_index(drop=True)
        print(f"G2 data (after OOM filter): {len(df_g2)} configurations")

    if has_ns3 and 'is_oom' in df_ns3.columns:
        df_ns3 = df_ns3[df_ns3['is_oom'] != True].reset_index(drop=True)
        print(f"NS3 data (after OOM filter): {len(df_ns3)} configurations")

    # ------------------------------------------------------------------ #
    # 1b. Keep only configurations common to ALL datasets
    # ------------------------------------------------------------------ #
    # Build intersection based on which datasets are available
    config_sets = [set(df_analytical['dp_mp_sp_pp_sharded'])]
    if has_g2:
        config_sets.append(set(df_g2['dp_mp_sp_pp_sharded']))
    if has_ns3:
        config_sets.append(set(df_ns3['dp_mp_sp_pp_sharded']))
    
    common_configs = set.intersection(*config_sets)
    
    if has_g2 and has_ns3:
        print(f"\nCommon configurations (G2 ∩ NS3 ∩ Analytical): {len(common_configs)}")
    elif has_ns3:
        print(f"\nCommon configurations (NS3 ∩ Analytical): {len(common_configs)}")
    else:
        print(f"\nCommon configurations (G2 ∩ Analytical): {len(common_configs)}")

    df_analytical = df_analytical[df_analytical['dp_mp_sp_pp_sharded'].isin(common_configs)].reset_index(drop=True)
    if has_g2:
        df_g2 = df_g2[df_g2['dp_mp_sp_pp_sharded'].isin(common_configs)].reset_index(drop=True)
    if has_ns3:
        df_ns3 = df_ns3[df_ns3['dp_mp_sp_pp_sharded'].isin(common_configs)].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 2. Sort & rank
    # ------------------------------------------------------------------ #
    df_analytical_sorted = df_analytical.sort_values('exec_cycles').reset_index(drop=True)
    df_analytical_sorted['analytical_rank'] = range(1, len(df_analytical_sorted) + 1)
    
    if has_g2:
        df_g2_sorted = df_g2.sort_values('exec_cycles').reset_index(drop=True)
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

    df_analytical_sorted, analytical_min, analytical_max = _normalize(df_analytical_sorted)
    analytical_range = analytical_max - analytical_min
    
    if has_g2:
        df_g2_sorted, g2_min, g2_max = _normalize(df_g2_sorted)
        g2_range = g2_max - g2_min
        g2_ana_scale_factor = g2_range / analytical_range if analytical_range > 0 else 1.0

    if has_ns3:
        df_ns3_sorted, ns3_min, ns3_max = _normalize(df_ns3_sorted)
        ns3_range = ns3_max - ns3_min
        ns3_ana_scale_factor = ns3_range / analytical_range if analytical_range > 0 else 1.0
        if has_g2:
            ns3_g2_scale_factor = ns3_range / g2_range if g2_range > 0 else 1.0

    # ------------------------------------------------------------------ #
    # 4. Pairwise merges & Kendall distances
    # ------------------------------------------------------------------ #
    # Determine which is the primary simulator (first column)
    # If has_g2, g2 is first; else ns3 is first
    primary_sim = 'g2' if has_g2 else 'ns3'
    
    if has_g2 and has_ns3:
        # 3-column: G2 vs Analytical
        merged_primary_ana = pd.merge(
            df_analytical_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'analytical_rank', 'normalized_cycles']],
            df_g2_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'g2_rank', 'normalized_cycles']],
            on='dp_mp_sp_pp_sharded', suffixes=('_analytical', '_primary')
        )
        merged_primary_ana['rank_diff'] = merged_primary_ana['analytical_rank'] - merged_primary_ana['g2_rank']
        merged_primary_ana['abs_rank_diff'] = merged_primary_ana['rank_diff'].abs()
        merged_primary_ana['cycles_diff'] = (merged_primary_ana['exec_cycles_analytical'] - merged_primary_ana['exec_cycles_primary']).abs()
        merged_primary_ana['normalized_diff'] = (merged_primary_ana['normalized_cycles_analytical'] - merged_primary_ana['normalized_cycles_primary']).abs()

        spearman_corr_primary_ana, _ = spearmanr(merged_primary_ana['analytical_rank'], merged_primary_ana['g2_rank'])
        ktau_primary_ana, kdist_primary_ana, kpairs_primary_ana, knorm_primary_ana = _kendall_stats(
            merged_primary_ana['analytical_rank'].values, merged_primary_ana['g2_rank'].values)

        # NS3 vs G2
        merged_ns3_g2 = pd.merge(
            df_ns3_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'ns3_rank', 'normalized_cycles']],
            df_g2_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'g2_rank', 'normalized_cycles']],
            on='dp_mp_sp_pp_sharded', suffixes=('_ns3', '_g2')
        )
        ktau_ns3_g2, kdist_ns3_g2, kpairs_ns3_g2, knorm_ns3_g2 = _kendall_stats(
            merged_ns3_g2['ns3_rank'].values, merged_ns3_g2['g2_rank'].values)
        spearman_corr_g2_ns3, _ = spearmanr(merged_ns3_g2['g2_rank'], merged_ns3_g2['ns3_rank'])

        # NS3 vs Analytical
        merged_ns3_ana = pd.merge(
            df_ns3_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'ns3_rank', 'normalized_cycles']],
            df_analytical_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'analytical_rank', 'normalized_cycles']],
            on='dp_mp_sp_pp_sharded', suffixes=('_ns3', '_analytical')
        )
        ktau_ns3_ana, kdist_ns3_ana, kpairs_ns3_ana, knorm_ns3_ana = _kendall_stats(
            merged_ns3_ana['ns3_rank'].values, merged_ns3_ana['analytical_rank'].values)
        spearman_corr_ana_ns3, _ = spearmanr(merged_ns3_ana['analytical_rank'], merged_ns3_ana['ns3_rank'])
    elif has_ns3:
        # 2-column: NS3 vs Analytical
        merged_primary_ana = pd.merge(
            df_analytical_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'analytical_rank', 'normalized_cycles']],
            df_ns3_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'ns3_rank', 'normalized_cycles']],
            on='dp_mp_sp_pp_sharded', suffixes=('_analytical', '_primary')
        )
        merged_primary_ana['rank_diff'] = merged_primary_ana['analytical_rank'] - merged_primary_ana['ns3_rank']
        merged_primary_ana['abs_rank_diff'] = merged_primary_ana['rank_diff'].abs()
        merged_primary_ana['cycles_diff'] = (merged_primary_ana['exec_cycles_analytical'] - merged_primary_ana['exec_cycles_primary']).abs()
        merged_primary_ana['normalized_diff'] = (merged_primary_ana['normalized_cycles_analytical'] - merged_primary_ana['normalized_cycles_primary']).abs()

        spearman_corr_primary_ana, _ = spearmanr(merged_primary_ana['analytical_rank'], merged_primary_ana['ns3_rank'])
        ktau_primary_ana, kdist_primary_ana, kpairs_primary_ana, knorm_primary_ana = _kendall_stats(
            merged_primary_ana['analytical_rank'].values, merged_primary_ana['ns3_rank'].values)
    else:
        # 2-column: G2 vs Analytical
        merged_primary_ana = pd.merge(
            df_analytical_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'analytical_rank', 'normalized_cycles']],
            df_g2_sorted[['dp_mp_sp_pp_sharded', 'exec_cycles', 'g2_rank', 'normalized_cycles']],
            on='dp_mp_sp_pp_sharded', suffixes=('_analytical', '_primary')
        )
        merged_primary_ana['rank_diff'] = merged_primary_ana['analytical_rank'] - merged_primary_ana['g2_rank']
        merged_primary_ana['abs_rank_diff'] = merged_primary_ana['rank_diff'].abs()
        merged_primary_ana['cycles_diff'] = (merged_primary_ana['exec_cycles_analytical'] - merged_primary_ana['exec_cycles_primary']).abs()
        merged_primary_ana['normalized_diff'] = (merged_primary_ana['normalized_cycles_analytical'] - merged_primary_ana['normalized_cycles_primary']).abs()

        spearman_corr_primary_ana, _ = spearmanr(merged_primary_ana['analytical_rank'], merged_primary_ana['g2_rank'])
        ktau_primary_ana, kdist_primary_ana, kpairs_primary_ana, knorm_primary_ana = _kendall_stats(
            merged_primary_ana['analytical_rank'].values, merged_primary_ana['g2_rank'].values)

    # ------------------------------------------------------------------ #
    # 5. Print statistics
    # ------------------------------------------------------------------ #
    n_configs = len(merged_primary_ana)
    print(f"Merged configurations: {n_configs}")

    print(f"\n{'='*80}")
    print(f"RANKING CORRELATION METRICS")
    print(f"{'='*80}")
    
    if has_g2 and has_ns3:
        print(f"\n--- G2 vs Analytical ({n_configs} common configs) ---")
        print(f"Spearman ρ: {spearman_corr_primary_ana:.6f}")
        print(f"Kendall τ: {ktau_primary_ana:.6f}")
        print(f"Kendall Distance: {kdist_primary_ana} discordant pairs (out of {kpairs_primary_ana})")
        print(f"Normalized Kendall Distance: {knorm_primary_ana:.4f}  ({(1-knorm_primary_ana)*100:.1f}% concordant)")
        print(f"Mean |Δ rank|: {merged_primary_ana['abs_rank_diff'].mean():.2f}")

        print(f"\n--- NS3 vs G2 ({len(merged_ns3_g2)} common configs) ---")
        print(f"Kendall τ: {ktau_ns3_g2:.6f}")
        print(f"Kendall Distance: {kdist_ns3_g2} discordant pairs (out of {kpairs_ns3_g2})")
        print(f"Normalized Kendall Distance: {knorm_ns3_g2:.4f}  ({(1-knorm_ns3_g2)*100:.1f}% concordant)")

        print(f"\n--- NS3 vs Analytical ({len(merged_ns3_ana)} common configs) ---")
        print(f"Kendall τ: {ktau_ns3_ana:.6f}")
        print(f"Kendall Distance: {kdist_ns3_ana} discordant pairs (out of {kpairs_ns3_ana})")
        print(f"Normalized Kendall Distance: {knorm_ns3_ana:.4f}  ({(1-knorm_ns3_ana)*100:.1f}% concordant)")
    elif has_ns3:
        print(f"\n--- NS3 vs Analytical ({n_configs} common configs) ---")
        print(f"Spearman ρ: {spearman_corr_primary_ana:.6f}")
        print(f"Kendall τ: {ktau_primary_ana:.6f}")
        print(f"Kendall Distance: {kdist_primary_ana} discordant pairs (out of {kpairs_primary_ana})")
        print(f"Normalized Kendall Distance: {knorm_primary_ana:.4f}  ({(1-knorm_primary_ana)*100:.1f}% concordant)")
        print(f"Mean |Δ rank|: {merged_primary_ana['abs_rank_diff'].mean():.2f}")
    else:
        print(f"\n--- G2 vs Analytical ({n_configs} common configs) ---")
        print(f"Spearman ρ: {spearman_corr_primary_ana:.6f}")
        print(f"Kendall τ: {ktau_primary_ana:.6f}")
        print(f"Kendall Distance: {kdist_primary_ana} discordant pairs (out of {kpairs_primary_ana})")
        print(f"Normalized Kendall Distance: {knorm_primary_ana:.4f}  ({(1-knorm_primary_ana)*100:.1f}% concordant)")
        print(f"Mean |Δ rank|: {merged_primary_ana['abs_rank_diff'].mean():.2f}")

    print(f"\n{'='*80}")
    print(f"SCALING FACTOR EXPLANATION")
    print(f"{'='*80}")
    print(f"Analytical range: {analytical_range:,.0f} cycles (from {analytical_min:,.0f} to {analytical_max:,.0f})")
    
    if has_g2:
        print(f"G2 range: {g2_range:,.0f} cycles (from {g2_min:,.0f} to {g2_max:,.0f})")
    if has_ns3:
        print(f"NS3 range: {ns3_range:,.0f} cycles (from {ns3_min:,.0f} to {ns3_max:,.0f})")
    
    if has_g2 and has_ns3:
        print(f"\nScaling Factor (G2 vs Analytical) = G2 range / Analytical range = {g2_ana_scale_factor:.4f}")
        print(f"Scaling Factor (NS3 vs Analytical) = NS3 range / Analytical range = {ns3_ana_scale_factor:.4f}")
        print(f"Scaling Factor (NS3 vs G2)         = NS3 range / G2 range        = {ns3_g2_scale_factor:.4f}")
    elif has_ns3:
        print(f"Scaling Factor = NS3 range / Analytical range = {ns3_ana_scale_factor:.4f}")
    elif has_g2:
        print(f"Scaling Factor = G2 range / Analytical range = {g2_ana_scale_factor:.4f}")
    
    print(f"\nInterpretation:")
    if has_g2 and has_ns3:
        if g2_ana_scale_factor > 1.05:
            print(f"  → G2 has {g2_ana_scale_factor:.2f}x MORE variability than Analytical")
        elif g2_ana_scale_factor < 0.95:
            print(f"  → G2 has {1/g2_ana_scale_factor:.2f}x LESS variability than Analytical")
        else:
            print(f"  → G2 and Analytical have approximately the SAME variability")
        
        if ns3_ana_scale_factor > 1.05:
            print(f"  → NS3 has {ns3_ana_scale_factor:.2f}x MORE variability than Analytical")
        elif ns3_ana_scale_factor < 0.95:
            print(f"  → NS3 has {1/ns3_ana_scale_factor:.2f}x LESS variability than Analytical")
        else:
            print(f"  → NS3 and Analytical have approximately the SAME variability")
    elif has_ns3:
        if ns3_ana_scale_factor > 1.05:
            print(f"  → NS3 has {ns3_ana_scale_factor:.2f}x MORE variability than Analytical")
        elif ns3_ana_scale_factor < 0.95:
            print(f"  → NS3 has {1/ns3_ana_scale_factor:.2f}x LESS variability than Analytical")
        else:
            print(f"  → NS3 and Analytical have approximately the SAME variability")
    elif has_g2:
        if g2_ana_scale_factor > 1.05:
            print(f"  → G2 has {g2_ana_scale_factor:.2f}x MORE variability than Analytical")
        elif g2_ana_scale_factor < 0.95:
            print(f"  → G2 has {1/g2_ana_scale_factor:.2f}x LESS variability than Analytical")
        else:
            print(f"  → G2 and Analytical have approximately the SAME variability")
    
    print(f"\nNote: To compare actual execution times, both datasets are normalized to 0-100 scale")
    print(f"      where 0 = best (lowest exec_cycles) and 100 = worst (highest exec_cycles)")

    # ------------------------------------------------------------------ #
    # 6. Build the slope graph (rank-based: each column ordered independently)
    # ------------------------------------------------------------------ #
    # Column x-positions based on plot mode
    if has_g2 and has_ns3:
        # 3-column mode
        g2_x = 0
        ns3_x = 0.6
        analytical_x = 1.2
        x_left_lim  = -0.3
        x_right_lim =  1.5
        col_labels = ['G2', 'ns-3', 'Analytical']
    elif has_ns3:
        # 2-column mode: NS3 and Analytical
        ns3_x = 0
        analytical_x = 0.6
        x_left_lim  = -0.3
        x_right_lim =  0.9
        col_labels = ['ns-3', 'Analytical']
    else:
        # 2-column mode: G2 and Analytical
        g2_x = 0
        analytical_x = 0.6
        x_left_lim  = -0.3
        x_right_lim =  0.9
        col_labels = ['G2', 'Analytical']

    fig_height = max(2.5, n_configs * 0.26)
    fig_width  = 7.2  # double-column width
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Color palette – one unique color per configuration
    all_configs_union = sorted(
        set(df_analytical_sorted['dp_mp_sp_pp_sharded']) |
        (set(df_g2_sorted['dp_mp_sp_pp_sharded']) if has_g2 else set()) |
        (set(df_ns3_sorted['dp_mp_sp_pp_sharded']) if has_ns3 else set())
    )
    colors_tab20  = plt.cm.tab20 (np.linspace(0, 1, 20))
    colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    colors_tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))
    all_colors = np.vstack([colors_tab20, colors_tab20b, colors_tab20c])
    config_to_color = {cfg: all_colors[i % len(all_colors)] for i, cfg in enumerate(all_configs_union)}

    # Rank-based y-positions: rank 0 = best, rank n-1 = worst; no overlap possible
    ana_config_to_pos = {cfg: i for i, cfg in enumerate(df_analytical_sorted['dp_mp_sp_pp_sharded'])}
    if has_g2:
        g2_config_to_pos  = {cfg: i for i, cfg in enumerate(df_g2_sorted['dp_mp_sp_pp_sharded'])}
    if has_ns3:
        ns3_config_to_pos = {cfg: i for i, cfg in enumerate(df_ns3_sorted['dp_mp_sp_pp_sharded'])}

    # ---- Draw connecting lines ----
    if has_g2 and has_ns3:
        # 3-column: draw g2-ns3 and ns3-analytical lines
        for cfg in set(df_g2_sorted['dp_mp_sp_pp_sharded']) & set(df_ns3_sorted['dp_mp_sp_pp_sharded']):
            ax.plot([g2_x, ns3_x], [g2_config_to_pos[cfg], ns3_config_to_pos[cfg]],
                    color=config_to_color[cfg], linewidth=1.5, alpha=0.7, zorder=1)
        for cfg in set(df_ns3_sorted['dp_mp_sp_pp_sharded']) & set(df_analytical_sorted['dp_mp_sp_pp_sharded']):
            ax.plot([ns3_x, analytical_x], [ns3_config_to_pos[cfg], ana_config_to_pos[cfg]],
                    color=config_to_color[cfg], linewidth=1.5, alpha=0.7, zorder=1)
    elif has_ns3:
        # 2-column: NS3 and Analytical
        for cfg in set(df_ns3_sorted['dp_mp_sp_pp_sharded']) & set(df_analytical_sorted['dp_mp_sp_pp_sharded']):
            ax.plot([ns3_x, analytical_x], [ns3_config_to_pos[cfg], ana_config_to_pos[cfg]],
                    color=config_to_color[cfg], linewidth=1.5, alpha=0.7, zorder=1)
    else:
        # 2-column: G2 and Analytical
        for cfg in set(df_g2_sorted['dp_mp_sp_pp_sharded']) & set(df_analytical_sorted['dp_mp_sp_pp_sharded']):
            ax.plot([g2_x, analytical_x], [g2_config_to_pos[cfg], ana_config_to_pos[cfg]],
                    color=config_to_color[cfg], linewidth=1.5, alpha=0.7, zorder=1)

    # ---- Draw dots ----
    dot_size = 30

    def _draw_dots(df_sorted, config_to_pos, x_pos):
        for _, row in df_sorted.iterrows():
            cfg = row['dp_mp_sp_pp_sharded']
            ax.scatter([x_pos], [config_to_pos[cfg]], s=dot_size,
                       color=config_to_color.get(cfg, 'gray'),
                       edgecolors='black', linewidth=0.4, zorder=2, alpha=0.9)

    if has_g2:
        _draw_dots(df_g2_sorted,          g2_config_to_pos,  g2_x)
    if has_ns3:
        _draw_dots(df_ns3_sorted, ns3_config_to_pos, ns3_x)
    _draw_dots(df_analytical_sorted,  ana_config_to_pos, analytical_x)

    # ---- Annotations ----
    top_n = len(merged_primary_ana)
    fsize = 10

    def _fmt_cfg(cfg, labeled=False):
        """Format dp_mp_sp_pp_sharded string.
        labeled=True  → DP:1,TP:64,SP:1,PP:1,FSDP:0  (first point only)
        labeled=False → 1,64,1,1,0
        """
        parts = cfg.split('_')
        if len(parts) == 5:
            dp, tp, sp, pp, fsdp = parts
            if labeled:
                return f"DP:{dp},TP:{tp},SP:{sp},PP:{pp},FSDP:{fsdp}"
            return f"| {dp},{tp},{sp},{pp},{fsdp}"
        return cfg  # fallback for unexpected format

    # Left column annotations
    if has_g2 and has_ns3:
        # 3-column: G2 on left
        for i, (_, row) in enumerate(df_g2_sorted.head(top_n).iterrows()):
            cfg = row['dp_mp_sp_pp_sharded']
            ms_str = f"{row['exec_cycles']/cycles_per_ms:.1f}ms"
            if i == 0:
                text = f"{ms_str}\n{_fmt_cfg(cfg, labeled=True)}"
            else:
                text = f"{ms_str} {_fmt_cfg(cfg)}"
            ax.text(g2_x - 0.08, g2_config_to_pos[cfg],
                    text,
                    ha='right', va='center', fontsize=fsize,
                    fontweight='bold' if i < 3 else 'normal')
    elif has_ns3:
        # 2-column: NS3 on left
        for i, (_, row) in enumerate(df_ns3_sorted.head(top_n).iterrows()):
            cfg = row['dp_mp_sp_pp_sharded']
            ms_str = f"{row['exec_cycles']/cycles_per_ms:.1f}ms"
            if i == 0:
                text = f"{ms_str}\n{_fmt_cfg(cfg, labeled=True)}"
            else:
                text = f"{ms_str} {_fmt_cfg(cfg)}"
            ax.text(ns3_x - 0.08, ns3_config_to_pos[cfg],
                    text,
                    ha='right', va='center', fontsize=fsize,
                    fontweight='bold' if i < 3 else 'normal')
    else:
        # 2-column: G2 on left
        for i, (_, row) in enumerate(df_g2_sorted.head(top_n).iterrows()):
            cfg = row['dp_mp_sp_pp_sharded']
            ms_str = f"{row['exec_cycles']/cycles_per_ms:.1f}ms"
            if i == 0:
                text = f"{ms_str}\n{_fmt_cfg(cfg, labeled=True)}"
            else:
                text = f"{ms_str} {_fmt_cfg(cfg)}"
            ax.text(g2_x - 0.08, g2_config_to_pos[cfg],
                    text,
                    ha='right', va='center', fontsize=fsize,
                    fontweight='bold' if i < 3 else 'normal')

    # Right column annotations (always Analytical)
    for i, (_, row) in enumerate(df_analytical_sorted.head(top_n).iterrows()):
        cfg = row['dp_mp_sp_pp_sharded']
        ax.text(analytical_x + 0.08, ana_config_to_pos[cfg],
                f"{row['exec_cycles']/cycles_per_ms:.1f}ms {_fmt_cfg(cfg)}",
                ha='left', va='center', fontsize=fsize,
                fontweight='bold' if i < 3 else 'normal')

    # Middle column annotations (only for 3-column mode)
    if has_g2 and has_ns3:
        for i, (_, row) in enumerate(df_ns3_sorted.head(top_n).iterrows()):
            cfg = row['dp_mp_sp_pp_sharded']
            ax.text(ns3_x + 0.06, ns3_config_to_pos[cfg],
                    f"{row['exec_cycles']/cycles_per_ms:.1f}ms",
                    ha='left', va='center', fontsize=fsize - 1, color='dimgray',
                    fontweight='bold' if i < 3 else 'normal')

    # ---- Axes ----
    ax.set_xlim([x_left_lim, x_right_lim])
    ax.set_ylim([n_configs - 0.5, -0.5])

    if has_g2 and has_ns3:
        x_ticks = [g2_x, ns3_x, analytical_x]
    elif has_ns3:
        x_ticks = [ns3_x, analytical_x]
    else:
        x_ticks = [g2_x, analytical_x]
    
    x_labels = col_labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    ax.yaxis.set_visible(False)
    for spine in ('left', 'right', 'top'):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2, linestyle='--')

    # ---- Title ----
    if has_g2 and has_ns3:
        title_text = f'{title}\nG2/Ana.: τ={ktau_primary_ana:.3f} D={kdist_primary_ana}/{kpairs_primary_ana} | ns-3/G2: τ={ktau_ns3_g2:.3f} D={kdist_ns3_g2}/{kpairs_ns3_g2} | ns-3/Ana: τ={ktau_ns3_ana:.3f} D={kdist_ns3_ana}/{kpairs_ns3_ana}'
    elif has_ns3:
        title_text = f'{title}\nNS3/Ana.: τ={ktau_primary_ana:.3f} D={kdist_primary_ana}/{kpairs_primary_ana}'
    else:
        title_text = f'{title}\nG2/Ana.: τ={ktau_primary_ana:.3f} D={kdist_primary_ana}/{kpairs_primary_ana}'
    #ax.set_title(title_text, fontsize=11, fontweight='bold', pad=6)

    plt.tight_layout(pad=0.3)

    # ------------------------------------------------------------------ #
    # 7. Save output
    # ------------------------------------------------------------------ #
    output_dir = output_path if output_path is not None else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if output_filename is None:
        # Generate filename based on which files are provided
        if has_g2 and has_ns3:
            g2_basename = os.path.basename(os.path.dirname(g2_file))
            ns3_basename = os.path.basename(os.path.dirname(ns3_file))
            analytical_basename = os.path.basename(os.path.dirname(analytical_file))
            base_name = f'slope_graph_{g2_basename}_vs_{ns3_basename}_vs_{analytical_basename}'
        elif has_ns3:
            ns3_basename = os.path.basename(os.path.dirname(ns3_file))
            analytical_basename = os.path.basename(os.path.dirname(analytical_file))
            base_name = f'slope_graph_{ns3_basename}_vs_{analytical_basename}'
        else:
            g2_basename = os.path.basename(os.path.dirname(g2_file))
            analytical_basename = os.path.basename(os.path.dirname(analytical_file))
            base_name = f'slope_graph_{g2_basename}_vs_{analytical_basename}'
    else:
        base_name = output_filename.replace('.png', '').replace('.pdf', '')

    output_file = os.path.join(output_dir, base_name + '.png')
    output_pdf  = os.path.join(output_dir, base_name + '.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf,           bbox_inches='tight')
    print(f"\nSlope graph saved to: {output_file}")
    print(f"PDF saved to:         {output_pdf}")

    base_filename = os.path.splitext(output_file)[0]
    output_csv = base_filename + '.csv'

    # ------------------------------------------------------------------ #
    # 8. Print top configs & findings
    # ------------------------------------------------------------------ #
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    
    if has_g2:
        print("\nG2 Top 10:")
        for i, row in df_g2_sorted.head(10).iterrows():
            print(f"  {i+1:2d}. {row['dp_mp_sp_pp_sharded']:20s} - {row['exec_cycles']:,} cycles")
    
    if has_ns3:
        print("\nNS3 Top 10:")
        for i, row in df_ns3_sorted.head(10).iterrows():
            print(f"  {i+1:2d}. {row['dp_mp_sp_pp_sharded']:20s} - {row['exec_cycles']:,} cycles")
    
    print("\nAnalytical Top 10:")
    for i, row in df_analytical_sorted.head(10).iterrows():
        print(f"  {i+1:2d}. {row['dp_mp_sp_pp_sharded']:20s} - {row['exec_cycles']:,} cycles")

    print("\n" + "="*80)
    if has_g2 and has_ns3:
        print("INTERESTING FINDINGS (G2 vs Analytical)")
    elif has_ns3:
        print("INTERESTING FINDINGS (NS3 vs Analytical)")
    else:
        print("INTERESTING FINDINGS (G2 vs Analytical)")
    print("="*80)
    
    improved_in_primary = merged_primary_ana.nsmallest(5, 'rank_diff')
    print("\nConfigurations that ranked BETTER in primary simulator (improved the most):")
    for _, row in improved_in_primary.iterrows():
        print(f"  {row['dp_mp_sp_pp_sharded']:20s}: Rank {row['analytical_rank']:2.0f} → {row['g2_rank'] if has_g2 else row['ns3_rank']:2.0f} (improved by {-row['rank_diff']:.0f})")
    
    degraded_in_primary = merged_primary_ana.nlargest(5, 'rank_diff')
    print("\nConfigurations that ranked WORSE in primary simulator (degraded the most):")
    for _, row in degraded_in_primary.iterrows():
        print(f"  {row['dp_mp_sp_pp_sharded']:20s}: Rank {row['analytical_rank']:2.0f} → {row['g2_rank'] if has_g2 else row['ns3_rank']:2.0f} (degraded by {row['rank_diff']:.0f})")
    
    stable_configs = merged_primary_ana[merged_primary_ana['abs_rank_diff'] <= 1].sort_values('analytical_rank')
    print(f"\nConfigurations with stable ranking (diff ≤ 1 positions): {len(stable_configs)}/{n_configs}")

    merged_primary_ana.sort_values('analytical_rank').to_csv(output_csv, index=False)
    print(f"\nComparison data saved to: {output_csv}")

    print("\n" + "="*80)
    print("FINAL CORRELATION METRICS")
    print("="*80)
    
    if has_g2 and has_ns3:
        print(f"--- G2 vs Analytical ---")
        print(f"Spearman Rank Correlation: {spearman_corr_primary_ana:.6f}")
        print(f"Kendall Tau Correlation:   {ktau_primary_ana:.6f}")
        print(f"Kendall Tau Distance:      {kdist_primary_ana} discordant pairs (out of {kpairs_primary_ana})")
        print(f"Normalized Kendall Dist:   {knorm_primary_ana:.4f}")
        
        print(f"\n--- NS3 vs G2 ---")
        print(f"Spearman Rank Correlation: {spearman_corr_g2_ns3:.6f}")
        print(f"Kendall Tau Correlation:   {ktau_ns3_g2:.6f}")
        print(f"Kendall Tau Distance:      {kdist_ns3_g2} discordant pairs (out of {kpairs_ns3_g2})")
        print(f"Normalized Kendall Dist:   {knorm_ns3_g2:.4f}")
        
        print(f"\n--- NS3 vs Analytical ---")
        print(f"Spearman Rank Correlation: {spearman_corr_ana_ns3:.6f}")
        print(f"Kendall Tau Correlation:   {ktau_ns3_ana:.6f}")
        print(f"Kendall Tau Distance:      {kdist_ns3_ana} discordant pairs (out of {kpairs_ns3_ana})")
        print(f"Normalized Kendall Dist:   {knorm_ns3_ana:.4f}")
    elif has_ns3:
        print(f"--- NS3 vs Analytical ---")
        print(f"Spearman Rank Correlation: {spearman_corr_primary_ana:.6f}")
        print(f"Kendall Tau Correlation:   {ktau_primary_ana:.6f}")
        print(f"Kendall Tau Distance:      {kdist_primary_ana} discordant pairs (out of {kpairs_primary_ana})")
        print(f"Normalized Kendall Dist:   {knorm_primary_ana:.4f}")
    else:
        print(f"--- G2 vs Analytical ---")
        print(f"Spearman Rank Correlation: {spearman_corr_primary_ana:.6f}")
        print(f"Kendall Tau Correlation:   {ktau_primary_ana:.6f}")
        print(f"Kendall Tau Distance:      {kdist_primary_ana} discordant pairs (out of {kpairs_primary_ana})")
        print(f"Normalized Kendall Dist:   {knorm_primary_ana:.4f}")
    
    print("="*80)

    return spearman_corr_primary_ana, output_file

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate slope graph from simulation CSV files."
    )
    parser.add_argument("--g2", default=None, help="Optional path to G2 CSV file")
    parser.add_argument("--analytical", required=True, help="Path to analytical CSV file (required)")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--ns3", default=None, help="Optional path to NS3 CSV file")
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
    
    # Validate that at least one simulator is provided
    if args.g2 is None and args.ns3 is None:
        parser.error("At least one of --g2 or --ns3 must be provided")

    generate_slope_graph(
        g2_file=args.g2,
        analytical_file=args.analytical,
        title=args.title,
        output_path=args.output_dir,
        output_filename=args.output_filename,
        ns3_file=args.ns3,
    )