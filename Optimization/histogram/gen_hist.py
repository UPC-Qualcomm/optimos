#!/usr/bin/env python3
"""
Flexible histogram generator for exec_cycles data with maximum value filtering capabilities
Usage: python gen_hist.py <csv_path> <output_name> [--dp MAX_VALUE] [--mp MAX_VALUE] [--sp MAX_VALUE] [--pp MAX_VALUE] [--sharding MAX_VALUE] [--oom true|false]
Note: All filters use <= (less than or equal) comparison. The --oom filter includes/excludes records based on is_oom field.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path


def create_hist_folder():
    """Create hist folder if it doesn't exist"""
    hist_folder = Path("data")
    hist_folder.mkdir(exist_ok=True)
    return hist_folder


def apply_filters(df, filters, oom_filter=None):
    """Apply filters to the dataframe based on maximum values (<=) and boolean filters"""
    filtered_df = df.copy()
    applied_filters = []
    
    # Apply numeric filters
    for column, max_value in filters.items():
        if max_value is not None:
            if column in df.columns:
                initial_count = len(filtered_df)
                filtered_df = filtered_df[filtered_df[column] <= max_value]
                final_count = len(filtered_df)
                applied_filters.append(f"{column}<={max_value} (kept {final_count}/{initial_count})")
            else:
                print(f"Warning: Column '{column}' not found in the data")
    
    # Apply is_oom filter if specified
    if oom_filter is not None:
        if 'is_oom' in df.columns:
            initial_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['is_oom'] == oom_filter]
            final_count = len(filtered_df)
            applied_filters.append(f"is_oom=={oom_filter} (kept {final_count}/{initial_count})")
        else:
            print(f"Warning: Column 'is_oom' not found in the data")
    
    return filtered_df, applied_filters


def generate_histogram(csv_path, output_name, filters=None, bins=None, oom_filter=None):
    """Generate histogram with optional filters"""
    
    # Load the data
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from: {csv_path}")
        print(f"Original dataset: {len(df)} configurations")
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        return False
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Check if exec_cycles column exists
    if 'exec_cycles' not in df.columns:
        print("Error: 'exec_cycles' column not found in the CSV file")
        print(f"Available columns: {', '.join(df.columns)}")
        return False
    
    # Apply filters if provided
    filtered_df = df
    filter_description = "No filters applied"
    
    if filters or oom_filter is not None:
        filtered_df, applied_filters = apply_filters(df, filters if filters else {}, oom_filter)
        if applied_filters:
            filter_description = "; ".join(applied_filters)
        else:
            filter_description = "No valid filters applied"
    
    if len(filtered_df) == 0:
        print("Error: No data remaining after applying filters")
        return False
    
    print(f"Filtered dataset: {len(filtered_df)} configurations")
    print(f"Filters: {filter_description}")
    
    # Extract exec_cycles data and convert to billions of cycles
    exec_cycles = filtered_df['exec_cycles'] / 1e9
    
    # Print statistics
    print("\n=== Time Statistics (in sec) ===")
    print(f"Count: {len(exec_cycles)}")
    print(f"Min: {exec_cycles.min():.2f}")
    print(f"Max: {exec_cycles.max():.2f}")
    print(f"Mean: {exec_cycles.mean():.2f}")
    print(f"Median: {exec_cycles.median():.2f}")
    print(f"Std Dev: {exec_cycles.std():.2f}")
    
    # Create the histogram
    plt.figure(figsize=(12, 8))
    
    # Determine number of bins (adaptive or user-specified)
    if bins is not None:
        n_bins = bins
        print(f"Using user-specified bins: {bins}")
    else:
        n_bins = min(50, max(10, len(exec_cycles) // 3))
        print(f"Using adaptive bins: {n_bins}")
    
    # Create histogram
    n, bins, patches = plt.hist(exec_cycles, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add trend line (kernel density estimation)
    try:
        from scipy import stats
        import numpy as np
        
        # Create a smooth curve using KDE
        kde = stats.gaussian_kde(exec_cycles)
        x_range = np.linspace(exec_cycles.min(), exec_cycles.max(), 200)
        kde_values = kde(x_range)
        
        # Scale KDE to match histogram scale
        kde_scaled = kde_values * len(exec_cycles) * (bins[1] - bins[0])
        
        # Plot the trend line
        plt.plot(x_range, kde_scaled, 'r-', linewidth=3, alpha=0.8, label='Trend Line (KDE)')
        
    except ImportError:
        print("Warning: scipy not available, trend line will not be shown")
    except Exception as e:
        print(f"Warning: Could not generate trend line: {e}")
    
    # Formatting
    title = f'Distribution of simulation time (sec) - {output_name}'
    if (filters and any(v is not None for v in filters.values())) or oom_filter is not None:
        active_filters = [f"{k}<={v}" for k, v in filters.items() if v is not None]
        if oom_filter is not None:
            active_filters.append(f"is_oom=={oom_filter}")
        title += f'\nFilters: {", ".join(active_filters)}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Simulation Time (sec)', fontsize=12)
    plt.ylabel('Number of Configurations', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    plt.axvline(exec_cycles.mean(), color='orange', linestyle='--', alpha=0.8, 
                label=f'Mean: {exec_cycles.mean():.2f}s', linewidth=2)
    plt.axvline(exec_cycles.median(), color='green', linestyle='--', alpha=0.8, 
                label=f'Median: {exec_cycles.median():.2f}s', linewidth=2)
    plt.legend(loc='upper right')
    
    # Add statistics text box
    stats_text = f'Count: {len(exec_cycles)}\nMean: {exec_cycles.mean():.2f}s\nMedian: {exec_cycles.median():.2f}s\nStd: {exec_cycles.std():.2f}s'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    hist_folder = create_hist_folder()
    output_path = hist_folder / f"{output_name}_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_path}")
    
    # Save filtered data to CSV file
    csv_path = hist_folder / f"{output_name}_filtered_data.csv"
    filtered_df.to_csv(csv_path, index=False)
    print(f"Filtered data saved to: {csv_path}")
    
    # Save exec_cycles values (in seconds) to a simple CSV file
    exec_cycles_csv_path = hist_folder / f"{output_name}_exec_cycles.csv"
    exec_cycles_df = pd.DataFrame({
        'configuration': filtered_df.get('dp_mp_sp_pp_sharded', range(len(filtered_df))),
        'exec_cycles_seconds': exec_cycles.values
    })
    exec_cycles_df.to_csv(exec_cycles_csv_path, index=False)
    print(f"Exec cycles data saved to: {exec_cycles_csv_path}")
    
    # Save detailed statistics to a text file
    stats_path = hist_folder / f"{output_name}_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Histogram Statistics for {output_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data source: {csv_path}\n")
        f.write(f"Original dataset size: {len(df)} configurations\n")
        f.write(f"Filtered dataset size: {len(filtered_df)} configurations\n")
        f.write(f"Filters applied: {filter_description}\n\n")
        
        f.write("Simulation Time Statistics (in sec):\n")
        f.write(f"  Count: {len(exec_cycles)}\n")
        f.write(f"  Min: {exec_cycles.min():.2f}\n")
        f.write(f"  Max: {exec_cycles.max():.2f}\n")
        f.write(f"  Mean: {exec_cycles.mean():.2f}\n")
        f.write(f"  Median: {exec_cycles.median():.2f}\n")
        f.write(f"  Std Dev: {exec_cycles.std():.2f}\n\n")
        
        f.write("Quartiles (in sec):\n")
        f.write(f"  25th percentile: {exec_cycles.quantile(0.25):.2f}\n")
        f.write(f"  50th percentile: {exec_cycles.quantile(0.50):.2f}\n")
        f.write(f"  75th percentile: {exec_cycles.quantile(0.75):.2f}\n\n")
        
        # Top and bottom performers
        if len(filtered_df) >= 5:
            f.write("Top 5 fastest configurations (lowest exec_cycles in time (sec)):\n")
            fastest = filtered_df.nsmallest(5, 'exec_cycles')
            for idx, row in fastest.iterrows():
                config_name = row.get('dp_mp_sp_pp_sharded', f"Row_{idx}")
                f.write(f"  {config_name}: {row['exec_cycles']/1e9:.2f}\n")
            
            f.write("\nTop 5 slowest configurations (highest exec_cycles in time (sec)):\n")
            slowest = filtered_df.nlargest(5, 'exec_cycles')
            for idx, row in slowest.iterrows():
                config_name = row.get('dp_mp_sp_pp_sharded', f"Row_{idx}")
                f.write(f"  {config_name}: {row['exec_cycles']/1e9:.2f}\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    # Show the plot
    #plt.show()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate histogram for exec_cycles with optional filtering')
    parser.add_argument('csv_path', help='Path to the CSV file')
    parser.add_argument('output_name', help='Name for the output files (without extension)')
    parser.add_argument('--dp', type=int, help='Filter by maximum dp (data parallelism) value (<=)')
    parser.add_argument('--mp', type=int, help='Filter by maximum mp (model parallelism) value (<=)')
    parser.add_argument('--sp', type=int, help='Filter by maximum sp (sequence parallelism) value (<=)')
    parser.add_argument('--pp', type=int, help='Filter by maximum pp (pipeline parallelism) value (<=)')
    parser.add_argument('--sharding', type=int, help='Filter by maximum sharding value (<= 0 or 1)')
    parser.add_argument('--oom', type=str, choices=['true', 'false'], help='Filter by is_oom field (true or false)')
    parser.add_argument('--bins', type=int, default=None, help='Number of bins for histogram (default: adaptive)')
    
    args = parser.parse_args()
    
    # Prepare filters dictionary
    filters = {
        'dp': args.dp,
        'mp': args.mp,
        'sp': args.sp,
        'pp': args.pp,
        'sharding': args.sharding
    }
    
    # Remove None values from filters
    filters = {k: v for k, v in filters.items() if v is not None}
    
    # Parse oom filter
    oom_filter = None
    if args.oom is not None:
        oom_filter = (args.oom.lower() == 'true')
    
    print(f"Generating histogram for: {args.csv_path}")
    print(f"Output name: {args.output_name}")
    if filters:
        print(f"Applying filters: {filters}")
    if oom_filter is not None:
        print(f"OOM filter: is_oom == {oom_filter}")
    
    success = generate_histogram(args.csv_path, args.output_name, filters, args.bins, oom_filter)
    
    if success:
        print("\nHistogram generation completed successfully!")
    else:
        print("\nHistogram generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
