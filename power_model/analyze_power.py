#!/usr/bin/env python3
"""
Power Model Analysis Tool

Main entry point for analyzing power consumption of AstraSim workloads.
Supports 4-mode LPM analysis and comparison.
"""

import argparse
import sys
import os
import json
import csv
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from power_config import PowerConfig
from compute_parser import parse_astrasim_log
from network_parser import parse_network_statistics_csv
from power_model import PowerModel, compare_all_modes

DATA_SET_SIZE = 300_000_000_000
def analyze_single_mode(astrasim_log: str,
                        network_csv: str,
                        mode: str = 'A',
                        output_json: str = None,
                        config_path: str = None,
                        nodemap_file: str = None,
                        topology_file: str = None,
                        num_steps: int = None):
    """
    Analyze power for a single LPM mode.

    Args:
        astrasim_log: Path to AstraSim log file
        network_csv: Path to network statistics CSV
        mode: Mode letter ('A', 'B', 'C', or 'D')
        output_json: Optional path to save results as JSON
        config_path: Optional path to JSON configuration file
        nodemap_file: Optional path to nodemap.json for topology-aware switch modelling
        topology_file: Optional path to NS3 topology file (used with nodemap_file)
        num_steps: Total training steps. When provided, sets
                   compute_stats.iterations so sample counts and energy
                   reflect the full training run.
    """
    print(f"\n🔍 Parsing input files...")
    print(f"  Compute:  {astrasim_log}")
    print(f"  Network:  {network_csv}")
    if nodemap_file:
        print(f"  Nodemap:  {nodemap_file}")
    if topology_file:
        print(f"  Topology: {topology_file}")
    if config_path:
        print(f"  Config:   {config_path}")

    # Load config first so prefix_map / host_prefix reach the nodemap parser
    if config_path:
        config = PowerConfig.from_json(config_path, mode=mode)
    else:
        all_modes = PowerConfig.get_all_modes()
        if mode not in all_modes:
            print(f"❌ Error: Invalid mode '{mode}'. Must be A, B, C, or D.")
            sys.exit(1)
        config = all_modes[mode]

    # Parse input files
    compute_stats, batch_size, seq_length = parse_astrasim_log(astrasim_log)
    if num_steps is not None and num_steps > 0:
        compute_stats.iterations = int(num_steps)
    else:
        compute_stats.iterations = DATA_SET_SIZE // (batch_size * seq_length)
    network_stats = parse_network_statistics_csv(
        network_csv,
        nodemap_file=nodemap_file,
        topology_file=topology_file,
        prefix_map=config.switch_type_prefixes,
        host_prefix=config.host_prefix,
    )
    
    # Run power model
    model = PowerModel(compute_stats, network_stats, config)
    model.print_summary()
    
    # Save results if requested
    if output_json:
        results = model.get_complete_breakdown()
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to: {output_json}\n")
    
    return model


def compare_modes(astrasim_log: str,
                  network_csv: str,
                  output_csv: str = None,
                  output_json: str = None,
                  config_path: str = None,
                  nodemap_file: str = None,
                  topology_file: str = None,
                  num_steps: int = None):
    """
    Compare all 4 LPM modes.

    Args:
        astrasim_log: Path to AstraSim log file
        network_csv: Path to network statistics CSV
        output_csv: Optional path to save comparison as CSV
        output_json: Optional path to save detailed results as JSON
        config_path: Optional path to JSON configuration file
        nodemap_file: Optional path to nodemap.json for topology-aware switch modelling
        topology_file: Optional path to NS3 topology file (used with nodemap_file)
        num_steps: Total training steps. When provided, sets
                   compute_stats.iterations so sample counts and energy
                   reflect the full training run.
    """
    print(f"\n🔍 Parsing input files...")
    print(f"  Compute:  {astrasim_log}")
    print(f"  Network:  {network_csv}")
    if nodemap_file:
        print(f"  Nodemap:  {nodemap_file}")
    if topology_file:
        print(f"  Topology: {topology_file}")
    if config_path:
        print(f"  Config:   {config_path}")

    # Load a base config to forward prefix settings to the nodemap parser.
    # Prefix/host_prefix are mode-independent, so mode='A' is fine here.
    if config_path:
        base_config = PowerConfig.from_json(config_path, mode='A')
    else:
        base_config = PowerConfig.get_all_modes()['A']

    # Parse input files
    compute_stats, batch_size, seq_length = parse_astrasim_log(astrasim_log)
    if num_steps is not None and num_steps > 0:
        compute_stats.iterations = int(num_steps)
    else:
        compute_stats.iterations = DATA_SET_SIZE // (batch_size * seq_length)
    network_stats = parse_network_statistics_csv(
        network_csv,
        nodemap_file=nodemap_file,
        topology_file=topology_file,
        prefix_map=base_config.switch_type_prefixes,
        host_prefix=base_config.host_prefix,
    )

    # Run all modes
    if config_path:
        # Load config for each mode
        results = {}
        for mode_letter in ['A', 'B', 'C', 'D']:
            config = PowerConfig.from_json(config_path, mode=mode_letter)
            results[mode_letter] = PowerModel(compute_stats, network_stats, config)
    else:
        results = compare_all_modes(compute_stats, network_stats)
    
    # Print summaries
    for mode_name in ['A', 'B', 'C', 'D']:
        results[mode_name].print_summary()
    
    # Create comparison table
    print("\n" + "="*120)
    print("📊 4-MODE COMPARISON SUMMARY")
    print("="*120)
    
    comparison_data = []
    for mode_name in ['A', 'B', 'C', 'D']:
        breakdown = results[mode_name].get_complete_breakdown()
        comparison_data.append({
            'Mode': mode_name,
            'Description': breakdown['mode'],
            'Compute_LPM': 'ON' if breakdown['compute_lpm'] else 'OFF',
            'Comm_LPM': 'ON' if breakdown['comm_lpm'] else 'OFF',
            'Total_Power_W': breakdown['total_power_W'],
            'GPU_Power_W': breakdown['gpu_power_W'],
            'Network_Power_W': breakdown['network_power_W'],
            'Total_Energy_J': breakdown['total_energy_J'],
            'Samples_per_J': breakdown['samples_per_joule'],
            'Throughput_samples_s': breakdown['throughput_samples_per_sec']
        })
    
    # Print comparison table
    print(f"\n{'Mode':<6} {'Compute LPM':<12} {'Comm LPM':<10} {'Power (W)':<12} {'Energy (J)':<12} {'Samples/J':<12}")
    print("-" * 120)
    for row in comparison_data:
        print(f"{row['Mode']:<6} {row['Compute_LPM']:<12} {row['Comm_LPM']:<10} "
              f"{row['Total_Power_W']:<12.2f} {row['Total_Energy_J']:<12.2f} {row['Samples_per_J']:<12.4f}")
    
    # Calculate improvements
    print(f"\n📈 EFFICIENCY IMPROVEMENTS vs MODE A (Baseline):")
    baseline_spj = comparison_data[0]['Samples_per_J']
    for row in comparison_data[1:]:
        improvement = ((row['Samples_per_J'] - baseline_spj) / baseline_spj) * 100
        print(f"  Mode {row['Mode']}: {improvement:+.2f}% samples/J improvement")
    
    baseline_energy = comparison_data[0]['Total_Energy_J']
    for row in comparison_data[1:]:
        saving = ((baseline_energy - row['Total_Energy_J']) / baseline_energy) * 100
        print(f"  Mode {row['Mode']}: {saving:.2f}% energy savings")
    
    print("="*120 + "\n")
    
    # Save CSV if requested
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=comparison_data[0].keys())
            writer.writeheader()
            writer.writerows(comparison_data)
        print(f"✅ Comparison table saved to: {output_csv}")
    
    # Save detailed JSON if requested
    if output_json:
        detailed_results = {
            mode: results[mode].get_complete_breakdown()
            for mode in ['A', 'B', 'C', 'D']
        }
        with open(output_json, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"✅ Detailed results saved to: {output_json}")
    
    print()
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Power Model Analysis for AstraSim Workloads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single mode
  python analyze_power.py --compute sim.log --network net_stats.csv --mode A
  
  # Compare all 4 modes
  python analyze_power.py --compute sim.log --network net_stats.csv --compare
  
  # Compare and save results
  python analyze_power.py --compute sim.log --network net_stats.csv --compare \\
      --output-csv comparison.csv --output-json results.json
        """
    )
    
    parser.add_argument('--compute', required=True,
                       help='Path to AstraSim log file')
    parser.add_argument('--network', required=True,
                       help='Path to network statistics CSV file')
    parser.add_argument('--mode', choices=['A', 'B', 'C', 'D'],
                       help='Single mode to analyze (A/B/C/D)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all 4 modes')
    parser.add_argument('--config', type=str,
                       help='Path to JSON power configuration file (uses defaults if not specified)')
    parser.add_argument('--nodemap', type=str,
                       help='Path to nodemap.json for topology-aware switch power modelling')
    parser.add_argument('--topology', type=str,
                       help='Path to NS3 topology file (used together with --nodemap)')
    parser.add_argument('--output-csv',
                       help='Path to save comparison CSV (with --compare)')
    parser.add_argument('--output-json',
                       help='Path to save results JSON')

    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.compute):
        print(f"❌ Error: Compute file not found: {args.compute}")
        sys.exit(1)
    if not os.path.exists(args.network):
        print(f"❌ Error: Network file not found: {args.network}")
        sys.exit(1)
    
    # Validate config file if provided
    if args.config and not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run analysis
    if args.compare:
        compare_modes(
            args.compute, args.network,
            args.output_csv, args.output_json, args.config,
            nodemap_file=args.nodemap,
            topology_file=args.topology,
        )
    elif args.mode:
        analyze_single_mode(
            args.compute, args.network, args.mode,
            args.output_json, args.config,
            nodemap_file=args.nodemap,
            topology_file=args.topology,
        )
    else:
        print("❌ Error: Must specify either --mode or --compare")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
