"""
power_estimator.py — Power estimation helper for the optimization framework.

Wraps the AstraSim power model (Mode D: Full LPM, Energy-Proportional) so
that SimulationRunner can call it with a single function after a successful
g2 simulation.

The function is intentionally kept stateless (no class needed) so it can be
imported and called from anywhere without a runner instance.
"""

import io
import os
import sys
import contextlib
from typing import Dict, Optional


def run_power_estimation(
    output_dir: str,
    network_config: str,
    base_dir: str,
    workload_file: str,
    power_config_path: Optional[str] = None,
    verbose: bool = False,
    num_steps: int = None
) -> Dict:
    """
    Run the power model (Mode D: Full LPM) for a completed g2 simulation.

    Locates the ``.log`` and ``_link_traffic.csv`` files produced by AstraSim
    inside *output_dir*, derives the nodemap from the ``topology_file`` field
    in the network YAML, and calls ``analyze_single_mode(..., mode='D')``.
    All power-model console output is suppressed unless *verbose* is ``True``.

    Args:
        output_dir:        Path to the AstraSim output directory
                           (same as ``--output_dir`` passed to run_astrasim).
        network_config:    Absolute or base-dir-relative path to the network
                           YAML used for the simulation.
        base_dir:          OPTIMOS base directory, used to resolve relative
                           *network_config* paths.
        workload_file:     Path to the workload file **without** its ``.0.et``
                           extension,
                           e.g. ``<output_dir>/4_8_2_2_1.seq_16384.batch_512``.
        power_config_path: Path to the power model JSON config.  Defaults to
                           ``<upc_root>/power_model/a100_config.json``.
        verbose:           If ``True``, print progress messages and forward the
                           power model's stdout.

    Returns:
        Dict with Mode-D power metrics on success::

            {
                'total_power_W'             : float,
                'gpu_power_W'               : float,
                'network_power_W'           : float,
                'total_energy_J'            : float,
                'gpu_energy_J'              : float,
                'network_energy_J'          : float,
                'throughput_samples_per_sec': float,
                'samples_per_joule'         : float,
                'power_mode'                : 'D',
            }

        Empty dict ``{}`` if any required input file is missing or an error
        occurs during estimation.
    """
    try:
        import yaml

        # Resolve OPTIMOS root robustly.
        # Prefer explicit base_dir from SimulationRunner, then ASTRA_SIM_ROOT.
        upc_root = base_dir if base_dir else os.environ.get('OPTIMOS_ROOT', '')
        upc_root = os.path.abspath(upc_root)
        power_model_dir = os.path.join(upc_root, 'power_model')

        # Make sure local power_model modules are imported from the expected location.
        # Importing analyze_power as a top-level module mirrors estimate_power.py and
        # guarantees sibling imports (e.g., nodemap_parser) resolve correctly.
        if power_model_dir not in sys.path:
            sys.path.insert(0, power_model_dir)
        if upc_root not in sys.path:
            sys.path.insert(0, upc_root)
        from analyze_power import analyze_single_mode

        workload_filename = os.path.basename(workload_file)
        log_file     = os.path.join(output_dir, workload_filename + ".log")
        link_traffic = os.path.join(output_dir, workload_filename + "_link_traffic.csv")
        output_json  = os.path.join(output_dir, workload_filename + "_power_est.json")
        if not os.path.isfile(log_file):
            if verbose:
                print(f"    ⚠️  Power estimation skipped: log file not found: {log_file}")
            return {}

        if not os.path.isfile(link_traffic):
            if verbose:
                print(f"    ⚠️  Power estimation skipped: link_traffic CSV not found: {link_traffic}")
            return {}

        # Derive nodemap path from topology_file inside the network YAML
        network_path = (network_config
                        if os.path.isabs(network_config)
                        else os.path.join(base_dir, network_config))
        nodemap = None
        try:
            with open(network_path, 'r') as f:
                net_cfg = yaml.safe_load(f)
            topo_file = net_cfg.get('topology_file', '').strip().strip('"').strip("'")
            if topo_file:
                candidate = topo_file + "_nodemap.json"
                if os.path.isfile(candidate):
                    nodemap = candidate
                elif verbose:
                    print(f"    ⚠️  Nodemap not found at {candidate}; "
                          "switch-level power will fall back to link-only estimation")
        except Exception as exc:
            if verbose:
                print(f"    ⚠️  Could not parse network YAML for nodemap: {exc}")

        # Power config: use provided path or default a100_config.json
        if power_config_path is None:
            power_config_path = os.path.join(upc_root, 'power_model', 'a100_config.json')

        if verbose:
            print(f"    ⚡ Running power estimation (Mode D - Full LPM)...")

        # Run power model, suppressing its stdout unless verbose
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = analyze_single_mode(
                log_file,
                link_traffic,
                mode='D',
                output_json=output_json,
                config_path=power_config_path,
                nodemap_file=nodemap,
                num_steps=num_steps
            )

        if verbose:
            print(buf.getvalue(), end='')

        breakdown = model.get_complete_breakdown()
        power_metrics = {
            'total_power_W':              float(breakdown.get('total_power_W', 0.0)),
            'gpu_power_W':                float(breakdown.get('gpu_power_W', 0.0)),
            'network_power_W':            float(breakdown.get('network_power_W', 0.0)),
            'total_energy_J':             float(breakdown.get('total_energy_J', 0.0)),
            'gpu_energy_J':               float(breakdown.get('gpu_energy_J', 0.0)),
            'network_energy_J':           float(breakdown.get('network_energy_J', 0.0)),
            'throughput_samples_per_sec': float(breakdown.get('throughput_samples_per_sec', 0.0)),
            'samples_per_joule':          float(breakdown.get('samples_per_joule', 0.0)),
            'samples_per_sec_per_mj':          float(breakdown.get('samples_per_sec_per_mj', 0.0)),
            'power_mode':                 'D',
        }

        if verbose:
            print(f"    ✓ Power: {power_metrics['total_power_W']:.2f} W, "
                  f"Energy: {power_metrics['total_energy_J']:.2f} J")

        return power_metrics

    except Exception as exc:
        if verbose:
            print(f"    ⚠️  Power estimation failed: {exc}")
        return {}
