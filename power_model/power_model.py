"""
Integrated Power Model

Top-level power model combining compute and network components.
Calculates total power, energy, and MLPerf Samples/J metric.
"""

from typing import Dict
from power_config import PowerConfig
from compute_parser import ComputeStats
from network_parser import NetworkStats
from compute_model import ComputeModel
from network_model import NetworkModel


class PowerModel:
    """
    Complete power model for AstraSim cluster.
    
    Supports 4 operational modes:
    - Mode A: No LPM (baseline)
    - Mode B: Compute LPM only
    - Mode C: Communication LPM only
    - Mode D: Full LPM (energy-proportional)
    """
    
    def __init__(self, 
                 compute_stats: ComputeStats,
                 network_stats: NetworkStats,
                 config: PowerConfig):
        """
        Initialize power model.
        
        Args:
            compute_stats: Parsed compute statistics from AstraSim log
            network_stats: Parsed network statistics from CSV
            config: Power configuration with LPM settings
        """
        self.compute_stats = compute_stats
        self.network_stats = network_stats
        self.config = config
        
        # Create component models
        self.compute_model = ComputeModel(compute_stats, config)
        self.network_model = NetworkModel(
            network_stats,
            compute_stats.total_exec_time,
            config,
            comm_time=compute_stats.avg_comm_time,
        )
    
    def total_power(self) -> float:
        """
        Calculate total system power consumption.
        
        P_total = P_GPU + P_network
        
        Returns:
            Total power in Watts
        """
        return self.compute_model.total_gpu_power() + \
               self.network_model.total_network_power()
    
    def total_energy(self) -> float:
        """
        Calculate total energy consumed.

        GPU energy is computed per-NPU (power_i × own_wall_time_i) and summed,
        which is more accurate than using a single cluster wall-clock time.
        Network infrastructure (switches, links) runs for the full job duration,
        so its energy still uses the cluster wall-clock time.

        E_total = Σ_i (P_GPU_i × T_wall_i) + P_network × T_exec

        Returns:
            Total energy in Joules
        """
        return (self.compute_model.total_gpu_energy() +
                self.network_model.total_network_energy())
    
    def samples_per_joule(self) -> float:
        """
        Calculate MLPerf Power metric: Samples per Joule.
        
        Samples/J = samples_per_batch / E_total
        
        Where:
            samples_per_batch = Batch_Size × Sequence_Length
        
        Returns:
            Samples per Joule
        """
        samples_per_batch = self.compute_stats.samples_per_batch
        energy = self.total_energy()
        
        return samples_per_batch / energy if energy > 0 else 0.0
    
    def throughput_samples_per_sec(self) -> float:
        """
        Calculate throughput in samples per second.
        
        Returns:
            Samples/second
        """
        return self.compute_stats.samples_per_batch / \
               self.compute_stats.total_exec_time \
               if self.compute_stats.total_exec_time > 0 else 0.0
    
    def samples_per_sec_per_megajoule(self) -> float:
        """
        Combined throughput-energy efficiency metric.

        samples / (second × MJ) = throughput_samples_per_sec / (total_energy_J / 1e6)

        Captures both speed and energy cost in one number.
        Higher is better.  Convenient scale: typical DNN training runs land
        in the range 0.01 – 100 samples/(s·MJ) depending on cluster size.

        Returns:
            Samples per (second × megajoule)
        """
        energy_mj = self.total_energy() / 1e6
        return self.throughput_samples_per_sec() / energy_mj if energy_mj > 0 else 0.0

    def power_efficiency_watts_per_sample_per_sec(self) -> float:
        """
        Alternative efficiency metric: W / (samples/sec).
        
        Returns:
            Watts per (sample/sec)
        """
        throughput = self.throughput_samples_per_sec()
        return self.total_power() / throughput if throughput > 0 else float('inf')
    
    def get_complete_breakdown(self) -> Dict:
        """
        Get comprehensive power, energy, and performance breakdown.
        
        Returns:
            Dictionary with all metrics
        """
        compute_breakdown = self.compute_model.get_power_breakdown()
        network_breakdown = self.network_model.get_power_breakdown()

        # Scale energy and timing to the full training run.
        # The underlying models compute per-step values; iterations carries
        # the number of training steps set by the caller.
        n = self.compute_stats.iterations
        total_energy  = self.total_energy() * n
        gpu_energy    = compute_breakdown['total_gpu_energy_J'] * n
        net_energy    = network_breakdown['total_network_energy_J'] * n
        total_samples = self.compute_stats.samples_per_batch          # batch_size × sequence_length
        total_time    = self.compute_stats.total_exec_time * n    # full training wall time

        throughput = total_samples / total_time if total_time > 0 else 0.0
        spj        = total_samples / total_energy if total_energy > 0 else 0.0
        energy_mj  = total_energy / 1e6
        sps_per_mj = throughput / energy_mj if energy_mj > 0 else 0.0
        
        return {
            # Mode information
            'mode': self.config.get_mode_name(),
            'compute_lpm': self.config.compute_lpm_enabled,
            'comm_lpm': self.config.comm_lpm_enabled,
            
            # Timing  (mean across NPUs; breakdown in compute_breakdown['per_npu'])
            'total_execution_time_s': total_time,
            'compute_time_s':         compute_breakdown['mean_compute_time_s'],
            'comm_time_s':            compute_breakdown['mean_comm_time_s'],

            # Utilization (mean / min / max across NPUs)
            'compute_utilization':    compute_breakdown['mean_compute_util'],
            'min_compute_util':       compute_breakdown['min_compute_util'],
            'max_compute_util':       compute_breakdown['max_compute_util'],
            'comm_utilization':       compute_breakdown['mean_comm_util'],
            'min_comm_util':          compute_breakdown['min_comm_util'],
            'max_comm_util':          compute_breakdown['max_comm_util'],
            
            # Power (Watts)
            'total_power_W': self.total_power(),
            'gpu_power_W': compute_breakdown['total_gpu_power_W'],
            'network_power_W': network_breakdown['total_network_power_W'],
            'link_power_W': network_breakdown['link_power_W'],
            'total_switch_power_W': network_breakdown['total_switch_power_W'],
            
            # Energy (Joules) — scaled to full training run
            'total_energy_J': total_energy,
            'gpu_energy_J': gpu_energy,
            'network_energy_J': net_energy,
            # Per-step energy (single simulation step)
            'total_energy_J_per_step': self.total_energy(),
            'gpu_energy_J_per_step': compute_breakdown['total_gpu_energy_J'],
            'network_energy_J_per_step': network_breakdown['total_network_energy_J'],
            
            # Performance
            'batch_size': self.compute_stats.batch_size,
            'iterations': self.compute_stats.iterations,
            'samples_per_batch_per_step': self.compute_stats.samples_per_batch,
            'throughput_samples_per_sec_per_step': self.throughput_samples_per_sec(),
            'total_samples': total_samples,
            'throughput_samples_per_sec': throughput,
            
            # Efficiency Metrics
            'samples_per_joule_per_step':      self.samples_per_joule(),
            'joules_per_sample_per_step':      1.0 / self.samples_per_joule() if self.samples_per_joule() > 0 else float('inf'),
            'samples_per_sec_per_mj_per_step': self.samples_per_sec_per_megajoule(),
            'samples_per_joule':      spj,
            'joules_per_sample':      1.0 / spj if spj > 0 else float('inf'),
            'samples_per_sec_per_mj': sps_per_mj,
            
            # Hardware counts
            'num_gpus': self.compute_stats.num_npus,
            'num_links': network_breakdown['num_links'],
            'total_switch_count': network_breakdown['total_switch_count'],
            # Per-type switch breakdown (generic – works for any topology)
            'switch_power_by_type': {
                sw_type: {
                    'power_W': network_breakdown[f'{sw_type}_power_W'],
                    'count':   network_breakdown[f'{sw_type}_count'],
                }
                for sw_type in self.network_model.switch_models
            },
        }
    
    def print_summary(self):
        """Print human-readable summary of results."""
        breakdown = self.get_complete_breakdown()
        
        print("\n" + "="*80)
        print(f"POWER MODEL RESULTS - {breakdown['mode']}")
        print("="*80)
        
        print(f"\nEXECUTION PROFILE:")
        print(f"  Total Time:          {breakdown['total_execution_time_s']:.3f} s")
        print(f"  Compute Time (mean): {breakdown['compute_time_s']:.3f} s  "
              f"util {breakdown['compute_utilization']:.1%} "
              f"[{breakdown['min_compute_util']:.1%} – {breakdown['max_compute_util']:.1%}]")
        print(f"  Comm Time (mean):    {breakdown['comm_time_s']:.3f} s  "
              f"util {breakdown['comm_utilization']:.1%} "
              f"[{breakdown['min_comm_util']:.1%} – {breakdown['max_comm_util']:.1%}]")
        
        print(f"\nPOWER CONSUMPTION:")
        print(f"  Total Power:         {breakdown['total_power_W']:.2f} W")
        print(f"    GPU Power:         {breakdown['gpu_power_W']:.2f} W ({breakdown['gpu_power_W']/breakdown['total_power_W']:.1%})")
        print(f"    Network Power:     {breakdown['network_power_W']:.2f} W ({breakdown['network_power_W']/breakdown['total_power_W']:.1%})")
        print(f"      Links:           {breakdown['link_power_W']:.2f} W")
        print(f"      Switches:        {breakdown['total_switch_power_W']:.2f} W")
        for sw_type, sw_info in breakdown['switch_power_by_type'].items():
            print(f"        {sw_type:<16} {sw_info['power_W']:.2f} W  ({sw_info['count']} units)")
        
        print(f"\nENERGY CONSUMPTION:")
        print(f"  Total Energy:        {breakdown['total_energy_J']:.2f} J")
        print(f"    GPU Energy:        {breakdown['gpu_energy_J']:.2f} J ({breakdown['gpu_energy_J']/breakdown['total_energy_J']:.1%})")
        print(f"    Network Energy:    {breakdown['network_energy_J']:.2f} J ({breakdown['network_energy_J']/breakdown['total_energy_J']:.1%})")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Total Samples:       {breakdown['total_samples']:,}")
        print(f"  Throughput:          {breakdown['throughput_samples_per_sec']:.2f} samples/s")
        
        print(f"\nEFFICIENCY (MLPerf Power Metric):")
        print(f"  Samples per Joule:   {breakdown['samples_per_joule']:.4f} samples/J")
        print(f"  Joules per Sample:   {breakdown['joules_per_sample']:.4f} J/sample")
        print(f"  Samples/(s·MJ):      {breakdown['samples_per_sec_per_mj']:.6f}")
        
        print(f"\nHARDWARE CONFIGURATION:")
        print(f"  GPUs:                {breakdown['num_gpus']}")
        print(f"  Links:               {breakdown['num_links']}")
        print(f"  Total Switches:      {breakdown['total_switch_count']}")
        for sw_type, sw_info in breakdown['switch_power_by_type'].items():
            print(f"    {sw_type:<20} {sw_info['count']}")
        
        print("="*80 + "\n")


def compare_all_modes(compute_stats: ComputeStats,
                      network_stats: NetworkStats) -> Dict[str, PowerModel]:
    """
    Run power analysis for all 4 LPM modes.
    
    Args:
        compute_stats: Compute statistics
        network_stats: Network statistics
        
    Returns:
        Dictionary mapping mode name to PowerModel instance
    """
    all_configs = PowerConfig.get_all_modes()
    results = {}
    
    for mode_name, config in all_configs.items():
        model = PowerModel(compute_stats, network_stats, config)
        results[mode_name] = model
    
    return results
