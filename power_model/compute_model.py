"""
Compute Power Model

Calculates GPU power consumption based on compute utilization and LPM settings.
Power is calculated per-NPU using individual utilization, then summed.
"""

from typing import Dict
from power_config import PowerConfig
from compute_parser import ComputeStats


class ComputeModel:
    """
    Models GPU compute power with LPM support.

    Per-NPU formula:
      LPM OFF: P_GPU(i) = P_idle  + (P_peak - P_idle)  * U_compute(i)
      LPM ON:  P_GPU(i) = P_sleep + (P_peak - P_sleep) * U_compute(i)

    Where U_compute(i) = T_compute(i) / T_exec(i) for each NPU individually.
    Total power = Σ P_GPU(i) over all NPUs.
    """

    def __init__(self, compute_stats: ComputeStats, config: PowerConfig):
        self.stats      = compute_stats
        self.config     = config
        self.num_gpus   = compute_stats.num_npus
        self.total_time = compute_stats.total_exec_time

    def compute_utilization(self, npu_id: int) -> float:
        """Compute utilization for a specific NPU (compute_time / wall_time)."""
        if self.stats.npu_stats and npu_id in self.stats.npu_stats:
            return self.stats.npu_stats[npu_id].compute_utilization
        return 0.0

    def _npu_ids(self):
        """Ordered list of NPU ids to iterate over."""
        if self.stats.npu_stats:
            return sorted(self.stats.npu_stats.keys())
        return list(range(self.num_gpus))

    def npu_wall_time(self, npu_id: int) -> float:
        """Wall-clock time for a specific NPU (seconds)."""
        if self.stats.npu_stats and npu_id in self.stats.npu_stats:
            return self.stats.npu_stats[npu_id].total_time
        return self.total_time  # fallback to cluster max

    def compute_gpu_power(self, npu_id: int = 0) -> float:
        """Power for a single NPU (Watts)."""
        U = self.compute_utilization(npu_id)
        if self.config.compute_lpm_enabled:
            return self.config.gpu_sleep_power + \
                   (self.config.gpu_peak_power - self.config.gpu_sleep_power) * U
        return self.config.gpu_idle_power + \
               (self.config.gpu_peak_power - self.config.gpu_idle_power) * U

    def compute_gpu_energy(self, npu_id: int = 0) -> float:
        """Energy for a single NPU (Joules) = power × that NPU's own wall time."""
        return self.compute_gpu_power(npu_id) * self.npu_wall_time(npu_id)

    def total_gpu_power(self) -> float:
        """Sum of per-NPU powers (Watts)."""
        return sum(self.compute_gpu_power(npu_id) for npu_id in self._npu_ids())

    def total_gpu_energy(self) -> float:
        """Total GPU energy (Joules) = sum of per-NPU (power × own wall time)."""
        return sum(self.compute_gpu_energy(npu_id) for npu_id in self._npu_ids())

    def get_power_breakdown(self) -> Dict:
        """
        Detailed power breakdown, including per-NPU utilization.
        """
        per_npu = {}
        for npu_id in self._npu_ids():
            ns = self.stats.npu_stats.get(npu_id)
            per_npu[npu_id] = {
                'compute_utilization': self.compute_utilization(npu_id),
                'comm_utilization':    ns.comm_utilization  if ns else 0.0,
                'compute_time_s':      ns.compute_time      if ns else 0.0,
                'comm_time_s':         ns.comm_time         if ns else 0.0,
                'wall_time_s':         self.npu_wall_time(npu_id),
                'power_W':             self.compute_gpu_power(npu_id),
                'energy_J':            self.compute_gpu_energy(npu_id),
            }

        vals = list(per_npu.values())
        def _mean(key): return sum(v[key] for v in vals) / len(vals) if vals else 0.0
        def _min(key):  return min(v[key] for v in vals)             if vals else 0.0
        def _max(key):  return max(v[key] for v in vals)             if vals else 0.0

        return {
            'total_gpu_power_W':      self.total_gpu_power(),
            'total_gpu_energy_J':     self.total_gpu_energy(),
            'num_gpus':               self.num_gpus,
            # compute
            'mean_compute_util':      _mean('compute_utilization'),
            'min_compute_util':       _min('compute_utilization'),
            'max_compute_util':       _max('compute_utilization'),
            'mean_compute_time_s':    _mean('compute_time_s'),
            # comm
            'mean_comm_util':         _mean('comm_utilization'),
            'min_comm_util':          _min('comm_utilization'),
            'max_comm_util':          _max('comm_utilization'),
            'mean_comm_time_s':       _mean('comm_time_s'),
            'total_time_s':           self.total_time,
            'lpm_enabled':            self.config.compute_lpm_enabled,
            'per_npu':                per_npu,
        }
