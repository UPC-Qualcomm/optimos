"""
Compute Trace Parser

Extracts compute and communication timing information from AstraSim log files.
"""

import re
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class NPUStats:
    """Per-NPU timing statistics extracted from a single AstraSim NPU."""
    npu_id: int
    total_time: float    # seconds
    compute_time: float  # seconds
    comm_time: float     # seconds

    @property
    def compute_utilization(self) -> float:
        """Fraction of time this NPU spent in compute."""
        return self.compute_time / self.total_time if self.total_time > 0 else 0.0

    @property
    def comm_utilization(self) -> float:
        """Fraction of time this NPU spent in communication."""
        return self.comm_time / self.total_time if self.total_time > 0 else 0.0


@dataclass
class ComputeStats:
    """Statistics extracted from AstraSim log file."""
    num_npus: int
    batch_size: int
    sequence_length: int
    iterations: int
    # Per-NPU breakdown.  Key = NPU id (0-based).
    npu_stats: Dict[int, NPUStats] = field(default_factory=dict)

    # ── Timing derived from per-NPU data ──────────────────────────────────

    @property
    def total_exec_time(self) -> float:
        """Wall-clock job time = max total_time across all NPUs (seconds).
        Used as the cluster duration for network energy and throughput.
        """
        if self.npu_stats:
            return max(n.total_time for n in self.npu_stats.values())
        return 0.0

    @property
    def avg_comm_time(self) -> float:
        """Average communication time across all NPUs (seconds).
        """
        if self.npu_stats:
            return sum(n.comm_time for n in self.npu_stats.values()) / len(self.npu_stats)
        return 0.0

    @property
    def samples_per_batch(self) -> int:
        """Total samples processed."""
        return self.batch_size * self.sequence_length


def parse_astrasim_log(log_file_path: str):
    """
    Parse AstraSim log file to extract timing and workload information.

    Args:
        log_file_path: Path to AstraSim .log file

    Returns:
        ComputeStats with per-NPU breakdown in npu_stats.

    Log format (spdlog structured output):
        [ts] [workload] [info] [SUMMARY] sys[X] finished, N cycles, exposed communication M cycles,
        [ts] [statistics] [info] sys[X], Wall time: N
        [ts] [statistics] [info] sys[X], GPU time: N
        [ts] [statistics] [info] sys[X], Comm time: N

    Mapping to NPUStats:
        Wall time  → total_time   (actual elapsed cycles for this NPU)
        GPU time   → compute_time (cycles spent executing compute operations)
        Comm time  → comm_time    (total communication time for this NPU)
    """
    clock_freq_hz = 1e9

    batch_size = None
    seq_length = None

    # Parse from filename: e.g. "8_4_1_2_1.seq_2048.batch_512.log"
    filename = log_file_path.split('/')[-1]
    batch_match = re.search(r'batch_(\d+)', filename)
    if batch_match:
        batch_size = int(batch_match.group(1))
    seq_match = re.search(r'seq_(\d+)', filename)
    if seq_match:
        seq_length = int(seq_match.group(1))

    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    # Per-NPU cycles extracted from [statistics] lines.
    # A given NPU id should only appear once for each metric, but we take
    # the max defensively in case the log is ever written more than once.
    npu_wall_cycles:    Dict[int, int] = {}
    npu_compute_cycles: Dict[int, int] = {}
    npu_comm_cycles:    Dict[int, int] = {}

    wall_re = re.compile(r'\[statistics\].*?sys\[(\d+)\],\s+Wall time:\s+(\d+)')
    gpu_re  = re.compile(r'\[statistics\].*?sys\[(\d+)\],\s+GPU time:\s+(\d+)')
    comm_re = re.compile(r'\[statistics\].*?sys\[(\d+)\],\s+Comm time:\s+(\d+)')

    for line in lines:
        m = wall_re.search(line)
        if m:
            nid, cyc = int(m.group(1)), int(m.group(2))
            npu_wall_cycles[nid] = max(npu_wall_cycles.get(nid, 0), cyc)
            continue
        m = gpu_re.search(line)
        if m:
            nid, cyc = int(m.group(1)), int(m.group(2))
            npu_compute_cycles[nid] = max(npu_compute_cycles.get(nid, 0), cyc)
            continue
        m = comm_re.search(line)
        if m:
            nid, cyc = int(m.group(1)), int(m.group(2))
            npu_comm_cycles[nid] = max(npu_comm_cycles.get(nid, 0), cyc)

    # ── Build per-NPU NPUStats ────────────────────────────────────────────
    # Fallback: no stats found → synthesise a single NPU with zero time
    if not npu_wall_cycles:
        npu_wall_cycles = {0: 0}

    npu_stats: Dict[int, NPUStats] = {}
    for npu_id, wall_cyc in npu_wall_cycles.items():
        comp_cyc = npu_compute_cycles.get(npu_id, 0)
        comm_cyc = npu_comm_cycles.get(npu_id, 0)
        npu_stats[npu_id] = NPUStats(
            npu_id=npu_id,
            total_time=wall_cyc   / clock_freq_hz,
            compute_time=comp_cyc / clock_freq_hz,
            comm_time=comm_cyc    / clock_freq_hz,
        )

    if batch_size is None:
        batch_size = 512
    if seq_length is None:
        seq_length = 2048

    return ComputeStats(
        num_npus=len(npu_stats),
        batch_size=batch_size,
        sequence_length=seq_length,
        iterations=1,
        npu_stats=npu_stats,
    ), batch_size, seq_length


def parse_compute_stats_from_dict(data: Dict) -> ComputeStats:
    """
    Create ComputeStats from dictionary (for testing or alternative formats).
    Timing aggregate keys (total_exec_time, compute_time, comm_time) are ignored
    because those are now computed properties derived from npu_stats.

    Args:
        data: Dictionary with keys matching ComputeStats fields

    Returns:
        ComputeStats object
    """
    filtered = {k: v for k, v in data.items()
                if k not in ('total_exec_time', 'compute_time', 'comm_time')}
    return ComputeStats(**filtered)
