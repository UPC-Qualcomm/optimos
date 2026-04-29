"""
TimeStatistics: Time profiling for optimization workflows.

Tracks time spent in different phases of optimization:
- Initialization
- GP model training
- Acquisition optimization
- Workload generation
- Simulation execution
- Result processing
- File cleanup
"""

import time
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class TimeStatistics:
    """
    Time profiling utility for optimizers.
    
    Tracks cumulative time spent in different optimization phases.
    Provides detailed breakdown and statistics.
    
    Usage:
        stats = TimeStatistics(enabled=True)
        
        with stats.timer("initialization"):
            # initialization code
            pass
        
        stats.print_summary()
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize time statistics tracker.
        
        Args:
            enabled: Whether to track and print statistics
        """
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.total_start_time: Optional[float] = None
        self.total_end_time: Optional[float] = None
    
    def start_total(self):
        """Start tracking total optimization time."""
        if self.enabled:
            self.total_start_time = time.time()
    
    def end_total(self):
        """End tracking total optimization time."""
        if self.enabled:
            self.total_end_time = time.time()
    
    def start(self, phase: str):
        """
        Start timing a phase.
        
        Args:
            phase: Name of the phase (e.g., "gp_training", "simulation")
        """
        if self.enabled:
            self.start_times[phase] = time.time()
    
    def end(self, phase: str):
        """
        End timing a phase and record duration.
        
        Args:
            phase: Name of the phase (must match start() call)
        """
        if self.enabled and phase in self.start_times:
            duration = time.time() - self.start_times[phase]
            self.timings[phase].append(duration)
            del self.start_times[phase]
    
    def timer(self, phase: str):
        """
        Context manager for timing a phase.
        
        Args:
            phase: Name of the phase
        
        Usage:
            with stats.timer("simulation"):
                # code to time
                pass
        """
        return _TimerContext(self, phase)
    
    def get_total_time(self, phase: str) -> float:
        """
        Get total time spent in a phase.
        
        Args:
            phase: Name of the phase
        
        Returns:
            Total time in seconds
        """
        if phase in self.timings:
            return sum(self.timings[phase])
        return 0.0
    
    def get_count(self, phase: str) -> int:
        """
        Get number of times a phase was executed.
        
        Args:
            phase: Name of the phase
        
        Returns:
            Number of executions
        """
        if phase in self.timings:
            return len(self.timings[phase])
        return 0
    
    def get_mean_time(self, phase: str) -> float:
        """
        Get mean time per execution for a phase.
        
        Args:
            phase: Name of the phase
        
        Returns:
            Mean time in seconds
        """
        if phase in self.timings and self.timings[phase]:
            return np.mean(self.timings[phase])
        return 0.0
    
    def get_std_time(self, phase: str) -> float:
        """
        Get standard deviation of execution time for a phase.
        
        Args:
            phase: Name of the phase
        
        Returns:
            Std dev in seconds
        """
        if phase in self.timings and len(self.timings[phase]) > 1:
            return np.std(self.timings[phase])
        return 0.0
    
    def get_stats(self, phase: str) -> Dict[str, float]:
        """
        Get comprehensive statistics for a phase.
        
        Args:
            phase: Name of the phase
        
        Returns:
            Dict with total, count, mean, std, min, max
        """
        if phase not in self.timings or not self.timings[phase]:
            return {
                "total": 0.0,
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        times = self.timings[phase]
        return {
            "total": sum(times),
            "count": len(times),
            "mean": np.mean(times),
            "std": np.std(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times)
        }
    
    def get_all_phases(self) -> List[str]:
        """
        Get list of all tracked phases.
        
        Returns:
            List of phase names
        """
        return sorted(self.timings.keys())
    
    def print_summary(self):
        """Print detailed timing summary."""
        if not self.enabled:
            return
        
        if not self.timings:
            print("\nNo timing data collected.")
            return
        
        print("\n" + "="*70)
        print("TIME PROFILING SUMMARY")
        print("="*70)
        
        # Total time
        if self.total_start_time and self.total_end_time:
            total_time = self.total_end_time - self.total_start_time
            print(f"\n⏱️  Total Optimization Time: {total_time:.2f}s")
        
        # Phase breakdown
        print("\n📊 PHASE BREAKDOWN:")
        print("-"*70)
        print(f"{'Phase':<25} {'Count':<8} {'Total(s)':<12} {'Mean(s)':<12} {'%':<8}")
        print("-"*70)
        
        # Calculate total tracked time
        total_tracked = sum(sum(times) for times in self.timings.values())
        
        # Sort phases by total time (descending)
        phases = sorted(
            self.timings.keys(),
            key=lambda p: sum(self.timings[p]),
            reverse=True
        )
        
        for phase in phases:
            stats = self.get_stats(phase)
            percentage = (stats["total"] / total_tracked * 100) if total_tracked > 0 else 0
            
            print(f"{phase:<25} {stats['count']:<8} "
                  f"{stats['total']:<12.2f} {stats['mean']:<12.3f} "
                  f"{percentage:<8.1f}")
        
        print("-"*70)
        print(f"{'TOTAL TRACKED':<25} {'':<8} {total_tracked:<12.2f} {'':<12} {'100.0':<8}")
        
        # Detailed statistics for key phases
        print("\n📈 DETAILED STATISTICS:")
        print("-"*70)
        print(f"{'Phase':<25} {'Min(s)':<10} {'Max(s)':<10} {'Std(s)':<10}")
        print("-"*70)
        
        for phase in phases:
            stats = self.get_stats(phase)
            if stats['count'] > 1:  # Only show std for multiple executions
                print(f"{phase:<25} {stats['min']:<10.3f} "
                      f"{stats['max']:<10.3f} {stats['std']:<10.3f}")
        
        print("="*70 + "\n")
    
    def get_summary_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing summary as dictionary.
        
        Returns:
            Dict mapping phase names to their statistics
        """
        return {phase: self.get_stats(phase) for phase in self.get_all_phases()}
    
    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.start_times.clear()
        self.total_start_time = None
        self.total_end_time = None
    
    def __repr__(self) -> str:
        """String representation."""
        n_phases = len(self.timings)
        total_time = sum(sum(times) for times in self.timings.values())
        return f"TimeStatistics(enabled={self.enabled}, phases={n_phases}, total={total_time:.2f}s)"


class _TimerContext:
    """Context manager for timing a phase."""
    
    def __init__(self, stats: TimeStatistics, phase: str):
        self.stats = stats
        self.phase = phase
    
    def __enter__(self):
        self.stats.start(self.phase)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stats.end(self.phase)
        return False
