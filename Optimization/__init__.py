"""
AstraSim Optimization Framework

A modular, scalable optimization framework for tuning AstraSim configurations.

Quick Start:
    >>> from core.search_space_builder import create_search_space
    >>> from optimizers.random_optimizer import RandomOptimizer
    >>> from optimizers.scikit_bayesian_optimizer import ScikitBayesianOptimizer
    >>> from core.sampler import RandomSampler
    >>> from core.simulation_runner import SimulationRunner
    
    >>> # Setup
    >>> search_space = create_search_space("search_space/parallelism_strategy_params.json")
    >>> sampler = RandomSampler(seed=42)
    >>> sim_runner = SimulationRunner(40, "GPT_40B", 128, "FoldedClos", "my_exp")
    
    >>> # Run
    >>> optimizer = RandomOptimizer(search_space, sampler, sim_runner, budget=20)
    >>> best_config, history = optimizer.run()

See examples/ directory for full documentation.
"""

# Core modules
from .core.search_space_builder import SearchSpaceBuilder, create_search_space
from .core.simulation_runner import SimulationRunner
from .core.simulation_tracker import SimulationTracker
from .core.base_optimizer import BaseOptimizer
from .helper import config_to_tuple, tuple_to_config

# Objective functions
from .core.objective import (
    ObjectiveFunction,
    MinimizeExecutionTime,
    MinimizeExecutionTimeAndNetworkBW,
    MinimizePower,
    MinimizeEnergy,
    MinimizePowerAndTime,
    MinimizeEnergyAndTime,
    MinimizeLatencyAndTotalNetworkBW,
    MinimizeLatencyAndNetworkBW,
    MinimizeLatencyAndMemory,
    MinimizeNetworkBWAndMemory,
    MinimizeLatencyNetworkBWAndMemory,
    MinimizeLatencyAndNetworkBWRaw,
    MinimizeLatencyAndNetworkBWMinMax,
    MinimizeLatencyAndNetworkBWSqrt,
    MinimizeLatencyAndNetworkBWPower,
    MinimizeEDPAndNetworkBW,
    MinimizeED2PAndNetworkBW,
    MinimizeE2DAndNetworkBW,
    MinimizeEnergyCyclesAndNetworkBW,
    MinimizePowerCyclesAndNetworkBW,
    MinimizeWeightedEDP,
    MinimizeTimeMaximizeThroughputPerEnergy,
    MaximizeMemoryMinimizeTime,
    CustomObjective,
    create_objective,
    get_available_objective_types,
)

# Samplers
from .core.sampler import (
    BaseSampler,
    RandomSampler,
    LatinHypercubeSampler,
    SobolSampler,
    GridSampler,
    StratifiedSampler,
    get_sampler
)


# Try to import DeepHyper optimizer
try:
    from .optimizers import DeepHyperOptimizer
    DEEPHYPER_AVAILABLE = True
except ImportError:
    DEEPHYPER_AVAILABLE = False

# Build exports list
if DEEPHYPER_AVAILABLE:
    __all__ = [
        # Core
        'SearchSpaceBuilder',
        'create_search_space',
        'SimulationRunner',
        'SimulationTracker',
        'BaseOptimizer',
        # Helpers
        'config_to_tuple',
        'tuple_to_config',
        # Objectives
        'ObjectiveFunction',
        'MinimizeExecutionTime',
        'MinimizeExecutionTimeAndNetworkBW',
        'MinimizePower',
        'MinimizeEnergy',
        'MinimizePowerAndTime',
        'MinimizeEnergyAndTime',
        'MinimizeLatencyAndTotalNetworkBW',
        'MinimizeLatencyAndNetworkBW',
        'MinimizeLatencyAndMemory',
        'MinimizeNetworkBWAndMemory',
        'MinimizeLatencyNetworkBWAndMemory',
        'MinimizeLatencyAndNetworkBWRaw',
        'MinimizeLatencyAndNetworkBWMinMax',
        'MinimizeLatencyAndNetworkBWSqrt',
        'MinimizeLatencyAndNetworkBWPower',
        'MinimizeEDPAndNetworkBW',
        'MinimizeED2PAndNetworkBW',
        'MinimizeE2DAndNetworkBW',
        'MinimizeEnergyCyclesAndNetworkBW',
        'MinimizePowerCyclesAndNetworkBW',
        'MinimizeWeightedEDP',
        'MinimizeTimeMaximizeThroughputPerEnergy',
        'MaximizeMemoryMinimizeTime',
        'CustomObjective',
        'create_objective',
        'get_available_objective_types',
        # Samplers
        'BaseSampler',
        'RandomSampler',
        'LatinHypercubeSampler',
        'SobolSampler',
        'GridSampler',
        'StratifiedSampler',
        'get_sampler',
        # Optimizers
        'RandomOptimizer',
        'ScikitBayesianOptimizer',
        'DeepHyperOptimizer',
        # Kernels
        'BaseKernel',
        'MaternKernel',
        'RBFKernel',
        'CustomKernel',
        'CompositeKernel',
        'get_kernel',
        # Acquisition
        'BaseAcquisitionFunction',
        'ExpectedImprovement',
        'UpperConfidenceBound',
        'ProbabilityOfImprovement',
        'ThompsonSampling',
        'get_acquisition',
    ]

__version__ = '1.0.0'
__author__ = 'AstraSim Optimization Team'
