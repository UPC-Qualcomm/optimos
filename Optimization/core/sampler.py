"""
Sampler: Generate initial sample points for optimization.

This module provides different sampling strategies:
1. Random sampling
2. Latin Hypercube Sampling (LHS)
3. Sobol sequences
4. Grid sampling
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import random
import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for all samplers.
    
    Samplers generate initial sample points to explore the design space.
    Different sampling strategies can provide better coverage or exploit
    problem structure.
    """
    
    @abstractmethod
    def sample(self, design_space: List[Dict], n_samples: int) -> List[Dict]:
        """
        Sample configurations from design space.
        
        Args:
            design_space: List of all valid configuration dictionaries
            n_samples: Number of samples to generate
        
        Returns:
            List of sampled configurations
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomSampler(BaseSampler):
    """
    Random sampling without replacement.
    
    Simple baseline strategy - uniformly samples configurations.
    
    Pros:
    - Simple and fast
    - No bias
    - Works well for large spaces
    
    Cons:
    - May cluster in some regions
    - No guarantee of good coverage
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize random sampler.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def sample(self, design_space: List[Dict], n_samples: int) -> List[Dict]:
        """Sample random configurations without replacement."""
        if n_samples >= len(design_space):
            return design_space.copy()
        
        return random.sample(design_space, n_samples)


class LatinHypercubeSampler(BaseSampler):
    """
    Latin Hypercube Sampling (LHS).
    
    Space-filling design that ensures good coverage across all dimensions.
    Divides each dimension into n equal intervals and samples one point
    from each interval.
    
    Pros:
    - Better coverage than random sampling
    - Ensures spread across all parameter ranges
    - Good for expensive evaluations
    
    Cons:
    - More complex than random sampling
    - Requires mapping to discrete design space
    
    Note: Since our design space is discrete, we approximate LHS by:
    1. Dividing design space into n_samples regions
    2. Sampling one point from each region
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize LHS sampler.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def sample(self, design_space: List[Dict], n_samples: int) -> List[Dict]:
        """
        Sample using Latin Hypercube approach.
        
        For discrete design space, we:
        1. Sort design space by first parameter (alphabetically)
        2. Divide into n_samples regions
        3. Sample one point from each region
        """
        if n_samples >= len(design_space):
            return design_space.copy()
        
        # Sort by first parameter (alphabetically by key name)
        first_key = sorted(design_space[0].keys())[0]
        sorted_space = sorted(design_space, key=lambda x: x[first_key])
        
        # Divide into n_samples strata
        samples = []
        stratum_size = len(sorted_space) // n_samples
        
        for i in range(n_samples):
            start = i * stratum_size
            end = (i + 1) * stratum_size if i < n_samples - 1 else len(sorted_space)
            
            # Sample one point from this stratum
            if start < len(sorted_space):
                stratum = sorted_space[start:end]
                if stratum:
                    samples.append(random.choice(stratum))
        
        return samples


class SobolSampler(BaseSampler):
    """
    Sobol sequence quasi-random sampling.
    
    Low-discrepancy sequence that provides better space coverage than
    random sampling. Particularly good for high-dimensional problems.
    
    Pros:
    - Excellent space coverage
    - Low discrepancy
    - Good for sensitivity analysis
    
    Cons:
    - Requires scipy
    - Requires mapping to discrete space
    
    Note: Requires scipy.stats.qmc.Sobol
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize Sobol sampler.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        
        # Check if scipy is available
        try:
            from scipy.stats import qmc
            self.qmc = qmc
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: scipy not available, falling back to random sampling")
    
    def sample(self, design_space: List[Dict], n_samples: int) -> List[Dict]:
        """
        Sample using Sobol sequence.
        
        Maps Sobol sequence points to discrete design space by:
        1. Generate Sobol sequence in [0,1]^d
        2. Map each point to nearest design space point
        """
        if not self.available:
            # Fallback to random sampling
            return random.sample(design_space, min(n_samples, len(design_space)))
        
        if n_samples >= len(design_space):
            return design_space.copy()
        
        # Generate Sobol sequence
        # Dimension is 1 (we'll map to design space indices)
        sampler = self.qmc.Sobol(d=1, seed=self.seed)
        sobol_points = sampler.random(n_samples)
        
        # Map to design space indices
        indices = (sobol_points[:, 0] * len(design_space)).astype(int)
        indices = np.clip(indices, 0, len(design_space) - 1)
        
        # Ensure unique indices
        indices = np.unique(indices)
        
        # If we don't have enough, add random samples
        while len(indices) < n_samples:
            new_idx = random.randint(0, len(design_space) - 1)
            if new_idx not in indices:
                indices = np.append(indices, new_idx)
        
        return [design_space[i] for i in indices[:n_samples]]


class GridSampler(BaseSampler):
    """
    Uniform grid sampling.
    
    Samples points uniformly across the design space by creating a grid.
    
    Pros:
    - Systematic coverage
    - Reproducible
    - Good for visualization
    
    Cons:
    - Can be inefficient for large spaces
    - May miss important regions between grid points
    """
    
    def __init__(self, shuffle: bool = True, seed: int = None):
        """
        Initialize grid sampler.
        
        Args:
            shuffle: Whether to shuffle the grid points
            seed: Random seed for shuffling
        """
        self.shuffle = shuffle
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def sample(self, design_space: List[Dict], n_samples: int) -> List[Dict]:
        """
        Sample uniformly from design space.
        
        Takes every k-th point to get approximately n_samples.
        """
        if n_samples >= len(design_space):
            samples = design_space.copy()
        else:
            # Calculate step size
            step = max(1, len(design_space) // n_samples)
            samples = design_space[::step][:n_samples]
        
        if self.shuffle:
            random.shuffle(samples)
        
        return samples


class StratifiedSampler(BaseSampler):
    """
    Stratified sampling based on parameter ranges.
    
    Divides parameter space into strata and samples from each.
    Ensures representation from all parameter combinations.
    
    Pros:
    - Ensures diverse parameter coverage
    - Good for analyzing parameter importance
    
    Cons:
    - May oversample some regions
    - More complex than random sampling
    """
    
    def __init__(self, stratify_param: int = 0, seed: int = None):
        """
        Initialize stratified sampler.
        
        Args:
            stratify_param: Which parameter index to stratify on (0=dp, 1=mp, etc.)
            seed: Random seed for reproducibility
        """
        self.stratify_param = stratify_param
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def sample(self, design_space: List[Dict], n_samples: int) -> List[Dict]:
        """
        Sample with stratification.
        
        Groups design space by stratify_param value and samples
        proportionally from each group.
        """
        if n_samples >= len(design_space):
            return design_space.copy()
        
        # Use the stratify_param-th key (sorted alphabetically)
        param_keys = sorted(design_space[0].keys())
        if self.stratify_param < len(param_keys):
            strat_key = param_keys[self.stratify_param]
        else:
            strat_key = param_keys[0]  # Fallback to first param
        
        # Group by stratify parameter
        strata = {}
        for config in design_space:
            key = config[strat_key]
            if key not in strata:
                strata[key] = []
            strata[key].append(config)
        
        # Calculate samples per stratum
        samples = []
        n_strata = len(strata)
        samples_per_stratum = max(1, n_samples // n_strata)
        
        for configs in strata.values():
            n = min(samples_per_stratum, len(configs))
            samples.extend(random.sample(configs, n))
        
        # If we need more samples, add random ones
        if len(samples) < n_samples:
            remaining = [c for c in design_space if c not in samples]
            if remaining:
                additional = random.sample(remaining, 
                                          min(n_samples - len(samples), len(remaining)))
                samples.extend(additional)
        
        return samples[:n_samples]


def get_sampler(name: str, **kwargs) -> BaseSampler:
    """
    Factory function to get sampler by name.
    
    Args:
        name: Sampler name ('random', 'lhs', 'sobol', 'grid', 'stratified')
        **kwargs: Additional arguments for sampler
    
    Returns:
        Sampler instance
    """
    samplers = {
        'random': RandomSampler,
        'lhs': LatinHypercubeSampler,
        'sobol': SobolSampler,
        'grid': GridSampler,
        'stratified': StratifiedSampler,
    }
    
    name = name.lower()
    if name not in samplers:
        raise ValueError(f"Unknown sampler: {name}. Choose from {list(samplers.keys())}")
    
    return samplers[name](**kwargs)
