"""
Configuration utilities for handling dictionary configurations.

This module provides common utilities for working with configuration dictionaries,
particularly for converting them to hashable tuples for use in sets and other
data structures that require hashable types.

Also provides worker functions for parallel evaluation using multiprocessing.
"""

from typing import Dict, Tuple, Optional


def enrich_config_with_clusters(config: Dict, clusters: Dict) -> Dict:
    """
    Enrich a config with cluster information and reconstruct collective implementations.
    
    This is a module-level function (can be pickled) that duplicates the logic
    from SearchSpaceBuilder.enrich_config_with_cluster_info() for use in
    multiprocessing workers.
    
    Args:
        config: Configuration dict potentially containing 'cluster' key and per-dimension collective params
        clusters: Dictionary mapping cluster names to cluster configs
    
    Returns:
        Enriched configuration with npu_count, npus_per_dim, num_dimensions, and reconstructed collectives
    """
    if 'cluster' not in config or not clusters:
        return config
    
    cluster_name = config['cluster']
    if cluster_name not in clusters:
        return config
    
    # Create a copy to avoid modifying the original
    enriched = config.copy()
    
    # Reconstruct collective implementations from per-dimension parameters
    collective_types = set()
    for key in list(enriched.keys()):
        if '-dim0' in key:
            collective_type = key.replace('-dim0', '')
            collective_types.add(collective_type)
    
    for collective_type in collective_types:
        algorithms = []
        dim = 0
        while f'{collective_type}-dim{dim}' in enriched:
            algorithms.append(enriched[f'{collective_type}-dim{dim}'])
            del enriched[f'{collective_type}-dim{dim}']
            dim += 1
        
        if algorithms:
            enriched[collective_type] = algorithms
    
    # Add cluster info to config
    cluster_config = clusters[cluster_name]
    enriched['npu_count'] = cluster_config['npu_count']
    enriched['npus_per_dim'] = cluster_config['npus_per_dim']
    enriched['num_dimensions'] = len(cluster_config['npus_per_dim'])
    
    return enriched


def config_to_tuple(config: Dict) -> tuple:
    """
    Convert a configuration dictionary to a hashable tuple.
    
    This is useful when configs need to be stored in sets or used as dictionary keys.
    The tuple contains sorted (key, value) pairs to ensure consistent ordering.
    
    Args:
        config: Configuration dictionary (e.g., {'dp': 1, 'mp': 2, 'sp': 8, ...})
    
    Returns:
        Hashable tuple of sorted (key, value) pairs
        
    Example:
        >>> config = {'dp': 1, 'mp': 2, 'sp': 8, 'pp': 8, 'sharded': True}
        >>> config_to_tuple(config)
        (('dp', 1), ('mp', 2), ('pp', 8), ('sharded', True), ('sp', 8))
    """
    return tuple(sorted(config.items()))


def tuple_to_config(config_tuple: tuple) -> Dict:
    """
    Convert a configuration tuple back to a dictionary.
    
    This is the inverse operation of config_to_tuple().
    
    Args:
        config_tuple: Tuple of (key, value) pairs
    
    Returns:
        Configuration dictionary
        
    Example:
        >>> config_tuple = (('dp', 1), ('mp', 2), ('pp', 8), ('sharded', True), ('sp', 8))
        >>> tuple_to_config(config_tuple)
        {'dp': 1, 'mp': 2, 'pp': 8, 'sharded': True, 'sp': 8}
    """
    return dict(config_tuple)


def evaluate_config_worker(config: Dict, simulation_runner, clusters: Optional[Dict] = None) -> Tuple[Dict, Optional[float], Dict, Dict]:
    """
    Worker function for parallel configuration evaluation.
    
    This is a module-level function so it can be pickled by multiprocessing.
    Used by optimizers that support parallel evaluation.
    
    Note: This function does NOT record results in the optimizer's history.
    It only evaluates and returns the raw results. The calling optimizer
    is responsible for storing results.
    
    Args:
        config: Configuration dictionary to evaluate
        simulation_runner: SimulationRunner instance to run the simulation
        clusters: Optional dict mapping cluster names to cluster configs for enrichment
    
    Returns:
        Tuple of (config, exec_time, file_paths, metadata):
            - config: The input configuration
            - exec_time: Execution time in seconds (None if evaluation failed)
            - file_paths: Dict of file paths generated during simulation
            - metadata: Dict of additional metadata from simulation
    
    Example:
        >>> from functools import partial
        >>> from multiprocessing import Pool
        >>> from Optimization.helper import evaluate_config_worker
        >>> 
        >>> worker = partial(evaluate_config_worker, 
        ...                  simulation_runner=optimizer.simulation_runner,
        ...                  clusters=optimizer.search_space.clusters)
        >>> with Pool(4) as pool:
        >>>     results = pool.map(worker, configs)
    """
    try:
        # Enrich config with cluster info if cluster parameter exists
        if clusters and 'cluster' in config:
            config = enrich_config_with_clusters(config, clusters)
        
        result = simulation_runner.run_simulation(config, return_paths=True)
        
        if result is not None:
            if isinstance(result, tuple) and len(result) == 4:
                exec_time, is_oom, file_paths, metadata = result
            elif isinstance(result, tuple) and len(result) == 3:
                exec_time, is_oom, file_paths = result
                metadata = {}
            else:
                exec_time, is_oom = result
                file_paths = {}
                metadata = {}
            
            return config, exec_time, is_oom, file_paths, metadata
        else:
            return config, None, None, {}, {}
            
    except Exception as e:
        print(f"⚠️  Evaluation error for config {config}: {e}")
        return config, None, None, {}, {}