"""
Config File Generator: Single source of truth for all AstraSim configuration files.

This module is the ONLY way to get configuration file paths in the optimization framework.
The optimizer and simulation runner ALWAYS use this generator - never user-provided paths.

If custom parameters (net_*, sys_*, mem_*) are provided in the search space config, they
override the defaults. Otherwise, configs are generated from defaults.

All generated configs are stored in: ./Optimization/temp/config/

Usage:
    from helper import config_generator
    
    # Generate configs (always, with or without custom params)
    system_path = config_generator.generate_system_config(config)
    network_path = config_generator.generate_network_config(config)
    memory_path = config_generator.generate_memory_config(config)
"""

import json
import yaml
import os
import tempfile
from typing import Dict, Any


# Get the Optimization directory (parent of helper directory)
_HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
_OPTIMIZATION_DIR = os.path.dirname(_HELPER_DIR)

# Config output directory (absolute path)
CONFIG_OUTPUT_DIR = os.path.join(_OPTIMIZATION_DIR, "temp", "config")

# Caches for config paths - reuse configs when parameters don't change
_SYSTEM_CONFIG_CACHE = {}  # Maps config hash to file path
_NETWORK_CONFIG_CACHE = {}  # Maps config hash to file path
_MEMORY_CONFIG_CACHE = {}  # Maps config hash to file path


def _hash_config(config_dict: Dict[str, Any]) -> str:
    """Create a hash of config dict for caching purposes."""
    import hashlib
    # Convert dict to sorted JSON string for consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def _map_topology_to_network_type(topology: str) -> str:
    """Map high-level topology name to AstraSim network type.
    
    Args:
        topology: High-level topology name ('FoldedClos', 'Dragonfly', 'Torus')
    
    Returns:
        AstraSim network type ('Switch', 'FullyConnected', 'Ring')
    """
    topology_map = {
        'FoldedClos': 'Switch',
        'Dragonfly': 'FullyConnected',
        'Torus': 'Ring'
    }
    return topology_map.get(topology, 'Switch')


def _normalize_values_per_dimension(values: Any, num_dims: int, default_value: Any) -> list:
    """Normalize scalar/list values to exactly num_dims entries.

    - Scalar -> repeated for all dims
    - List with exact length -> unchanged
    - List longer than dims -> truncated
    - List shorter than dims -> padded with its last value
    - Empty/invalid -> default repeated
    """
    if num_dims <= 0:
        return []

    if isinstance(values, list):
        if not values:
            return [default_value] * num_dims
        if len(values) >= num_dims:
            return values[:num_dims]
        return values + [values[-1]] * (num_dims - len(values))

    if isinstance(values, str):
        return [values] * num_dims

    return [default_value] * num_dims


def _enforce_system_collective_dimensions(system_config: Dict[str, Any], num_dims: int) -> Dict[str, Any]:
    """Ensure all collective implementation arrays exactly match num_dims."""
    fixed = dict(system_config)
    for collective in ['all-reduce', 'all-gather', 'reduce-scatter', 'all-to-all']:
        impl_key = f'{collective}-implementation'
        fixed[impl_key] = _normalize_values_per_dimension(
            fixed.get(impl_key, ['halvingDoubling']),
            num_dims,
            'halvingDoubling'
        )
    return fixed


def _build_collective_implementations(config: Dict[str, Any], num_dims: int) -> Dict[str, list]:
    """Build collective implementation arrays for each dimension.
    
    Args:
        config: Configuration dictionary that may contain:
                - 'all-reduce': single value or list per dimension
                - 'all-gather': single value or list per dimension
                - 'reduce-scatter': single value or list per dimension
                - 'all-to-all': single value or list per dimension
        num_dims: Number of network dimensions
    
    Returns:
        Dictionary with collective implementation lists for each collective type
    """
    collectives = ['all-reduce', 'all-gather', 'reduce-scatter', 'all-to-all']
    result = {}
    
    for collective in collectives:
        key = f'{collective}-implementation'
        if collective in config:
            result[key] = _normalize_values_per_dimension(
                config[collective],
                num_dims,
                'halvingDoubling'
            )
        else:
            # Not specified, use default
            result[key] = ['halvingDoubling'] * num_dims
    
    return result


# Default configurations
DEFAULT_SYSTEM_CONFIG = {
    "scheduling-policy": "LIFO",
    "endpoint-delay": 10,
    "active-chunks-per-dimension": 1,
    "preferred-dataset-splits": 4,
    "all-reduce-implementation": ["halvingDoubling", "halvingDoubling"],
    "all-gather-implementation": ["halvingDoubling", "halvingDoubling"],
    "reduce-scatter-implementation": ["halvingDoubling", "halvingDoubling"],
    "all-to-all-implementation": ["halvingDoubling", "halvingDoubling"],
    "collective-optimization": "localBWAware",
    "local-mem-bw": 3350,
    "local-mem-size": 80,
    "enable_network_logger": 1,
    "boost-mode": 0,
    "peak-perf": 989,
    "roofline-enabled": 1,
    "trace-enabled": 1,
    "track-local-mem": 1,
    "local-mem-trace-filename": "mem_trace.json",
    "dump-local-mem-trace": 0
}


DEFAULT_NETWORK_CONFIG = {
    "topology": ["Switch", "Switch", "Switch"],
    "npus_count": [8, 4, 2],
    "bandwidth": [900.0, 200.0, 200.0],
    "bandwidth_unit": "GB/s",
    "latency": [0.0, 0.0, 0.0],
    "packet_size": 1500,
    "header_size": 48
}

DEFAULT_MEMORY_CONFIG = {
    "memory-type": "NO_MEMORY_EXPANSION"
}


def generate_system_config(config: Dict[str, Any]) -> str:
    """
    Generate system configuration JSON file.
    
    Caches and reuses configs when parameters don't change.
    Supports collective implementations per dimension.
    
    Args:
        config: Configuration dictionary that may contain:
                - Basic params: 'scheduling-policy', 'endpoint-delay', etc.
                - Collective algos: 'all-reduce', 'all-gather', 'reduce-scatter', 'all-to-all'
                  (can be string for all dims, or list per dimension)
                - 'npus_per_dim': List defining network dimensions (e.g., [8, 8] for 2D)
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _SYSTEM_CONFIG_CACHE
    
    # Start with defaults
    system_config = DEFAULT_SYSTEM_CONFIG.copy()
    
    # Determine number of dimensions from npus_per_dim or default to 2
    npus_per_dim = config.get('npus_per_dim', [8, 8])
    num_dims = len(npus_per_dim)
    
    # Build collective implementations for each dimension
    collective_impls = _build_collective_implementations(config, num_dims)
    system_config.update(collective_impls)
    system_config = _enforce_system_collective_dimensions(system_config, num_dims)
    
    # Override with custom values from config if present
    param_mapping = [
        'scheduling-policy',
        'endpoint-delay',
        'active-chunks-per-dimension',
        'preferred-dataset-splits',
        'collective-optimization',
        'local-mem-bw',
        'local-mem-size',
        'boost-mode',
        'peak-perf'
    ]
    
    for key in param_mapping:
        if key in config:
            system_config[key] = config[key]
    
    # Check cache - reuse if same config exists
    config_hash = _hash_config(system_config)
    if config_hash in _SYSTEM_CONFIG_CACHE:
        cached_path = _SYSTEM_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with hash (not timestamp)
    output_path = os.path.join(CONFIG_OUTPUT_DIR, f"system_{config_hash}.json")
    
    # If another worker already wrote the same file, reuse it directly.
    if os.path.exists(output_path):
        _SYSTEM_CONFIG_CACHE[config_hash] = output_path
        return output_path

    # Write atomically: write to a temp file then rename so the C++ simulator
    # never sees a partially-written file (fixes parallel-worker race condition).
    with tempfile.NamedTemporaryFile('w', dir=CONFIG_OUTPUT_DIR, suffix='.json', delete=False) as tmp:
        json.dump(system_config, tmp, indent=4)
        tmp_path = tmp.name
    os.replace(tmp_path, output_path)
    
    # Cache the path
    _SYSTEM_CONFIG_CACHE[config_hash] = output_path
    
    return output_path


def generate_network_config(config: Dict[str, Any]) -> str:
    """
    Generate network configuration YAML file.
    
    Supports topology-based network types:
    - FoldedClos → Switch
    - Dragonfly → FullyConnected  
    - Torus → Ring
    
    Args:
        config: Configuration dictionary that may contain:
                - 'topology': 'FoldedClos', 'Dragonfly', or 'Torus'
                - 'npus_per_dim': Network dimensions (e.g., [8, 8])
                - 'intra-node-bw': Intra-node bandwidth
                - 'inter-node-bw': Inter-node bandwidth
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _NETWORK_CONFIG_CACHE
    
    # Start with defaults
    network_config = DEFAULT_NETWORK_CONFIG.copy()
    
    # Get network dimensions from npus_per_dim
    if 'npus_per_dim' in config:
        npus_per_dim = list(config['npus_per_dim'])
        network_config['npus_count'] = npus_per_dim
    elif 'npu_count' in config and isinstance(config['npu_count'], (list, tuple)):
        network_config['npus_count'] = list(config['npu_count'])
    elif 'npu_count' in config and isinstance(config['npu_count'], int):
        network_config['npus_count'] = [config['npu_count']]
    
    num_dims = len(network_config['npus_count'])    
    # Map topology to network type for each dimension
    if 'topology' in config:
        topology = config['topology']
        network_type = _map_topology_to_network_type(topology)
        network_config['topology'] = [network_type] * num_dims
    else:
        # Default to Switch for all dimensions
        network_config['topology'] = ['Switch'] * num_dims

    # Ensure bandwidth array always matches dimensions
    network_config['bandwidth'] = _normalize_values_per_dimension(
        network_config.get('bandwidth', [900.0]),
        num_dims,
        900.0
    )
    
    # Override with bandwidth values from config if present
    if 'intra-node-bw' in config or 'inter-node-bw' in config:
        bandwidth = [network_config['bandwidth'][0]] * num_dims if num_dims > 0 else [900.0]
        if 'intra-node-bw' in config:
            bandwidth[0] = config['intra-node-bw']
        if 'inter-node-bw' in config:
            for i in range(1, num_dims):
                bandwidth[i] = config['inter-node-bw']
        network_config['bandwidth'] = bandwidth
    
    # Ensure latency array matches dimensions
    if len(network_config.get('latency', [])) != num_dims:
        network_config['latency'] = [0.0] * num_dims
    
    
    # Check cache - reuse if same config exists
    config_hash = _hash_config(network_config)
    if config_hash in _NETWORK_CONFIG_CACHE:
        cached_path = _NETWORK_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with hash (not timestamp)
    output_path = os.path.join(CONFIG_OUTPUT_DIR, f"network_{config_hash}.yml")
    
    # If another worker already wrote the same file, reuse it directly.
    if os.path.exists(output_path):
        _NETWORK_CONFIG_CACHE[config_hash] = output_path
        return output_path

    # Write atomically to avoid readers seeing partial YAML under parallel runs.
    with tempfile.NamedTemporaryFile('w', dir=CONFIG_OUTPUT_DIR, suffix='.yml', delete=False) as tmp:
        yaml.dump(network_config, tmp, default_flow_style=None)
        tmp_path = tmp.name
    os.replace(tmp_path, output_path)
    
    # Cache the path
    _NETWORK_CONFIG_CACHE[config_hash] = output_path
    
    return output_path


def generate_memory_config(config: Dict[str, Any]) -> str:
    """
    Generate memory configuration JSON file.
    
    Caches and reuses configs when parameters don't change.
    If config contains mem_* parameters, they override defaults.
    Otherwise, uses DEFAULT_MEMORY_CONFIG.
    
    Args:
        config: Configuration dictionary (may contain mem_* prefixed parameters)
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _MEMORY_CONFIG_CACHE
    
    # Start with defaults
    memory_config = DEFAULT_MEMORY_CONFIG.copy()
    
    # Override with custom values from config if present
    if 'mem_memory-type' in config:
        memory_config['memory-type'] = config['mem_memory-type']
    
    # Check cache - reuse if same config exists
    config_hash = _hash_config(memory_config)
    if config_hash in _MEMORY_CONFIG_CACHE:
        cached_path = _MEMORY_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with hash (not timestamp)
    output_path = os.path.join(CONFIG_OUTPUT_DIR, f"memory_{config_hash}.json")
    
    # If another worker already wrote the same file, reuse it directly.
    if os.path.exists(output_path):
        _MEMORY_CONFIG_CACHE[config_hash] = output_path
        return output_path

    # Write atomically to avoid readers seeing partial JSON under parallel runs.
    with tempfile.NamedTemporaryFile('w', dir=CONFIG_OUTPUT_DIR, suffix='.json', delete=False) as tmp:
        json.dump(memory_config, tmp, indent=4)
        tmp_path = tmp.name
    os.replace(tmp_path, output_path)
    
    # Cache the path
    _MEMORY_CONFIG_CACHE[config_hash] = output_path
    
    return output_path


def generate_all_configs(config: Dict[str, Any], net_sim_config: Dict[str, Any] = {"sim_type": "analytical_unaware"}) -> tuple[str, str, str]:
    """
    Generate all three configuration files at once.
    
    This is the recommended way to get config paths for simulation.
    Always generates fresh configs with unique timestamps.
    
    Args:
        config: Configuration dictionary (may contain net_*, sys_*, mem_* prefixed parameters)
        net_sim_config: Network simulation configuration dict with 'sim_type' and topology info
    
    Returns:
        Tuple of (system_path, network_path, memory_path) - all absolute paths
    
    Example:
        system, network, memory = config_generator.generate_all_configs(config)
        # Pass these paths to SimulationRunner
    """
    memory_path = generate_memory_config(config)
    system_path = generate_system_config(config)
    network_path = generate_network_config(config)
   
    
    return system_path, network_path, memory_path
