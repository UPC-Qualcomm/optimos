"""
Configuration file parser for AstraSim configs.

Parses system, network, and memory configuration files to extract
all parameters for logging and analysis.
"""

import json
import yaml
from typing import Dict, Any


def parse_system_config(filepath: str) -> Dict[str, Any]:
    """
    Parse system configuration JSON file.
    
    Args:
        filepath: Path to system config JSON file
        
    Returns:
        Dictionary with all system parameters (prefixed with sys_)
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Flatten and prefix with 'sys_'
        flattened = {}
        for key, value in config.items():
            # Handle list values by converting to string or taking first element
            if isinstance(value, list):
                if len(value) == 1:
                    flattened[f'sys_{key}'] = value[0]
                else:
                    # For multi-value lists, convert to comma-separated string
                    flattened[f'sys_{key}'] = ','.join(map(str, value))
            else:
                flattened[f'sys_{key}'] = value
        
        return flattened
    except Exception as e:
        print(f"Warning: Could not parse system config {filepath}: {e}")
        return {}


def parse_network_config(filepath: str) -> Dict[str, Any]:
    """
    Parse network configuration YAML file.
    
    Args:
        filepath: Path to network config YAML file
        
    Returns:
        Dictionary with all network parameters (prefixed with net_)
    """
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Flatten and prefix with 'net_'
        flattened = {}
        for key, value in config.items():
            # Handle list values
            if isinstance(value, list):
                if len(value) == 1:
                    flattened[f'net_{key}'] = value[0]
                elif len(value) == 2 and key in ['topology', 'npus_count', 'bandwidth', 'latency']:
                    # Common pattern: [level0, level1]
                    flattened[f'net_{key}_l0'] = value[0]
                    flattened[f'net_{key}_l1'] = value[1]
                else:
                    # For other multi-value lists, convert to string
                    flattened[f'net_{key}'] = ','.join(map(str, value))
            else:
                flattened[f'net_{key}'] = value
        
        return flattened
    except Exception as e:
        print(f"Warning: Could not parse network config {filepath}: {e}")
        return {}


def parse_memory_config(filepath: str) -> Dict[str, Any]:
    """
    Parse memory configuration JSON file.
    
    Args:
        filepath: Path to memory config JSON file
        
    Returns:
        Dictionary with all memory parameters (prefixed with mem_)
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Flatten and prefix with 'mem_'
        flattened = {}
        for key, value in config.items():
            if isinstance(value, list):
                flattened[f'mem_{key}'] = ','.join(map(str, value))
            else:
                flattened[f'mem_{key}'] = value
        
        return flattened
    except Exception as e:
        print(f"Warning: Could not parse memory config {filepath}: {e}")
        return {}


def parse_all_configs(system_path: str, network_path: str, memory_path: str) -> Dict[str, Any]:
    """
    Parse all configuration files and return combined dictionary.
    
    Args:
        system_path: Path to system config file
        network_path: Path to network config file
        memory_path: Path to memory config file
        
    Returns:
        Combined dictionary with all parameters
    """
    result = {}
    result.update(parse_system_config(system_path))
    result.update(parse_network_config(network_path))
    result.update(parse_memory_config(memory_path))
    return result
