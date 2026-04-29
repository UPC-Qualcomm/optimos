#!/usr/bin/env python3
"""
Network Configuration Helper for AstraSim Optimizers

This module provides utilities for managing network configurations
across different optimizers, following the DRY (Don't Repeat Yourself) principle.

Available Networks:
- FoldedClos: Folded Clos topology (default)
- Switch: Switch-based topology
- Ring: Ring topology
- FullyConnected: Fully connected topology
- 2D_Torus: 2D Torus topology
- 3D_Torus: 3D Torus topology
- Dragonfly: Dragonfly topology
- DGX1: NVIDIA DGX-1 topology
"""

import os
from typing import Tuple, List


class NetworkConfig:
    """
    Configuration helper for AstraSim network topologies.
    
    This class encapsulates network configuration paths and validation,
    making it reusable across different optimizer implementations.
    """
    
    # Available network topologies
    AVAILABLE_NETWORKS = [
        "FoldedClos",
        "Switch", 
        "Ring",
        "FullyConnected",
        "2D_Torus",
        "3D_Torus",
        "Dragonfly",
    ]
    
    def __init__(self, network_name: str = "FoldedClos", config_dir: str = "./configuration"):
        """
        Initialize network configuration.
        
        Args:
            network_name: Name of the network topology
            config_dir: Directory containing configuration files
            
        Raises:
            ValueError: If network_name is not valid
        """
        self.network_name = network_name
        self.config_dir = config_dir
        
        # Validate network exists
        if not self.is_valid_network(network_name):
            raise ValueError(
                f"Invalid network: {network_name}. "
                f"Available networks: {', '.join(self.AVAILABLE_NETWORKS)}"
            )
        
        # Build configuration paths
        self.system_config = f"{config_dir}/{network_name}_sys.json"
        self.network_config = f"{config_dir}/{network_name}.yml"
    
    @classmethod
    def is_valid_network(cls, network_name: str) -> bool:
        """Check if network name is valid."""
        return network_name in cls.AVAILABLE_NETWORKS
    
    @classmethod
    def list_available_networks(cls) -> List[str]:
        """Get list of available network topologies."""
        return cls.AVAILABLE_NETWORKS.copy()
    
    def get_paths(self) -> Tuple[str, str]:
        """
        Get system and network configuration file paths.
        
        Returns:
            Tuple of (system_config_path, network_config_path)
        """
        return self.system_config, self.network_config
    
    def get_output_dirs(self, base_dir: str, model_dir: str) -> Tuple[str, str]:
        """
        Get output and network log directories for this network.
        
        Args:
            base_dir: Base directory for outputs
            model_dir: Model-specific directory name
            
        Returns:
            Tuple of (output_dir, network_log_dir)
        """
        output_dir = f"{base_dir}/output/{model_dir}/{self.network_name}"
        network_log_dir = f"{base_dir}/network_log/{model_dir}/{self.network_name}"
        return output_dir, network_log_dir
    
    def __str__(self) -> str:
        """String representation."""
        return f"NetworkConfig(network={self.network_name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"NetworkConfig(network_name='{self.network_name}', "
                f"system_config='{self.system_config}', "
                f"network_config='{self.network_config}')")


def validate_network_files(network_name: str, config_dir: str = "./configuration") -> bool:
    """
    Validate that network configuration files exist.
    
    Args:
        network_name: Name of the network topology
        config_dir: Directory containing configuration files
        
    Returns:
        True if both files exist, False otherwise
    """
    system_file = f"{config_dir}/{network_name}_sys.json"
    network_file = f"{config_dir}/{network_name}.yml"
    
    system_exists = os.path.exists(system_file)
    network_exists = os.path.exists(network_file)
    
    if not system_exists:
        print(f"Warning: System config not found: {system_file}")
    if not network_exists:
        print(f"Warning: Network config not found: {network_file}")
    
    return system_exists and network_exists


# Convenience function for command-line tools
def get_network_config_from_args(network_name: str) -> NetworkConfig:
    """
    Create NetworkConfig from command-line argument with validation.
    
    Args:
        network_name: Network name from argparse
        
    Returns:
        NetworkConfig instance
        
    Raises:
        SystemExit: If network is invalid (prints helpful message)
    """
    try:
        return NetworkConfig(network_name)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nAvailable networks:")
        for net in NetworkConfig.AVAILABLE_NETWORKS:
            print(f"  - {net}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    # Demo usage
    print("Network Configuration Helper Demo")
    print("=" * 60)
    
    print("\nAvailable Networks:")
    for network in NetworkConfig.list_available_networks():
        print(f"  - {network}")
    
    print("\nExample Configuration:")
    config = NetworkConfig("FoldedClos")
    print(f"  {config}")
    print(f"  System config: {config.system_config}")
    print(f"  Network config: {config.network_config}")
    
    print("\nValidation Example:")
    is_valid = NetworkConfig.is_valid_network("FoldedClos")
    print(f"  Is 'FoldedClos' valid? {is_valid}")
    is_valid = NetworkConfig.is_valid_network("InvalidNetwork")
    print(f"  Is 'InvalidNetwork' valid? {is_valid}")
