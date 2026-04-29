"""
SearchSpaceBuilder: Flexible builder for creating search spaces from JSON configurations.

This module provides a builder pattern for creating search spaces with:
1. Support for all parameter types (clusters, parallelism_strategies, network, system, model)
2. Dynamic constraint parsing and evaluation
3. Flexible parameter subsets (doesn't require all parameters)
4. Integration with various sampling strategies
5. Configuration validation and space generation
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Callable
from itertools import product

try:
    from .sampler import get_sampler
except ImportError:
    from sampler import get_sampler


class SearchSpaceBuilder:
    """
    Builder for creating flexible search spaces from JSON configurations.
    
    Features:
    - Parses JSON with arbitrary parameter subsets
    - Extracts and evaluates constraints dynamically
    - Generates valid configurations based on constraints
    - Integrates with sampling strategies
    - Returns configurations as dictionaries for easy use
    
    Example usage:
        builder = SearchSpaceBuilder('config.json', num_npus=128)
        builder.parse_parameters()
        builder.apply_constraints()
        configs = builder.sample(n_samples=20, strategy='random')
    """
    
    def __init__(self, config_path: str):
        """
        Initialize search space builder.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Parse cluster configurations
        if "clusters" in self.config:
            self.clusters = self.config["clusters"]
            # For backward compatibility, set num_npus to first cluster's npu_count
            first_cluster = next(iter(self.clusters.values()))
            self.num_npus = first_cluster.get("npu_count")

        else:
            raise ValueError("clusters' must be specified in the configuration file")
        
        # Storage for parsed data
        self.parameters: Dict[str, List[Any]] = {}
        self.constraints: List[Callable] = []
        self.constraint_strings: List[str] = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load JSON configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def parse_parameters(self, include_categories: Optional[List[str]] = None,
                        exclude_categories: Optional[List[str]] = None) -> 'SearchSpaceBuilder':
        """
        Parse parameters from configuration.
        
        Args:
            include_categories: Only include these categories (e.g., ['parallelism_strategy', 'network'])
            exclude_categories: Exclude these categories
        
        Returns:
            Self for method chaining
        """
        self.parameters = {}
        
        # Add cluster names as a parameter (keys, not objects - for DeepHyper compatibility)
        # Cluster objects can be retrieved later via get_cluster_config()
        self.parameters['cluster'] = list(self.clusters.keys())
        
        # Define parameter categories
        categories = {
            'parallelism_strategy': self._parse_parallelism_strategy,
            'system': self._parse_system,
            'network': self._parse_network,
            'model': self._parse_model,
        }
        
        # Filter categories
        if include_categories:
            categories = {k: v for k, v in categories.items() if k in include_categories}
        if exclude_categories:
            categories = {k: v for k, v in categories.items() if k not in exclude_categories}
        
        # Parse each category
        for category, parser in categories.items():
            if category in self.config:
                parser(self.config[category])
        
        return self
    
    def _parse_parallelism_strategy(self, params: Dict) -> None:
        """Parse parallelism strategy parameters."""
        if 'dp' in params:
            self.parameters['dp'] = params['dp']
        if 'mp' in params:
            self.parameters['mp'] = params['mp']
        if 'sp' in params:
            self.parameters['sp'] = params['sp']
        if 'pp' in params:
            self.parameters['pp'] = params['pp']
        if 'FSDP' in params:
            self.parameters['sharded'] = [bool(x) for x in params['FSDP']]
    
    def _parse_system(self, params: Dict) -> None:
        """Parse collective communication parameters."""
        if 'scheduling-policy' in params:
            self.parameters['scheduling-policy'] = params['scheduling-policy']
        
        if 'collective-implementation' in params:
            impl = params['collective-implementation']
            # Determine max dimensions across clusters so all clusters are representable.
            # Per-cluster trimming is applied later during config enrichment.
            cluster_dims = [
                len(cluster.get('npus_per_dim', []))
                for cluster in self.clusters.values()
            ]
            num_dims = max(cluster_dims) if cluster_dims else 2
            if num_dims <= 0:
                num_dims = 2
            
            # Parse each collective type with per-dimension parameters
            for collective_type, algorithms in impl.items():
                # Create a parameter for each dimension of this collective
                for dim in range(num_dims):
                    param_name = f'{collective_type}-dim{dim}'
                    self.parameters[param_name] = algorithms
        
        if 'active-chunks-per-dimension' in params:
            self.parameters['active-chunks-per-dimension'] = params['active-chunks-per-dimension']
        
        if 'preferred-dataset-splits' in params:
            self.parameters['preferred-dataset-splits'] = params['preferred-dataset-splits']
        
        if 'collective-optimization' in params:
            self.parameters['collective-optimization'] = params['collective-optimization']

        if 'local-mem-bw' in params:
            self.parameters['local-mem-bw'] = params['local-mem-bw']
        if 'local-mem-size' in params:
            self.parameters['local-mem-size'] = params['local-mem-size']
        if 'peak-perf' in params:
            self.parameters['peak-perf'] = params['peak-perf']
    
    def _parse_network(self, params: Dict) -> None:
        """Parse network parameters."""
        if 'topology' in params:
            self.parameters['topology'] = params['topology']
        if 'inter-node-bw' in params:
            self.parameters['inter-node-bw'] = params['inter-node-bw']
        if 'intra-node-bw' in params:
            self.parameters['intra-node-bw'] = params['intra-node-bw']
        if 'npus-per-node' in params:
            self.parameters['npus-per-node'] = params['npus-per-node']
    
    def _parse_model(self, params: Dict) -> None:
        """Parse model parameters."""
        if 'batch_size' in params:
            self.parameters['batch_size'] = self._get_batch_size_values(params['batch_size'])
        if 'micro_batch_size' in params:
            self.parameters['micro_batch_size'] = params['micro_batch_size']
        if 'mixed_precision' in params:
            self.parameters['mixed_precision'] = [bool(x) for x in params['mixed_precision']]
        if 'sequence_length' in params:
            self.parameters['sequence_length'] = params['sequence_length']
        if 'FNN_hidden_size' in params:
            self.parameters['FNN_hidden_size'] = params['FNN_hidden_size']
        if 'num_attention_heads' in params:
            self.parameters['num_attention_heads'] = params['num_attention_heads']
        if 'num_layers' in params:
            self.parameters['num_layers'] = params['num_layers']
    
    def apply_constraints(self, custom_constraints: Optional[List[str]] = None) -> 'SearchSpaceBuilder':
        """
        Parse and apply constraints from configuration.
        
        Args:
            custom_constraints: Additional constraint strings to apply
        
        Returns:
            Self for method chaining
        """
        self.constraints = []
        self.constraint_strings = []
        
        # Parse constraints from config
        if 'constraints' in self.config:
            for name, constraint_str in self.config['constraints'].items():
                self._parse_constraint(constraint_str)
        
        # Add custom constraints
        if custom_constraints:
            for constraint_str in custom_constraints:
                self._parse_constraint(constraint_str)
        
        return self
    
    def _get_batch_size_values(self, batch_size_param) -> List[int]:
        """Helper to get batch size values, supporting both int and list."""
        bs = batch_size_param
        if isinstance(bs, dict) and "min" in bs and "max" in bs:
            start = bs["min"]
            end = bs["max"]
            # Power-of-2 mode
            if bs.get("mode") == "power2":
                values = []
                v = start
                # move to first power of 2 >= start
                if v < 1:
                    raise ValueError("min must be >= 1 for power2 mode")
                while (v & (v - 1)) != 0:  # not power of 2
                    v += 1
                while v <= end:
                    values.append(v)
                    v *= 2
            # Linear step mode (default)
            else:
                step = bs.get("step", 1)
                if step <= 0:
                    raise ValueError("step must be > 0")
                values = list(range(start, end + 1, step))
            return values
        else:
            return bs
    
    def _parse_constraint(self, constraint_str: str) -> None:
        """
        Parse a constraint string and create a validation function.
        
        Supported constraint formats:
        - "dp * mp * sp * pp = npu_count"
        - "dp <= npu_count"
        - "batch_size % micro_batch_size = 0"
        - "micro_batch_size <= batch_size"
        
        Args:
            constraint_str: String representation of constraint
        """
        # Store original constraint string
        self.constraint_strings.append(constraint_str)

        # Normalize single '=' to '==' while preserving <=, >=, !=, ==
        normalized_expr = re.sub(r'(?<![<>=!])=(?!=)', '==', constraint_str)

        # Validate expression syntax once at parse time
        try:
            compiled_expr = compile(normalized_expr, '<constraint>', 'eval')
        except SyntaxError as e:
            raise SyntaxError(f"Invalid constraint syntax '{constraint_str}': {e}") from e

        # Create a function that evaluates the expression
        def constraint_func(config: Dict[str, Any]) -> bool:
            # Create a local scope for eval, including config and built-ins
            local_scope = config.copy()

            # Special handling for 'npu_count' - resolve from cluster if needed
            if 'npu_count' in normalized_expr and 'cluster' in local_scope:
                cluster_name = local_scope['cluster']
                local_scope['npu_count'] = self.get_cluster_npu_count(cluster_name)

            try:
                return eval(compiled_expr, {"__builtins__": {}}, local_scope)
            except (NameError, TypeError) as e:
                # This can happen if a parameter in the constraint is not in the config
                # For now, we treat this as a non-violation (or could be strict)
                # print(f"Warning: Constraint '{constraint_str}' could not be evaluated: {e}")
                return True
        
        self.constraints.append(constraint_func)
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about parsed parameters.
        
        Returns:
            Dictionary with parameter names as keys and their info as values.
        """
        info = {}
        for name, values in self.parameters.items():
            info[name] = {
                'count': len(values),
                'values': values,
                'type': type(values[0]).__name__ if values else 'N/A'
            }
        return info

    def get_cluster_config(self, cluster_name: str) -> Dict[str, Any]:
        """
        Get the full configuration for a specific cluster.
        
        Args:
            cluster_name: The name of the cluster (e.g., 'cl1')
            
        Returns:
            Dictionary with cluster configuration
        """
        if cluster_name not in self.clusters:
            raise ValueError(f"Cluster '{cluster_name}' not found. Available: {list(self.clusters.keys())}")
        return self.clusters[cluster_name]
    
    def get_cluster_npu_count(self, cluster_name: str) -> int:
        """
        Get NPU count for a specific cluster.
        
        Args:
            cluster_name: Name of the cluster
        
        Returns:
            Number of NPUs in the cluster
        """
        cluster_config = self.get_cluster_config(cluster_name)
        return cluster_config.get('npu_count', 0)
    
    def get_cluster_npus_per_dim(self, cluster_name: str) -> List[int]:
        """
        Get NPUs per dimension for a specific cluster.
        
        Args:
            cluster_name: Name of the cluster
        
        Returns:
            List of NPUs per dimension (e.g., [8, 8] for 2D topology)
        """
        cluster_config = self.get_cluster_config(cluster_name)
        return cluster_config.get('npus_per_dim', [])
    
    def get_cluster_dimensions(self, cluster_name: str) -> int:
        """
        Get number of dimensions in a cluster's topology.
        
        Args:
            cluster_name: Name of the cluster
        
        Returns:
            Number of dimensions (deduced from npus_per_dim length)
        """
        npus_per_dim = self.get_cluster_npus_per_dim(cluster_name)
        return len(npus_per_dim)
    
    def get_all_clusters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all cluster configurations.
        
        Returns:
            Dictionary mapping cluster names to their configurations
        """
        return self.clusters.copy()
    
    def reconstruct_collective_implementations(self, config: Dict[str, Any], num_dims: Optional[int] = None) -> Dict[str, Any]:
        """
        Reconstruct collective implementation arrays from per-dimension parameters.
        
        Converts parameters like 'all-reduce-dim0', 'all-reduce-dim1' back into
        'all-reduce': ['ring', 'halvingDoubling'] format for config generator.
        
        Args:
            config: Configuration dictionary with per-dimension parameters
            num_dims: Optional dimension cap. If provided, only dim0..dim(num_dims-1)
                     are reconstructed and higher-dimension parameters are discarded.
        
        Returns:
            Configuration with reconstructed collective arrays
        """
        enriched = config.copy()
        
        # Find all collective types by looking for -dim0 parameters
        collective_types = set()
        for key in config.keys():
            if '-dim0' in key:
                collective_type = key.replace('-dim0', '')
                collective_types.add(collective_type)
        
        # Reconstruct arrays for each collective type
        for collective_type in collective_types:
            algorithms = []
            # Gather all available dimension indices for this collective.
            available_dims = []
            for key in list(config.keys()):
                prefix = f'{collective_type}-dim'
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    if suffix.isdigit():
                        available_dims.append(int(suffix))

            if not available_dims:
                continue

            limit = (max(available_dims) + 1) if num_dims is None else max(0, num_dims)

            for dim in sorted(available_dims):
                param_key = f'{collective_type}-dim{dim}'
                if dim < limit:
                    algorithms.append(config[param_key])
                # Remove per-dimension parameters regardless; they are internal search params.
                if param_key in enriched:
                    del enriched[param_key]
            
            # Add the reconstructed array
            if algorithms:
                enriched[collective_type] = algorithms
        
        return enriched
    
    def enrich_config_with_cluster_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a configuration dictionary with cluster information.
        
        If the config contains a 'cluster' key, this method adds the cluster's
        npu_count, npus_per_dim, and num_dimensions to the config.
        Also reconstructs collective implementation arrays from per-dimension parameters.
        
        Args:
            config: Configuration dictionary (may contain 'cluster' key)
        
        Returns:
            Enriched configuration dictionary with cluster info added
        """
        enriched = config.copy()

        if 'cluster' in enriched:
            cluster_name = enriched['cluster']
            cluster_config = self.get_cluster_config(cluster_name)
            cluster_dims = len(cluster_config.get('npus_per_dim', []))

            # Reconstruct collectives constrained to this cluster's dimensions.
            enriched = self.reconstruct_collective_implementations(config, num_dims=cluster_dims)
            
            # Add cluster info to config
            enriched['npu_count'] = cluster_config.get('npu_count')
            enriched['npus_per_dim'] = cluster_config.get('npus_per_dim', [])
            enriched['num_dimensions'] = len(enriched['npus_per_dim'])
        elif len(self.clusters) == 1:
            # If only one cluster and no cluster key, use the single cluster's info
            cluster_name = next(iter(self.clusters.keys()))
            cluster_config = self.get_cluster_config(cluster_name)
            cluster_dims = len(cluster_config.get('npus_per_dim', []))

            # Reconstruct collectives constrained to this cluster's dimensions.
            enriched = self.reconstruct_collective_implementations(config, num_dims=cluster_dims)
            
            enriched['npu_count'] = cluster_config.get('npu_count')
            enriched['npus_per_dim'] = cluster_config.get('npus_per_dim', [])
            enriched['num_dimensions'] = len(enriched['npus_per_dim'])
        else:
            # No cluster context: preserve all reconstructed dimensions.
            enriched = self.reconstruct_collective_implementations(config)
        
        return enriched
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SearchSpaceBuilder(params={len(self.parameters)}, "
                f"constraints={len(self.constraints)})")

    def __str__(self) -> str:
        """Human-readable string."""
        lines = ["Search Space Summary:", "="*25]
        for name, values in self.parameters.items():
            lines.append(f"  - {name}: {len(values)} values")
        if self.constraint_strings:
            lines.append("Constraints:")
            for cs in self.constraint_strings:
                lines.append(f"  - {cs}")
        return "\n".join(lines)


# Convenience function for quick usage
def create_search_space(config_path: str, 
                       include_categories: Optional[List[str]] = None,
                       exclude_categories: Optional[List[str]] = None,
                       custom_constraints: Optional[List[str]] = None,
                       max_configs: Optional[int] = None) -> SearchSpaceBuilder:
    """
    Convenience function to create and build a search space in one call.
    
    Args:
        config_path: Path to JSON configuration file
        num_npus: Number of NPUs
        include_categories: Categories to include
        exclude_categories: Categories to exclude
        custom_constraints: Additional constraints
        max_configs: Maximum configurations to generate
    
    Returns:
        Built SearchSpaceBuilder instance
    """
    builder = SearchSpaceBuilder(config_path)
    builder.parse_parameters(include_categories, exclude_categories)
    builder.apply_constraints(custom_constraints)
    
    return builder
