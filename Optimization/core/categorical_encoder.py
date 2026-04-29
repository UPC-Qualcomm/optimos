"""
CategoricalEncoder: Simple enum-based encoder for categorical parameters.

Provides efficient mapping between categorical (string) values and numerical indices.
"""

from typing import Any

# Define enums for categorical parameters
CATEGORICAL_MAPPINGS = {
    'scheduling-policy': {
        'FIFO': 0,
        'LIFO': 1,
    },
    'collective-optimization': {
        'baseline': 0,
        'localBWAware': 1,
    },
    'all_reduce': {
        'ring': 0,
        'halvingDoubling': 1,
        'direct': 2,
        'doubleBinaryTree': 3,
    },
    'all_gather': {
        'ring': 0,
        'halvingDoubling': 1,
        'direct': 2,
        'doubleBinaryTree': 3,
    },
    'reduce_scatter': {
        'ring': 0,
        'halvingDoubling': 1,
        'direct': 2,
        'doubleBinaryTree': 3,
    },
    'all_to_all': {
        'ring': 0,
        'halvingDoubling': 1,
        'direct': 2,
        'doubleBinaryTree': 3,
    },
    'topology': {
        'FoldedClos': 0,
        'Dragonfly': 1,
        'Torus': 2,
    }
}

# Build reverse mappings
CATEGORICAL_REVERSE_MAPPINGS = {
    param: {v: k for k, v in mapping.items()}
    for param, mapping in CATEGORICAL_MAPPINGS.items()
}


def get_numerical(param: str, value: Any) -> float:
    """
    Convert a parameter value to numerical representation.
    
    Args:
        param: Parameter name
        value: Parameter value (string for categorical, numeric for numerical)
    
    Returns:
        Float representation of the value
    """
    if param in CATEGORICAL_MAPPINGS:
        # Categorical parameter - map to numerical
        if value not in CATEGORICAL_MAPPINGS[param]:
            raise ValueError(
                f"Unknown categorical value '{value}' for parameter '{param}'. "
                f"Valid values: {list(CATEGORICAL_MAPPINGS[param].keys())}"
            )
        return float(CATEGORICAL_MAPPINGS[param][value])
    else:
        # Numerical parameter - convert to float
        if isinstance(value, bool):
            return float(value)
        return float(value)


def get_str(param: str, numerical_value: float) -> Any:
    """
    Convert a numerical value back to its original representation.
    
    Args:
        param: Parameter name
        numerical_value: Numerical value
    
    Returns:
        Original value (string for categorical, number for numerical)
    """
    if param in CATEGORICAL_REVERSE_MAPPINGS:
        # Categorical parameter - map back to string
        idx = int(round(numerical_value))
        if idx not in CATEGORICAL_REVERSE_MAPPINGS[param]:
            raise ValueError(
                f"Invalid numerical value {numerical_value} (rounded to {idx}) "
                f"for parameter '{param}'. "
                f"Valid indices: {list(CATEGORICAL_REVERSE_MAPPINGS[param].keys())}"
            )
        return CATEGORICAL_REVERSE_MAPPINGS[param][idx]
    else:
        # Numerical parameter - return as-is
        return numerical_value
