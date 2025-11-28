"""
Configuration loader for neural reconstruction modules.

This module provides YAML-based configuration management with:
- Pydantic models for type validation
- Environment variable substitution
- CLI argument override support
- Multiple configuration profile support
"""

import os
import re
from pathlib import Path
from typing import Optional, Any, Dict

import yaml
from pydantic import BaseModel, Field, field_validator


class ConnectedComponentsConfig(BaseModel):
    """Configuration for connected components analysis."""

    connectivity: int = Field(8, description="Connectivity (4 or 8)")
    min_area: int = Field(10, description="Minimum component area (pixels)")

    @field_validator('connectivity')
    @classmethod
    def connectivity_must_be_valid(cls, v):
        if v not in [4, 8]:
            raise ValueError('connectivity must be 4 or 8')
        return v

    @field_validator('min_area')
    @classmethod
    def min_area_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('min_area must be non-negative (>= 0)')
        return v


class SeedExtractionConfig(BaseModel):
    """Configuration for seed extraction."""

    base_segment_length: float = Field(10.0, description="Base segment length for uniform seed placement (pixels)")

    @field_validator('base_segment_length')
    @classmethod
    def segment_length_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('base_segment_length must be positive (> 0)')
        return v


class ComponentPairingConfig(BaseModel):
    """Configuration for component pairing analysis."""

    max_distance_threshold: float = Field(100.0, description="Maximum distance threshold (pixels)")
    max_cost_threshold: float = Field(150.0, description="Maximum path cost threshold")

    @field_validator('max_distance_threshold')
    @classmethod
    def max_distance_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('max_distance_threshold must be positive (> 0)')
        return v

    @field_validator('max_cost_threshold')
    @classmethod
    def max_cost_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('max_cost_threshold must be positive (> 0)')
        return v


class CostWeightsConfig(BaseModel):
    """Configuration for cost calculation weights."""

    alpha: float = Field(1.0, description="Weight for distance cost")
    beta: float = Field(1.0, description="Weight for intensity cost")
    gamma: float = Field(1.0, description="Weight for curvature cost")


class NetworkConfig(BaseModel):
    """Configuration for network building."""

    k_neighbors: int = Field(5, description="Number of k-nearest neighbors for graph construction")
    max_edge_cost: float = Field(100.0, description="Maximum edge cost threshold")


class NetworkBuildingConfig(BaseModel):
    """Configuration for network building stage."""

    cost_weights: CostWeightsConfig = Field(default_factory=CostWeightsConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)


class MSTReconstructionConfig(BaseModel):
    """Configuration for MST reconstruction stage."""

    max_edge_cost: Optional[float] = Field(None, description="Maximum edge cost for MST filtering (optional)")


class NeuralReconstructionConfig(BaseModel):
    """Root configuration for neural reconstruction modules."""

    connected_components: ConnectedComponentsConfig = Field(default_factory=ConnectedComponentsConfig)
    seed_extraction: SeedExtractionConfig = Field(default_factory=SeedExtractionConfig)
    component_pairing: ComponentPairingConfig = Field(default_factory=ComponentPairingConfig)
    network_building: NetworkBuildingConfig = Field(default_factory=NetworkBuildingConfig)
    reconstruction: MSTReconstructionConfig = Field(default_factory=MSTReconstructionConfig)


# Alias for backward compatibility
IENFConfig = NeuralReconstructionConfig


def substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively substitute environment variables in config dictionary.

    Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Dictionary with environment variables substituted
    """
    def _substitute_value(value):
        if isinstance(value, str):
            # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ''
                return os.environ.get(var_name, default_value)

            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: _substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_substitute_value(item) for item in value]
        else:
            return value

    return _substitute_value(config_dict)


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> NeuralReconstructionConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. If None, uses default.yaml
        overrides: Dictionary of override values (e.g., from CLI arguments)

    Returns:
        Validated NeuralReconstructionConfig object

    Example:
        >>> config = load_config('config/high_quality.yaml')
        >>> config = load_config(overrides={'seed_extraction.base_segment_length': 15.0})
    """
    # Determine config file path
    if config_path is None:
        # Default to config/default.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Substitute environment variables
    config_dict = substitute_env_vars(config_dict)

    # Apply overrides if provided
    if overrides:
        config_dict = apply_overrides(config_dict, overrides)

    # Validate and create config object
    config = NeuralReconstructionConfig(**config_dict)

    return config


def apply_overrides(config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply override values to configuration dictionary.

    Supports dot notation for nested keys: 'seed_extraction.base_segment_length'

    Args:
        config_dict: Base configuration dictionary
        overrides: Override values with dot notation keys

    Returns:
        Updated configuration dictionary
    """
    for key, value in overrides.items():
        keys = key.split('.')
        current = config_dict

        # Navigate to the nested location
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

    return config_dict


def save_config(config: NeuralReconstructionConfig, output_path: str):
    """
    Save configuration to YAML file.

    Useful for saving modified configurations or creating new profiles.

    Args:
        config: NeuralReconstructionConfig object to save
        output_path: Output YAML file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary
    config_dict = config.model_dump()

    # Write YAML
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)


# Convenience function to get default config
def get_default_config() -> NeuralReconstructionConfig:
    """Get default configuration."""
    return load_config()


if __name__ == "__main__":
    # Test configuration loading
    print("Loading default configuration...")
    config = get_default_config()

    print("\nConnected Components Config:")
    print(f"  Connectivity: {config.connected_components.connectivity}")
    print(f"  Min area: {config.connected_components.min_area}")

    print("\nSeed Extraction Config:")
    print(f"  Base segment length: {config.seed_extraction.base_segment_length}")

    print("\nComponent Pairing Config:")
    print(f"  Max distance threshold: {config.component_pairing.max_distance_threshold}")
    print(f"  Max cost threshold: {config.component_pairing.max_cost_threshold}")

    print("\n✓ Configuration loaded successfully!")
