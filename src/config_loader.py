"""
Configuration loader for IENF quantification pipeline.

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
from pydantic import BaseModel, Field, validator


class SeedExtractionConfig(BaseModel):
    """Configuration for Stage 02: Seed Extraction."""

    window_size: int = Field(5, description="Curvature calculation window size (must be odd)")
    base_segment_length: float = Field(10.0, description="Base segmentation length (pixels)")
    max_segment_length: float = Field(20.0, description="Maximum segment length (pixels)")
    curvature_threshold: float = Field(30.0, description="Curvature threshold (degrees)")
    skip_branchpoint_range: int = Field(5, description="Skip range around branch points")
    min_path_points: int = Field(10, description="Minimum path points for processing")

    @validator('window_size')
    def window_size_must_be_odd(cls, v):
        if v % 2 == 0:
            raise ValueError('window_size must be odd')
        return v

    @validator('max_segment_length')
    def max_greater_than_base(cls, v, values):
        if 'base_segment_length' in values and v < values['base_segment_length']:
            raise ValueError('max_segment_length must be >= base_segment_length')
        return v


class CostWeightsConfig(BaseModel):
    """Cost calculation weights for network building."""

    alpha: float = Field(0.05, description="Geometric cost weight", ge=0.0, le=1.0)
    beta: float = Field(0.9, description="Image cost weight", ge=0.0, le=1.0)
    gamma: float = Field(0.05, description="Curvature cost weight", ge=0.0, le=1.0)

    @validator('gamma')
    def weights_should_sum_to_one(cls, v, values):
        if 'alpha' in values and 'beta' in values:
            total = values['alpha'] + values['beta'] + v
            if not (0.95 <= total <= 1.05):  # Allow small floating-point tolerance
                raise ValueError(f'Cost weights should sum to ~1.0, got {total}')
        return v


class DensityConfig(BaseModel):
    """Density-based adaptive pairing configuration."""

    dense_threshold: float = Field(30, description="Threshold for dense regions (pixels)")
    moderate_threshold: float = Field(70, description="Threshold for moderate regions (pixels)")
    dense_radius: float = Field(30, description="Pairing radius in dense regions (pixels)")
    moderate_radius: float = Field(50, description="Pairing radius in moderate regions (pixels)")
    sparse_radius: float = Field(80, description="Pairing radius in sparse regions (pixels)")


class PathfindingConfig(BaseModel):
    """A* pathfinding parameters."""

    max_distance_multiplier: float = Field(200, description="Max search distance multiplier")
    distance_from_start_cutoff: float = Field(30, description="Early termination distance (pixels)")


class NetworkConfig(BaseModel):
    """General network building configuration."""

    k_neighbors: int = Field(10, description="Number of neighbors for density estimation")
    max_edge_cost: float = Field(150.0, description="Maximum edge cost threshold")
    verbose: bool = Field(False, description="Display detailed information")


class NetworkBuildingConfig(BaseModel):
    """Configuration for Stage 03: Network Building."""

    network: NetworkConfig = Field(default_factory=NetworkConfig)
    cost_weights: CostWeightsConfig = Field(default_factory=CostWeightsConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    pathfinding: PathfindingConfig = Field(default_factory=PathfindingConfig)


class ReconstructionConfig(BaseModel):
    """Configuration for Stage 04: Neural Reconstruction."""

    max_edge_cost: float = Field(150, description="MST edge cost threshold")
    min_branch_angle: float = Field(30, description="Sharp branch angle threshold (degrees)")
    min_quality_threshold: float = Field(80, description="Minimum path quality threshold")
    verbose: bool = Field(True, description="Display detailed progress")


class PipelinePathsConfig(BaseModel):
    """Pipeline input/output paths."""

    data_dir: str = Field("data", description="Data directory")
    output_dir: str = Field("output", description="Output directory")


class PipelineStagesConfig(BaseModel):
    """Pipeline stage execution flags."""

    run_seed_extraction: bool = Field(True, description="Run seed extraction")
    run_network_building: bool = Field(True, description="Run network building")
    run_reconstruction: bool = Field(True, description="Run reconstruction")


class PipelineIntermediatesConfig(BaseModel):
    """Pipeline intermediate files configuration."""

    save: bool = Field(False, description="Save intermediate outputs (seeds, network)")


class PipelineLoggingConfig(BaseModel):
    """Pipeline logging configuration."""

    level: str = Field("INFO", description="Logging level")
    log_to_file: bool = Field(True, description="Enable file logging")
    log_file: str = Field("output/pipeline.log", description="Log file path")


class PipelineConfig(BaseModel):
    """Pipeline orchestration configuration."""

    paths: PipelinePathsConfig = Field(default_factory=PipelinePathsConfig)
    stages: PipelineStagesConfig = Field(default_factory=PipelineStagesConfig)
    intermediates: PipelineIntermediatesConfig = Field(default_factory=PipelineIntermediatesConfig)
    logging: PipelineLoggingConfig = Field(default_factory=PipelineLoggingConfig)


class IENFConfig(BaseModel):
    """Root configuration for IENF quantification pipeline."""

    seed_extraction: SeedExtractionConfig = Field(default_factory=SeedExtractionConfig)
    network_building: NetworkBuildingConfig = Field(default_factory=NetworkBuildingConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


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


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> IENFConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. If None, uses default.yaml
        overrides: Dictionary of override values (e.g., from CLI arguments)

    Returns:
        Validated IENFConfig object

    Example:
        >>> config = load_config('config/high_quality.yaml')
        >>> config = load_config(overrides={'seed_extraction.window_size': 7})
    """
    # Determine config file path
    if config_path is None:
        # Default to config/default.yaml in project root
        project_root = Path(__file__).parent.parent
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
    config = IENFConfig(**config_dict)

    return config


def apply_overrides(config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply override values to configuration dictionary.

    Supports dot notation for nested keys: 'seed_extraction.window_size'

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


def save_config(config: IENFConfig, output_path: str):
    """
    Save configuration to YAML file.

    Useful for saving modified configurations or creating new profiles.

    Args:
        config: IENFConfig object to save
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
def get_default_config() -> IENFConfig:
    """Get default configuration."""
    return load_config()


if __name__ == "__main__":
    # Test configuration loading
    print("Loading default configuration...")
    config = get_default_config()

    print("\nSeed Extraction Config:")
    print(f"  Window size: {config.seed_extraction.window_size}")
    print(f"  Base segment length: {config.seed_extraction.base_segment_length}")

    print("\nNetwork Building Config:")
    print(f"  K-neighbors: {config.network_building.network.k_neighbors}")
    print(f"  Cost weights: α={config.network_building.cost_weights.alpha}, "
          f"β={config.network_building.cost_weights.beta}, "
          f"γ={config.network_building.cost_weights.gamma}")

    print("\nReconstruction Config:")
    print(f"  Max edge cost: {config.reconstruction.max_edge_cost}")
    print(f"  Min branch angle: {config.reconstruction.min_branch_angle}")

    print("\n✓ Configuration loaded successfully!")
