from dataclasses import dataclass, field
from typing import Dict, Any, Literal


@dataclass
class MorphologyConfig:
    """Configuration for morphological operations.

    Attributes:
        closing_kernel: Kernel size for closing operation. Default: 3
        opening_kernel: Kernel size for opening operation. Default: 3
    """

    closing_kernel: int = 3
    opening_kernel: int = 3


@dataclass
class MaskConfig:
    """Configuration for mask operations.

    Attributes:
        dilate_offset: Vertical dilation offset in pixels. Default: 100
    """

    dilate_offset: int = 100


@dataclass
class BackgroundConfig:
    """Configuration for background correction.

    Attributes:
        method: Background correction method ('morphology', 'rolling_ball'). Default: 'rolling_ball'
        radius: Ball radius for rolling ball/morphology methods. Default: 4
        sato_weight: Blend weight for Sato filter (0=disabled, >0=enabled). Default: 0.0
        sato_sigmas: Iterable of floats for Sato filter scales. Default: (1.0, 2.0, 3.0)
    """

    method: Literal["morphology", "rolling_ball"] = "rolling_ball"
    radius: int = 4
    sato_weight: float = 0.0
    sato_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.sato_weight <= 1:
            raise ValueError(f"sato_weight must be in [0, 1], got {self.sato_weight}")


@dataclass
class ThresholdConfig:
    """Configuration for thresholding operations.

    Attributes:
        method: Threshold type ('binary' or 'binary_inv'). Default: 'binary'
        use_full_roi: Use full ROI for pseudo-label instead of masked region. Default: False
    """

    use_full_roi: bool = False


@dataclass
class NormalizationConfig:
    """Configuration for normalization operations.

    Attributes:
        enabled: Whether to enable regional normalization. Default: True
        method: Normalization method ('minmax' or 'clahe'). Default: 'minmax'
        clip_limit: CLAHE clip limit for contrast limiting. Default: 2.0
        tile_grid_size: CLAHE tile grid size (width, height). Default: (8, 8)
    """

    enabled: bool = True
    method: Literal["minmax", "clahe"] = "minmax"
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


@dataclass
class PipelineConfig:
    """Complete configuration for skin analysis pipeline.

    This dataclass encapsulates all configuration parameters for the SkinAnalysisPipeline.

    Attributes:
        morphology: Morphological operations configuration
        mask: Mask operations configuration
        background: Background correction configuration
        threshold: Thresholding configuration
        normalization: Normalization configuration

    Example:
        >>> config = PipelineConfig(
        ...     morphology=MorphologyConfig(closing_kernel=3, opening_kernel=3),
        ...     mask=MaskConfig(dilate_offset=50),
        ...     background=BackgroundConfig(method='rolling_ball', radius=4, sato_weight=0.2, sato_sigmas=(1.0, 2.0)),
        ...     threshold=ThresholdConfig(method='binary')
        ... )
        >>> pipeline = SkinAnalysisPipeline(config)
    """

    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from a dictionary.

        Args:
            config_dict: Dictionary with 'morphology', 'mask', 'background',
                        'threshold', and 'normalization' keys

        Returns:
            PipelineConfig instance

        Raises:
            ValueError: If required configuration keys are missing
        """
        required_keys = ["morphology", "mask", "background", "threshold"]
        for key in required_keys:
            if key not in config_dict:
                raise ValueError(f"Missing required config key: '{key}'")

        morphology_cfg = config_dict.get("morphology", {})
        mask_cfg = config_dict.get("mask", {})
        background_cfg = config_dict.get("background", {})
        threshold_cfg = config_dict.get("threshold", {})
        normalization_cfg = config_dict.get("normalization", {})

        return cls(
            morphology=MorphologyConfig(**morphology_cfg),
            mask=MaskConfig(**mask_cfg),
            background=BackgroundConfig(**background_cfg),
            threshold=ThresholdConfig(**threshold_cfg),
            normalization=NormalizationConfig(**normalization_cfg),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert PipelineConfig to a dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "morphology": {
                "closing_kernel": self.morphology.closing_kernel,
                "opening_kernel": self.morphology.opening_kernel,
            },
            "mask": {
                "dilate_offset": self.mask.dilate_offset,
            },
            "background": {
                "method": self.background.method,
                "radius": self.background.radius,
                "sato_weight": self.background.sato_weight,
                "sato_sigmas": self.background.sato_sigmas,
            },
            "threshold": {
                "use_full_roi": self.threshold.use_full_roi,
            },
            "normalization": {
                "enabled": self.normalization.enabled,
                "method": self.normalization.method,
                "clip_limit": self.normalization.clip_limit,
                "tile_grid_size": self.normalization.tile_grid_size,
            },
        }
