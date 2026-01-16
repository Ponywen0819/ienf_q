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
        dilate_offset: Vertical dilation offset in pixels. Default: 50
    """

    dilate_offset: int = 50


@dataclass
class BackgroundConfig:
    """Configuration for background correction.

    Attributes:
        method: Background correction method ('morphology', 'rolling_ball', or 'gaussian'). Default: 'morphology'
        radius: Ball radius for rolling ball/morphology methods. Default: 12
        sigma: Gaussian sigma for gaussian method. Default: 0
        light_background: Whether background is brighter than foreground. Default: False
    """

    method: Literal["morphology", "rolling_ball"] = "rolling_ball"
    radius: int = 2
    sigma: float = 0


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
        enabled: Whether to enable regional normalization. Default: False
    """

    enabled: bool = True


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
        ...     background=BackgroundConfig(method='rolling_ball', radius=12),
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
                "sigma": self.background.sigma,
            },
            "threshold": {
                "use_full_roi": self.threshold.use_full_roi,
            },
            "normalization": {
                "enabled": self.normalization.enabled,
            },
        }
