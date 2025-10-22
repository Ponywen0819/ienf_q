"""
配置管理模組

定義前處理流程的所有參數
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChannelDecouplingConfig:
    """通道去耦合配置"""
    alpha: float = 0.2  # 基礎去耦合係數（紅色→綠色）
    beta: float = 0.0   # 藍色去耦合係數（藍色→綠色）
    adaptive: bool = False  # 是否使用自適應策略
    alpha_min: float = 0.2  # 自適應最小 alpha
    alpha_max: float = 0.5  # 自適應最大 alpha
    use_blue_channel: bool = False  # 是否考慮藍色通道影響


@dataclass
class BackgroundCorrectionConfig:
    """背景校正配置"""
    ball_radius: int = 50  # Rolling Ball 半徑 (40-60)
    method: str = "rolling_ball"  # 背景校正方法


@dataclass
class ContrastEnhancementConfig:
    """對比度增強配置"""
    tile_size: tuple[int, int] = (8, 8)  # CLAHE tile size (8x8 或 16x16)
    clip_limit: float = 2.0  # CLAHE clip limit (2.0-4.0)


@dataclass
class NormalizationConfig:
    """標準化配置"""
    lower_percentile: float = 1.0  # 下百分位數
    upper_percentile: float = 99.0  # 上百分位數
    output_range: tuple[int, int] = (0, 255)  # 輸出範圍


@dataclass
class PreprocessingConfig:
    """完整的前處理配置"""

    # 各步驟配置
    channel_decoupling: ChannelDecouplingConfig = field(default_factory=ChannelDecouplingConfig)
    background_correction: BackgroundCorrectionConfig = field(default_factory=BackgroundCorrectionConfig)
    contrast_enhancement: ContrastEnhancementConfig = field(default_factory=ContrastEnhancementConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    # 流程控制
    enable_channel_decoupling: bool = True
    enable_background_correction: bool = True
    enable_contrast_enhancement: bool = True
    enable_normalization: bool = True

    # 輸出控制
    save_intermediate: bool = False  # 是否保存中間結果
    verbose: bool = True  # 是否輸出詳細信息

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PreprocessingConfig":
        """
        從字典創建配置

        Args:
            config_dict: 配置字典

        Returns:
            PreprocessingConfig 實例
        """
        pass

    def to_dict(self) -> dict:
        """
        轉換為字典

        Returns:
            配置字典
        """
        pass

    @classmethod
    def load_from_file(cls, config_path: str) -> "PreprocessingConfig":
        """
        從 YAML/JSON 文件載入配置

        Args:
            config_path: 配置文件路徑

        Returns:
            PreprocessingConfig 實例
        """
        pass

    def save_to_file(self, config_path: str) -> None:
        """
        保存配置到文件

        Args:
            config_path: 配置文件路徑
        """
        pass
