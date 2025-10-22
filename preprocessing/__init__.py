"""
IENF 前處理套件

提供神經纖維影像的前處理功能，包括：
- 通道去耦合
- 背景不均校正
- 局部對比度增強
- 標準化

主要類別和函數：
- PreprocessingPipeline: 完整的前處理流程
- PreprocessingConfig: 配置參數
"""

from .pipeline import PreprocessingPipeline
from .config import PreprocessingConfig
from .image_loader import load_image, save_image

__version__ = "1.0.0"
__all__ = [
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "load_image",
    "save_image",
]
