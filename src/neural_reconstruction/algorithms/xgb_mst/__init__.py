"""
純 MST 重建模組 (Pure MST Reconstruction Module)

實作基於 MST 的傳統神經纖維重建算法。

主控制器：
    - PureMstLinker: 完整流程控制器

使用範例：
---------
from neural_reconstruction.algorithms.pure_mst import PureMstLinker

linker = PureMstLinker(
    segment_length=5.0,
    search_radius=50.0,
)
mst_forest = linker.run(label_image, green_channel)
"""

from .linker import XgbMstLinker

__all__ = ["XgbMstLinker"]

__version__ = "1.0.0"
