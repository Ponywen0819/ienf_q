"""
背景校正模組

實現 Rolling Ball 背景扣除算法
"""

import numpy as np
from .config import BackgroundCorrectionConfig


class BackgroundCorrector:
    """背景校正處理器"""

    def __init__(self, config: BackgroundCorrectionConfig):
        """
        初始化背景校正處理器

        Args:
            config: 背景校正配置
        """
        self.config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        執行背景校正

        Args:
            image: 輸入圖像

        Returns:
            背景校正後的圖像
        """
        pass

    def _rolling_ball_background(
        self,
        image: np.ndarray,
        radius: int
    ) -> np.ndarray:
        """
        計算 Rolling Ball 背景

        原理：
        - 模擬球體在影像強度表面下方滾動
        - 球接觸的表面視為背景

        Args:
            image: 輸入圖像
            radius: 球體半徑

        Returns:
            估計的背景圖像
        """
        pass

    def _create_ball_kernel(self, radius: int) -> np.ndarray:
        """
        創建球形結構元素

        Args:
            radius: 球體半徑

        Returns:
            球形核心
        """
        pass

    def _subtract_background(
        self,
        image: np.ndarray,
        background: np.ndarray
    ) -> np.ndarray:
        """
        從原圖扣除背景

        Args:
            image: 原始圖像
            background: 背景圖像

        Returns:
            背景扣除後的圖像
        """
        pass
