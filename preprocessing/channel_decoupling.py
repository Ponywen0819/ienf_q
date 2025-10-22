"""
通道去耦合模組

負責校正紅色通道對綠色通道的干擾
"""

import numpy as np
from scipy import ndimage
from .config import ChannelDecouplingConfig


class ChannelDecoupler:
    """通道去耦合處理器"""

    def __init__(self, config: ChannelDecouplingConfig):
        """
        初始化通道去耦合處理器

        Args:
            config: 通道去耦合配置
        """
        self.config = config

    def process(
        self,
        green_channel: np.ndarray,
        red_channel: np.ndarray,
        blue_channel: np.ndarray = None
    ) -> np.ndarray:
        """
        執行通道去耦合

        Args:
            green_channel: 綠色通道圖像
            red_channel: 紅色通道圖像
            blue_channel: 藍色通道圖像（可選，用於處理紫色干擾）

        Returns:
            校正後的綠色通道圖像
        """
        # 驗證輸入
        if green_channel.shape != red_channel.shape:
            raise ValueError(
                f"綠色和紅色通道尺寸不匹配: "
                f"{green_channel.shape} vs {red_channel.shape}"
            )

        # 確保是 2D 圖像
        if len(green_channel.shape) != 2:
            raise ValueError(f"輸入必須是 2D 圖像，收到 {len(green_channel.shape)}D")

        # 如果需要使用藍色通道但未提供
        if self.config.use_blue_channel and blue_channel is None:
            raise ValueError("配置要求使用藍色通道，但未提供 blue_channel")

        # 根據配置選擇使用固定或自適應 alpha
        if self.config.adaptive:
            alpha = self._calculate_adaptive_alpha(red_channel)
        else:
            alpha = self.config.alpha

        # 應用線性校正
        if self.config.use_blue_channel and blue_channel is not None:
            corrected = self._apply_multiband_correction(
                green_channel, red_channel, blue_channel, alpha, self.config.beta
            )
        else:
            corrected = self._apply_linear_correction(green_channel, red_channel, alpha)

        return corrected

    def _calculate_adaptive_alpha(
        self,
        red_channel: np.ndarray,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """
        計算自適應 alpha 值

        Args:
            red_channel: 紅色通道圖像
            mask: 可選的遮罩，用於區分表皮上下方

        Returns:
            alpha 映射圖 (與輸入同尺寸)
        """
        # 計算紅色通道的局部密度（使用高斯平滑）
        # 使用較大的 sigma 來獲得區域性的紅色密度
        from scipy.ndimage import gaussian_filter

        sigma = 20  # 控制密度計算的區域大小
        red_density = gaussian_filter(red_channel.astype(float), sigma=sigma)

        # 正規化紅色密度到 [0, 1]
        red_min = red_density.min()
        red_max = red_density.max()

        if red_max - red_min > 0:
            normalized_density = (red_density - red_min) / (red_max - red_min)
        else:
            # 如果紅色通道是均勻的，使用基礎 alpha
            return np.full_like(red_channel, self.config.alpha, dtype=float)

        # 根據紅色密度計算 alpha
        # 密度高的區域（表皮下方）使用較大的 alpha
        # 密度低的區域（表皮上方）使用較小的 alpha
        alpha_map = (
            self.config.alpha_min +
            (self.config.alpha_max - self.config.alpha_min) * normalized_density
        )

        # 如果提供了遮罩，可以進一步調整
        if mask is not None:
            # 在遮罩外的區域使用基礎 alpha
            alpha_map = np.where(mask > 0, alpha_map, self.config.alpha)

        # 應用平滑過渡，避免邊界偽影
        alpha_map = gaussian_filter(alpha_map, sigma=10)

        return alpha_map

    def _apply_linear_correction(
        self,
        green_channel: np.ndarray,
        red_channel: np.ndarray,
        alpha: float | np.ndarray
    ) -> np.ndarray:
        """
        應用線性校正: Green_corrected = Green_original - α × Red

        Args:
            green_channel: 綠色通道圖像
            red_channel: 紅色通道圖像
            alpha: 去耦合係數 (標量或與圖像同尺寸的數組)

        Returns:
            校正後的綠色通道圖像
        """
        # 轉換為 float 以避免溢出
        green_float = green_channel.astype(float)
        red_float = red_channel.astype(float)

        # 應用線性校正公式: Green_corrected = Green_original - α × Red
        corrected = green_float - alpha * red_float

        # 裁剪到有效範圍 [0, 原始最大值]
        # 保持原始數據的動態範圍
        original_max = green_channel.max()
        corrected = np.clip(corrected, 0, original_max)

        # 轉換回原始數據類型
        corrected = corrected.astype(green_channel.dtype)

        return corrected

    def _apply_multiband_correction(
        self,
        green_channel: np.ndarray,
        red_channel: np.ndarray,
        blue_channel: np.ndarray,
        alpha: float | np.ndarray,
        beta: float | np.ndarray
    ) -> np.ndarray:
        """
        應用多通道校正: Green_corrected = Green - α × Red - β × Blue

        用於處理紫色/洋紅色干擾（紅色+藍色混合）

        Args:
            green_channel: 綠色通道圖像
            red_channel: 紅色通道圖像
            blue_channel: 藍色通道圖像
            alpha: 紅色去耦合係數
            beta: 藍色去耦合係數

        Returns:
            校正後的綠色通道圖像
        """
        # 轉換為 float 以避免溢出
        green_float = green_channel.astype(float)
        red_float = red_channel.astype(float)
        blue_float = blue_channel.astype(float)

        # 應用多通道校正公式
        corrected = green_float - alpha * red_float - beta * blue_float

        # 裁剪到有效範圍
        original_max = green_channel.max()
        corrected = np.clip(corrected, 0, original_max)

        # 轉換回原始數據類型
        corrected = corrected.astype(green_channel.dtype)

        return corrected
