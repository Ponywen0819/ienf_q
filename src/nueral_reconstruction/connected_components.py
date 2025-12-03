#!/usr/bin/env python3
"""
連通元件分析模組 (Connected Components Analysis)

提供 ConnectedComponentsAnalyzer 類別，用於從二值標註影像中提取所有獨立的白色區塊（連通元件），
返回 scikit-image RegionProperties 物件列表，供後續骨架化和種子提取使用。

功能:
- 自動檢測並分離二值影像中的所有獨立區塊
- 支援 4-連通或 8-連通分析
- 可過濾小面積雜訊元件
- 提供詳細的統計資訊（元件數量、面積等）
- 自動二值化非標準二值影像

使用範例:
    from connected_components import ConnectedComponentsAnalyzer

    # 建立分析器（8-連通，過濾小於 50 像素的元件）
    analyzer = ConnectedComponentsAnalyzer(connectivity=8, min_area=50)

    # 處理影像，返回 RegionProperties 物件列表
    regions = analyzer.process('path/to/annotation.png')

    # 使用返回的 region 物件
    for region in regions:
        print(f"元件面積: {region.area}")
        print(f"邊界框: {region.bbox}")
        print(f"質心: {region.centroid}")

類別參數:
    connectivity: int = 8
        連通性，可選 4 或 8（預設：8）
    min_area: int = 10
        最小元件面積（像素），小於此值會被過濾（預設：10）

返回值:
    List[RegionProperties]: scikit-image 的 RegionProperties 物件列表
        每個物件包含元件的各種屬性（area, bbox, centroid, perimeter 等）

作者: Generated with Claude Code
日期: 2025-11-15
"""

import logging
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from skimage import measure

# 設定 logger
logger = logging.getLogger(__name__)

class ConnectedComponentsAnalyzer:
    """連通元件分析器"""

    def __init__(
        self,
        connectivity: int = 8,
        min_area: int = 10,
    ):
        """
        初始化連通元件分析器

        Args:
            connectivity: 連通性 (4 或 8)
            min_area: 最小元件面積（像素），小於此值的元件會被過濾
            verbose: 是否輸出詳細資訊
        """

        self.connectivity = connectivity
        self.min_area = min_area

        # 驗證參數
        if connectivity not in [4, 8]:
            raise ValueError(f"connectivity 必須是 4 或 8，但收到 {connectivity}")

        # scikit-image 的 connectivity 參數: 1 = 4-連通, 2 = 8-連通
        self.skimage_connectivity = 1 if connectivity == 4 else 2

    def load_binary_image(self, image_path: str) -> np.ndarray:
        """
        載入並驗證二值影像

        Args:
            image_path: 影像檔案路徑

        Returns:
            二值影像陣列 (0 或 255)

        Raises:
            FileNotFoundError: 檔案不存在
            ValueError: 影像格式不正確
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"影像檔案不存在: {image_path}")

        # 載入為灰階影像
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        # 檢查是否為二值影像
        unique_values = np.unique(image)
        if not (len(unique_values) <= 2 and all(v in [0, 255] for v in unique_values)):
            logger.warning(f"影像不是標準二值影像（只包含 0 和 255）")
            logger.warning(f"  唯一像素值: {unique_values}")
            logger.warning(f"  將進行二值化處理（閾值 = 127）")
            # 自動二值化
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        logger.info(f"✓ 成功載入二值影像: {image_path}")
        logger.info(f"  影像尺寸: {image.shape[1]}x{image.shape[0]} (寬x高)")
        logger.info(f"  白色像素數: {np.sum(image == 255)} ({np.sum(image == 255) / image.size * 100:.2f}%)")

        return image

    def analyze(self, binary_image: np.ndarray) -> List[Dict]:
        """
        執行連通元件分析

        Args:
            binary_image: 二值影像 (0 或 255)

        Returns:
            (標籤影像, 元件列表)
            - 標籤影像: 每個像素值代表所屬元件的 ID (0 = 背景)
            - 元件列表: 每個元件的屬性字典
        """
        logger.info(f"\n執行連通元件分析...")
        logger.info(f"  連通性: {self.connectivity}-連通")
        logger.info(f"  最小面積過濾: {self.min_area} 像素")

        # 1. 執行連通元件標記
        # 將二值影像轉為 0/1
        binary = (binary_image > 0).astype(np.uint8)

        # 使用 scikit-image 標記連通元件
        labeled_image = measure.label(binary, connectivity=self.skimage_connectivity)

        initial_num_components = labeled_image.max()

        logger.info(f"\n  初始檢測到 {initial_num_components} 個連通元件")

        # 2. 提取元件屬性
        regions = measure.regionprops(labeled_image)

        # 3. 過濾小元件並重新編號
        valid_components = []
        filtered_count = 0

        for region in regions:
            if region.area >= self.min_area:
                valid_components.append(region)
            else:
                filtered_count += 1

        logger.info(f"  過濾掉 {filtered_count} 個小元件（面積 < {self.min_area}）")
        logger.info(f"  保留 {len(valid_components)} 個有效元件")

        # 4. 創建新的標籤影像（重新編號 1, 2, 3, ...）
        components_list = []

        for new_id, region in enumerate(valid_components, start=1):
            # 提取元件屬性
            # component_info = self.extract_component_properties(region, new_id)
            components_list.append(region)

        logger.info(f"\n✓ 連通元件分析完成")
        logger.info(f"  最終元件數: {len(components_list)}")
        if components_list:
            total_area = sum(c['area'] for c in components_list)
            avg_area = total_area / len(components_list)
            logger.info(f"  總標註面積: {total_area} 像素")
            logger.info(f"  平均元件面積: {avg_area:.1f} 像素")
            logger.info(f"  最大元件面積: {max(c['area'] for c in components_list)} 像素")
            logger.info(f"  最小元件面積: {min(c['area'] for c in components_list)} 像素")

        return components_list

    def process(
        self,
        image: np.ndarray
    ) -> List[Dict]:
        """
        完整的連通元件分析流程

        Args:
            image_path: 輸入二值標註影像路徑

        Returns:
            (標籤影像, 元件列表)
        """

        # 1. 載入二值影像
        binary_image = self.load_binary_image(image_path)

        # 2. 執行連通元件分析
        components = self.analyze(binary_image)

        return components
