#!/usr/bin/env python3
"""
骨架化分析模組 (Skeletonization Analysis)

提供 SkeletonAnalyzer 類別，用於對連通元件執行形態學骨架化，
並檢測骨架的端點和分支點。

功能:
- 使用 Zhang-Suen 演算法進行骨架化
- 檢測骨架端點（鄰居數 = 1）
- 檢測骨架分支點（鄰居數 >= 3）

使用範例:
    from skeletonization import SkeletonAnalyzer
    from connected_components import ConnectedComponentsAnalyzer

    # 先提取連通元件
    cc_analyzer = ConnectedComponentsAnalyzer(connectivity=8, min_area=50)
    regions = cc_analyzer.process('path/to/annotation.png')

    # 進行骨架化分析
    skel_analyzer = SkeletonAnalyzer()
    skeleton_results = skel_analyzer.process(regions)

    # 使用返回的骨架資訊
    for result in skeleton_results:
        print(f"元件 {result['region'].label}:")
        print(f"  端點數: {result['num_endpoints']}")
        print(f"  分支點數: {result['num_branchpoints']}")

返回值:
    List[Dict]: 骨架分析結果列表，每個字典包含：
        - region: RegionProperties - 原始 region 物件
        - skeleton: np.ndarray - 骨架影像 (0 或 255)
        - num_endpoints: int - 端點數量
        - num_branchpoints: int - 分支點數量
        - endpoints: List[Dict] - 端點座標列表 [{'x': int, 'y': int}, ...]
        - branchpoints: List[Dict] - 分支點座標列表 [{'x': int, 'y': int}, ...]

作者: Generated with Claude Code
日期: 2025-11-15
"""

import logging
from typing import List, Dict, Any

import cv2
import numpy as np
from skimage import morphology

# 設定 logger
logger = logging.getLogger(__name__)


class SkeletonAnalyzer:
    """骨架化與結構分析器"""

    def __init__(self):
        """初始化骨架分析器"""
        self.method = 'zhang-suen'

    def skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """
        對單個 mask 執行骨架化 (使用 Zhang-Suen 演算法)

        Args:
            mask: 二值 mask (0 或 255)

        Returns:
            骨架影像 (0 或 255)
        """
        # 確保 mask 是二值的
        binary = (mask > 0).astype(np.uint8)

        # scikit-image 的 Zhang-Suen 演算法
        skeleton = morphology.skeletonize(binary)
        skeleton = (skeleton * 255).astype(np.uint8)

        return skeleton

    def detect_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """
        檢測骨架端點（鄰居數 = 1）

        Args:
            skeleton: 骨架影像 (0 或 255)

        Returns:
            端點座標陣列 [[y1, x1], [y2, x2], ...]
        """
        # 二值化
        binary = (skeleton > 0).astype(np.uint8)

        # 8-鄰域卷積核
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)

        # 計算每個骨架點的鄰居數
        neighbor_count = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # 端點: 骨架點 且 鄰居數 = 1
        endpoints_mask = (binary > 0) & (neighbor_count == 1)
        endpoints = np.argwhere(endpoints_mask)

        return endpoints

    def detect_branchpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """
        檢測骨架分支點（鄰居數 >= 3）

        Args:
            skeleton: 骨架影像 (0 或 255)

        Returns:
            分支點座標陣列 [[y1, x1], [y2, x2], ...]
        """
        # 二值化
        binary = (skeleton > 0).astype(np.uint8)

        # 8-鄰域卷積核
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)

        # 計算每個骨架點的鄰居數
        neighbor_count = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # 分支點: 骨架點 且 鄰居數 >= 3
        branchpoints_mask = (binary > 0) & (neighbor_count >= 3)
        branchpoints = np.argwhere(branchpoints_mask)

        return branchpoints

    def analyze_skeleton(
        self,
        skeleton: np.ndarray,
        region: Any
    ) -> Dict:
        """
        分析骨架結構

        Args:
            skeleton: 骨架影像
            region: 原始 region 物件（scikit-image RegionProperties）

        Returns:
            骨架分析結果字典
        """
        # 檢測端點和分支點
        endpoints = self.detect_endpoints(skeleton)
        branchpoints = self.detect_branchpoints(skeleton)

        # 構建結果
        result = {
            'region': region,
            'skeleton': skeleton,
            'num_endpoints': len(endpoints),
            'num_branchpoints': len(branchpoints),
            'endpoints': [{'x': int(pt[1]), 'y': int(pt[0])} for pt in endpoints],
            'branchpoints': [{'x': int(pt[1]), 'y': int(pt[0])} for pt in branchpoints],
        }

        return result
    
    def process(self, region: Any) -> Dict:
        """
        對單個 region 執行骨架化分析

        Args:
            region: scikit-image RegionProperties 物件

        Returns:
            骨架分析結果字典
        """
        logger.info(f"處理元件 (label={region.label}) 的骨架化...")

        # 提取該元件的 mask (相對於 bbox 的區域座標)
        mask = region.image.astype(np.uint8) * 255

        # 執行骨架化
        skeleton = self.skeletonize(mask)

        # 分析骨架結構
        skeleton_info = self.analyze_skeleton(skeleton, region)

        logger.info(f"  端點數: {skeleton_info['num_endpoints']}")
        logger.info(f"  分支點數: {skeleton_info['num_branchpoints']}")

        return skeleton_info

    def batch_process(self, regions: List[Any]) -> List[Dict]:
        """
        對 region 列表執行骨架化分析

        Args:
            regions: scikit-image RegionProperties 物件列表

        Returns:
            骨架分析結果列表，每個元素包含骨架影像和結構資訊
        """
        logger.info(f"開始處理 {len(regions)} 個元件的骨架化...")

        skeleton_results = []

        for i, region in enumerate(regions, 1):
            skeleton_info = self.process(region)
            skeleton_results.append(skeleton_info)

            logger.debug(f"  端點數: {skeleton_info['num_endpoints']}")
            logger.debug(f"  分支點數: {skeleton_info['num_branchpoints']}")

        logger.info(f"骨架化處理完成! 共處理 {len(skeleton_results)} 個元件")

        return skeleton_results
