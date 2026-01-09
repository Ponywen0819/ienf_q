"""
元件分析模組 (Component Analyzer Module)

提供單一連通元件的完整分析功能：
1. 骨架化
2. 關鍵點偵測（端點/分支點）
3. 拓樸建構
4. 種子萃取

所有分析結果使用局部座標系統（相對於元件的 bbox）。
"""

import logging
from typing import List, Any

import numpy as np
from skimage import morphology

from neural_reconstruction.data_types import (
    SeedPoint,
    TopologyResult,
    ComponentAnalysisResult,
)
from .topology import KeyPointDetector, TopologyBuilder
from .seed_extraction import EdgeSeedExtractor

logger = logging.getLogger(__name__)


class ComponentAnalyzer:
    """
    單一元件的完整分析器

    對每個連通元件執行：骨架化 → 關鍵點偵測 → 拓樸建構 → 種子萃取
    所有結果使用局部座標系統。
    """

    def __init__(
        self,
        segment_length: float = 10.0,
        min_edge_length: float = 10.0
    ):
        """
        Args:
            segment_length: 種子間隔長度（像素）
            min_edge_length: 最小邊長度閾值（像素）
        """
        self.segment_length = segment_length
        self.keypoint_detector = KeyPointDetector()
        self.topology_builder = TopologyBuilder()
        self.seed_extractor = EdgeSeedExtractor(min_edge_length=min_edge_length)

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """
        對 mask 執行骨架化（Zhang-Suen 演算法）

        Args:
            mask: 二值 mask (0 或 255)

        Returns:
            骨架影像 (0 或 255)
        """
        binary = (mask > 0).astype(np.uint8)
        skeleton = morphology.skeletonize(binary)
        return (skeleton * 255).astype(np.uint8)

    def analyze(self, region: Any) -> ComponentAnalysisResult:
        """
        對單一元件執行完整分析

        Args:
            region: scikit-image RegionProperties 物件

        Returns:
            ComponentAnalysisResult: 包含骨架、拓樸、種子的完整分析結果（局部座標）
        """
        component_id = region.label
        bbox = region.bbox  # (minr, minc, maxr, maxc)

        logger.debug(f"分析元件 {component_id}...")

        # 提取該元件的 mask（局部座標）
        mask = region.image.astype(np.uint8) * 255

        # 1. 骨架化
        skeleton = self._skeletonize(mask)

        # 2. 關鍵點偵測
        endpoints, branchpoints = self.keypoint_detector.detect_all(skeleton)

        logger.debug(f"  端點數: {len(endpoints)}, 分支點數: {len(branchpoints)}")

        # 3. 拓樸建構（局部座標）
        topology = self.topology_builder.build_topology(
            skeleton, endpoints, branchpoints
        )

        # 4. 種子萃取（局部座標）
        seeds: List[SeedPoint] = self.seed_extractor.extract_seeds_from_topology(
            topology, self.segment_length, component_id
        )

        # 5. 將拓樸節點加入種子
        for node in topology.nodes:
            seeds.append(SeedPoint(
                position=node.position,
                seed_type=node.node_type,
                component_id=component_id
            ))

        # 6. 空種子處理：使用質心
        if not seeds:
            h, w = mask.shape
            centroid = (h // 2, w // 2)
            seeds.append(SeedPoint(
                position=centroid,
                seed_type='centroid',
                component_id=component_id
            ))
            logger.debug(f"  元件 {component_id} 無種子，使用質心 {centroid}")

        logger.debug(f"  種子數: {len(seeds)}")

        return ComponentAnalysisResult(
            component_id=component_id,
            bbox=bbox,
            skeleton=skeleton,
            topology=topology,
            seeds=seeds
        )

    def batch_analyze(self, regions: List[Any]) -> List[ComponentAnalysisResult]:
        """
        批次分析多個元件

        Args:
            regions: scikit-image RegionProperties 物件列表

        Returns:
            List[ComponentAnalysisResult]: 分析結果列表
        """
        logger.info(f"開始分析 {len(regions)} 個元件...")

        results = []
        for region in regions:
            result = self.analyze(region)
            results.append(result)

        total_seeds = sum(len(r.seeds) for r in results)
        logger.info(f"元件分析完成！共 {len(results)} 個元件，{total_seeds} 個種子")

        return results
