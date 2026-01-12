"""
元件分析模組 (Component Analyzer Module)

提供單一連通元件的完整分析功能：
1. 拓樸建構（整合骨架化、關鍵點偵測、剪枝）
2. 種子萃取

所有分析結果使用局部座標系統（相對於元件的 bbox）。
"""

import logging
from typing import Any, Tuple

import numpy as np

from neural_reconstruction.common.data_types import (
    ComponentAnalysisResult,
)
from .topology import ComponentTopologyBuilder
from .seed_extraction import EdgeSeedGenerator

logger = logging.getLogger(__name__)


class ComponentAnalyzer:
    """
    單一元件的完整分析器

    使用 ComponentTopologyBuilder 進行拓樸建構（整合骨架化、分析、剪枝）
    所有結果使用局部座標系統。
    """

    def __init__(
        self,
        segment_length: float = 10.0,
        min_edge_length: float = 10.0,
        prune_threshold: float = 5.0,
        spacing: float = 1.0,
    ):
        """
        Args:
            segment_length: 種子間隔長度（像素）
            min_edge_length: 最小邊長度閾值（像素）
            prune_threshold: 剪枝閾值 - 移除長度小於此值的分支（像素）
            spacing: 像素間距（用於距離計算）
        """
        self.segment_length = segment_length
        self.topology_builder = ComponentTopologyBuilder(
            prune_threshold=prune_threshold, spacing=spacing
        )
        self.seed_extractor = EdgeSeedGenerator(min_edge_length=min_edge_length)

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

        # 1. 使用 ComponentTopologyBuilder 建構拓樸（整合骨架化、分析、剪枝）
        topology = self.topology_builder.build_topology(mask)

        topology = self.seed_extractor.extract_seeds_from_topology(
            topology, self.segment_length
        )

        # 4. 空種子處理：使用質心
        if topology.number_of_edges() <= 0:
            centroid = self._get_component_centroid(mask)
            topology.add_node(centroid)
            logger.debug(f"  元件 {component_id} 無種子，使用質心 {centroid}")

        logger.debug(f"  種子數: {topology.number_of_edges()}")

        return ComponentAnalysisResult(
            component_id=component_id, bbox=bbox, topology=topology
        )

    def _get_component_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        h, w = mask.shape
        centroid = (h // 2, w // 2)
        return centroid
