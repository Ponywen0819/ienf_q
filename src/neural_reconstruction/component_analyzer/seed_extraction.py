"""
種子提取模組 (Seed Extraction Module)

提供從拓樸邊上均勻抽取種子點的功能。

主要用途：
- 在骨架路徑上均勻放置種子點，用於後續神經重建

注意：SkeletonTopologyBuilder 已移動至 topology.py 模組
"""

import logging
from typing import Tuple, List

import numpy as np

from ..data_types import SeedPoint, TopologyResult, TopologyEdge

logger = logging.getLogger(__name__)


class EdgeSeedExtractor:
    """邊種子提取器 - 從拓樸邊上均勻抽取種子"""

    def __init__(self, min_edge_length: float = 10.0):
        """
        Args:
            min_edge_length: 最小邊長度閾值（像素），短於此長度的邊不抽取種子
        """
        self.min_edge_length = min_edge_length

    def extract_seeds_from_edge(
        self,
        edge: TopologyEdge,
        segment_length: float,
        component_id: int
    ) -> List[SeedPoint]:
        """
        從單條邊均勻抽取種子

        Args:
            edge: TopologyEdge 物件
            segment_length: 分段長度閾值（像素）
            component_id: 元件 ID

        Returns:
            seeds: SeedPoint 列表
        """
        path = edge.path
        length = edge.length

        # 判斷邊長度是否小於閾值
        if length < self.min_edge_length:
            logger.debug(f"邊長度 {length:.2f} < 閾值 {self.min_edge_length}，不抽取種子")
            return []

        # 計算需要抽取的種子數量（下界）
        num_seeds = int(length // segment_length)

        if num_seeds <= 0:
            logger.debug(f"邊長度 {length:.2f}，計算出種子數 {num_seeds}，不抽取種子")
            return []

        logger.debug(f"邊長度 {length:.2f}，抽取 {num_seeds} 個種子")

        # 均勻放置種子
        positions = self._place_seeds_uniformly(path, num_seeds)

        # 轉換為 SeedPoint
        seeds = []
        for pos in positions:
            seeds.append(SeedPoint(
                position=pos,
                seed_type='edge',
                component_id=component_id,
                edge_id=edge.source_id  # 使用 source_id 作為邊的標識
            ))

        return seeds

    def _place_seeds_uniformly(
        self,
        path: List[Tuple[int, int]],
        num_seeds: int
    ) -> List[Tuple[int, int]]:
        """
        在路徑上均勻放置種子

        Args:
            path: 路徑座標列表
            num_seeds: 要放置的種子數量

        Returns:
            seeds: 種子座標列表
        """
        if num_seeds <= 0 or len(path) < 2:
            return []

        # 計算路徑上每個點的累積距離
        cumulative_distances = [0.0]
        for i in range(1, len(path)):
            prev = np.array(path[i-1])
            curr = np.array(path[i])
            dist = np.linalg.norm(curr - prev)
            cumulative_distances.append(cumulative_distances[-1] + dist)

        total_length = cumulative_distances[-1]

        if total_length < 1e-6:
            logger.warning("路徑總長度接近零，無法均勻放置種子")
            return []

        # 計算種子應該放置的目標距離
        # 將路徑分成 (num_seeds + 1) 段，種子放在分段點上
        seeds = []
        for i in range(1, num_seeds + 1):
            target_distance = (i * total_length) / (num_seeds + 1)

            # 找到最接近目標距離的路徑點
            seed_position = self._find_point_at_distance(
                path, cumulative_distances, target_distance
            )

            seeds.append(seed_position)

        return seeds

    def _find_point_at_distance(
        self,
        path: List[Tuple[int, int]],
        cumulative_distances: List[float],
        target_distance: float
    ) -> Tuple[int, int]:
        """
        找到路徑上累積距離最接近目標距離的點

        Args:
            path: 路徑座標列表
            cumulative_distances: 累積距離列表
            target_distance: 目標距離

        Returns:
            position: 最接近目標距離的點座標
        """
        # 找到第一個累積距離 >= target_distance 的點
        for i in range(len(cumulative_distances)):
            if cumulative_distances[i] >= target_distance:
                return path[i]

        # 如果沒找到（理論上不應發生），返回最後一個點
        return path[-1]

    def extract_seeds_from_topology(
        self,
        topology: TopologyResult,
        segment_length: float,
        component_id: int
    ) -> List[SeedPoint]:
        """
        從整個拓樸結構的所有邊抽取種子

        Args:
            topology: TopologyResult 物件
            segment_length: 分段長度閾值（像素）
            component_id: 元件 ID

        Returns:
            seeds: SeedPoint 列表
        """
        all_seeds: List[SeedPoint] = []

        logger.debug(f"從 {len(topology.edges)} 條邊抽取種子...")

        for edge in topology.edges:
            seeds = self.extract_seeds_from_edge(edge, segment_length, component_id)
            all_seeds.extend(seeds)

        logger.debug(f"總計抽取 {len(all_seeds)} 個種子")

        return all_seeds
