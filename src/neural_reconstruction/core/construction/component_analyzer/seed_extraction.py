"""
種子提取模組 (Seed Extraction Module)

提供從拓樸邊上均勻抽取種子點的功能。

主要用途：
- 在骨架路徑上均勻放置種子點，用於後續神經重建

輸入：NetworkX MultiGraph (由 ComponentTopologyBuilder 產生)
"""

import logging
from typing import Tuple, List

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


class EdgeSeedGenerator:
    """邊種子提取器 - 從拓樸邊上均勻抽取種子"""

    def __init__(self, min_edge_length: float = 10.0):
        """
        Args:
            min_edge_length: 最小邊長度閾值（像素），短於此長度的邊不抽取種子
        """
        self.min_edge_length = min_edge_length

    def extract_seeds_from_topology(
        self, graph: nx.MultiGraph, segment_length: float
    ) -> nx.MultiGraph:
        """
        從整個拓樸結構（NetworkX graph）的所有邊抽取種子

        Args:
            graph: NetworkX MultiGraph（由 ComponentTopologyBuilder 產生）
            segment_length: 分段長度閾值（像素）
            component_id: 元件 ID

        Returns:
            seeds: SeedPoint 列表
        """
        result_graph = nx.MultiGraph()

        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()
        logger.debug(f"從 {num_edges} 條邊抽取種子...")

        # 遍歷所有邊
        for u, v, data in graph.edges(data=True):
            # 加入原始節點
            result_graph.add_node(u, **graph.nodes[u])
            result_graph.add_node(v, **graph.nodes[v])

            # 從邊的屬性中獲取路徑和長度
            path_coords = data.get("path", [])
            path_coords = [u] + path_coords  # 加上 source 座標，因為原始 path 不包含
            length = data.get("branch-distance", 0.0)

            # 抽取種子
            edges = self.extract_seeds_from_edge(path_coords, segment_length, length)

            for edge in edges:
                source_node = edge[0]
                target_node = edge[-1]
                path = edge[1:]  # 不包含 source node
                result_graph.add_edge(source_node, target_node, path=path)

        logger.debug(f"總計生成 {result_graph.number_of_edges() - num_nodes} 個種子")

        return result_graph

    def extract_seeds_from_edge(
        self, path: List[Tuple[int, int]], segment_length: float, length: float
    ) -> List[Tuple[int, int]]:
        """
        從單條邊均勻抽取種子

        Args:
            path: 路徑座標列表 [(y, x), ...]
            segment_length: 分段長度閾值（像素）
            length: 邊長度（像素）
        Returns:
            edges: 路徑座標列表 [(y, x), ...] 或 None（如果未抽取種子）
        """
        # 判斷邊長度是否小於閾值
        if length < self.min_edge_length:
            logger.debug(
                f"邊長度 {length:.2f} < 閾值 {self.min_edge_length}，不抽取種子"
            )
            return []

        # 計算需要抽取的種子數量（下界）
        num_seeds = int(length // segment_length)

        if num_seeds <= 0:
            logger.debug(f"邊長度 {length:.2f}，計算出種子數 {num_seeds}，不抽取種子")
            return []
        edges = []

        cumulative_distances = self._compute_cumulative_distances(path)
        last_subpath_end_index = 0
        for i in range(num_seeds):
            target_distance = (i + 1) * segment_length

            # 找到最接近目標距離的路徑點
            seed_index = self._find_point_at_distance(
                path, cumulative_distances, target_distance
            )

            subpath = path[last_subpath_end_index : seed_index + 1]
            last_subpath_end_index = seed_index
            edges.append(subpath)
        return edges

    def _compute_cumulative_distances(self, path: List[Tuple[int, int]]) -> List[float]:
        """
        計算路徑上每個點的累積距離

        Args:
            path: 路徑座標列表

        Returns:
            cumulative_distances: 累積距離列表
        """
        cumulative_distances = [0.0]

        for i in range(1, len(path)):
            p1 = np.array(path[i - 1])
            p2 = np.array(path[i])
            dist = float(np.linalg.norm(p2 - p1))
            cumulative_distances.append(cumulative_distances[-1] + dist)

        return cumulative_distances

    def _find_point_at_distance(
        self,
        path: List[Tuple[int, int]],
        cumulative_distances: List[float],
        target_distance: float,
    ) -> int:
        """
        找到路徑上累積距離最接近目標距離的點

        Args:
            path: 路徑座標列表
            cumulative_distances: 累積距離列表
            target_distance: 目標距離

        Returns:
            index: 路徑上最接近目標距離的點的索引
        """
        # 找到第一個累積距離 >= target_distance 的點
        for i in range(len(cumulative_distances)):
            if cumulative_distances[i] >= target_distance:
                return i

        # 如果沒找到（理論上不應發生），返回最後一個點
        return len(path) - 1
