#!/usr/bin/env python3
"""
神經網路建構模組 (Neural Network Builder Module)

此模組負責從元件分析結果建構全局種子連接圖。

主要流程：
1. 從所有元件收集全局種子點，建構 cKDTree 空間索引
2. 針對每個種子，搜尋半徑 R 內的其他種子（不同元件）
3. 使用 A* 計算種子對之間的連接路徑與成本
4. 建構無向圖，每條邊包含節點對、路徑與成本

輸出：
- 無向圖結構，節點為種子點，邊為可行連接

使用範例:
    from src.neural_reconstruction.pair_analyzer import NetworkBuilder

    builder = NetworkBuilder(
        image=green_channel_image,
        search_radius=50.0,
        max_cost_threshold=0.98
    )

    # 從元件分析結果建構圖
    graph = builder.build_graph(component_results)

作者: Generated with Claude Code
日期: 2026-01-09
"""

import logging
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from scipy.spatial import KDTree
import networkx as nx

from .path_finder import Pathfinder
from neural_reconstruction.common.data_types import (
    ComponentAnalysisResult,
    ConnectionGraphBuilderResult,
)

# 設定 logger
logger = logging.getLogger(__name__)


class NetworkBuilder:
    """
    網路建構器

    基於 KDTree 建構全局種子連接圖。

    Attributes:
        pathfinder: A* 路徑搜尋器
        search_radius: 搜尋半徑（像素）
        max_cost_threshold: 最大成本閾值（成本超過此值則不建立連接）
        kdtree: 全局種子點空間索引
        global_seeds: 全局種子座標陣列 (N, 2)
        component_ids: 對應的元件 ID 陣列 (N,)
    """

    def __init__(
        self,
        image: np.ndarray,
        search_radius: float = 50.0,
        max_cost_threshold: float = 0.98,
        intensity_weight: float = 0.6,
        shape_weight: float = 0.4,
    ):
        """
        初始化網路建構器

        Args:
            image: 影像 (uint8, 0-255)
            search_radius: 搜尋半徑（像素），在此半徑內搜尋候選連接
            max_cost_threshold: 最大成本閾值比例 (0-1)
            intensity_weight: 強度權重
            shape_weight: 形狀權重
        """
        self.pathfinder = Pathfinder(image, intensity_weight, shape_weight)
        self.search_radius = search_radius
        self.max_cost_threshold = max_cost_threshold

        # 全局索引結構（稍後建構）
        self.kdtree: Optional[KDTree] = None
        self.global_seeds: Optional[np.ndarray] = None
        self.component_ids: Optional[np.ndarray] = None

        logger.info("=" * 70)
        logger.info("網路建構器初始化完成")
        logger.info("=" * 70)
        logger.info(f"搜尋半徑: {search_radius} 像素")
        logger.info(f"最大成本閾值: {max_cost_threshold}")
        logger.info(f"強度權重: {intensity_weight}, 形狀權重: {shape_weight}")
        logger.info("=" * 70)

    def build_graph(
        self, component_results: List[ComponentAnalysisResult]
    ) -> ConnectionGraphBuilderResult:
        """
        建構全局種子連接圖

        流程：
        1. 建構全局 cKDTree 索引
        2. 對每個種子，搜尋半徑內的其他種子（不同元件）
        3. 計算路徑與成本，建立邊（無向圖，避免重複）

        Args:
            component_results: 元件分析結果列表

        Returns:
            圖結構：
                {
                    'nodes': np.ndarray,  # 全局種子座標 (N, 2)
                    'component_ids': np.ndarray,  # 對應的元件 ID (N,)
                    'edges': List[Dict],  # 邊列表
                    'num_nodes': int,
                    'num_edges': int,
                    'num_components': int
                }
        """

        logger.info("\n" + "=" * 70)
        logger.info("開始建構全局種子連接圖")
        logger.info("=" * 70)

        global_topology = self._build_global_topology(component_results)
        print(global_topology.number_of_edges())
        self._build_global_index(global_topology)

        res = ConnectionGraphBuilderResult()
        if (
            self.kdtree is None
            or self.global_seeds is None
            or len(self.global_seeds) == 0
        ):
            logger.warning("沒有種子點，返回空圖")
            return res

        # 步驟 2: 對每個節點搜尋鄰居並建立邊
        processed_pairs: Set[Tuple[int, int]] = set()  # 記錄已處理的節點對（無向圖）

        num_nodes = len(self.global_seeds)
        logger.info(f"\n開始搜尋節點鄰居（半徑 {self.search_radius} 像素）...")

        total_new_edges = 0
        for i in range(num_nodes):
            new_edges = self._compute_edges_from_source(i, processed_pairs)
            total_new_edges += len(new_edges)
            # edges.extend(new_edges)
            self._update_edges_to_graph(new_edges, global_topology)

            # 進度報告
            if (i + 1) % 100 == 0 or (i + 1) == num_nodes:
                logger.info(
                    f"  已處理 {i + 1}/{num_nodes} 個節點，找到 {total_new_edges} 條新邊，當前圖邊數: {global_topology.number_of_edges()}"
                )

        logger.info("\n" + "=" * 70)
        logger.info("圖建構完成")
        logger.info("=" * 70)
        logger.info(f"節點總數: {num_nodes}")
        logger.info(f"邊總數: {global_topology.number_of_edges()}")
        logger.info(f"元件總數: {len(component_results)}")
        logger.info("=" * 70)

        res.nodes = self.global_seeds
        res.graph = global_topology
        return res

    def _get_component_global_topology(
        self, component_result: ComponentAnalysisResult
    ) -> nx.MultiGraph:
        """
        獲取元件的全局拓樸結構

        Args:
            component_result: 元件分析結果

        Returns:
            nx.MultiGraph: 元件的全局拓樸圖
        """
        bbox = component_result.bbox
        topology = component_result.topology
        minr, minc, _, _ = bbox

        # 建立座標轉換映射 (local -> global)
        mapping = {node: (node[0] + minr, node[1] + minc) for node in topology.nodes()}

        # 使用 nx.relabel_nodes 建立新圖，不修改原圖
        # copy=True 避免修改輸入的 topology
        topology = nx.relabel_nodes(topology, mapping, copy=True)

        # 更新 Edge 中的 path 資訊 (如果有的話)
        for u, v, data in topology.edges(data=True):
            if "path" in data:
                # 將 path 中的每個點也轉換為全局座標
                local_path = data["path"]
                global_path = [(p[0] + minr, p[1] + minc) for p in local_path]
                data["path"] = global_path
            # 在邊屬性中新增 weight 屬性，預設為 1e-5
            data["weight"] = 1e-5

        #  在所有節點中新增 component_id 屬性
        for node in topology.nodes():
            topology.nodes[node]["component_id"] = component_result.component_id
        return topology

    def _build_global_topology(
        self, component_result_list: List[ComponentAnalysisResult]
    ) -> nx.MultiGraph:
        """
        建構元件的全局拓樸結構

        Args:
            component_result: 元件分析結果

        Returns:
            nx.MultiGraph: 元件的全局拓樸圖
        """
        result_graph = nx.MultiGraph()
        for component_result in component_result_list:
            topology = self._get_component_global_topology(component_result)

            result_graph.update(topology)
        return result_graph

    def _build_global_index(self, global_graph: nx.MultiGraph) -> None:
        """
        建構全局種子點索引

        從所有元件收集種子點，建立 cKDTree 空間索引與對應的 component_id 陣列。

        Args:
            component_results: 元件分析結果列表
        """
        componenet_ids = []
        global_seeds = []

        for node in global_graph.nodes():
            global_seeds.append(node)
            componenet_ids.append(global_graph.nodes[node]["component_id"])

        # 建立全局陣列
        self.component_ids = np.array(
            componenet_ids,
            dtype=np.int32,
        )

        self.global_seeds = np.array(
            global_seeds,
            dtype=np.int32,
        )

        # 建立 KD-Tree（空陣列時無法建立，需要至少有一個點）
        if len(self.global_seeds) > 0:
            self.kdtree = KDTree(self.global_seeds)
        else:
            self.kdtree = None

        logger.info(f"建構全局索引: {len(self.global_seeds)} 個種子點")

    def _compute_edges_from_source(
        self, source_index: int, visited: Set[Tuple[int, int]]
    ) -> List[Dict]:
        """
        從單一源點計算可行邊

        Args:
            source_index: 源點索引
            visited: 已訪問的目標點集合

        Returns:
            可行邊列表
        """
        if (
            self.component_ids is None
            or self.global_seeds is None
            or self.kdtree is None
        ):
            raise RuntimeError("請先建構全局索引")

        edges = []
        unprocessed_neighbor_indices = self._get_unprocessed_neighbor_indices(
            source_index, visited
        )

        new_edges = self._resolve_candidate_paths(
            source_index, unprocessed_neighbor_indices, visited
        )
        edges.extend(new_edges)

        return edges

    def _get_unprocessed_neighbor_indices(
        self, source_index: int, visited: Set[Tuple[int, int]]
    ) -> List[int]:
        if (
            self.component_ids is None
            or self.global_seeds is None
            or self.kdtree is None
        ):
            raise RuntimeError("請先建構全局索引")

        # 過濾掉自己與同元件的節點（使用 NumPy 布林索引）
        source_component_id = self.component_ids[source_index]
        source_pos = self.global_seeds[source_index]

        # 搜尋半徑內的鄰居
        neighbor_indices = self.kdtree.query_ball_point(
            source_pos, r=self.search_radius
        )
        neighbor_indices = np.array(neighbor_indices, dtype=np.int32)

        # 建立篩選條件：不是自己 且 不是同元件
        mask = (neighbor_indices != source_index) & (
            self.component_ids[neighbor_indices] != source_component_id
        )
        valid_neighbor_indices = neighbor_indices[mask]

        # 過濾掉已處理的節點對（無向圖）
        unprocessed_neighbor_indices = []
        for valid_neighbor_index in valid_neighbor_indices:
            pair = (
                min(source_index, valid_neighbor_index),
                max(source_index, valid_neighbor_index),
            )
            if pair not in visited:
                unprocessed_neighbor_indices.append(pair)

        return unprocessed_neighbor_indices

    def _resolve_candidate_paths(
        self,
        source_index: int,
        target_index_list: List[Tuple[int, int]],
        visited: Set[Tuple[int, int]],
    ) -> List[Dict]:
        if (
            self.component_ids is None
            or self.global_seeds is None
            or self.kdtree is None
        ):
            raise RuntimeError("請先建構全局索引")

        # 批次計算從根節點到所有未處理鄰居的路徑
        source_pos = tuple(self.global_seeds[source_index].astype(int))

        # target_index_list 是 tuple pair 列表 [(min_idx, max_idx), ...]
        # 需要從 pair 中提取實際的 target index
        target_indices = [
            pair[1] if pair[0] == source_index else pair[0]
            for pair in target_index_list
        ]
        target_position_list = [
            tuple(self.global_seeds[j].astype(int)) for j in target_indices
        ]

        path_results = self.pathfinder.find_paths_from_source(
            source_pos, target_position_list
        )

        edges = []
        num_no_path = 0
        num_cost_filtered = 0
        # 處理路徑結果
        for pair, target_index, target_pos in zip(
            target_index_list, target_indices, target_position_list
        ):
            visited.add(pair)
            result = path_results.get(target_pos)

            if result is None:
                num_no_path += 1
                continue

            path, cost = result
            distance = np.linalg.norm(
                self.global_seeds[source_index] - self.global_seeds[target_index]
            )

            # 判斷成本是否在閾值內
            # max_cost = (
            #     self.pathfinder.cost_map.max() * distance * self.max_cost_threshold
            # )
            # if cost > max_cost:
            #     num_cost_filtered += 1
            #     continue

            edges.append(
                {
                    "node_a": source_index,
                    "node_b": target_index,
                    "distance": distance,
                    "cost": cost,
                    "path": path,
                }
            )

        return edges

    def _update_edges_to_graph(self, edges: List[Dict], graph: nx.Graph) -> None:
        """
        將邊列表加入 NetworkX 無向圖

        Args:
            edges: 邊列表
            graph: NetworkX 無向圖
        """
        if (
            self.component_ids is None
            or self.global_seeds is None
            or self.kdtree is None
        ):
            raise RuntimeError("請先建構全局索引")

        for edge in edges:
            node_a_index = edge["node_a"]
            node_b_index = edge["node_b"]

            node_a_pos = tuple(map(int, self.global_seeds[node_a_index]))
            node_b_pos = tuple(map(int, self.global_seeds[node_b_index]))

            graph.add_edge(
                node_a_pos,
                node_b_pos,
                distance=edge["distance"],
                weight=edge["cost"],
                path=edge["path"],
            )
