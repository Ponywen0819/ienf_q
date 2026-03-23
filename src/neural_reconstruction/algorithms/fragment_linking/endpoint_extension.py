"""
端點延伸模組 (Endpoint Extension Module)

實作階段1：高信心端點延伸算法。
對圖中所有端點（degree == 1）進行延伸，使用嚴格的角度和距離約束，
每個端點只選取最佳候選點。
"""

from typing import List, Tuple, Dict
import logging

import numpy as np
import networkx as nx
from scipy.spatial import KDTree

from .utils import compute_vector_angle, is_direction_too_similar

logger = logging.getLogger(__name__)


def extend_endpoints(
    graph: nx.Graph,
    topology_points: np.ndarray,
    kdtree: KDTree,
    path_lookup: Dict,
    search_radius: float,
    max_angle_degrees: float = 75.0,
    max_angle_penalty: float = 0.5,
    direction_threshold: float = 5.0,
) -> List[Tuple]:
    """
    對 graph 中所有端點（degree == 1）進行延伸

    Args:
        graph: NetworkX Graph
        topology_points: 所有拓樸點座標
        kdtree: KDTree
        path_lookup: 路徑字典
        search_radius: KDTree 搜尋半徑
        max_angle_degrees: 最大允許角度
        max_angle_penalty: 最大角度懲罰 (0.5 = 50%)
        direction_threshold: 方向相似度閾值（度）
        verbose: 詳細輸出

    Returns:
        new_edges: [(endpoint, target, cost, path), ...]
    """
    # 找出所有端點
    endpoints = {node for node in graph.nodes() if graph.degree(node) == 1}

    logger.info(f"  ✓ 找到 {len(endpoints)} 個端點")

    new_edges = []

    for endpoint in endpoints:
        # 1. 取得端點的唯一鄰居，計算延伸方向
        neighbors = list(graph.neighbors(endpoint))
        if len(neighbors) != 1:
            continue

        neighbor = neighbors[0]
        extend_vector = np.array(endpoint) - np.array(neighbor)

        # 2. KDTree 查詢候選節點
        endpoint_arr = np.array(endpoint)
        candidate_indices = kdtree.query_ball_point(endpoint_arr, r=search_radius)

        # 3. 計算每個候選的距離並排序（從近到遠）
        candidates_with_dist = []
        for idx in candidate_indices:
            candidate = tuple(topology_points[idx])

            if candidate == endpoint or candidate == neighbor:
                continue

            if graph.has_edge(endpoint, candidate):
                continue

            dist = np.linalg.norm(topology_points[idx] - endpoint_arr)
            candidates_with_dist.append((candidate, dist, idx))

        candidates_with_dist.sort(key=lambda x: x[1])

        # 4. 評估候選節點
        selected_directions = []
        best_candidate = None
        best_cost = float("inf")
        best_path = None

        for candidate, dist, idx in candidates_with_dist:
            key1 = (endpoint, candidate)
            key2 = (candidate, endpoint)

            if key1 in path_lookup:
                path, base_cost = path_lookup[key1]
            elif key2 in path_lookup:
                path, base_cost = path_lookup[key2]
            else:
                continue

            # 4.1 計算 AC 向量
            ac_vector = np.array(candidate) - endpoint_arr

            # 4.2 檢查方向是否與已通過篩選的候選方向太相近
            if is_direction_too_similar(
                ac_vector, selected_directions, direction_threshold
            ):
                continue

            # 4.3 通過篩選，加入方向清單
            selected_directions.append(ac_vector)

            # 4.4 計算夾角
            angle = compute_vector_angle(extend_vector, ac_vector)

            if angle > max_angle_degrees:
                continue

            # 角度懲罰
            penalty = max_angle_penalty * (angle / max_angle_degrees)
            final_cost = base_cost * (1 + penalty)

            # 更新最佳候選
            if final_cost < best_cost:
                best_cost = final_cost
                best_candidate = candidate
                best_path = path

        # 5. 記錄最佳延伸
        if best_candidate is not None:
            new_edges.append((endpoint, best_candidate, best_cost, best_path))

    logger.info(f"✓ 階段1完成: 找到 {len(new_edges)} 個可延伸的端點")

    return new_edges
