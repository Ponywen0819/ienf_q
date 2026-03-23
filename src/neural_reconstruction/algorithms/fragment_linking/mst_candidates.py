"""
MST 候選邊生成模組 (MST Candidates Module)

實作階段2：寬鬆約束下的 MST 候選邊生成。
為每個端點生成到鄰近節點的候選邊，使用較大的搜尋半徑和角度閾值，
產生多個候選供 MST 演算法選擇最優連接。
"""

from typing import List, Tuple, Dict

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import logging

from .utils import compute_vector_angle

logger = logging.getLogger(__name__)


def generate_mst_candidates(
    graph: nx.Graph,
    topology_points: np.ndarray,
    kdtree: KDTree,
    path_lookup: Dict,
    search_radius: float = 20,
    max_angle_degrees: float = 90,
    angle_penalty_weight: float = 0.5,
    distance_weight: float = 0.2,
    max_cost_threshold: float = 0.75,
    verbose: bool = False,
) -> List[Tuple]:
    """
    為每個端點生成到所有其他節點的候選邊

    Args:
        graph: 當前圖（包含階段1的邊）
        topology_points: 所有拓樸點座標
        kdtree: KDTree
        path_lookup: 路徑字典
        search_radius: 搜尋半徑
        max_angle_degrees: 最大允許角度
        angle_penalty_weight: 角度懲罰權重
        distance_weight: 距離懲罰權重
        max_cost_threshold: 成本閾值
        verbose: 詳細輸出

    Returns:
        [(endpoint, target_node, cost, path), ...]
    """
    nodes = list(graph.nodes())
    endpoints = [node for node in nodes if graph.degree(node) == 1]
    isolated = [node for node in nodes if graph.degree(node) == 0]

    logger.info(f"階段2：找到 {len(endpoints)} 個端點, {len(isolated)} 個孤立節點")

    candidate_edges = []

    # --- 端點延伸（有方向資訊，使用完整角度限制）---
    for endpoint in endpoints:
        neighbors = list(graph.neighbors(endpoint))
        if len(neighbors) != 1:
            continue

        neighbor = neighbors[0]
        extend_vector = np.array(endpoint) - np.array(neighbor)

        endpoint_arr = np.array(endpoint)
        candidate_indices = kdtree.query_ball_point(endpoint_arr, r=search_radius)

        for candidate_idx in candidate_indices:
            target_node = tuple(topology_points[candidate_idx])

            if target_node == endpoint or target_node == neighbor:
                continue

            if (endpoint, target_node) in path_lookup:
                path, base_cost = path_lookup[(endpoint, target_node)]
            elif (target_node, endpoint) in path_lookup:
                path, base_cost = path_lookup[(target_node, endpoint)]
            else:
                continue

            if graph.has_edge(endpoint, target_node):
                continue

            ac_vector = np.array(target_node) - endpoint_arr
            distance = np.linalg.norm(ac_vector)
            angle = compute_vector_angle(extend_vector, ac_vector)

            if angle > max_angle_degrees:
                continue

            # 計算路徑長度
            path_arr = np.array(path)
            diffs = np.diff(path_arr, axis=0)
            segment_dists = np.linalg.norm(diffs, axis=1)
            path_length = np.sum(segment_dists)

            # 計算最終成本
            distance_penalty = distance_weight * (distance / search_radius)
            angle_penalty = angle_penalty_weight * (angle / max_angle_degrees)
            final_cost = base_cost * (1 + angle_penalty) * (1 + distance_penalty)

            if final_cost <= max_cost_threshold * path_length:
                candidate_edges.append((endpoint, target_node, final_cost, path))

    # --- 孤立節點延伸（無方向資訊，搜尋半徑減半，不做角度限制）---
    isolated_search_radius = search_radius / 2.0

    for node in isolated:
        node_arr = np.array(node)
        candidate_indices = kdtree.query_ball_point(node_arr, r=isolated_search_radius)

        for candidate_idx in candidate_indices:
            target_node = tuple(topology_points[candidate_idx])

            if target_node == node:
                continue

            if graph.has_edge(node, target_node):
                continue

            if (node, target_node) in path_lookup:
                path, base_cost = path_lookup[(node, target_node)]
            elif (target_node, node) in path_lookup:
                path, base_cost = path_lookup[(target_node, node)]
            else:
                continue

            ac_vector = np.array(target_node) - node_arr
            distance = np.linalg.norm(ac_vector)

            # 計算路徑長度
            path_arr = np.array(path)
            diffs = np.diff(path_arr, axis=0)
            segment_dists = np.linalg.norm(diffs, axis=1)
            path_length = np.sum(segment_dists)

            # 無角度懲罰
            distance_penalty = distance_weight * (distance / isolated_search_radius)
            final_cost = base_cost * (1 + distance_penalty)
            if final_cost <= max_cost_threshold * path_length:
                candidate_edges.append((node, target_node, final_cost, path))

    logger.info(f"✓ 階段2完成: 生成 {len(candidate_edges)} 條候選邊")
    logger.info(
        f"  - 端點→端點: {sum(1 for _, target, _, _ in candidate_edges if graph.degree(target) == 1)}"
    )
    logger.info(
        f"  - 端點→孤立節點: {sum(1 for _, target, _, _ in candidate_edges if graph.degree(target) == 0)}"
    )
    logger.info(
        f"  - 端點→中間節點: {sum(1 for _, target, _, _ in candidate_edges if graph.degree(target) >= 2)}"
    )
    return candidate_edges
