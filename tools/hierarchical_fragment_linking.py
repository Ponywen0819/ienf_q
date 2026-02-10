#!/usr/bin/env python3
"""
階層式片段連接算法（含完整預處理）

這個算法實現了完整的神經纖維重建流程：
1. 預處理：ROI提取、背景減除、伪標註生成
2. 階段1：高信心端點延伸（嚴格角度限制、小搜索半徑）
3. 階段2：生成MST候選邊（寬鬆角度、大搜索半徑）+ MST優化

用法:
    python tools/hierarchical_fragment_linking.py \
        --image data/S1585-2_a/image.png \
        --mask data/S1585-2_a/mask.png \
        --annotation data/S1585-2_a/annotation.png \
        --output output/hierarchical_linking/S1585-2_a_result.pkl
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import networkx as nx
from scipy.spatial import KDTree
from PIL import Image
import skimage as ski
from skimage import morphology, segmentation
from skimage.measure import label, regionprops
from skan import Skeleton, summarize
from skan.csr import skeleton_to_nx

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.compare_topologies import TopologyLoader
from src.neural_reconstruction.core.preprocessing import SkinAnalysisPipeline
from src.neural_reconstruction.core.preprocessing.utils import ensure_grayscale


# =============================================================================
# 幾何計算輔助函數
# =============================================================================


def compute_vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    計算兩個向量之間的夾角（度）

    Args:
        v1: 第一個向量
        v2: 第二個向量

    Returns:
        夾角 [0, 180] 度
    """
    # 計算向量長度
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # 避免除以零
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    # 計算 cos(θ)
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)

    # 限制在 [-1, 1] 範圍內（處理浮點誤差）
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # 轉換為角度
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def is_direction_too_similar(
    new_direction: np.ndarray,
    existing_directions: List[np.ndarray],
    threshold_degrees: float,
) -> bool:
    """
    檢查新方向是否與已存在的任一方向太相近

    Args:
        new_direction: 新候選的方向向量
        existing_directions: 已通過篩選的方向向量列表
        threshold_degrees: 角度閾值（度）

    Returns:
        True 如果太相近（應該跳過），False 如果可以考慮
    """
    if not existing_directions:
        return False

    for existing in existing_directions:
        angle = compute_vector_angle(new_direction, existing)
        if angle < threshold_degrees:
            return True

    return False


# =============================================================================
# 拓撲構建
# =============================================================================


class TopologyBuilder:
    """從標註圖像構建初始拓撲"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def build_skeleton_graph(
        self, annotation: np.ndarray, equalized_img: np.ndarray
    ) -> nx.MultiGraph:
        """
        從標註構建骨架圖

        Args:
            annotation: 二值標註圖像
            equalized_img: 均衡化後的原始圖像（用於補齊缺失節點）

        Returns:
            NetworkX MultiGraph，節點為 (y, x) 座標
        """
        # 1. 二值化和骨架化
        binary = (annotation > 0).astype(np.uint8)
        skeleton = morphology.skeletonize(binary).astype(np.uint8)

        # 2. 使用 skan 構建圖
        skel_obj = Skeleton(skeleton, keep_images=False)
        summary = summarize(skel_obj)
        skeleton_graph = skeleton_to_nx(skel_obj, summary=summary)

        # 3. 過濾短邊
        filtered_graph = nx.MultiGraph()
        for u, v, data in skeleton_graph.edges(data=True):
            path = data["path"]
            # 保留長邊或端點連接的短邊
            if len(path) > 2 or (
                skeleton_graph.degree(u) != 1 and skeleton_graph.degree(v) != 1
            ):
                filtered_graph.add_edge(u, v, **data)

        # 4. 將節點 ID 轉回座標
        mapping = {
            i: tuple(skel_obj.coordinates[i].astype(int))
            for i in filtered_graph.nodes()
        }
        filtered_graph = nx.relabel_nodes(filtered_graph, mapping)

        # 5. 合併中間點（degree == 2）
        filtered_graph = self._merge_middle_points(filtered_graph)

        # 6. 補齊缺失的節點（孤立的連通分量）
        filtered_graph = self._fill_missing_nodes(filtered_graph, binary, equalized_img)

        if self.verbose:
            print(
                f"✓ 骨架圖構建完成: {filtered_graph.number_of_nodes()} 節點, "
                f"{filtered_graph.number_of_edges()} 邊"
            )

        return filtered_graph

    def _merge_middle_points(self, graph: nx.MultiGraph) -> nx.MultiGraph:
        """合併中間點（degree == 2）"""
        middle_points = [
            point for point in graph.nodes() if len(list(graph.neighbors(point))) == 2
        ]

        for mp in middle_points:
            neighbors = list(graph.neighbors(mp))
            # Skip if node no longer has exactly 2 neighbors (graph may have changed)
            if len(neighbors) != 2:
                continue
            u, v = neighbors

            # 獲取兩條邊的路徑
            path1 = graph[u][mp][0]["path"]
            path2 = graph[mp][v][0]["path"]

            u_y, u_x = u
            mp_y, mp_x = mp
            v_y, v_x = v

            # 拼接路徑
            result_path = []
            if tuple(path1[-1]) == (mp_y, mp_x) and tuple(path1[0]) == (u_y, u_x):
                result_path.extend(path1)
            else:
                result_path.extend(path1[::-1])

            if tuple(path2[0]) == (mp_y, mp_x) and tuple(path2[-1]) == (v_y, v_x):
                result_path.extend(path2[1:])
            else:
                result_path.extend(path2[-2::-1])

            # 移除中間點，添加新邊
            graph.remove_node(mp)
            graph.add_edge(u, v, path=result_path)

        return graph

    def _fill_missing_nodes(
        self, graph: nx.MultiGraph, binary: np.ndarray, equalized_img: np.ndarray
    ) -> nx.MultiGraph:
        """補齊缺失的節點（孤立的連通分量）"""
        label_img = label(binary, connectivity=2)
        regions = regionprops(label_img)

        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox

            # 檢查這個 bbox 中是否已有節點
            bbox_nodes = [
                node
                for node in graph.nodes()
                if node[0] >= min_row
                and node[0] < max_row
                and node[1] >= min_col
                and node[1] < max_col
            ]

            if len(bbox_nodes) != 0:
                continue

            # 找到這個區域中最亮的像素
            brightest_pixel = None
            brightest_value = -1

            for r in range(min_row, max_row):
                for c in range(min_col, max_col):
                    if equalized_img[r, c] > brightest_value:
                        brightest_value = equalized_img[r, c]
                        brightest_pixel = (r, c)

            # 添加節點
            if brightest_pixel is not None:
                graph.add_node(brightest_pixel)

        return graph


# =============================================================================
# 種子圖生成
# =============================================================================


class SeedGraphBuilder:
    """將骨架圖轉換為種子圖"""

    def __init__(self, segment_length: float = 3.0, verbose: bool = False):
        self.segment_length = segment_length
        self.verbose = verbose

    def build(self, skeleton_graph: nx.MultiGraph) -> nx.MultiGraph:
        """
        構建種子圖

        Args:
            skeleton_graph: 骨架圖

        Returns:
            種子圖，沿邊切分為更小的片段
        """
        seed_graph = nx.MultiGraph()

        # 添加所有節點
        for u in skeleton_graph.nodes():
            seed_graph.add_node(u)

        # 處理每條邊
        for u, v, data in skeleton_graph.edges(data=True):
            path = data["path"]

            # 擺正方向
            corrected_path = (
                path[:] if tuple(path[0]) == u and tuple(path[-1]) == v else path[::-1]
            )

            # 計算路徑長度
            path_arr = np.array(corrected_path)
            diffs = np.diff(path_arr, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
            path_length = cumulative_distances[-1]

            # 計算分段數
            num_segments = int(path_length // self.segment_length)

            if num_segments <= 0:
                seed_graph.add_edge(u, v, path=path)
                continue

            # 按距離切分
            last_index = 0
            for i in range(num_segments):
                target_distance = (i + 1) * path_length / num_segments
                segment_end_index = 0

                for idx, cumulative_distance in enumerate(
                    cumulative_distances[last_index:]
                ):
                    if cumulative_distance >= target_distance:
                        segment_end_index = idx + last_index
                        break

                segment_path = corrected_path[last_index : segment_end_index + 1]
                if len(segment_path) == 0:
                    continue

                seed_graph.add_edge(
                    tuple(segment_path[0]), tuple(segment_path[-1]), path=segment_path
                )
                last_index = segment_end_index

            # 添加最後一段
            if last_index < len(corrected_path) - 1:
                final_segment_path = corrected_path[last_index:]
                seed_graph.add_edge(
                    tuple(final_segment_path[0]),
                    tuple(final_segment_path[-1]),
                    path=final_segment_path,
                )

        if self.verbose:
            print(
                f"✓ 種子圖構建完成: {seed_graph.number_of_nodes()} 節點, "
                f"{seed_graph.number_of_edges()} 邊"
            )

        return seed_graph


# =============================================================================
# 路徑查找
# =============================================================================


class PathFinder:
    """使用 MCP 查找路徑"""

    def __init__(
        self,
        cost_map: np.ndarray,
        seed_map: np.ndarray,
        label_img: np.ndarray,
        bbox_padding: int = 10,
        verbose: bool = False,
    ):
        self.cost_map = cost_map
        self.seed_map = seed_map
        self.label_img = label_img
        self.bbox_padding = bbox_padding
        self.verbose = verbose
        self.cost_map_h, self.cost_map_w = cost_map.shape

    def find_paths_from_seeds(
        self, topology_points: np.ndarray, kdtree: KDTree, search_radius: float
    ) -> Dict[Tuple, Tuple[List, float]]:
        """
        從所有種子點查找路徑

        Args:
            topology_points: 所有拓撲點座標
            kdtree: KDTree
            search_radius: 搜索半徑

        Returns:
            path_lookup: {(start, end): (path, cost)}
        """
        path_lookup = {}

        for u_idx in range(len(topology_points)):
            u = topology_points[u_idx]

            # 查詢鄰居
            neighbor_indices = kdtree.query_ball_point(u, r=search_radius)
            targets = [topology_points[v_idx] for v_idx in neighbor_indices]

            # 過濾
            targets = [t for t in targets if tuple(t) != tuple(u)]

            current_component_id = self.label_img[u[0], u[1]]
            targets = [
                t for t in targets if self.label_img[t[0], t[1]] != current_component_id
            ]

            targets = [
                t
                for t in targets
                if (tuple(u), tuple(t)) not in path_lookup
                and (tuple(t), tuple(u)) not in path_lookup
            ]

            if not targets:
                continue

            # 計算包含所有點的最小 bbox
            all_points = [u] + targets
            all_y = [p[0] for p in all_points]
            all_x = [p[1] for p in all_points]

            min_y = max(0, min(all_y) - self.bbox_padding)
            max_y = min(self.cost_map_h - 1, max(all_y) + self.bbox_padding)
            min_x = max(0, min(all_x) - self.bbox_padding)
            max_x = min(self.cost_map_w - 1, max(all_x) + self.bbox_padding)

            # 裁剪 cost map
            cropped_cost_map = self.cost_map[min_y : max_y + 1, min_x : max_x + 1]

            # 轉換為局部座標
            local_points = [
                (pos_global[0] - min_y, pos_global[1] - min_x)
                for pos_global in all_points
            ]

            # 使用 MCP 查找路徑
            mcp = ski.graph.MCP_Geometric(cropped_cost_map, fully_connected=True)
            cumulative_costs, traceback = mcp.find_costs(
                starts=local_points[:1], ends=local_points[1:]
            )

            # 提取路徑
            for target_local, target_global in zip(local_points[1:], targets):
                if np.isinf(cumulative_costs[target_local]):
                    continue

                path = mcp.traceback(target_local)
                cost = cumulative_costs[target_local]

                # 轉換回全局座標
                global_path = [(p[0] + min_y, p[1] + min_x) for p in path]
                global_start = (u[0], u[1])
                global_target = (target_global[0], target_global[1])

                # 檢查中間點是否經過其他種子點
                middle_points = np.array(path[1:-1])
                if len(middle_points) > 0:
                    if np.any(self.seed_map[middle_points[:, 0], middle_points[:, 1]]):
                        continue

                # 存儲路徑
                if (global_target, global_start) in path_lookup:
                    min_cost = min(path_lookup[(global_target, global_start)][1], cost)
                    path_lookup[(global_target, global_start)] = (global_path, min_cost)
                else:
                    path_lookup[(global_start, global_target)] = (global_path, cost)

        if self.verbose:
            print(f"✓ 路徑查找完成: {len(path_lookup)} 條路徑")

        return path_lookup


# =============================================================================
# 階段1：端點延伸
# =============================================================================


def extend_endpoints(
    graph: nx.Graph,
    topology_points: np.ndarray,
    kdtree: KDTree,
    path_lookup: Dict,
    search_radius: float,
    max_angle_degrees: float = 75.0,
    max_angle_penalty: float = 0.5,
    direction_threshold: float = 5.0,
    verbose: bool = False,
) -> List[Tuple]:
    """
    對 graph 中所有端點（degree == 1）進行延伸

    Args:
        graph: NetworkX Graph
        topology_points: 所有拓撲點座標
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

    if verbose:
        print(f"階段1：找到 {len(endpoints)} 個端點")

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

            # 4.5 從 path_lookup 取得成本
            key1 = (endpoint, candidate)
            key2 = (candidate, endpoint)

            if key1 in path_lookup:
                path, base_cost = path_lookup[key1]
            elif key2 in path_lookup:
                path, base_cost = path_lookup[key2]
            else:
                continue

            # 計算路徑長度
            path_arr = np.array(path)
            diffs = np.diff(path_arr, axis=0)
            segment_dists = np.linalg.norm(diffs, axis=1)
            path_length = np.sum(segment_dists)

            # 歸一化成本
            normalized_cost = 0.8 * base_cost / path_length + 0.2 * (
                dist / search_radius
            )

            # 角度懲罰
            penalty = max_angle_penalty * (angle / max_angle_degrees)
            final_cost = normalized_cost * (1 + penalty) / (1 + max_angle_penalty)

            # 更新最佳候選
            if final_cost < best_cost:
                best_cost = final_cost
                best_candidate = candidate
                best_path = path

        # 5. 記錄最佳延伸
        if best_candidate is not None:
            new_edges.append((endpoint, best_candidate, best_cost, best_path))

    if verbose:
        print(f"✓ 階段1完成: 找到 {len(new_edges)} 個可延伸的端點")

    return new_edges


# =============================================================================
# 階段2：生成 MST 候選邊
# =============================================================================


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
        topology_points: 所有拓撲點座標
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
    endpoints = [node for node in graph.nodes() if graph.degree(node) == 1]

    if verbose:
        print(f"階段2：找到 {len(endpoints)} 個端點")

    candidate_edges = []

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

            if graph.has_edge(endpoint, target_node):
                continue

            ac_vector = np.array(target_node) - endpoint_arr
            distance = np.linalg.norm(ac_vector)
            angle = compute_vector_angle(extend_vector, ac_vector)

            if angle > max_angle_degrees:
                continue

            # 查找路徑
            path = None
            base_cost = None

            if (endpoint, target_node) in path_lookup:
                path, base_cost = path_lookup[(endpoint, target_node)]
            elif (target_node, endpoint) in path_lookup:
                path, base_cost = path_lookup[(target_node, endpoint)]
            else:
                continue

            # 計算路徑長度
            path_arr = np.array(path)
            diffs = np.diff(path_arr, axis=0)
            segment_dists = np.linalg.norm(diffs, axis=1)
            path_length = np.sum(segment_dists)

            # 計算最終成本
            distance_penalty = distance / search_radius
            normalized_cost = (
                1 - distance_weight
            ) * base_cost / path_length + distance_weight * distance_penalty
            angle_penalty = angle_penalty_weight * (angle / 180.0)

            final_cost = (
                normalized_cost * (1 + angle_penalty) / (1 + angle_penalty_weight)
            )

            if final_cost <= max_cost_threshold:
                candidate_edges.append((endpoint, target_node, final_cost, path))

    if verbose:
        target_endpoints = sum(
            1 for _, target, _, _ in candidate_edges if graph.degree(target) == 1
        )
        target_middle = sum(
            1 for _, target, _, _ in candidate_edges if graph.degree(target) >= 2
        )
        print(f"✓ 階段2完成: 生成 {len(candidate_edges)} 條候選邊")
        print(f"  - 端點→端點: {target_endpoints}")
        print(f"  - 端點→中間節點: {target_middle}")

    return candidate_edges


# =============================================================================
# 主要流程
# =============================================================================


class HierarchicalFragmentLinker:
    """階層式片段連接器（含預處理）"""

    def __init__(
        self,
        # 預處理參數
        offset_px: int = 100,
        rolling_ball_radius: int = 2,
        sato_weight: float = 0.0,
        opening_kernel_size: int = 3,
        # 種子圖參數
        segment_length: float = 3.0,
        # 路徑查找參數
        search_radius_pathfinding: float = 50.0,
        # 階段1參數
        search_radius_phase1: float = 10.0,
        max_angle_phase1: float = 75.0,
        angle_penalty_phase1: float = 0.5,
        direction_threshold_phase1: float = 5.0,
        # 階段2參數
        search_radius_phase2: float = 20.0,
        max_angle_phase2: float = 90.0,
        angle_penalty_phase2: float = 0.5,
        distance_weight_phase2: float = 0.2,
        max_cost_threshold_phase2: float = 0.75,
        # MST參數
        phase1_weight_discount: float = 0.5,
        verbose: bool = False,
    ):
        # 預處理參數
        self.offset_px = offset_px
        self.rolling_ball_radius = rolling_ball_radius
        self.sato_weight = sato_weight
        self.opening_kernel_size = opening_kernel_size

        # 重建參數
        self.segment_length = segment_length
        self.search_radius_pathfinding = search_radius_pathfinding

        self.search_radius_phase1 = search_radius_phase1
        self.max_angle_phase1 = max_angle_phase1
        self.angle_penalty_phase1 = angle_penalty_phase1
        self.direction_threshold_phase1 = direction_threshold_phase1

        self.search_radius_phase2 = search_radius_phase2
        self.max_angle_phase2 = max_angle_phase2
        self.angle_penalty_phase2 = angle_penalty_phase2
        self.distance_weight_phase2 = distance_weight_phase2
        self.max_cost_threshold_phase2 = max_cost_threshold_phase2

        self.phase1_weight_discount = phase1_weight_discount

        self.verbose = verbose

    def run(
        self, image: np.ndarray, mask: np.ndarray, annotation: np.ndarray
    ) -> nx.Graph:
        """
        運行完整的階層式片段連接算法（含預處理）

        Args:
            image: 原始圖像 (H, W) 或 (H, W, 3)
            mask: 表皮遮罩 (H, W)
            annotation: 手工標註 (H, W)

        Returns:
            MST 圖
        """
        if self.verbose:
            print("=" * 60)
            print("開始階層式片段連接（含預處理）")
            print("=" * 60)

        # 1. 預處理
        if self.verbose:
            print("\n1. 圖像預處理...")

        # 使用 SkinAnalysisPipeline 進行預處理
        preprocessing_config = {
            'morphology': {
                'closing_kernel': 0,  # 不使用 closing
                'opening_kernel': self.opening_kernel_size,
            },
            'mask': {
                'dilate_offset': self.offset_px,
            },
            'background': {
                'method': 'rolling_ball',
                'radius': self.rolling_ball_radius,
                'sato_weight': self.sato_weight,
                'sato_sigmas': (1.0, 2.0),
            },
            'threshold': {
                'use_full_roi': False,
            },
            'normalization': {
                'enabled': False,
            }
        }

        pipeline = SkinAnalysisPipeline(preprocessing_config)

        # 提取綠色通道
        if len(image.shape) == 3:
            orig_img = image[:, :, 1]  # 綠色通道
        else:
            orig_img = image

        if self.verbose:
            print("  - 提取綠色通道")
            print("  - 構建 ROI mask")
            print("  - 背景減除")
            print("  - 提取 ROI")
            print("  - 生成偽標註")
            print("  - 形態學處理")

        roi_annotation, roi_image = pipeline.run(annotation, mask, orig_img)

        if self.verbose:
            print("  ✓ 預處理完成")

        # CLAHE 均衡化用於種子點提取
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_img = clahe.apply(roi_image)

        # 2. 構建骨架圖
        if self.verbose:
            print("\n2. 構建骨架圖...")

        topology_builder = TopologyBuilder(verbose=self.verbose)
        skeleton_graph = topology_builder.build_skeleton_graph(
            roi_annotation, equalized_img
        )

        # 3. 構建種子圖
        if self.verbose:
            print("\n3. 構建種子圖...")

        seed_builder = SeedGraphBuilder(
            segment_length=self.segment_length, verbose=self.verbose
        )
        seed_graph = seed_builder.build(skeleton_graph)

        # 為種子圖的邊設置低成本
        for u, v, data in seed_graph.edges(data=True):
            data["weight"] = 1e-5

        # 4. 準備路徑查找
        if self.verbose:
            print("\n4. 準備路徑查找...")

        topology_points = np.array(list(seed_graph.nodes()))
        kdtree = KDTree(topology_points)

        # 構建成本地圖和種子地圖
        cost_map = ((255 - roi_image.astype(np.float64)) / 255.0) ** 1.5
        seed_map = np.zeros_like(cost_map, dtype=np.uint8)
        for p in topology_points:
            seed_map[p[0], p[1]] = True

        # 構建 label 圖（用於排除同一連通分量）
        binary = (roi_annotation > 0).astype(np.uint8)
        label_img = label(binary, connectivity=2)

        # 5. 路徑查找
        if self.verbose:
            print("\n5. 路徑查找...")

        path_finder = PathFinder(
            cost_map=cost_map,
            seed_map=seed_map,
            label_img=label_img,
            verbose=self.verbose,
        )

        path_lookup = path_finder.find_paths_from_seeds(
            topology_points=topology_points,
            kdtree=kdtree,
            search_radius=self.search_radius_pathfinding,
        )

        # 6. 階段1：端點延伸
        if self.verbose:
            print("\n6. 階段1：端點延伸...")

        phase1_edges = extend_endpoints(
            graph=seed_graph,
            topology_points=topology_points,
            kdtree=kdtree,
            path_lookup=path_lookup,
            search_radius=self.search_radius_phase1,
            max_angle_degrees=self.max_angle_phase1,
            max_angle_penalty=self.angle_penalty_phase1,
            direction_threshold=self.direction_threshold_phase1,
            verbose=self.verbose,
        )

        # 添加階段1的邊
        extended_graph = seed_graph.copy()
        for endpoint, target, cost, path in phase1_edges:
            if not extended_graph.has_edge(endpoint, target):
                extended_graph.add_edge(
                    endpoint, target, weight=cost, path=path, phase=1
                )

        if self.verbose:
            endpoints_after_phase1 = sum(
                1 for n in extended_graph.nodes() if extended_graph.degree(n) == 1
            )
            print(f"  延伸後端點數: {endpoints_after_phase1}")

        # 7. 階段2：生成 MST 候選邊
        if self.verbose:
            print("\n7. 階段2：生成 MST 候選邊...")

        phase2_candidates = generate_mst_candidates(
            graph=extended_graph,
            topology_points=topology_points,
            kdtree=kdtree,
            path_lookup=path_lookup,
            search_radius=self.search_radius_phase2,
            max_angle_degrees=self.max_angle_phase2,
            angle_penalty_weight=self.angle_penalty_phase2,
            distance_weight=self.distance_weight_phase2,
            max_cost_threshold=self.max_cost_threshold_phase2,
            verbose=self.verbose,
        )

        # 8. 構建候選圖並執行 MST
        if self.verbose:
            print("\n8. 執行 MST 優化...")

        # 複製圖並對階段1的邊打折
        mst_tree = extended_graph.copy()
        for u, v, data in mst_tree.edges(data=True):
            data["weight"] = data["weight"] * self.phase1_weight_discount
            data["phase"] = 1

        # 添加階段2候選邊
        for endpoint, target, cost, path in phase2_candidates:
            if not mst_tree.has_edge(endpoint, target):
                mst_tree.add_edge(endpoint, target, weight=cost, path=path, phase=2)

        # 執行 MST
        mst_result = nx.minimum_spanning_tree(mst_tree, weight="weight")

        if self.verbose:
            phase1_in_mst = sum(
                1 for u, v in mst_result.edges() if mst_result[u][v].get("phase") == 1
            )
            phase2_in_mst = sum(
                1 for u, v in mst_result.edges() if mst_result[u][v].get("phase") == 2
            )
            final_endpoints = sum(
                1 for n in mst_result.nodes() if mst_result.degree(n) == 1
            )

            print(f"\n✓ MST 結果:")
            print(f"  - 總邊數: {mst_result.number_of_edges()}")
            print(f"  - 來自階段1: {phase1_in_mst}")
            print(f"  - 來自階段2: {phase2_in_mst}")
            print(f"  - 最終端點數: {final_endpoints}")
            print(f"  - 連通分量數: {nx.number_connected_components(mst_result)}")

        return mst_result


# =============================================================================
# 命令行接口
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="階層式片段連接算法（含完整預處理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python tools/hierarchical_fragment_linking.py \\
      --image data/S1585-2_a/image.png \\
      --mask data/S1585-2_a/mask.png \\
      --annotation data/S1585-2_a/annotation.png \\
      --output output/hierarchical_linking/S1585-2_a_result.pkl

  # 詳細模式
  python tools/hierarchical_fragment_linking.py \\
      --image data/S1585-2_a/image.png \\
      --mask data/S1585-2_a/mask.png \\
      --annotation data/S1585-2_a/annotation.png \\
      --output output/hierarchical_linking/S1585-2_a_result.pkl \\
      --verbose

  # 自定義參數
  python tools/hierarchical_fragment_linking.py \\
      --image data/S1585-2_a/image.png \\
      --mask data/S1585-2_a/mask.png \\
      --annotation data/S1585-2_a/annotation.png \\
      --output output/hierarchical_linking/S1585-2_a_result.pkl \\
      --offset-px 150 \\
      --segment-length 5.0 \\
      --search-radius-phase1 15.0 \\
      --search-radius-phase2 25.0 \\
      --verbose
        """,
    )

    # 必需參數
    parser.add_argument(
        "--image", type=Path, required=True, help="輸入原始圖像路徑 (RGB或灰度)"
    )

    parser.add_argument("--mask", type=Path, required=True, help="輸入表皮遮罩圖像路徑")

    parser.add_argument(
        "--annotation", type=Path, required=True, help="輸入手工標註圖像路徑"
    )

    parser.add_argument(
        "--output", type=Path, required=True, help="輸出拓撲文件路徑 (.pkl)"
    )

    # 預處理參數
    parser.add_argument(
        "--offset-px",
        type=int,
        default=100,
        help="ROI垂直膨脹偏移量 (default: 100)",
    )

    parser.add_argument(
        "--rolling-ball-radius",
        type=int,
        default=2,
        help="背景減除 rolling ball 半徑 (default: 2)",
    )

    parser.add_argument(
        "--sato-weight",
        type=float,
        default=0.0,
        help="Sato濾波器權重 (default: 0.0, 不使用)",
    )

    parser.add_argument(
        "--opening-kernel-size",
        type=int,
        default=3,
        help="形態學 opening 核大小 (default: 3)",
    )

    # 重建參數
    parser.add_argument(
        "--segment-length",
        type=float,
        default=3.0,
        help="種子圖分段長度 (default: 3.0)",
    )

    parser.add_argument(
        "--search-radius-pathfinding",
        type=float,
        default=50.0,
        help="路徑查找搜索半徑 (default: 50.0)",
    )

    parser.add_argument(
        "--search-radius-phase1",
        type=float,
        default=10.0,
        help="階段1搜索半徑 (default: 10.0)",
    )

    parser.add_argument(
        "--max-angle-phase1",
        type=float,
        default=75.0,
        help="階段1最大角度 (default: 75.0)",
    )

    parser.add_argument(
        "--search-radius-phase2",
        type=float,
        default=20.0,
        help="階段2搜索半徑 (default: 20.0)",
    )

    parser.add_argument(
        "--max-angle-phase2",
        type=float,
        default=90.0,
        help="階段2最大角度 (default: 90.0)",
    )

    parser.add_argument(
        "--max-cost-threshold-phase2",
        type=float,
        default=0.75,
        help="階段2成本閾值 (default: 0.75)",
    )

    parser.add_argument(
        "--phase1-weight-discount",
        type=float,
        default=0.5,
        help="階段1邊權重折扣 (default: 0.5)",
    )

    parser.add_argument("--verbose", action="store_true", help="詳細輸出")

    args = parser.parse_args()

    # 檢查輸入文件
    if not args.image.exists():
        print(f"錯誤: 圖像文件不存在: {args.image}")
        return 1

    if not args.mask.exists():
        print(f"錯誤: 遮罩文件不存在: {args.mask}")
        return 1

    if not args.annotation.exists():
        print(f"錯誤: 標註文件不存在: {args.annotation}")
        return 1

    # 創建輸出目錄
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 載入圖像
    if args.verbose:
        print(f"載入圖像: {args.image}")
        print(f"載入遮罩: {args.mask}")
        print(f"載入標註: {args.annotation}")

    # 載入原始圖像 (支持 RGB 和灰度)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        # 嘗試灰度模式
        image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    annotation = cv2.imread(str(args.annotation), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"錯誤: 無法載入圖像: {args.image}")
        return 1

    if mask is None:
        print(f"錯誤: 無法載入遮罩: {args.mask}")
        return 1

    if annotation is None:
        print(f"錯誤: 無法載入標註: {args.annotation}")
        return 1

    # 創建連接器
    linker = HierarchicalFragmentLinker(
        # 預處理參數
        offset_px=args.offset_px,
        rolling_ball_radius=args.rolling_ball_radius,
        sato_weight=args.sato_weight,
        opening_kernel_size=args.opening_kernel_size,
        # 重建參數
        segment_length=args.segment_length,
        search_radius_pathfinding=args.search_radius_pathfinding,
        search_radius_phase1=args.search_radius_phase1,
        max_angle_phase1=args.max_angle_phase1,
        search_radius_phase2=args.search_radius_phase2,
        max_angle_phase2=args.max_angle_phase2,
        max_cost_threshold_phase2=args.max_cost_threshold_phase2,
        phase1_weight_discount=args.phase1_weight_discount,
        verbose=args.verbose,
    )

    # 運行算法
    mst_result = linker.run(image, mask, annotation)

    # 保存結果
    if args.verbose:
        print(f"\n保存結果到: {args.output}")

    loader = TopologyLoader()
    loader.save(mst_result, args.output, format="pickle")

    print(f"\n✓ 完成！結果已保存到: {args.output}")
    print(f"  - 節點數: {mst_result.number_of_nodes()}")
    print(f"  - 邊數: {mst_result.number_of_edges()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
