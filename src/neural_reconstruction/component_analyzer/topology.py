"""
骨架拓樸模組 (Skeleton Topology Module)

提供骨架關鍵點偵測與拓樸建構功能：
1. KeypointDetector - 偵測端點與分支點
2. TopologyBuilder - 建構骨架網路拓樸
"""

import logging
from typing import Tuple, List, Set

import cv2
import numpy as np

from ..data_types import TopologyNode, TopologyEdge, TopologyResult

logger = logging.getLogger(__name__)


class KeyPointDetector:
    """骨架關鍵點偵測器 - 偵測端點與分支點"""

    def detect_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """
        檢測骨架端點（鄰居數 = 1）

        Args:
            skeleton: 骨架影像 (0 或 255)

        Returns:
            端點座標陣列 [[y1, x1], [y2, x2], ...]
        """
        neighbor_count = self._get_connected_neighbor_map(skeleton)
        binary = (skeleton > 0).astype(np.uint8)

        # 端點: 骨架點 且 鄰居數 = 1
        endpoints_mask = (binary > 0) & (neighbor_count == 1)
        endpoints = np.argwhere(endpoints_mask)

        return endpoints
    
    def detect_endpoints(self, binary: np.ndarray, neighbor_count: np.ndarray) -> np.ndarray:
        # 端點: 骨架點 且 鄰居數 = 1
        endpoints_mask = (binary > 0) & (neighbor_count == 1)
        endpoints = np.argwhere(endpoints_mask)

        return endpoints

    def detect_branch_points(self, binary: np.ndarray, neighbor_count: np.ndarray) -> np.ndarray:
        """
        檢測骨架分支點（鄰居數 >= 3）

        Args:
            skeleton: 骨架影像 (0 或 255)

        Returns:
            分支點座標陣列 [[y1, x1], [y2, x2], ...]
        """
        # 分支點: 骨架點 且 鄰居數 >= 3
        branch_points_mask = (binary > 0) & (neighbor_count >= 3)
        branch_points = np.argwhere(branch_points_mask)

        return branch_points
    
    def _get_connected_neighbor_map(self, skeleton: np.ndarray) -> np.ndarray:
        # 8-鄰域卷積核
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)

        # 計算每個骨架點的鄰居數
        neighbor_count = cv2.filter2D(skeleton, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        return neighbor_count

    def detect_all(self, skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        同時偵測端點與分支點

        Args:
            skeleton: 骨架影像 (0 或 255)

        Returns:
            (endpoints, branchpoints) 座標陣列
            
        """
        binary = (skeleton > 0).astype(np.uint8)
        neighbor_count = self._get_connected_neighbor_map(binary)

        return self.detect_endpoints(binary, neighbor_count), self.detect_branch_points(binary, neighbor_count)


class TopologyBuilder:
    """骨架拓樸建構器 - 建構端點與分支點之間的網路拓樸"""

    def build_topology(
        self,
        skeleton: np.ndarray,
        endpoints: np.ndarray,
        branchpoints: np.ndarray
    ) -> TopologyResult:
        """
        建構骨架的網路拓樸

        Args:
            skeleton: 骨架 mask (二值影像)
            endpoints: 端點座標陣列 [[y, x], ...]
            branchpoints: 分支點座標陣列 [[y, x], ...]

        Returns:
            TopologyResult: 拓樸結構（局部座標）
        """
        logger.debug("建構骨架拓樸...")

        # 1. 建立節點列表（端點 + 分支點）
        nodes: List[TopologyNode] = []
        position_to_id = {}
        node_id = 0

        # 統一處理端點與分支點的節點建立
        for points, node_type in [(endpoints, 'endpoint'), (branchpoints, 'branchpoint')]:
            for p in points:
                p_tuple = (int(p[0]), int(p[1]))
                nodes.append(TopologyNode(
                    node_id=node_id,
                    position=p_tuple,
                    node_type=node_type
                ))
                position_to_id[p_tuple] = node_id
                node_id += 1

        logger.debug(f"節點總數: {len(nodes)} (端點: {len(endpoints)}, 分支點: {len(branchpoints)})")

        # 2. 建立邊列表（從每個節點追蹤到相鄰節點）
        edges: List[TopologyEdge] = []
        visited_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        keypoints_set = set(position_to_id.keys())

        for start_pos in keypoints_set:
            start_id = position_to_id[start_pos]

            # 獲取起點的骨架鄰居
            neighbors = self._get_skeleton_neighbors(skeleton, start_pos)

            for neighbor in neighbors:
                # 避免重複邊
                edge_key = self._make_edge_key(start_pos, neighbor)
                if edge_key in visited_edges:
                    continue

                # 從鄰居追蹤到下一個關鍵點
                path_coords = self._trace_path_to_keypoint(
                    skeleton, neighbor, start_pos, keypoints_set
                )

                if not path_coords:
                    continue

                # 完整路徑 = [start] + path
                full_path = [start_pos] + path_coords
                end_pos = full_path[-1]

                # 檢查終點是否是關鍵點
                if end_pos not in position_to_id:
                    logger.warning(f"路徑終點 {end_pos} 不是關鍵點，跳過")
                    continue

                end_id = position_to_id[end_pos]

                # 計算路徑長度
                path_length = self._calculate_path_length(full_path)

                # 標記所有經過的邊為已訪問
                for i in range(len(full_path) - 1):
                    edge_key = self._make_edge_key(full_path[i], full_path[i+1])
                    visited_edges.add(edge_key)

                # 添加邊
                edges.append(TopologyEdge(
                    source_id=start_id,
                    target_id=end_id,
                    path=full_path,
                    length=path_length
                ))

        logger.debug(f"邊總數: {len(edges)}")

        return TopologyResult(nodes=nodes, edges=edges)

    def _get_skeleton_neighbors(
        self,
        skeleton: np.ndarray,
        point: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """獲取骨架點的 8-鄰域骨架鄰居"""
        y, x = point
        neighbors = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = y + dy, x + dx

                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx] > 0:
                        neighbors.append((ny, nx))

        return neighbors

    def _make_edge_key(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """創建唯一的邊鍵（排序後的點對）"""
        return tuple(sorted([point1, point2]))

    def _trace_path_to_keypoint(
        self,
        skeleton: np.ndarray,
        start: Tuple[int, int],
        came_from: Tuple[int, int],
        keypoints: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        從 start 追蹤路徑直到遇到關鍵點

        Args:
            skeleton: 骨架 mask
            start: 起始點
            came_from: 來源點（避免回頭）
            keypoints: 關鍵點集合

        Returns:
            path: 路徑座標列表（包含終點關鍵點）
        """
        path = []
        current = start
        visited = {came_from}

        while True:
            path.append(current)
            visited.add(current)

            # 如果當前點是關鍵點（且不是起點），停止
            if current in keypoints and current != start:
                break

            # 獲取未訪問的鄰居
            neighbors = self._get_skeleton_neighbors(skeleton, current)
            unvisited = [n for n in neighbors if n not in visited]

            # 沒有未訪問的鄰居，停止
            if not unvisited:
                break

            # 繼續追蹤（對於非分支點，應該只有一個未訪問鄰居）
            current = unvisited[0]

        return path

    def _calculate_path_length(self, path_coords: List[Tuple[int, int]]) -> float:
        """計算路徑長度（考慮對角線距離）"""
        if len(path_coords) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path_coords)):
            prev = np.array(path_coords[i-1])
            curr = np.array(path_coords[i])
            total_length += np.linalg.norm(curr - prev)

        return total_length
