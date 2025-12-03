"""
骨架種子提取模組 (Skeleton Seed Extraction Module)

提供骨架拓樸分析與種子點提取功能：
1. SkeletonTopologyBuilder - 建構骨架的網路拓樸（節點與邊）
2. EdgeSeedExtractor - 從拓樸邊上均勻抽取種子點

主要用途：
- 分析骨架結構，識別端點、分支點及其連接關係
- 在骨架路徑上均勻放置種子點，用於後續神經重建
"""

import logging
from typing import Tuple, List, Dict, Set

import numpy as np

logger = logging.getLogger(__name__)


class SkeletonTopologyBuilder:
    """骨架拓樸建構器 - 建構端點與分支點之間的網路拓樸"""

    def __init__(self):
        pass

    def build_topology(
        self,
        skeleton: np.ndarray,
        endpoints: List[Tuple[int, int]],
        branchpoints: List[Tuple[int, int]]
    ) -> Dict:
        """
        建構骨架的網路拓樸

        Args:
            skeleton: 骨架 mask (二值影像)
            endpoints: 端點列表 [(y, x), ...]
            branchpoints: 分支點列表 [(y, x), ...]

        Returns:
            topology: 拓樸結構字典
                {
                    'nodes': [{'id': int, 'position': (y, x), 'type': str}, ...],
                    'edges': [{'source': int, 'target': int, 'path': [...], 'length': float}, ...]
                }
        """
        logger.info("建構骨架拓樸...")

        # 1. 建立節點列表（端點 + 分支點）
        nodes = []
        node_id = 0
        position_to_id = {}

        # 添加端點
        for ep in endpoints:
            ep_tuple = tuple(ep)
            nodes.append({
                'id': node_id,
                'position': ep_tuple,
                'type': 'endpoint'
            })
            position_to_id[ep_tuple] = node_id
            node_id += 1

        # 添加分支點
        for bp in branchpoints:
            bp_tuple = tuple(bp)
            nodes.append({
                'id': node_id,
                'position': bp_tuple,
                'type': 'branchpoint'
            })
            position_to_id[bp_tuple] = node_id
            node_id += 1

        logger.info(f"節點總數: {len(nodes)} (端點: {len(endpoints)}, 分支點: {len(branchpoints)})")

        # 2. 建立邊列表（從每個節點追蹤到相鄰節點）
        edges = []
        visited_edges = set()
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
                edges.append({
                    'source': start_id,
                    'target': end_id,
                    'path': full_path,
                    'length': path_length
                })

        logger.info(f"邊總數: {len(edges)}")
        
        topology = {
            'nodes': nodes,
            'edges': edges
        }

        return topology

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
        edge: Dict,
        segment_length: float = None
    ) -> List[Tuple[int, int]]:
        """
        從單條邊均勻抽取種子

        Args:
            edge: 邊資訊字典，包含 'path' 和 'length'
            segment_length: 分段長度閾值（像素），若為 None 則使用 min_edge_length

        Returns:
            seeds: 種子座標列表 [(y, x), ...]，不包含端點
        """
        if segment_length is None:
            segment_length = self.min_edge_length

        path = edge['path']
        length = edge['length']

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
        seeds = self._place_seeds_uniformly(path, num_seeds)

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
        topology: Dict,
        segment_length: float = None
    ) -> List[Dict]:
        """
        從整個拓樸結構的所有邊抽取種子

        Args:
            topology: 拓樸結構字典（包含 'nodes' 和 'edges'）
            segment_length: 分段長度閾值（像素）

        Returns:
            seeds: 種子列表，每個種子包含 {'position': (y, x), 'edge_id': int}
        """
        edges = topology.get('edges', [])
        all_seeds = []

        logger.info(f"從 {len(edges)} 條邊抽取種子...")

        for edge_id, edge in enumerate(edges):
            seed_positions = self.extract_seeds_from_edge(edge, segment_length)

            for pos in seed_positions:
                all_seeds.append({
                    'position': pos,
                    'edge_id': edge_id
                })

        logger.info(f"總計抽取 {len(all_seeds)} 個種子")

        return all_seeds

