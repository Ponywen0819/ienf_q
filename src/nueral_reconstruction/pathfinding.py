"""
A* 路徑搜尋模組 (A* Pathfinding Module)

基於影像成本地圖的 A* 路徑搜尋。
提供兩個核心功能：
1. 尋找兩點之間的最短路徑
2. 計算路徑成本
"""

import numpy as np
import heapq
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class AStarPathfinder:
    """
    A* 路徑搜尋器

    使用 A* 演算法在成本地圖上尋找最短路徑。
    成本地圖 = 255 - green_channel（綠色強度越高，成本越低）
    """

    def __init__(self, green_channel: np.ndarray):
        """
        初始化路徑搜尋器

        Args:
            green_channel: 綠色通道影像 (uint8, 0-255)
        """
        self.green_channel = green_channel
        self.cost_map = 255 - green_channel.astype(np.float32)
        self.height, self.width = self.cost_map.shape

        logger.info(f"初始化 A* 路徑搜尋器: {self.height} x {self.width}")

    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        使用 A* 演算法尋找最短路徑

        Args:
            start: 起點 (y, x)
            end: 終點 (y, x)

        Returns:
            路徑座標列表 [(y, x), ...]，如果找不到路徑則返回 None
        """
        # 邊界檢查
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            logger.warning(f"起點 {start} 或終點 {end} 超出影像範圍")
            return None

        # A* 資料結構
        open_set = []  # 優先佇列: (f_score, counter, position)
        counter = 0
        heapq.heappush(open_set, (0, counter, start))
        counter += 1

        came_from = {}  # 路徑重建用
        g_score = {start: 0}  # 從起點到目前位置的實際成本
        f_score = {start: self._heuristic(start, end)}  # g + h

        visited = set()

        # A* 主迴圈
        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            # 到達目標
            if current == end:
                path = self._reconstruct_path(came_from, current)
                logger.debug(f"找到路徑: 長度 {len(path)}, 成本 {g_score[current]:.2f}")
                return path

            # 跳過已訪問的節點
            if current in visited:
                continue
            visited.add(current)

            # 探索 8-鄰域鄰居
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue

                # 計算新的 g_score
                edge_cost = self._edge_cost(current, neighbor)
                tentative_g = g_score[current] + edge_cost

                # 更新或新增鄰居
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        # 找不到路徑
        logger.warning(f"無法找到從 {start} 到 {end} 的路徑")
        return None

    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """
        計算路徑總成本

        Args:
            path: 路徑座標列表

        Returns:
            總成本
        """
        if len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._edge_cost(path[i], path[i+1])
        
        return total_cost

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """啟發式函數：歐幾里得距離（可接受的下界估計）"""
        return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def _edge_cost(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> float:
        """
        計算邊成本
        成本 = 移動距離 × 目標位置成本地圖值
        """
        # 移動距離
        dy = abs(pos1[0] - pos2[0])
        dx = abs(pos1[1] - pos2[1])
        distance = 1.414 if dy == 1 and dx == 1 else 1.0

        # 目標位置成本
        pixel_cost = self.cost_map[pos2[0], pos2[1]]



        normoalized_pixel_cost = ((pixel_cost / 255.0)** 2)*(4)
        return (distance * normoalized_pixel_cost)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """獲取 8-鄰域鄰居"""
        y, x = pos
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if self._is_valid_position((ny, nx)):
                    neighbors.append((ny, nx))
        return neighbors

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """檢查位置是否在影像範圍內"""
        y, x = pos
        return 0 <= y < self.height and 0 <= x < self.width

    def _reconstruct_path(
        self,
        came_from: dict,
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """回溯重建路徑"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

