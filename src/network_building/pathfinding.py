"""
A* 路徑搜索
在影像成本地圖上尋找最短路徑
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Any


class ImagePathfinder:
    """
    基於 A* 演算法的影像路徑搜索器

    在成本地圖上尋找從起點到終點的最低成本路徑
    成本地圖 = 255 - green_channel (綠色越強,成本越低)
    """

    def __init__(self, green_channel: np.ndarray, verbose: bool = False):
        """
        初始化路徑搜索器

        Args:
            green_channel: 綠色通道影像 (uint8, 0-255)
            verbose: 是否輸出詳細資訊
        """
        self.green_channel = green_channel
        self.cost_map = 255 - green_channel.astype(np.float32)
        self.height, self.width = self.cost_map.shape
        self.verbose = verbose

        if verbose:
            print(f"✓ 初始化 A* 路徑搜索器")
            print(f"  影像尺寸: {self.height} x {self.width}")
            print(f"  成本範圍: [{self.cost_map.min():.1f}, {self.cost_map.max():.1f}]")

    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        max_g_cost: float = None,
        max_distance_from_start: float = 30.0 # 使用您提到的 30px 作為可配置的預設值
    ) -> Dict[str, Any]:
        """
        使用 A* 演算法尋找最短路徑

        Args:
            start: 起點 (y, x)
            end: 終點 (y, x)
            max_g_cost: 最大路徑成本 (g_score) 限制
            max_distance_from_start: 離起點的最大歐氏距離

        Returns:
            一個包含狀態和結果的字典:
            - {'status': 'success', 'path': path, 'cost': total_cost}
            - {'status': 'cutoff', 'path': partial_path, 'distance': distance}
            - {'status': 'no_path'}
        """
        # 邊界檢查
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            return {'status': 'no_path', 'reason': 'Invalid start or end position'}

        # A* 資料結構
        open_set = []  # 優先佇列: (f_score, counter, position)
        counter = 0  # 確保佇列中項目唯一
        heapq.heappush(open_set, (0, counter, start))
        counter += 1

        came_from = {}  # 路徑記錄
        g_score = {start: 0}  # 起點到當前點的實際成本
        f_score = {start: self._heuristic(start, end)}  # g + h

        visited = set()

        # A* 主迴圈
        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            # 到達終點
            if current == end:
                path = self._reconstruct_path(came_from, current)
                total_cost = g_score[current]
                return {'status': 'success', 'path': path, 'cost': total_cost}

            # 跳過已訪問的節點
            if current in visited:
                continue
            visited.add(current)

            # 提前終止: 距離起點太遠 (您提到的 30px 限制)
            distance_from_start = self._euclidean_distance(start, current)
            if max_distance_from_start is not None and distance_from_start > max_distance_from_start:
                partial_path = self._reconstruct_path(came_from, current)
                return {
                    'status': 'cutoff',
                    'reason': 'distance_from_start',
                    'path': partial_path,
                    'distance': distance_from_start
                }

            # 提前終止: 總成本過高
            if max_g_cost is not None and g_score[current] > max_g_cost:
                partial_path = self._reconstruct_path(came_from, current)
                return {
                    'status': 'cutoff',
                    'reason': 'max_g_cost',
                    'path': partial_path,
                    'cost': g_score[current]
                }


            # 探索8-連通鄰居
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue

                # 計算新的 g_score
                edge_cost = self._edge_cost(current, neighbor)
                tentative_g = g_score[current] + edge_cost

                # 更新或添加鄰居
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        # 找不到路徑
        return {'status': 'no_path', 'reason': 'Open set exhausted'}

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        啟發式函數: 歐氏距離 (可採納的下界估計)
        """
        return self._euclidean_distance(pos, goal)

    def _euclidean_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> float:
        """計算歐氏距離"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _edge_cost(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> float:
        """
        計算邊成本
        成本 = 移動距離 × 目標位置的成本地圖值
        """
        # 移動距離
        dy = abs(pos1[0] - pos2[0])
        dx = abs(pos1[1] - pos2[1])
        distance = 1.414 if dy == 1 and dx == 1 else 1.0

        # 目標位置的成本
        pixel_cost = self.cost_map[pos2[0], pos2[1]]

        return (distance * (pixel_cost**2)) / (255*255)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        獲取8-連通鄰居
        """
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
        """
        回溯重建路徑
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """
        計算路徑的總成本
        """
        if len(path) < 2:
            return 0.0
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._edge_cost(path[i], path[i+1])
        return total_cost

    def get_path_intensity_profile(
        self,
        path: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        獲取路徑上的綠色通道強度分布
        """
        intensities = np.array([self.green_channel[y, x] for y, x in path])
        return intensities


if __name__ == '__main__':
    # 測試程式碼
    import cv2

    # 載入測試影像
    try:
        green_channel = cv2.imread('data/Original/S163-2_a_corrected_normalized.tif', cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        green_channel = None

    if green_channel is not None:
        pathfinder = ImagePathfinder(green_channel, verbose=True)

        # 測試路徑搜索
        start = (349, 3489)
        end = (353, 3484)

        print(f"\n搜索路徑: {start} → {end}")
        result = pathfinder.find_path(start, end)

        if result['status'] == 'success':
            path = result['path']
            cost = result['cost']
            print(f"✓ 找到路徑")
            print(f"  路徑長度: {len(path)} 像素")
            print(f"  總成本: {cost:.2f}")
            print(f"  平均成本: {cost/len(path):.2f}")

            # 路徑強度
            intensities = pathfinder.get_path_intensity_profile(path)
            print(f"  平均綠色強度: {intensities.mean():.1f}")
        else:
            print(f"✗ 找不到路徑，狀態: {result['status']}, 原因: {result.get('reason', 'N/A')}")
    else:
        print("測試影像不存在,跳過測試")
