"""
A* 路徑搜索
在影像成本地圖上尋找最短路徑
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional


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
        max_distance: float = None
    ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """
        使用 A* 演算法尋找最短路徑
        
        Args:
            start: 起點 (y, x)
            end: 終點 (y, x)
            max_distance: 最大搜索距離 (None 表示無限制)
        
        Returns:
            path: 路徑座標列表 [(y, x), ...], None 表示找不到路徑
            total_cost: 路徑總成本
        """
        # 邊界檢查
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            return None, float('inf')
        
        # 計算直線距離
        straight_distance = self._euclidean_distance(start, end)
        
        if max_distance is None:
            max_distance = straight_distance * 200.0  # 預設最大為直線距離的50倍
        
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
                return path, total_cost
            
            # 跳過已訪問的節點
            if current in visited:
                continue
            visited.add(current)
            
            # 提前終止: 距離起點太遠
            # if g_score[current] > max_distance:  # 成本閾值
            #     continue
                
            distance_from_start = self._euclidean_distance(start, current)
            if distance_from_start > 30:
                continue

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
        return None, float('inf')
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        啟發式函數: 歐氏距離 (可採納的下界估計)
        
        Args:
            pos: 當前位置
            goal: 目標位置
        
        Returns:
            h: 啟發式值
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
        
        對角線移動距離 = sqrt(2) ≈ 1.414
        正交移動距離 = 1.0
        """
        # 移動距離
        dy = abs(pos1[0] - pos2[0])
        dx = abs(pos1[1] - pos2[1])
        
        if dy == 1 and dx == 1:
            # 對角線移動
            distance = 1.414
        else:
            # 正交移動
            distance = 1.0
        
        # 目標位置的成本
        pixel_cost = self.cost_map[pos2[0], pos2[1]]

        return (distance * (pixel_cost**2)) / (255*255)  # 成本平方根加權

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        獲取8-連通鄰居
        
        Args:
            pos: 當前位置 (y, x)
        
        Returns:
            neighbors: 鄰居列表
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
        
        Args:
            came_from: 路徑記錄字典
            current: 終點
        
        Returns:
            path: 從起點到終點的路徑
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
        
        Args:
            path: 路徑座標列表
        
        Returns:
            total_cost: 總成本
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
        
        Args:
            path: 路徑座標列表
        
        Returns:
            intensities: 強度陣列
        """
        intensities = np.array([self.green_channel[y, x] for y, x in path])
        return intensities


if __name__ == '__main__':
    # 測試程式碼
    import cv2
    
    # 載入測試影像
    green_channel = cv2.imread('data/Original/S163-2_a_corrected_normalized.tif', cv2.IMREAD_GRAYSCALE)
    
    if green_channel is not None:
        pathfinder = ImagePathfinder(green_channel, verbose=True)
        
        # 測試路徑搜索
        start = (349, 3489)
        end = (353, 3484)
        
        print(f"\n搜索路徑: {start} → {end}")
        path, cost = pathfinder.find_path(start, end)
        
        if path:
            print(f"✓ 找到路徑")
            print(f"  路徑長度: {len(path)} 像素")
            print(f"  總成本: {cost:.2f}")
            print(f"  平均成本: {cost/len(path):.2f}")
            
            # 路徑強度
            intensities = pathfinder.get_path_intensity_profile(path)
            print(f"  平均綠色強度: {intensities.mean():.1f}")
        else:
            print("✗ 找不到路徑")
    else:
        print("測試影像不存在,跳過測試")
