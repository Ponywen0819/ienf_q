"""
A* 路徑搜尋模組 (A* Pathfinding Module)

基於影像成本地圖的 A* 路徑搜尋。
提供兩個核心功能：
1. 尋找兩點之間的最短路徑
2. 計算路徑成本
"""

import numpy as np
import logging
import skimage.graph
from skimage.filters import sato
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class Pathfinder:
    """
    路徑搜尋器

    在成本地圖上尋找最短路徑。
    成本地圖 = 255 - image ( 亮度強度越高，成本越低 )
    """

    def __init__(
        self,
        image: np.ndarray,
        intensity_weight: float = 0.6,
        shape_weight: float = 0.4,
    ):
        """
        初始化路徑搜尋器

        Args:
            image: 單通道影像 (uint8, 0-255)
        """

        self.intensity_weight = intensity_weight
        self.shape_weight = shape_weight
        self._create_cost_map(image)

        logger.info(f"初始化 A* 路徑搜尋器: {self.height} x {self.width}")

    def _create_cost_map(self, image: np.ndarray) -> None:
        """建立成本地圖"""
        # intensity_cost_map = 1 / (
        #     image.astype(np.float64) + 1
        # )  # 反轉強度 (亮度越高成本越低)
        intensity_cost_map = 255 - image.astype(
            np.float64
        )  # 反轉強度 (亮度越高成本越低)

        self.cost_map = (
            intensity_cost_map**self.intensity_weight / 255**self.intensity_weight
        )
        self.height, self.width = self.cost_map.shape

    def find_paths_from_source(
        self,
        start: Tuple[int, int],
        targets: List[Tuple[int, int]],
        bbox_padding: int = 20,
    ):
        """
        從單一源點計算到多個目標點的最短路徑（使用 Dijkstra）

        使用動態 bbox 裁剪來減少計算範圍，提升效能。

        Args:
            start: 源點座標 (y, x)
            targets: 目標點座標列表 [(y, x), ...]
            bbox_padding: bbox 的 padding（像素）

        Returns:
            字典，key 為目標點座標，value 為 (路徑, 成本) 或 None
            {
                (y1, x1): ([path], cost),
                (y2, x2): None,  # 無法到達
                ...
            }
        """
        result: Dict[Tuple[int, int], Optional[Tuple[List[Tuple[int, int]], float]]] = {
            target: None for target in targets
        }
        # 邊界檢查
        if not self._is_valid_position(start):
            logger.warning(f"源點 {start} 超出影像範圍")
            return result

        # 過濾有效目標點
        valid_targets = [t for t in targets if self._is_valid_position(t)]
        if len(valid_targets) != len(targets):
            logger.warning("某些目標點超出影像範圍")

        if len(valid_targets) == 0:
            return result

        # 取得裁剪後的成本地圖
        min_y, max_y, min_x, max_x, cropped_cost_map = self._get_path_finding_roi(
            start, valid_targets, bbox_padding
        )

        # 將全局座標轉換為局部座標
        start_local = self._convert_position_global_to_local(start, min_y, min_x)
        targets_local = [
            self._convert_position_global_to_local(t, min_y, min_x)
            for t in valid_targets
        ]

        logger.debug(
            f"使用動態 bbox: ({min_y}:{max_y}, {min_x}:{max_x}), "
            f"大小: {max_y - min_y + 1}x{max_x - min_x + 1}, "
            f"原始: {self.height}x{self.width}"
        )

        try:
            # 使用 MCP_Geometric 在裁剪後的 cost map 上計算最短路徑
            path_results = self._execute_mcp_traceback(
                cropped_cost_map, start_local, targets_local
            )
            # 將局部結果轉換回全局座標
            offset = np.array([min_y, min_x])
            for target_local, path_result in path_results.items():
                target_global = self._convert_position_local_to_global(
                    target_local, min_y, min_x
                )
                if path_result is None:
                    continue

                path_local, cost = path_result
                path_global = (np.array(path_local) + offset).tolist()

                result[target_global] = (path_global, cost)

        except Exception as e:
            logger.warning(f"批次路徑搜尋失敗: {e}")
            return {target: None for target in targets}

        return result

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """檢查位置是否在影像範圍內"""
        y, x = pos
        return 0 <= y < self.height and 0 <= x < self.width

    def _get_path_finding_roi(
        self,
        start: Tuple[int, int],
        targets: List[Tuple[int, int]],
        bbox_padding: int = 20,
    ) -> Tuple[int, int, int, int, np.ndarray]:
        """計算最小 bbox"""

        # 計算包含所有點的最小 bbox + padding
        all_points = [start] + targets
        all_y = [p[0] for p in all_points]
        all_x = [p[1] for p in all_points]

        min_y = max(0, min(all_y) - bbox_padding)
        max_y = min(self.height - 1, max(all_y) + bbox_padding)
        min_x = max(0, min(all_x) - bbox_padding)
        max_x = min(self.width - 1, max(all_x) + bbox_padding)

        # 裁剪 cost map
        cropped_cost_map = self.cost_map[min_y : max_y + 1, min_x : max_x + 1]

        return min_y, max_y, min_x, max_x, cropped_cost_map

    def _execute_mcp_traceback(
        self,
        cost_map: np.ndarray,
        source: Tuple[int, int],
        targets: List[Tuple[int, int]],
    ):
        mcp = skimage.graph.MCP_Geometric(cost_map, fully_connected=True)
        cumulative_costs, traceback = mcp.find_costs(starts=[source])

        results = {}
        for target in targets:
            # 檢查是否可達
            if np.isinf(cumulative_costs[target]):
                logger.debug(f"目標點 {target} 無法從 {source} 到達")
                results[target] = None
                continue

            # 使用 MCP 的 traceback 方法回溯路徑
            path = mcp.traceback(target)
            cost = cumulative_costs[target]

            results[target] = (path, cost)
            logger.debug(
                f"找到路徑 {source} -> {target}: 長度 {len(path)}, 成本 {cost:.4f}"
            )

        return results

    def _convert_position_local_to_global(
        self, pos_local: Tuple[int, int], min_y: int, min_x: int
    ) -> Tuple[int, int]:
        """將局部座標轉換為全局座標"""
        return (pos_local[0] + min_y, pos_local[1] + min_x)

    def _convert_position_global_to_local(
        self, pos_global: Tuple[int, int], min_y: int, min_x: int
    ) -> Tuple[int, int]:
        """將全局座標轉換為局部座標"""
        return (pos_global[0] - min_y, pos_global[1] - min_x)
