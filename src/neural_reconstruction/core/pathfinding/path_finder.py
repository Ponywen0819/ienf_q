"""
路徑查找模組 (Path Finder Module)

基於 MCP_Geometric 的統一路徑查找器。
接受外部傳入的 cost_map，作為基底型別，未來可加入特定的 cost_map 子類。

支援兩種使用模式：
    1. find_paths_from_source: 從單一源點到多個目標點（用於 connection_graph_builder）
    2. find_paths_from_seeds: 預計算所有鄰近種子點對路徑（用於 fragment_linking）

使用範例：
---------
# 建立 cost_map（由呼叫端負責）
cost_map = (255 - image.astype(np.float64)) / 255.0

# 建立查找器
finder = PathFinder(cost_map)

# 模式1：單源到多目標
results = finder.find_paths_from_source(start=(10, 20), targets=[(50, 60), (80, 90)])

# 模式2：種子點對全域查找
path_lookup = finder.find_paths_from_seeds(
    topology_points, kdtree, search_radius=50.0
)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import skimage.graph
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class PathFinder:
    """
    基於 MCP_Geometric 的路徑查找器

    接受外部建立的 cost_map，不在內部建立。
    可作為基底型別，未來可透過繼承加入特定用途的 cost_map 初始化邏輯。

    Examples:
        >>> cost_map = (255 - image.astype(np.float64)) / 255.0
        >>> finder = PathFinder(cost_map)
        >>> results = finder.find_paths_from_source((10, 20), [(50, 60)])
    """

    def __init__(self, cost_map: np.ndarray, bbox_padding: int = 10):
        """
        Args:
            cost_map: 成本地圖，值越低表示路徑越優
            bbox_padding: bbox 裁剪的邊距（像素）
        """
        self.cost_map = cost_map
        self.bbox_padding = bbox_padding
        self.height, self.width = cost_map.shape

    def find_paths_from_source(
        self,
        start: Tuple[int, int],
        targets: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], Optional[Tuple[List, float]]]:
        """
        從單一源點到多個目標點查找路徑（使用動態 bbox 裁剪）

        Args:
            start: 源點座標 (y, x)
            targets: 目標點座標列表 [(y, x), ...]

        Returns:
            {target: (path, cost)} 或 {target: None}（無法到達時）
        """
        result: Dict[Tuple[int, int], Optional[Tuple[List, float]]] = {
            t: None for t in targets
        }

        if not self._is_valid_position(start):
            logger.warning(f"源點 {start} 超出影像範圍")
            return result

        valid_targets = [t for t in targets if self._is_valid_position(t)]
        if not valid_targets:
            return result

        min_y, max_y, min_x, max_x, cropped = self._get_bbox_roi(start, valid_targets)

        start_local = (start[0] - min_y, start[1] - min_x)
        targets_local = [(t[0] - min_y, t[1] - min_x) for t in valid_targets]

        logger.debug(
            f"動態 bbox: ({min_y}:{max_y}, {min_x}:{max_x}), "
            f"大小: {max_y - min_y + 1}x{max_x - min_x + 1}"
        )

        try:
            mcp_results = self._execute_mcp_traceback(
                cropped, start_local, targets_local
            )
            for t_local, t_global in zip(targets_local, valid_targets):
                path_result = mcp_results.get(t_local)
                if path_result is None:
                    continue
                path_local, cost = path_result
                path_global = [(p[0] + min_y, p[1] + min_x) for p in path_local]
                result[t_global] = (path_global, cost)
        except Exception as e:
            logger.warning(f"路徑搜尋失敗: {e}")
            return {t: None for t in targets}

        return result

    def find_paths_from_seeds(
        self,
        topology_points: np.ndarray,
        kdtree: KDTree,
        search_radius: float,
        seed_map: Optional[np.ndarray] = None,
        label_img: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Dict[Tuple, Tuple[List, float]]:
        """
        預計算所有鄰近種子點對之間的路徑

        Args:
            topology_points: 所有種子點座標，形狀 (N, 2)
            kdtree: 對應 topology_points 的 KDTree
            search_radius: 搜索半徑（像素）
            seed_map: 種子點二值圖（防止路徑穿越種子點）；None 表示不過濾
            label_img: 連通元件標籤圖（過濾同一元件）；None 表示不過濾

        Returns:
            path_lookup: {(start, end): (path_coords, cost)}
        """
        path_lookup = {}

        for u_idx in range(len(topology_points)):
            u = topology_points[u_idx]

            neighbor_indices = kdtree.query_ball_point(u, r=search_radius)
            targets = [topology_points[v_idx] for v_idx in neighbor_indices]

            # 過濾自身
            targets = [t for t in targets if tuple(t) != tuple(u)]

            # 過濾同一連通元件
            if label_img is not None:
                current_label = label_img[u[0], u[1]]
                targets = [t for t in targets if label_img[t[0], t[1]] != current_label]

            # 過濾已計算的點對（雙向去重）
            targets = [
                t
                for t in targets
                if (tuple(u), tuple(t)) not in path_lookup
                and (tuple(t), tuple(u)) not in path_lookup
            ]

            if not targets:
                continue

            all_points = [u] + targets
            min_y, max_y, min_x, max_x, cropped = self._get_bbox_roi(
                (u[0], u[1]), [(t[0], t[1]) for t in targets]
            )

            local_points = [(int(p[0]) - min_y, int(p[1]) - min_x) for p in all_points]

            mcp = skimage.graph.MCP_Geometric(cropped, fully_connected=True)
            cumulative_costs, _ = mcp.find_costs(
                starts=local_points[:1], ends=local_points[1:]
            )

            for target_local, target_global in zip(local_points[1:], targets):
                if np.isinf(cumulative_costs[target_local]):
                    continue

                path = mcp.traceback(target_local)
                cost = cumulative_costs[target_local]

                # 過濾穿越其他種子點的路徑
                if seed_map is not None:
                    middle_points = np.array(path[1:-1])
                    if len(middle_points) > 0:
                        global_middle = middle_points + np.array([min_y, min_x])
                        if np.any(seed_map[global_middle[:, 0], global_middle[:, 1]]):
                            continue

                global_path = [(p[0] + min_y, p[1] + min_x) for p in path]
                global_start = (int(u[0]), int(u[1]))
                global_target = (int(target_global[0]), int(target_global[1]))

                # 若反向已存在，保留較小成本
                if (global_target, global_start) in path_lookup:
                    existing_cost = path_lookup[(global_target, global_start)][1]
                    if cost < existing_cost:
                        path_lookup[(global_target, global_start)] = (global_path, cost)
                else:
                    path_lookup[(global_start, global_target)] = (global_path, cost)

        if verbose:
            print(f"✓ 路徑查找完成: {len(path_lookup)} 條路徑")

        return path_lookup

    # ========== 私有輔助方法 ==========

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """檢查座標是否在影像範圍內"""
        y, x = pos
        return 0 <= y < self.height and 0 <= x < self.width

    def _get_bbox_roi(
        self,
        start: Tuple[int, int],
        targets: List[Tuple[int, int]],
    ) -> Tuple[int, int, int, int, np.ndarray]:
        """計算包含所有點的最小 bbox 並裁剪 cost_map"""
        all_y = [start[0]] + [t[0] for t in targets]
        all_x = [start[1]] + [t[1] for t in targets]

        min_y = max(0, min(all_y) - self.bbox_padding)
        max_y = min(self.height - 1, max(all_y) + self.bbox_padding)
        min_x = max(0, min(all_x) - self.bbox_padding)
        max_x = min(self.width - 1, max(all_x) + self.bbox_padding)

        cropped = self.cost_map[min_y : max_y + 1, min_x : max_x + 1]
        return min_y, max_y, min_x, max_x, cropped

    def _execute_mcp_traceback(
        self,
        cost_map: np.ndarray,
        source: Tuple[int, int],
        targets: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], Optional[Tuple[List, float]]]:
        """執行 MCP_Geometric 並回溯路徑"""
        mcp = skimage.graph.MCP_Geometric(cost_map, fully_connected=True)
        cumulative_costs, _ = mcp.find_costs(starts=[source])

        results = {}
        for target in targets:
            if np.isinf(cumulative_costs[target]):
                logger.debug(f"目標 {target} 無法從 {source} 到達")
                results[target] = None
                continue
            path = mcp.traceback(target)
            cost = cumulative_costs[target]
            results[target] = (path, cost)
            logger.debug(
                f"路徑 {source} -> {target}: 長度 {len(path)}, 成本 {cost:.4f}"
            )

        return results
