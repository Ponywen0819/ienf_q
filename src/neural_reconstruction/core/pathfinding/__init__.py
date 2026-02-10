"""
路徑查找模組 (Pathfinding Module)

提供基於 MCP_Geometric 的統一路徑查找功能。

核心類別：
    - PathFinder: 基底路徑查找器，接受外部 cost_map

支援兩種使用模式：
    - find_paths_from_source: 從單一源點到多個目標點
    - find_paths_from_seeds: 預計算所有鄰近種子點對路徑

使用範例：
---------
from neural_reconstruction.core.pathfinding import PathFinder

# cost_map 由呼叫端建立
cost_map = (255 - image.astype(float)) / 255.0
finder = PathFinder(cost_map)

# 單源到多目標
results = finder.find_paths_from_source(start, targets)

# 全域種子點對查找
path_lookup = finder.find_paths_from_seeds(topology_points, kdtree, search_radius)
"""

from .path_finder import PathFinder

__all__ = ['PathFinder']

__version__ = '1.0.0'
