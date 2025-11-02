"""
成本計算器
階段B: 完整版 (多因素成本)
"""

import numpy as np
from typing import Dict, Any

# 類型提示 (避免循環匯入)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .pathfinding import ImagePathfinder
    from .seed_loader import Seed


class CostCalculator:
    """
    多因素成本計算器

    成本公式:
    total_cost = α×geometric + β×image + γ×curvature
    """

    def __init__(
        self,
        pathfinder: 'ImagePathfinder',
        alpha: float = 0.05,
        beta: float = 0.9,
        gamma: float = 0.05,
        verbose: bool = False
    ):
        """
        初始化成本計算器

        Args:
            pathfinder: ImagePathfinder 物件,用於 A* 路徑搜索
            alpha: 幾何成本權重 (預設 0.3)
            beta: 影像成本權重 (預設 0.5)
            gamma: 曲率成本權重 (預設 0.2)
            verbose: 是否輸出詳細資訊
        """
        self.pathfinder = pathfinder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.verbose = verbose

    def calculate_total_cost(
        self,
        seed_i: 'Seed',
        seed_j: 'Seed'
    ) -> Dict[str, Any]:
        """
        計算兩個種子之間的完整多因素成本

        Args:
            seed_i: 起始種子 (Seed 物件)
            seed_j: 目標種子 (Seed 物件)

        Returns:
            dict: 包含各種成本和總成本的字典
        """
        # 1. 幾何成本 (總是先計算)
        geometric_cost = np.sqrt(
            (seed_i.x - seed_j.x)**2 +
            (seed_i.y - seed_j.y)**2
        )

        # 如果幾何距離為0,成本為0
        if geometric_cost == 0:
            return {
                'total_cost': 0.0,
                'geometric_cost': 0.0,
                'image_cost': 0.0,
                'curvature_cost': 0.0,
                'tortuosity': 1.0,
                'path': [(seed_i.y, seed_i.x)],
                'path_cost': 0.0
            }

        # 2. 影像成本 (透過 A* 搜索)
        start_pos = (seed_i.y, seed_i.x)
        end_pos = (seed_j.y, seed_j.x)
        path, path_cost = self.pathfinder.find_path(start_pos, end_pos)

        # 如果找不到路徑,回傳無限大成本
        if path is None:
            return {
                'total_cost': float('inf'),
                'geometric_cost': geometric_cost,
                'image_cost': float('inf'),
                'curvature_cost': float('inf'),
                'tortuosity': float('inf'),
                'path': None,
                'path_cost': float('inf')
            }

        path_length = len(path)
        
        # 標準化影像成本
        image_cost = path_cost / path_length if path_length > 0 else 0.0

        # 3. 曲率成本
        tortuosity = path_length / geometric_cost
        
        if tortuosity < 1.5:
            curvature_cost = 0.0
        else:
            curvature_cost = (tortuosity - 1) * 100

        # 4. 計算總成本
        total_cost = (
            self.alpha * geometric_cost +
            self.beta * image_cost +
            self.gamma * curvature_cost
        )

        return {
            'total_cost': total_cost,
            'geometric_cost': geometric_cost,
            'image_cost': image_cost,
            'curvature_cost': curvature_cost,
            'tortuosity': tortuosity,
            'path': path,
            'path_cost': path_cost
        }