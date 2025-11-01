"""
密度估算器 (空殼實作)
"""

import numpy as np


class DensityEstimator:
    """局部密度估算與自適應半徑決定"""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def calculate_local_density(self, seed, kdtree, k: int = None):
        """
        計算種子的局部密度

        Args:
            seed: Seed 物件
            kdtree: sklearn KDTree 物件
            k: 近鄰數(若為 None 則使用 self.k)

        Returns:
            density: 局部密度(平均距離,單位:像素)
        """
        if k is None:
            k = self.k

        # 查詢最近的 k+1 個鄰居(包含自己)
        query_point = np.array([[seed.x, seed.y]])
        distances, indices = kdtree.query(query_point, k=k+1)

        # 排除第一個距離(自己到自己 = 0),計算平均值
        local_density = np.mean(distances[0][1:])

        return local_density
    
    def determine_adaptive_radius(self, local_density: float) -> float:
        """
        根據局部密度決定自適應半徑

        Args:
            local_density: 局部密度值(平均距離,單位:像素)

        Returns:
            radius: 配對半徑(像素)

        規則:
            - density < 30px  → radius = 30px  (密集區)
            - 30 ≤ density < 70px → radius = 50px  (適中區)
            - density ≥ 70px → radius = 80px  (稀疏區)
        """
        if local_density < 30:
            return 30
        elif local_density < 70:
            return 50
        else:
            return 80
