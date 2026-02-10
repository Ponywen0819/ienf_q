"""
幾何計算工具模組 (Geometry Utilities)

提供階層式片段連接算法所需的幾何計算輔助函數：
- 向量夾角計算
- 方向相似度檢查
"""

from typing import List

import numpy as np


def compute_vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    計算兩個向量之間的夾角（度）

    Args:
        v1: 第一個向量
        v2: 第二個向量

    Returns:
        夾角 [0, 180] 度
    """
    # 計算向量長度
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # 避免除以零
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    # 計算 cos(θ)
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)

    # 限制在 [-1, 1] 範圍內（處理浮點誤差）
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # 轉換為角度
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def is_direction_too_similar(
    new_direction: np.ndarray,
    existing_directions: List[np.ndarray],
    threshold_degrees: float,
) -> bool:
    """
    檢查新方向是否與已存在的任一方向太相近

    Args:
        new_direction: 新候選的方向向量
        existing_directions: 已通過篩選的方向向量列表
        threshold_degrees: 角度閾值（度）

    Returns:
        True 如果太相近（應該跳過），False 如果可以考慮
    """
    if not existing_directions:
        return False

    for existing in existing_directions:
        angle = compute_vector_angle(new_direction, existing)
        if angle < threshold_degrees:
            return True

    return False
