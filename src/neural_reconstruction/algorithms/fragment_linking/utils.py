"""
幾何計算工具模組 (Geometry Utilities)

提供階層式片段連接算法所需的幾何計算輔助函數：
- 向量夾角計算
- 方向相似度檢查
- Hessian-based 纖維方向場計算
"""

from typing import List, Tuple

import numpy as np
from skimage.feature import hessian_matrix


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


def compute_fiber_orientation_field(
    image: np.ndarray, sigma: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算 Hessian-based 纖維方向場。

    對亮纖維、暗背景的圖像：Hessian 的較大特徵值 λ2 對應沿纖維方向，
    其特徵向量 v ∝ [Hrc, λ2 − Hrr]（以 (y, x) 座標表示）。

    Args:
        image: 灰階輸入影像 (H, W)，uint8 或 float。
        sigma: Gaussian 平滑 sigma，控制偵測纖維的尺度（預設 2.0）。

    Returns:
        orient_y: (H, W) 單位向量的 y 分量。
        orient_x: (H, W) 單位向量的 x 分量。

    Examples:
        >>> oy, ox = compute_fiber_orientation_field(roi_image, sigma=2.0)
        >>> direction = np.array([oy[y, x], ox[y, x]])   # unit vector at (y, x)
    """
    img_float = image.astype(np.float64)
    Hrr, Hrc, Hcc = hessian_matrix(
        img_float, sigma=sigma, order="rc", use_gaussian_derivatives=False
    )

    discriminant = np.sqrt(((Hrr - Hcc) / 2) ** 2 + Hrc ** 2)
    lam2 = (Hrr + Hcc) / 2 + discriminant   # larger eigenvalue → along-fiber direction

    fiber_vy = Hrc
    fiber_vx = lam2 - Hrr
    fiber_norm = np.sqrt(fiber_vy ** 2 + fiber_vx ** 2)
    fiber_norm = np.where(fiber_norm < 1e-10, 1.0, fiber_norm)

    orient_y = fiber_vy / fiber_norm
    orient_x = fiber_vx / fiber_norm
    return orient_y, orient_x


def get_point_orientation(
    orient_y: np.ndarray, orient_x: np.ndarray, pt: Tuple[float, float]
) -> np.ndarray:
    """
    從預先計算的方向場中取得單一點的纖維方向單位向量。

    Args:
        orient_y: compute_fiber_orientation_field 回傳的 y 分量場 (H, W)。
        orient_x: compute_fiber_orientation_field 回傳的 x 分量場 (H, W)。
        pt: 查詢點座標 (y, x)。

    Returns:
        形狀 (2,) 的單位向量 [dy, dx]。
    """
    y, x = int(pt[0]), int(pt[1])
    return np.array([orient_y[y, x], orient_x[y, x]])
