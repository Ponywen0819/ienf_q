"""
評估度量計算模組 (Evaluation Metrics Module)

提供拓樸比對的核心度量計算功能：
- Average Hausdorff Distance: 平均豪斯多夫距離
- 未來可擴展其他度量（如 Chamfer Distance）

設計原則：
- 純函數式設計，無狀態
- 僅依賴 numpy 和 scipy
- 高效向量化計算
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, Union


def compute_point_min_distances(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算兩點集間每個點到對方點集的最小距離

    Args:
        points_a: 點集 A，形狀 (M, 2)，每行為 [y, x]
        points_b: 點集 B，形狀 (N, 2)，每行為 [y, x]

    Returns:
        (min_dist_a_to_b, min_dist_b_to_a)
        - min_dist_a_to_b: 形狀 (M,)，points_a[i] 到 points_b 最近點的距離
        - min_dist_b_to_a: 形狀 (N,)，points_b[j] 到 points_a 最近點的距離

    Raises:
        ValueError: 如果任一點集為空或形狀不正確
    """
    _validate_point_set(points_a, "points_a")
    _validate_point_set(points_b, "points_b")

    dist_matrix = cdist(points_a, points_b, metric='euclidean')  # (M, N)
    min_dist_a_to_b = np.min(dist_matrix, axis=1)  # (M,)
    min_dist_b_to_a = np.min(dist_matrix, axis=0)  # (N,)

    return min_dist_a_to_b, min_dist_b_to_a


def compute_average_hausdorff_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    return_components: bool = False
) -> Union[float, Tuple[float, float, float]]:
    """
    計算兩個點集之間的平均 Hausdorff 距離

    Average Hausdorff Distance 定義:
    - d(A→B) = mean(min_distance(a, B) for a in A)
    - d(B→A) = mean(min_distance(b, A) for b in B)
    - avg_hausdorff(A, B) = (d(A→B) + d(B→A)) / 2

    與傳統的 Hausdorff 距離（取最大值）不同，平均版本對離群點
    更加穩健，能更好地反映整體的相似度。

    Args:
        points_a: 點集 A，形狀 (M, 2)，每行為 [y, x]
        points_b: 點集 B，形狀 (N, 2)，每行為 [y, x]
        return_components: 是否返回雙向距離分量

    Returns:
        如果 return_components=False（預設）:
            平均 Hausdorff 距離（像素單位）
        如果 return_components=True:
            (avg_distance, d_a_to_b, d_b_to_a)

    Raises:
        ValueError: 如果任一點集為空或形狀不正確

    Examples:
        >>> points_a = np.array([[0, 0], [1, 1], [2, 2]])
        >>> points_b = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
        >>> dist = compute_average_hausdorff_distance(points_a, points_b)
        >>> print(f"{dist:.4f}")
        0.1414

        >>> # 獲取雙向距離分量
        >>> avg_dist, d_ab, d_ba = compute_average_hausdorff_distance(
        ...     points_a, points_b, return_components=True
        ... )
    """
    # 取得每個點的最小距離陣列
    min_dist_a_to_b, min_dist_b_to_a = compute_point_min_distances(points_a, points_b)

    # 計算雙向平均距離
    d_a_to_b = np.mean(min_dist_a_to_b)
    d_b_to_a = np.mean(min_dist_b_to_a)

    # 返回對稱的平均距離
    avg_distance = (d_a_to_b + d_b_to_a) / 2.0

    if return_components:
        return float(avg_distance), float(d_a_to_b), float(d_b_to_a)
    return float(avg_distance)


def compute_directed_hausdorff_distance(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> float:
    """
    計算單向 Hausdorff 距離 d(A→B)

    單向距離定義為:
    d(A→B) = mean(min_distance(a, B) for a in A)

    Args:
        points_a: 源點集，形狀 (M, 2)
        points_b: 目標點集，形狀 (N, 2)

    Returns:
        單向距離 mean(min_distance(a, B) for a in A)

    Raises:
        ValueError: 如果任一點集為空或形狀不正確

    Examples:
        >>> points_a = np.array([[0, 0], [1, 1]])
        >>> points_b = np.array([[0.1, 0.1], [1.1, 1.1], [2, 2]])
        >>> dist = compute_directed_hausdorff_distance(points_a, points_b)
        >>> print(f"{dist:.4f}")
        0.1414
    """
    min_dist_a_to_b, _ = compute_point_min_distances(points_a, points_b)
    return float(np.mean(min_dist_a_to_b))


def _validate_point_set(points: np.ndarray, name: str) -> None:
    """
    驗證點集格式

    Args:
        points: 點集陣列
        name: 參數名稱（用於錯誤訊息）

    Raises:
        TypeError: 如果不是 numpy.ndarray
        ValueError: 如果為空或形狀不正確
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(f"{name} 必須是 numpy.ndarray，實際為 {type(points)}")

    if points.size == 0:
        raise ValueError(f"{name} 不能為空")

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"{name} 應為形狀 (N, 2)，實際為 {points.shape}"
        )


# 未來可擴展的度量
def compute_chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> float:
    """
    計算 Chamfer Distance（未來實作）

    Args:
        points_a: 點集 A
        points_b: 點集 B

    Returns:
        Chamfer Distance

    Raises:
        NotImplementedError: 尚未實作
    """
    raise NotImplementedError("Chamfer Distance 尚未實作")
