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
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize, disk
from typing import Tuple, Union


def compute_point_min_distances(
    points_a: np.ndarray, points_b: np.ndarray
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

    dist_matrix = cdist(points_a, points_b, metric="euclidean")  # (M, N)
    min_dist_a_to_b = np.min(dist_matrix, axis=1)  # (M,)
    min_dist_b_to_a = np.min(dist_matrix, axis=0)  # (N,)

    return min_dist_a_to_b, min_dist_b_to_a


def compute_average_hausdorff_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> Tuple[float, float, float]:
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

    return float(avg_distance), float(d_a_to_b), float(d_b_to_a)


def compute_hd95(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> Tuple[float, float, float]:
    """
    計算 95th Percentile Hausdorff Distance (HD95)

    HD95 定義：
    - d95(A→B) = 95th percentile of {min_dist(a, B) for a in A}
    - d95(B→A) = 95th percentile of {min_dist(b, A) for b in B}
    - HD95(A, B) = max(d95(A→B), d95(B→A))

    相比傳統最大 Hausdorff 距離，HD95 排除最遠的 5% 離群點，
    對少數嚴重偏差的點更加穩健。

    Args:
        points_a: 點集 A，形狀 (M, 2)，每行為 [y, x]
        points_b: 點集 B，形狀 (N, 2)，每行為 [y, x]

    Returns:
        (hd95, d95_a_to_b, d95_b_to_a)
        - hd95: 對稱 HD95，= max(d95_a_to_b, d95_b_to_a)
        - d95_a_to_b: A→B 方向的第 95 百分位最小距離
        - d95_b_to_a: B→A 方向的第 95 百分位最小距離

    Raises:
        ValueError: 如果任一點集為空或形狀不正確
    """
    min_dist_a_to_b, min_dist_b_to_a = compute_point_min_distances(points_a, points_b)

    d95_a_to_b = float(np.percentile(min_dist_a_to_b, 95))
    d95_b_to_a = float(np.percentile(min_dist_b_to_a, 95))
    hd95 = max(d95_a_to_b, d95_b_to_a)

    return hd95, d95_a_to_b, d95_b_to_a


def compute_directed_hausdorff_distance(
    points_a: np.ndarray, points_b: np.ndarray
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
        raise ValueError(f"{name} 應為形狀 (N, 2)，實際為 {points.shape}")


def compute_cldice(
    pred_graph: nx.Graph,
    gt_label: np.ndarray,
    tolerance_px: int = 3,
) -> Tuple[float, float, float]:
    """
    計算 clDice（Centerline Dice）

    針對圖結構預測的 clDice 實作：
      - Pred skeleton  = 預測圖的所有點（節點 + 邊路徑），直接作為中心線
      - GT skeleton    = skimage.skeletonize(gt_label > 0) 的骨架像素
      - tolerance_px   = 膨脹半徑（像素），允許些微偏移

    公式：
      Tprec  = |skel_pred  ∩  dilate(gt_mask,   tol)| / |skel_pred|
      Tsens  = |skel_gt    ∩  dilate(pred_mask, tol)| / |skel_gt|
      clDice = 2 × Tprec × Tsens / (Tprec + Tsens)

    Args:
        pred_graph:   NetworkX 圖，節點為 (y, x)，邊可帶 'path' 屬性
        gt_label:     GT 二值遮罩 (H, W)，0 = 背景，>0 = 纖維
        tolerance_px: 膨脹半徑（預設 3 px），增大可寬容空間偏移

    Returns:
        (clDice, Tprec, Tsens)  — 均為 0.0–1.0 之間的浮點數

    Examples:
        >>> cld, tprec, tsens = compute_cldice(pred_graph, gt_label, tolerance_px=3)
        >>> print(f"clDice={cld:.4f}  Tprec={tprec:.4f}  Tsens={tsens:.4f}")
    """
    H, W = gt_label.shape

    # ── Step 1: 取得預測骨架點（graph points），rasterize 成二值遮罩 ──────────
    pred_pts: list[tuple[int, int]] = []
    for node in pred_graph.nodes():
        y, x = int(round(node[0])), int(round(node[1]))
        if 0 <= y < H and 0 <= x < W:
            pred_pts.append((y, x))

    for u, v, data in pred_graph.edges(data=True):
        path = data.get("path")
        if path is None:
            path = data.get("path-coordinates")
        if path is not None and len(path) > 0:
            for pt in path:
                y, x = int(round(pt[0])), int(round(pt[1]))
                if 0 <= y < H and 0 <= x < W:
                    pred_pts.append((y, x))

    if not pred_pts:
        return 0.0, 0.0, 0.0

    skel_pred_mask = np.zeros((H, W), dtype=bool)
    ys, xs = zip(*pred_pts)
    skel_pred_mask[list(ys), list(xs)] = True

    # ── Step 2: GT 骨架 ────────────────────────────────────────────────────────
    gt_binary = gt_label > 0
    skel_gt_mask: np.ndarray = skeletonize(gt_binary)

    if not skel_gt_mask.any():
        return 0.0, 0.0, 0.0

    # ── Step 3: 膨脹（tolerance）─────────────────────────────────────────────
    structuring_element = disk(tolerance_px)
    dilated_gt = binary_dilation(gt_binary, structure=structuring_element)
    dilated_pred = binary_dilation(skel_pred_mask, structure=structuring_element)

    # ── Step 4: Tprec / Tsens ─────────────────────────────────────────────────
    n_skel_pred = int(skel_pred_mask.sum())
    n_skel_gt = int(skel_gt_mask.sum())

    tprec = float((skel_pred_mask & dilated_gt).sum()) / n_skel_pred
    tsens = float((skel_gt_mask & dilated_pred).sum()) / n_skel_gt

    # ── Step 5: clDice ────────────────────────────────────────────────────────
    denom = tprec + tsens
    cldice = (2.0 * tprec * tsens / denom) if denom > 0.0 else 0.0

    return cldice, tprec, tsens


# 未來可擴展的度量
def compute_chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
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
