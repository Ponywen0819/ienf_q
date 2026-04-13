"""
Multi-source Dijkstra with per-component adaptive stopping.

Each annotation component expands from its seed pixels simultaneously.
Expansion halts when accumulated cost exceeds mean + k*std of the
component's own seed pixel costs.
"""

import heapq
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops


def get_components(binary_img: np.ndarray, min_area: int = 0) -> np.ndarray:
    """
    Label connected components, removing those below min_area.

    Returns:
        labeled: int array (H, W), 0=background, 1..N=component IDs
    """
    labeled = label(binary_img, connectivity=1)
    for p in regionprops(labeled):
        if p.area < min_area:
            labeled[labeled == p.label] = 0
    labeled, _ = ndimage.label(labeled > 0)
    return labeled


def compute_per_component_thresholds(
    cost_map: np.ndarray,
    annot_labeled: np.ndarray,
    k: float = 1.5,
) -> dict[int, float]:
    """
    Compute per-component adaptive stopping thresholds.

    threshold[cid] = mean(cost_map[seed_pixels]) + k * std(cost_map[seed_pixels])

    Args:
        cost_map:      float32 (H, W) traversal cost
        annot_labeled: int (H, W) component labels
        k:             multiplier for std (larger = expands further)

    Returns:
        thresholds: dict mapping component ID -> stopping threshold
    """
    thresholds: dict[int, float] = {}
    n = int(annot_labeled.max())
    for cid in range(1, n + 1):
        seed_costs = cost_map[annot_labeled == cid]
        mu = float(seed_costs.mean())
        sig = float(seed_costs.std())
        thresholds[cid] = mu + k * sig
    return thresholds


def multi_source_dijkstra(
    cost_map: np.ndarray,
    annot_labeled: np.ndarray,
    connectivity: int = 8,
    thresholds: Optional[dict[int, float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-source Dijkstra on a 2D cost map.

    All annotation pixels start at accumulated cost 0 simultaneously.
    Each pixel records the component that reached it first and the cost.

    Args:
        cost_map:      float32 (H, W) per-pixel traversal cost
        annot_labeled: int (H, W) component labels (0=background)
        connectivity:  4 or 8
        thresholds:    if provided, expansion from component cid stops
                       when accumulated cost exceeds thresholds[cid]

    Returns:
        owner_map: int32 (H, W) — which component owns each pixel (0=unvisited)
        dist_map:  float32 (H, W) — accumulated cost to reach each pixel
        prev_y:    int32 (H, W) — predecessor row (-1 for annotation seeds)
        prev_x:    int32 (H, W) — predecessor col (-1 for annotation seeds)
    """
    H, W = cost_map.shape

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

    dist_map = np.full((H, W), np.inf, dtype=np.float32)
    owner_map = np.zeros((H, W), dtype=np.int32)
    prev_y = np.full((H, W), -1, dtype=np.int32)
    prev_x = np.full((H, W), -1, dtype=np.int32)

    heap: list = []
    ys, xs = np.where(annot_labeled > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cid = int(annot_labeled[y, x])
        dist_map[y, x] = 0.0
        owner_map[y, x] = cid
        heapq.heappush(heap, (0.0, y, x, cid))

    while heap:
        d, y, x, cid = heapq.heappop(heap)

        if d > dist_map[y, x]:
            continue
        if thresholds is not None and d > thresholds[cid]:
            continue

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue

            new_dist = d + float(cost_map[ny, nx])

            if thresholds is not None and new_dist > thresholds[cid]:
                continue

            if new_dist < dist_map[ny, nx]:
                dist_map[ny, nx] = new_dist
                owner_map[ny, nx] = cid
                prev_y[ny, nx] = y
                prev_x[ny, nx] = x
                heapq.heappush(heap, (new_dist, ny, nx, cid))

    return owner_map, dist_map, prev_y, prev_x
