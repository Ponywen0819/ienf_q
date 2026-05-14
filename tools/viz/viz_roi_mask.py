from collections import defaultdict
import heapq

import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import pandas as pd

from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.pathfinding import PathFinder
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.algorithms.annotation_grow.dijkstra import get_components
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (
    find_meeting_points,
    build_component_graph,
    prune_edges,
    minimum_spanning_forest,
)
from neural_reconstruction.algorithms.annotation_grow.skeleton import (
    build_result_graph,
)
from skimage.measure import regionprops
import matplotlib.cm as mcm
from PIL import Image
import skimage as ski

# from skimage.measure import label
# from skimage.restoration import rolling_ball
from scipy.spatial import KDTree
from neural_reconstruction.core.evaluation import (
    extract_graph_points,
    compute_average_hausdorff_distance,
    compute_point_min_distances,
)
from neural_reconstruction.core.crosses_detection import (
    SegmentDetector,
    RegionLabeler,
    CrossingCounter,
)

BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200


def crop(arr):
    """Return the fixed crop used by the paper visualizations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


image = np.array(cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB))

image = image[:, :, 1]  # 只取綠色通道
bg_image = image.copy()

mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
annotation = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)
label_img = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)

OFFSET_PX = 50
roi_mask = dilate_epidermis_vertically(mask, offset_px=OFFSET_PX)


def _compute_aux_mask(m: np.ndarray) -> np.ndarray:
    """重算 dilate_epidermis_vertically 內部使用的輔助遮罩。"""
    binary = m > 0
    H = m.shape[0]
    col_has_mask = binary.any(axis=0)
    min_y = np.where(col_has_mask, np.argmax(binary, axis=0), H)
    y_indices = np.arange(H).reshape(-1, 1)
    return np.where(
        y_indices >= min_y[np.newaxis, :], np.uint8(255), np.uint8(0)
    ).astype(np.uint8)


def _dilate_unconstrained(m: np.ndarray, offset_px: int) -> np.ndarray:
    """純圓形 SE 膨脹,不做向下限制 (對應 dilate_epidermis_vertically 內的 dilated)。"""
    d = 2 * offset_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    return cv2.dilate(m, kernel, iterations=1)


def _overlay_masks(
    bg: np.ndarray,
    masks_and_colors: list,
    alpha: float = 0.5,
    outlines: list | None = None,
) -> np.ndarray:
    """以灰階 bg 為底,將多個遮罩以指定 RGB 顏色 alpha 混合疊上;可選擇加上輪廓線。

    masks_and_colors: [(mask_uint8, (r, g, b))] — 填色,後者覆蓋於前者之上。
    outlines: [(mask_uint8, (r, g, b), thickness)] — 輪廓線,畫在填色之後。
    """
    bg_rgb = np.stack([bg, bg, bg], axis=-1).astype(np.float32)
    for m, color in masks_and_colors:
        m_bool = m > 0
        color_arr = np.array(
            [color[0] * 255.0, color[1] * 255.0, color[2] * 255.0],
            dtype=np.float32,
        )
        bg_rgb[m_bool] = bg_rgb[m_bool] * (1.0 - alpha) + color_arr * alpha
    out = bg_rgb.clip(0, 255).astype(np.uint8)
    if outlines:
        for m, color, thick in outlines:
            contours, _ = cv2.findContours(
                (m > 0).astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE,
            )
            color_255 = (
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
            )
            cv2.drawContours(out, contours, -1, color_255, thick)
    return out


aux_mask = _compute_aux_mask(mask)
unconstrained_dilated = _dilate_unconstrained(mask, OFFSET_PX)

DILATED_COLOR = (0.20, 0.90, 0.30)  # 綠 — 膨脹結果 (受限/不受限)
ORIG_MASK_COLOR = (1.0, 0.20, 0.80)  # 洋紅 — 原始遮罩 (輪廓)
AUX_COLOR = (1.0, 0.55, 0.0)  # 橙 — 輔助遮罩 (印刷比黃色清楚)
OUTLINE_THICKNESS = 10

# 視覺化裁切:含原始遮罩 + 膨脹環 + 額外上緣空間以看清 aux 邊界
ys, xs = np.where(mask > 0)
PAD_ABOVE = OFFSET_PX + 40
PAD_BELOW = OFFSET_PX + 30
PAD_LATERAL = OFFSET_PX + 30
y0 = max(0, int(ys.min()) - PAD_ABOVE)
y1 = min(mask.shape[0], int(ys.max()) + PAD_BELOW)
x0 = max(0, int(xs.min()) - PAD_LATERAL)
x1 = min(mask.shape[1], int(xs.max()) + PAD_LATERAL)


def _viz_crop(arr: np.ndarray) -> np.ndarray:
    return arr[y0:y1, x0:x1]


# 圖 1: aux_mask + 原始遮罩輪廓 (讓讀者看出 aux 上緣就貼著遮罩頂端)
viz_aux = _overlay_masks(
    bg_image,
    masks_and_colors=[(aux_mask, AUX_COLOR)],
    alpha=0.45,
    outlines=[(mask, ORIG_MASK_COLOR, OUTLINE_THICKNESS)],
)
Image.fromarray(_viz_crop(viz_aux)).save("./output/viz_aux_mask.png")

# 圖 2: 不受限膨脹 + 原始遮罩輪廓
viz_uncon = _overlay_masks(
    bg_image,
    masks_and_colors=[(unconstrained_dilated, DILATED_COLOR)],
    alpha=0.5,
    outlines=[(mask, ORIG_MASK_COLOR, OUTLINE_THICKNESS)],
)
Image.fromarray(_viz_crop(viz_uncon)).save("./output/viz_unconstrained_dilation.png")

# 圖 3: roi_mask + 原始遮罩輪廓
viz_roi = _overlay_masks(
    bg_image,
    masks_and_colors=[(roi_mask, DILATED_COLOR)],
    alpha=0.5,
    outlines=[(mask, ORIG_MASK_COLOR, OUTLINE_THICKNESS)],
)
Image.fromarray(_viz_crop(viz_roi)).save("./output/viz_roi_mask.png")