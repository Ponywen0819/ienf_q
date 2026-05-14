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
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

image = np.array(cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB))
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200


def crop(arr):
    """Return the fixed crop used by the paper visualizations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


image = image[:, :, 1]  # 只取綠色通道
mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
annotation = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)
label_img = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)

roi_mask = dilate_epidermis_vertically(mask, offset_px=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

image = cv2.subtract(image, background)

tileGridSize = 768
clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(tileGridSize, tileGridSize))
image = clahe.apply(image)

image = cv2.bitwise_and(image, image, mask=roi_mask)
image = ski.filters.sato(image, sigmas=range(3, 8), black_ridges=False)
image = (image - image.min()) / (image.max() - image.min()) * 255
image = image.astype(np.uint8)
cost_map = np.exp(1.0 - (image.astype(np.float32) / 255.0)) - 1.0

# 將 ROI 外設為 masked,並以 viridis_r 反轉:低 cost 亮 (=偏好路徑)、高 cost 暗
masked_cost = np.ma.masked_where(roi_mask == 0, cost_map)
cmap = plt.get_cmap("viridis_r").copy()
cmap.set_bad(color="#1a1a1a")  # ROI 外塗深灰

# 兩張圖共用同一 colorbar 動態範圍以利視覺比較
VMIN = float(masked_cost.min())
VMAX = float(masked_cost.max())


def _render_cost_map(
    arr: np.ndarray,
    out_path: str,
    fig_x: float,
    rect_xywh: tuple | None = None,
    tick_size: int = 14,
    rect_linewidth: float = 8.0,
    show_colorbar: bool = True,
) -> None:
    fig_y = (arr.shape[0] / arr.shape[1]) * fig_x
    fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))
    im = ax.imshow(arr, cmap=cmap, vmin=VMIN, vmax=VMAX)
    ax.axis("off")
    if rect_xywh is not None:
        rx, ry, rw, rh = rect_xywh
        ax.add_patch(
            Rectangle(
                (rx, ry), rw, rh,
                linewidth=rect_linewidth,
                edgecolor="red",
                facecolor="none",
            )
        )
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# 全圖:在 crop 位置畫紅框
_render_cost_map(
    masked_cost,
    "./output/viz_cost_map.png",
    fig_x=64,
    rect_xywh=(CROP_X0, CROP_Y0, CROP_W, CROP_H),
    tick_size=48,
    rect_linewidth=12.0,
)

# 局部放大:同樣 colormap 與 vmin/vmax,不畫紅框、不放 colorbar (色階從全圖讀)
_render_cost_map(
    crop(masked_cost),
    "./output/viz_cost_map_zoom.png",
    fig_x=12,
    show_colorbar=False,
)
