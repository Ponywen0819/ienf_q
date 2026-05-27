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
from neural_reconstruction.algorithms.annotation_grow.dijkstra import (
    get_components,
    multi_source_dijkstra,
)
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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-source Dijkstra expansion
# ─────────────────────────────────────────────────────────────────────────────

annotation_roi = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
annotation_bin = (annotation_roi > 127).astype(np.uint8)
annot_labeled = get_components(annotation_bin)

owner_map, dist_map, prev_y, prev_x = multi_source_dijkstra(
    cost_map=cost_map,
    annot_labeled=annot_labeled,
    connectivity=8,
    roi_mask=(roi_mask > 127),
)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation: 3-panel cropped view + full-image inset
# ─────────────────────────────────────────────────────────────────────────────


def _component_palette(n_comp: int, seed: int = 0) -> np.ndarray:
    """Distinct qualitative colours for component IDs 1..n_comp. ID 0 → black."""
    rng = np.random.default_rng(seed)
    base = np.array(
        [plt.get_cmap("tab20")(i / 20.0)[:3] for i in range(20)]
        + [plt.get_cmap("tab20b")(i / 20.0)[:3] for i in range(20)]
        + [plt.get_cmap("tab20c")(i / 20.0)[:3] for i in range(20)],
        dtype=np.float32,
    )
    if n_comp > len(base):
        extra = rng.uniform(0.15, 0.95, size=(n_comp - len(base), 3)).astype(np.float32)
        base = np.vstack([base, extra])
    colours = base[:n_comp]
    rng.shuffle(colours)
    palette = np.zeros((n_comp + 1, 3), dtype=np.float32)
    palette[1:] = colours
    return palette


def _components_overlay(green: np.ndarray, labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Solid component colours composited on top of the (cropped) green channel."""
    rgb = np.stack([green, green, green], axis=-1).astype(np.float32) / 255.0
    fg = palette[np.clip(labels, 0, len(palette) - 1)]
    mask_fg = (labels > 0)[..., None]
    return np.where(mask_fg, fg, rgb)


def _voronoi_overlay(
    green: np.ndarray,
    owner: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    roi: np.ndarray,
    expanded_alpha: float = 0.5,
) -> np.ndarray:
    """Owner map painted on top of green channel: solid for source pixels, alpha for expansion.
    Pixels outside the ROI are painted black."""
    rgb = np.stack([green, green, green], axis=-1).astype(np.float32) / 255.0
    owner_rgb = palette[np.clip(owner, 0, len(palette) - 1)]
    expanded_mask = ((owner > 0) & (labels == 0))[..., None]
    out = np.where(expanded_mask, expanded_alpha * owner_rgb + (1 - expanded_alpha) * rgb, rgb)
    source_mask = (labels > 0)[..., None]
    out = np.where(source_mask, owner_rgb, out)
    # Voronoi boundary lines (where any 4-neighbour has different non-zero owner)
    boundary = np.zeros_like(owner, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(owner, shift=(dy, dx), axis=(0, 1))
        diff = (shifted != owner) & (owner > 0) & (shifted > 0)
        boundary |= diff
    out[boundary] = (0.0, 0.0, 0.0)
    out[roi == 0] = (0.0, 0.0, 0.0)
    return out


# Crop everything we need
green_full = image  # post-Sato uint8, but used here as a faint background
green_raw = np.array(cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB))[:, :, 1]
green_crop = crop(green_raw)
labels_crop = crop(annot_labeled)
owner_crop = crop(owner_map)
dist_crop = crop(dist_map)
roi_crop = crop(roi_mask)

# Palette is sized to whole image so colours are consistent if we ever re-crop
palette = _component_palette(int(annot_labeled.max()))

panel_components = _components_overlay(green_crop, labels_crop, palette)
panel_voronoi = _voronoi_overlay(green_crop, owner_crop, labels_crop, palette, roi_crop)

# Distance heatmap: clip to ROI and to 99th percentile of finite values for contrast
dist_view = dist_crop.astype(np.float32).copy()
finite_mask = np.isfinite(dist_view) & (roi_crop > 0)
if finite_mask.any():
    vmax = float(np.percentile(dist_view[finite_mask], 99))
else:
    vmax = 1.0
dist_view[~finite_mask] = np.nan

out_dir = Path(__file__).parent
panel_size = (5.0, 5.0)

# Panel (a): components overlay
fig_a, ax_a = plt.subplots(figsize=panel_size, constrained_layout=True)
ax_a.imshow(panel_components)
ax_a.axis("off")
out_a = out_dir / "viz_region_grow_components.png"
fig_a.savefig(out_a, dpi=200, bbox_inches="tight")
plt.close(fig_a)

# Panel (b): cumulative cost heatmap
fig_b, ax_b = plt.subplots(figsize=panel_size, constrained_layout=True)
cmap_cost = plt.get_cmap("cool").copy()
cmap_cost.set_bad(color="black")
im_cost = ax_b.imshow(dist_view, cmap=cmap_cost, vmin=0.0, vmax=vmax)
ax_b.axis("off")
divider = make_axes_locatable(ax_b)
cax = divider.append_axes("right", size="4%", pad=0.04)
cbar = fig_b.colorbar(im_cost, cax=cax)
cbar.ax.tick_params(labelsize=16)
out_b = out_dir / "viz_region_grow_cost.png"
fig_b.savefig(out_b, dpi=200, bbox_inches="tight")
plt.close(fig_b)

# Panel (c): Voronoi owner map
fig_c, ax_c = plt.subplots(figsize=panel_size, constrained_layout=True)
ax_c.imshow(panel_voronoi)
ax_c.axis("off")
out_c = out_dir / "viz_region_grow_voronoi.png"
fig_c.savefig(out_c, dpi=200, bbox_inches="tight")
plt.close(fig_c)

print(f"Saved: {out_a}")
print(f"Saved: {out_b}")
print(f"Saved: {out_c}")
