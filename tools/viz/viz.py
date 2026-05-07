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

Image.fromarray(crop(image)).save(f"./output/viz_input_crop.png")

image = image[:, :, 1]  # 只取綠色通道
Image.fromarray(image).save(f"./output/viz_original.png")
Image.fromarray(crop(image)).save(f"./output/viz_original_crop.png")
bg_image = image.copy()

mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
annotation = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)
label_img = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)

Image.fromarray(crop(mask)).save(f"./output/viz_mask_crop.png")
Image.fromarray(crop(annotation)).save(f"./output/viz_annotation_crop.png")
Image.fromarray(crop(label_img)).save(f"./output/viz_label_crop.png")


roi_mask = dilate_epidermis_vertically(mask, offset_px=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# background = ski.restoration.rolling_ball(image, radius=50)

image = cv2.subtract(image, background)
Image.fromarray(background).save(f"./output/viz_bg_only.png")
Image.fromarray(crop(background)).save(f"./output/viz_bg_only_crop.png")
Image.fromarray(image).save(f"./output/viz_bg.png")
Image.fromarray(crop(image)).save(f"./output/viz_bg_sb_crop.png")
tileGridSize = 1024
clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(tileGridSize, tileGridSize))
image = clahe.apply(image)

# image = cv2.morphologyEx(
#     image,
#     cv2.MORPH_OPEN,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
#     iterations=2,
# )
Image.fromarray(crop(image)).save(f"./output/viz_enhenced_crop.png")

Image.fromarray(image).save(f"./output/viz_contrast_enhanced.png")

# Snapshot of the contrast-enhanced grayscale image, used as the base
# canvas for every overlay produced later in this script.

viz_img = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2RGB)
orig_mask_bin = mask > 127
grown_bin = (roi_mask > 0) & ~orig_mask_bin
tint = viz_img.copy()
tint[orig_mask_bin] = (255, 255, 0)
tint[grown_bin] = (0, 255, 0)
viz_img = cv2.addWeighted(tint, 0.45, viz_img, 0.55, 0)

Image.fromarray(viz_img).save(f"./output/viz_mask_overlay.png")
Image.fromarray(crop(viz_img)).save(f"./output/viz_mask_overlay_crop.png")

# Unrestricted dilation: plain circular dilation without vertical clipping.
unrestricted_offset_px = 50
_d = 2 * unrestricted_offset_px + 1
_kernel_unrestricted = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_d, _d))
unrestricted_roi_mask = cv2.dilate(mask, _kernel_unrestricted, iterations=1)
unrestricted_grown_bin = (unrestricted_roi_mask > 0) & ~orig_mask_bin

viz_img_unrestricted = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2RGB)
tint_unrestricted = viz_img_unrestricted.copy()
tint_unrestricted[orig_mask_bin] = (255, 255, 0)
tint_unrestricted[unrestricted_grown_bin] = (0, 255, 255)
viz_img_unrestricted = cv2.addWeighted(
    tint_unrestricted, 0.45, viz_img_unrestricted, 0.55, 0
)

Image.fromarray(viz_img_unrestricted).save("./output/viz_mask_overlay_unrestricted.png")
Image.fromarray(crop(viz_img_unrestricted)).save(
    "./output/viz_mask_overlay_unrestricted_crop.png"
)

# Side-by-side comparison: orig (yellow), restricted-only growth (green),
# unrestricted-only growth that the restriction trimmed away (red),
# growth shared by both (cyan).
restricted_only = grown_bin & ~unrestricted_grown_bin
unrestricted_only = unrestricted_grown_bin & ~grown_bin
both_grown = grown_bin & unrestricted_grown_bin

viz_img_compare = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2RGB)
tint_compare = viz_img_compare.copy()
tint_compare[orig_mask_bin] = (255, 255, 0)
tint_compare[both_grown] = (0, 255, 255)
tint_compare[restricted_only] = (0, 255, 0)
tint_compare[unrestricted_only] = (255, 50, 50)
viz_img_compare = cv2.addWeighted(tint_compare, 0.45, viz_img_compare, 0.55, 0)

Image.fromarray(viz_img_compare).save("./output/viz_mask_overlay_compare.png")
Image.fromarray(crop(viz_img_compare)).save(
    "./output/viz_mask_overlay_compare_crop.png"
)

image = cv2.bitwise_and(image, image, mask=roi_mask)

Image.fromarray(image).save(f"./output/viz_roi_image.png")
Image.fromarray(crop(image)).save(f"./output/viz_roi_image_crop.png")

start_sato = ski.filters.sato(image, sigmas=[3], black_ridges=False)
start_sato = (
    (start_sato - start_sato.min()) / (start_sato.max() - start_sato.min()) * 255
)
start_sato = start_sato.astype(np.uint8)
mid_sato = ski.filters.sato(image, sigmas=[5], black_ridges=False)
mid_sato = (mid_sato - mid_sato.min()) / (mid_sato.max() - mid_sato.min()) * 255
mid_sato = mid_sato.astype(np.uint8)
end_sato = ski.filters.sato(image, sigmas=[8], black_ridges=False)
end_sato = (end_sato - end_sato.min()) / (end_sato.max() - end_sato.min()) * 255
end_sato = end_sato.astype(np.uint8)

image = ski.filters.sato(image, sigmas=range(3, 8), black_ridges=False)


# image = ski.filters.meijering(image, sigmas=range(3, 12), black_ridges=False)
image = (image - image.min()) / (image.max() - image.min()) * 255
image = image.astype(np.uint8)


Image.fromarray(image).save(f"./output/viz_sato.png")

Image.fromarray(crop(image)).save(f"./output/viz_sato_crop.png")
Image.fromarray(crop(start_sato)).save(f"./output/viz_sato_start_crop.png")
Image.fromarray(crop(mid_sato)).save(f"./output/viz_sato_mid_crop.png")
Image.fromarray(crop(end_sato)).save(f"./output/viz_sato_end_crop.png")

cost_map = np.exp(1.0 - (image.astype(np.float32) / 255.0)) - 1.0

fig_x = 64
fig_y = (cost_map.shape[0] / cost_map.shape[1]) * fig_x
fig, axes = plt.subplots(1, 1, figsize=(fig_x, fig_y))
axes.imshow(cost_map)
# add colorbar
cbar = plt.colorbar(axes.imshow(cost_map), ax=axes, fraction=0.046, pad=0.04)
axes.axis("off")
plt.tight_layout()
plt.savefig(f"./output/viz_cost_map.png", dpi=300)


# ── Multi-source Dijkstra with snapshots ────────────────────────────────
annotation_bin = (annotation > 127).astype(np.uint8)
annotation_bin = cv2.bitwise_and(annotation_bin, annotation_bin, mask=roi_mask)
annot_labeled = get_components(annotation_bin)
n_components = int(annot_labeled.max())
print(f"Annotation components: {n_components}")


def multi_source_dijkstra_snapshots(
    cost_map,
    annot_labeled,
    connectivity=8,
    snapshot_fractions=(0.0, 0.2, 1.0),
    total_target=None,
):
    """Dijkstra that records owner_map copies at given progress fractions."""
    H, W = cost_map.shape
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    dist_map = np.full((H, W), np.inf, dtype=np.float32)
    owner_map = np.zeros((H, W), dtype=np.int32)
    prev_y = np.full((H, W), -1, dtype=np.int32)
    prev_x = np.full((H, W), -1, dtype=np.int32)

    heap = []
    ys, xs = np.where(annot_labeled > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cid = int(annot_labeled[y, x])
        dist_map[y, x] = 0.0
        owner_map[y, x] = cid
        heapq.heappush(heap, (0.0, y, x, cid))

    if total_target is None:
        total_target = H * W
    snapshot_targets = [max(1, int(f * total_target)) for f in snapshot_fractions]
    snapshots = []
    snap_idx = 0
    assigned = int((owner_map > 0).sum())

    while heap:
        d, y, x, cid = heapq.heappop(heap)
        if d > dist_map[y, x]:
            continue
        for dy, dx in neighbors:
            ny, nxc = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nxc < W):
                continue
            new_dist = d + float(cost_map[ny, nxc])
            if new_dist < dist_map[ny, nxc]:
                if owner_map[ny, nxc] == 0:
                    assigned += 1
                dist_map[ny, nxc] = new_dist
                owner_map[ny, nxc] = cid
                prev_y[ny, nxc] = y
                prev_x[ny, nxc] = x
                heapq.heappush(heap, (new_dist, ny, nxc, cid))
                while (
                    snap_idx < len(snapshot_targets)
                    and assigned >= snapshot_targets[snap_idx]
                ):
                    snapshots.append(owner_map.copy())
                    snap_idx += 1

    while len(snapshots) < len(snapshot_targets):
        snapshots.append(owner_map.copy())

    return owner_map, dist_map, prev_y, prev_x, snapshots


roi_pixel_count = int((roi_mask > 0).sum())
snapshot_fractions = (0.0, 0.2, 1.0)
owner_map, dist_map, prev_y, prev_x, snapshots = multi_source_dijkstra_snapshots(
    cost_map,
    annot_labeled,
    connectivity=8,
    snapshot_fractions=snapshot_fractions,
    total_target=roi_pixel_count,
)
print(f"Captured {len(snapshots)} snapshots")


def cost_map_to_gray(cost_map, roi_mask=None):
    """Normalize cost_map to a uint8 grayscale RGB background."""
    cm = cost_map.astype(np.float32)
    if roi_mask is not None:
        inside = cm[roi_mask > 0]
        lo, hi = float(inside.min()), float(inside.max())
    else:
        lo, hi = float(cm.min()), float(cm.max())
    if hi <= lo:
        hi = lo + 1.0
    gray = 1 - np.clip((cm - lo) / (hi - lo), 0.0, 1.0)
    gray = (gray * 255).astype(np.uint8)
    if roi_mask is not None:
        gray[roi_mask == 0] = 0
    return np.stack([gray] * 3, axis=-1)


def image_to_gray_bg(bg, roi_mask=None):
    """Wrap a uint8 grayscale image into an RGB background, zeroing non-ROI."""
    gray = bg.astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1).copy()
    if roi_mask is not None:
        rgb[roi_mask == 0] = 0
    return rgb


def make_border_overlay(
    owner_map, cost_map, annot_labeled=None, roi_mask=None, seed=42
):
    """
    Draw per-component region borders over a grayscale cost_map background.
    If annot_labeled is given, paint annotation pixels as solid regions
    using the same per-component color scheme.
    """
    n = int(owner_map.max())
    if annot_labeled is not None:
        n = max(n, int(annot_labeled.max()))
    rng = np.random.default_rng(seed)
    colors = rng.integers(60, 255, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = 0

    # Boundary: owned pixel whose owner differs from any 4-neighbor (incl. unowned).
    o = owner_map
    boundary = np.zeros_like(o, dtype=bool)
    boundary[:-1, :] |= o[:-1, :] != o[1:, :]
    boundary[1:, :] |= o[1:, :] != o[:-1, :]
    boundary[:, :-1] |= o[:, :-1] != o[:, 1:]
    boundary[:, 1:] |= o[:, 1:] != o[:, :-1]
    boundary &= o > 0
    if roi_mask is not None:
        boundary &= roi_mask > 0

    rgb = image_to_gray_bg(bg_image, roi_mask=roi_mask)
    rgb[boundary] = colors[o[boundary]]

    if annot_labeled is not None:
        annot_mask = annot_labeled > 0
        rgb[annot_mask] = colors[annot_labeled[annot_mask]]

    return rgb


def make_owner_voronoi_crop(
    owner_map,
    annot_labeled,
    bg_image,
    roi_mask,
    seed=42,
    alpha=0.52,
):
    """Fill the fixed crop by final Dijkstra owner, i.e. cost-metric Voronoi."""
    owner_c = crop(owner_map)
    annot_c = crop(annot_labeled)
    roi_c = crop(roi_mask) > 0
    bg_c = crop(bg_image)

    n = max(int(owner_map.max()), int(annot_labeled.max()))
    rng = np.random.default_rng(seed)
    colors = rng.integers(45, 245, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = 0

    rgb = np.stack([bg_c] * 3, axis=-1).astype(np.float32)
    color_img = colors[owner_c].astype(np.float32)
    owned = (owner_c > 0) & roi_c
    rgb[owned] = (1.0 - alpha) * rgb[owned] + alpha * color_img[owned]

    boundary = np.zeros_like(owner_c, dtype=bool)
    boundary[:-1, :] |= owner_c[:-1, :] != owner_c[1:, :]
    boundary[1:, :] |= owner_c[1:, :] != owner_c[:-1, :]
    boundary[:, :-1] |= owner_c[:, :-1] != owner_c[:, 1:]
    boundary[:, 1:] |= owner_c[:, 1:] != owner_c[:, :-1]
    boundary &= owned
    rgb[boundary] = (255, 255, 255)

    annot_mask = annot_c > 0
    rgb[annot_mask] = colors[annot_c[annot_mask]]
    rgb[~roi_c] *= 0.25

    return np.clip(rgb, 0, 255).astype(np.uint8)


stage_names = ["begin", "middle", "end"]
for stage, frac, snap in zip(stage_names, snapshot_fractions, snapshots):
    rgb = make_border_overlay(
        snap, cost_map, annot_labeled=annot_labeled, roi_mask=roi_mask
    )
    Image.fromarray(rgb).save(f"./output/viz_dijkstra_{stage}.png")
    Image.fromarray(crop(rgb)).save(f"./output/viz_dijkstra_{stage}_crop.png")
    print(f"  {stage} (~{int(frac * 100)}%): owned={(snap > 0).sum()}")

owner_voronoi_crop = make_owner_voronoi_crop(
    owner_map,
    annot_labeled,
    bg_image,
    roi_mask,
)
Image.fromarray(owner_voronoi_crop).save("./output/viz_owner_voronoi_crop.png")


# ── Meeting points ──────────────────────────────────────────────────────
connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
print(f"Meeting pairs: {len(connections)}")

centroids = {}
for p in regionprops(annot_labeled):
    cy, cx = p.centroid
    centroids[p.label] = (int(cy), int(cx))


def filter_by_crop(connections, y0, x0, h, w):
    """Keep only edges whose meeting point falls inside the crop window."""
    y1, x1 = y0 + h, x0 + w
    return {
        k: v
        for k, v in connections.items()
        if (
            (y0 <= v["y"] < y1 and x0 <= v["x"] < x1)
            or (y0 <= v["y_b"] < y1 and x0 <= v["x_b"] < x1)
        )
    }


def backtrack_path(y, x, prev_y, prev_x):
    """Trace one side of a meeting point back to its Dijkstra seed."""
    path = []
    while y >= 0 and x >= 0:
        path.append((int(y), int(x)))
        py, px = int(prev_y[y, x]), int(prev_x[y, x])
        if py < 0:
            break
        y, x = py, px
    return path


def points_to_crop_xy(path):
    """Convert global (y, x) path points to crop-local OpenCV (x, y)."""
    pts = []
    y1, x1 = CROP_Y0 + CROP_H, CROP_X0 + CROP_W
    for y, x in path:
        if CROP_Y0 <= y < y1 and CROP_X0 <= x < x1:
            pts.append((int(x - CROP_X0), int(y - CROP_Y0)))
    return pts


def draw_crop_path_with_arrow(rgb, path, color, thickness=1):
    """Draw the visible crop portion of a backtrack path with an arrow to seed."""
    pts = points_to_crop_xy(path)
    if len(pts) < 2:
        return
    pts_np = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(rgb, [pts_np], False, color, thickness, cv2.LINE_AA)
    cv2.arrowedLine(
        rgb,
        pts[-2],
        pts[-1],
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.28,
    )


def make_meeting_backtrack_crop(
    connections,
    owner_map,
    annot_labeled,
    bg_image,
    roi_mask,
    prev_y,
    prev_x,
):
    """Show lowest-cost adjacent meeting pairs and their backtracked paths."""
    bg_c = crop(bg_image)
    roi_c = crop(roi_mask) > 0
    annot_c = crop(annot_labeled)
    rgb = np.stack([bg_c] * 3, axis=-1).copy()
    rgb[~roi_c] = (0, 0, 0)
    n = int(annot_labeled.max())
    rng = np.random.default_rng(42)
    colors = rng.integers(45, 245, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = 0
    annot_mask = annot_c > 0
    rgb[annot_mask] = colors[annot_c[annot_mask]]

    if len(connections) == 0:
        return rgb

    costs = np.array([v["cost"] for v in connections.values()], dtype=np.float32)
    order = np.argsort(costs)
    items = list(connections.items())

    for idx in order:
        (_, _), info = items[idx]
        path_a = backtrack_path(info["y"], info["x"], prev_y, prev_x)
        path_b = backtrack_path(info["y_b"], info["x_b"], prev_y, prev_x)

        draw_crop_path_with_arrow(rgb, path_a, (255, 80, 40), thickness=1)
        draw_crop_path_with_arrow(rgb, path_b, (40, 210, 255), thickness=1)

        pa = (int(info["x"] - CROP_X0), int(info["y"] - CROP_Y0))
        pb = (int(info["x_b"] - CROP_X0), int(info["y_b"] - CROP_Y0))
        # for p, color in ((pa, (255, 255, 0)), (pb, (255, 255, 255))):
        #     if 0 <= p[0] < CROP_W and 0 <= p[1] < CROP_H:
        #         cv2.circle(rgb, p, 1, color, -1, cv2.LINE_AA)
        #         cv2.circle(rgb, p, 2, (0, 0, 0), 1, cv2.LINE_AA)
        if (
            0 <= pa[0] < CROP_W
            and 0 <= pa[1] < CROP_H
            and 0 <= pb[0] < CROP_W
            and 0 <= pb[1] < CROP_H
        ):
            cv2.line(rgb, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)

    return rgb


def make_meeting_overlay(
    connections,
    centroids,
    cost_map,
    annot_labeled=None,
    roi_mask=None,
    cost_cap=None,
    line_thickness=1,
    dot_radius=1,
    annot_seed=42,
):
    """
    Draw meeting-point edges over grayscale cost_map.

    Each edge is drawn as centroid_A → meeting_A → meeting_B → centroid_B,
    so the line actually passes through the meeting pixel.
    Edge color encodes cost via 'viridis' (low=purple, high=yellow).
    If annot_labeled is given, paint annotation pixels as solid regions
    using the same per-component color scheme as make_border_overlay.
    """
    rgb = image_to_gray_bg(bg_image, roi_mask=roi_mask)

    if len(connections) == 0:
        return rgb

    costs = np.array([v["cost"] for v in connections.values()], dtype=np.float32)
    if cost_cap is None:
        cost_cap = float(np.percentile(costs, 99))
    cost_cap = max(cost_cap, float(costs.min()) + 1e-6)
    # cmap = mcm.get_cmap("viridis")

    # Sort so expensive edges render on top (visible even when overlapping).
    order = np.argsort(costs)
    items = list(connections.items())
    for idx in order:
        (a, b), info = items[idx]
        if a not in centroids or b not in centroids:
            continue
        ya, xa = centroids[a]
        yb, xb = centroids[b]
        my, mx = int(info["y"]), int(info["x"])
        myb, mxb = int(info["y_b"]), int(info["x_b"])

        t = float(np.clip(info["cost"] / cost_cap, 0.0, 1.0))
        # color = tuple(int(c * 255) for c in cmap(t)[:3])
        color = (20, 200, 50)

        cv2.line(rgb, (xa, ya), (mx, my), color, line_thickness, cv2.LINE_AA)
        cv2.line(rgb, (mxb, myb), (xb, yb), color, line_thickness, cv2.LINE_AA)
        cv2.circle(rgb, (mx, my), dot_radius, (255, 255, 0), -1)

    if annot_labeled is not None:
        n = int(annot_labeled.max())
        rng = np.random.default_rng(annot_seed)
        colors = rng.integers(60, 255, size=(n + 1, 3), dtype=np.uint8)
        colors[0] = 0
        annot_mask = annot_labeled > 0
        rgb[annot_mask] = colors[annot_labeled[annot_mask]]

    return rgb


# Only keep edges whose meeting point lies inside the crop window.
connections_crop = filter_by_crop(connections, CROP_Y0, CROP_X0, CROP_H, CROP_W)
print(f"Meeting pairs inside crop: {len(connections_crop)} / {len(connections)}")

meeting_backtrack_crop = make_meeting_backtrack_crop(
    connections_crop,
    owner_map,
    annot_labeled,
    bg_image,
    roi_mask,
    prev_y,
    prev_x,
)
Image.fromarray(meeting_backtrack_crop).save("./output/viz_meeting_backtrack_crop.png")

# Shared color range so both images use the same scale.
all_costs = np.array([v["cost"] for v in connections_crop.values()], dtype=np.float32)
cost_cap = float(np.percentile(all_costs, 99)) if len(all_costs) else 1.0

rgb_all = make_meeting_overlay(
    connections_crop,
    centroids,
    cost_map,
    annot_labeled=annot_labeled,
    roi_mask=roi_mask,
    cost_cap=cost_cap,
)
Image.fromarray(rgb_all).save("./output/viz_meeting_all.png")
Image.fromarray(crop(rgb_all)).save("./output/viz_meeting_all_crop.png")

prune_threshold = 20.0
connections_kept = {
    k: v for k, v in connections_crop.items() if v["cost"] <= prune_threshold
}
print(
    f"Filtered pairs inside crop (cost <= {prune_threshold}): "
    f"{len(connections_kept)} / {len(connections_crop)}"
)

rgb_kept = make_meeting_overlay(
    connections_kept,
    centroids,
    cost_map,
    annot_labeled=annot_labeled,
    roi_mask=roi_mask,
    cost_cap=cost_cap,
)
Image.fromarray(rgb_kept).save("./output/viz_meeting_filtered.png")
Image.fromarray(crop(rgb_kept)).save("./output/viz_meeting_filtered_crop.png")


# ── Component graph paths ───────────────────────────────────────────────
G = build_component_graph(connections_kept, n_components)
G = nx.minimum_spanning_tree(G, weight="cost")

print(f"Component graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def make_path_overlay(
    G,
    cost_map,
    annot_labeled=None,
    roi_mask=None,
    path_color=(0, 255, 0),
    annot_seed=42,
):
    """
    Draw per-edge Dijkstra-backtracked paths in a single color (default green)
    over grayscale cost_map. Annotation pixels are painted solid on top
    using the same seeded color scheme as the other overlays.
    """
    rgb = image_to_gray_bg(bg_image, roi_mask=roi_mask)
    H, W = rgb.shape[:2]

    path_mask = np.zeros((H, W), dtype=np.uint8)
    for _, _, data in G.edges(data=True):
        path = data.get("path", [])
        if not path:
            continue
        pts = np.asarray(path, dtype=np.int32)
        ys_p, xs_p = pts[:, 0], pts[:, 1]
        valid = (ys_p >= 0) & (ys_p < H) & (xs_p >= 0) & (xs_p < W)
        path_mask[ys_p[valid], xs_p[valid]] = 1

    # Dilate path by 3px so it's visible at full resolution.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    path_mask = cv2.dilate(path_mask, kernel, iterations=1)
    rgb[path_mask > 0] = path_color

    if annot_labeled is not None:
        n = int(annot_labeled.max())
        rng = np.random.default_rng(annot_seed)
        colors = rng.integers(60, 255, size=(n + 1, 3), dtype=np.uint8)
        colors[0] = 0
        annot_mask = annot_labeled > 0
        rgb[annot_mask] = colors[annot_labeled[annot_mask]]

    return rgb


rgb_paths = make_path_overlay(
    G, cost_map, annot_labeled=annot_labeled, roi_mask=roi_mask
)
Image.fromarray(rgb_paths).save("./output/viz_component_graph.png")
Image.fromarray(crop(rgb_paths)).save("./output/viz_component_graph_crop.png")


# ── Complete result graph (prune + MST + skeleton) ─────────────────────
G_full = build_component_graph(connections, n_components)
G_pruned = prune_edges(G_full, threshold=20.0)
print(
    f"Pruned component graph: {G_pruned.number_of_edges()} edges "
    f"(was {G_full.number_of_edges()})"
)

mst = minimum_spanning_forest(G_pruned)


def _path_length(data, u, v):
    path = data.get("path", [u, v])
    return len(path) - 1  # number of pixel steps


print(f"MST: {mst.number_of_edges()} edges")

result_graph = build_result_graph(mst, annotation_bin, segment_length=100.0)
print(
    f"Result graph: {result_graph.number_of_nodes()} nodes, "
    f"{result_graph.number_of_edges()} edges"
)


def make_graph_overlay(
    graph,
    cost_map,
    annot_labeled=None,
    roi_mask=None,
    edge_color=(0, 255, 255),
    node_color=(255, 50, 50),
    edge_dilate=1,
    node_radius=2,
    annot_seed=42,
):
    """
    Draw a pixel-level graph (nodes keyed by (y, x)) over grayscale cost_map.
    Edges (with 'path' attribute) are drawn dilated; nodes as filled circles.
    Annotation is overlaid on top using the shared seeded color scheme.
    """
    rgb = image_to_gray_bg(bg_image, roi_mask=roi_mask)
    H, W = rgb.shape[:2]

    edge_mask = np.zeros((H, W), dtype=np.uint8)
    for _, _, data in graph.edges(data=True):
        path = data.get("path", [])

        pts = np.asarray(path, dtype=np.int32)
        ys_p, xs_p = pts[:, 0], pts[:, 1]
        valid = (ys_p >= 0) & (ys_p < H) & (xs_p >= 0) & (xs_p < W)
        edge_mask[ys_p[valid], xs_p[valid]] = 1
    if edge_dilate > 0:
        ksize = edge_dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        edge_mask = cv2.dilate(edge_mask, kernel)
    rgb[edge_mask > 0] = edge_color

    # if annot_labeled is not None:
    #     n = int(annot_labeled.max())
    #     rng = np.random.default_rng(annot_seed)
    #     colors = rng.integers(60, 255, size=(n + 1, 3), dtype=np.uint8)
    #     colors[0] = 0
    #     annot_mask = annot_labeled > 0
    #     rgb[annot_mask] = colors[annot_labeled[annot_mask]]

    for node in graph.nodes:
        y, x = node
        if 0 <= y < H and 0 <= x < W:
            cv2.circle(rgb, (int(x), int(y)), node_radius, node_color, -1)

    return rgb


rgb_final = make_graph_overlay(
    result_graph, cost_map, annot_labeled=annot_labeled, roi_mask=roi_mask
)
Image.fromarray(rgb_final).save("./output/viz_result_graph.png")
Image.fromarray(crop(rgb_final)).save("./output/viz_result_graph_crop.png")


# ── Crossing detection (segment / region / effective count) ────────────
g_segmented = SegmentDetector.detect_segments(result_graph)

seg_edges = defaultdict(list)
for u, v, data in g_segmented.edges(data=True):
    seg_id = data.get("segment_id")
    if seg_id is not None:
        seg_edges[seg_id].append((u, v, data))

edges_to_remove = []
for seg_id, edges in seg_edges.items():
    # Collect boundary nodes of this segment
    boundary_nodes = set()
    for u, v, _ in edges:
        if g_segmented.nodes[u].get("node_type") in (
            "endpoint",
            "branchpoint",
        ):
            boundary_nodes.add(u)
        if g_segmented.nodes[v].get("node_type") in (
            "endpoint",
            "branchpoint",
        ):
            boundary_nodes.add(v)

    # Only prune if at least one boundary node is an endpoint (dangling stub)
    has_endpoint = any(
        g_segmented.nodes[n].get("node_type") == "endpoint" for n in boundary_nodes
    )
    if not has_endpoint:
        continue

    total_length = sum(_path_length(data, u, v) for u, v, data in edges)
    if total_length < 5:
        edges_to_remove.extend((u, v) for u, v, _ in edges)
g_segmented.remove_edges_from(edges_to_remove)
g_segmented.remove_nodes_from(list(nx.isolates(g_segmented)))
g_labeled, n_crossing_segments = RegionLabeler().label_topology(g_segmented, mask)
count_result = CrossingCounter().count_effective_crossings(
    g_labeled, epidermis_mask=mask, min_region_length=5.0
)
print(CrossingCounter().get_crossing_summary(count_result))


def _draw_edge_path(rgb, path, color, thickness=2):
    """Stroke a polyline along an edge's path list."""
    if len(path) < 2:
        return
    pts = np.asarray(path, dtype=np.int32)[:, ::-1]  # (y,x) -> (x,y)
    cv2.polylines(rgb, [pts], False, color, thickness, cv2.LINE_AA)


def _blend_epidermis_tint(rgb, epidermis_mask, tint=(255, 180, 120), alpha=0.18):
    """Lightly tint the epidermis region and outline its boundary."""
    if epidermis_mask is None:
        return rgb
    epi = epidermis_mask > 127
    overlay = rgb.copy()
    overlay[epi] = tint
    rgb = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)
    boundary = cv2.morphologyEx(
        epi.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    rgb[boundary > 0] = (255, 255, 255)
    return rgb


def make_segments_overlay(
    graph,
    cost_map,
    epidermis_mask=None,
    roi_mask=None,
    edge_thickness=2,
    node_radius=3,
    seed=42,
):
    """
    Colorize each segment (unique segment_id) with a distinct color.
    Endpoints drawn as green dots, branchpoints as magenta triangles.
    """
    rgb = image_to_gray_bg(bg_image, roi_mask=roi_mask)
    rgb = _blend_epidermis_tint(rgb, epidermis_mask)
    H, W = rgb.shape[:2]

    seg_ids = [
        d.get("segment_id")
        for _, _, d in graph.edges(data=True)
        if d.get("segment_id") is not None
    ]
    max_seg = max(seg_ids) if seg_ids else 0
    rng = np.random.default_rng(seed)
    seg_colors = rng.integers(60, 255, size=(max_seg + 2, 3), dtype=np.uint8)

    for u, v, data in graph.edges(data=True):
        seg = data.get("segment_id")
        if seg is None:
            continue
        color = tuple(int(c) for c in seg_colors[seg])
        _draw_edge_path(rgb, data.get("path", [u, v]), color, edge_thickness)

    for node, data in graph.nodes(data=True):
        y, x = node
        if not (0 <= y < H and 0 <= x < W):
            continue
        ntype = data.get("node_type")
        if ntype == "endpoint":
            cv2.circle(rgb, (int(x), int(y)), node_radius, (50, 255, 50), -1)
        elif ntype == "branchpoint":
            cv2.drawMarker(
                rgb,
                (int(x), int(y)),
                (255, 50, 255),
                cv2.MARKER_TRIANGLE_UP,
                node_radius * 3,
                thickness=2,
            )
    return rgb


def make_region_overlay(
    graph,
    cost_map,
    epidermis_mask=None,
    roi_mask=None,
    edge_thickness=2,
    node_radius=3,
):
    """
    Show region labeling: epidermis tint + crossing edges in red,
    non-crossing edges in light gray. Nodes colored by region.
    """
    rgb = image_to_gray_bg(bg_image, roi_mask=roi_mask)
    rgb = _blend_epidermis_tint(rgb, epidermis_mask)
    H, W = rgb.shape[:2]

    for u, v, data in graph.edges(data=True):
        color = (50, 50, 255) if data.get("is_crossing") else (170, 170, 170)
        _draw_edge_path(rgb, data.get("path", [u, v]), color, edge_thickness)

    for node, data in graph.nodes(data=True):
        y, x = node
        if not (0 <= y < H and 0 <= x < W):
            continue
        region = data.get("region", "dermis")
        color = (255, 120, 60) if region == "epidermis" else (60, 180, 255)
        cv2.circle(rgb, (int(x), int(y)), node_radius, color, -1)
    return rgb


def make_effective_crossing_overlay(
    graph,
    count_result,
    cost_map,
    epidermis_mask=None,
    roi_mask=None,
    edge_thickness=2,
    node_radius=3,
):
    """
    Highlight effective-crossing segments.
    - valid crossing segment (counted):   green
    - invalid crossing segment (too short in one region): orange
    - non-crossing segment:               light gray
    Crossing edge midpoints marked cyan; valid-segment centroids marked yellow.
    """
    rgb = image_to_gray_bg(bg_image, roi_mask=roi_mask)
    rgb = _blend_epidermis_tint(rgb, epidermis_mask)
    H, W = rgb.shape[:2]

    seg_status = {d["segment_id"]: d for d in count_result["segment_details"]}

    edges_by_seg = {}
    for u, v, data in graph.edges(data=True):
        seg = data.get("segment_id")
        if seg is None:
            continue
        edges_by_seg.setdefault(seg, []).append((u, v, data))

    for seg_id, edges in edges_by_seg.items():
        info = seg_status.get(seg_id)
        if info is None:
            color = (170, 170, 170)
        elif info["is_valid"]:
            color = (50, 255, 50)
        elif info["has_crossing"]:
            color = (255, 150, 50)
        else:
            color = (170, 170, 170)
        for u, v, data in edges:
            _draw_edge_path(rgb, data.get("path", [u, v]), color, edge_thickness)

    for u, v, data in graph.edges(data=True):
        if not data.get("is_crossing"):
            continue
        path = data.get("path", [u, v])
        my, mx = path[len(path) // 2]
        if 0 <= my < H and 0 <= mx < W:
            cv2.circle(rgb, (int(mx), int(my)), 3, (0, 255, 255), -1)

    for seg_id, edges in edges_by_seg.items():
        info = seg_status.get(seg_id)
        if info is None or not info["is_valid"]:
            continue
        all_pts = []
        for u, v, data in edges:
            all_pts.extend(data.get("path", [u, v]))
        if not all_pts:
            continue
        pts = np.asarray(all_pts)
        cy, cx = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        if 0 <= cy < H and 0 <= cx < W:
            cv2.drawMarker(
                rgb,
                (cx, cy),
                (255, 255, 0),
                cv2.MARKER_STAR,
                12,
                thickness=2,
            )
    return rgb


rgb_segments = make_segments_overlay(
    g_segmented, cost_map, epidermis_mask=mask, roi_mask=roi_mask
)
Image.fromarray(rgb_segments).save("./output/viz_segments.png")
Image.fromarray(crop(rgb_segments)).save("./output/viz_segments_crop.png")


def make_valid_segments_overlay(
    graph,
    count_result,
    cost_map,
    epidermis_mask=None,
    roi_mask=None,
    edge_thickness=2,
    node_radius=3,
    seed=42,
):
    """
    Same per-segment random coloring as make_segments_overlay,
    but only draws segments marked as valid (counted as effective
    crossing) in count_result. Non-valid segments stay on the
    grayscale cost_map background.
    """
    rgb = image_to_gray_bg(bg_image, roi_mask=roi_mask)
    rgb = _blend_epidermis_tint(rgb, epidermis_mask)
    H, W = rgb.shape[:2]

    valid_seg_ids = {
        d["segment_id"] for d in count_result["segment_details"] if d.get("is_valid")
    }

    seg_ids = [
        d.get("segment_id")
        for _, _, d in graph.edges(data=True)
        if d.get("segment_id") is not None
    ]
    max_seg = max(seg_ids) if seg_ids else 0
    rng = np.random.default_rng(seed)
    seg_colors = rng.integers(60, 255, size=(max_seg + 2, 3), dtype=np.uint8)

    invalid_color = (170, 170, 170)

    # Draw invalid segments first so valid ones render on top.
    valid_edges = []
    for u, v, data in graph.edges(data=True):
        seg = data.get("segment_id")
        if seg is None:
            continue
        if seg in valid_seg_ids:
            valid_edges.append((u, v, data, seg))
        else:
            _draw_edge_path(
                rgb, data.get("path", [u, v]), invalid_color, edge_thickness
            )

    valid_nodes = set()
    for u, v, data, seg in valid_edges:
        color = tuple(int(c) for c in seg_colors[seg])
        _draw_edge_path(rgb, data.get("path", [u, v]), color, edge_thickness)
        valid_nodes.add(u)
        valid_nodes.add(v)

    for node in valid_nodes:
        data = graph.nodes[node]
        y, x = node
        if not (0 <= y < H and 0 <= x < W):
            continue
        ntype = data.get("node_type")
        if ntype == "endpoint":
            cv2.circle(rgb, (int(x), int(y)), node_radius, (50, 255, 50), -1)
        elif ntype == "branchpoint":
            cv2.drawMarker(
                rgb,
                (int(x), int(y)),
                (255, 50, 255),
                cv2.MARKER_TRIANGLE_UP,
                node_radius * 3,
                thickness=2,
            )
    return rgb


rgb_regions = make_region_overlay(
    g_labeled, cost_map, epidermis_mask=mask, roi_mask=roi_mask
)
Image.fromarray(rgb_regions).save("./output/viz_regions.png")
Image.fromarray(crop(rgb_regions)).save("./output/viz_regions_crop.png")

rgb_cross = make_effective_crossing_overlay(
    g_labeled, count_result, cost_map, epidermis_mask=mask, roi_mask=roi_mask
)
Image.fromarray(rgb_cross).save("./output/viz_effective_crossings.png")
Image.fromarray(crop(rgb_cross)).save("./output/viz_effective_crossings_crop.png")

rgb_valid_segments = make_valid_segments_overlay(
    g_labeled, count_result, image, epidermis_mask=mask, roi_mask=roi_mask
)
Image.fromarray(rgb_valid_segments).save("./output/viz_valid_segments.png")
Image.fromarray(crop(rgb_valid_segments)).save("./output/viz_valid_segments_crop.png")

print(
    f"Crossing-segment count (raw): {n_crossing_segments}, "
    f"effective: {count_result['effective_crossing_count']}"
)
