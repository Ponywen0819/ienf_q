"""Visualise crossing-point detection and effective-crossing counting.

Reuses the exact crop and preprocessing pipeline of viz_region_grow.py, builds
the same pixel-level topology skeleton as viz_bridge_skeleton.py (figure 6),
then operates on it as a pixel graph (every skeleton pixel is a node, 8-adjacency
gives edges):

  viz_crossing_points.png    — crossing points: skeleton pixels adjacent to a
                               skeleton pixel of the other region (epidermis /
                               dermis, per the epidermis mask M)
  viz_crossing_effective.png — effective crossing segments coloured; a segment
                               is effective iff it crosses the DEJ and has
                               >= L_MIN pixel-path length in BOTH regions.

Region of a pixel: epidermis if M > 0, dermis otherwise.
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy.ndimage import convolve

from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.topology import TopologyBuilder
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

# ─────────────────────────────────────────────────────────────────────────────
# Config / crop (shared with viz_region_grow.py)
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

PRUNE_THRESHOLD = 20.0   # tau
DILATE_RADIUS = 1        # r_d
SEGMENT_LENGTH = 100.0   # build_seed_graph segmentation
L_MIN = 5.0              # minimum per-region path length for an effective crossing
MIN_TREE_COMPONENTS = 5  # min annotation components a subtree must cover to count
COVER_NEIGHBORHOOD = 3   # half-window when checking annotation coverage

SKELETON_COLOR = np.array([1.00, 1.00, 1.00], dtype=np.float32)   # white
EFFECTIVE_COLOR = np.array([0.20, 0.90, 0.35], dtype=np.float32)  # green
EXCLUDED_COLOR = np.array([0.55, 0.55, 0.60], dtype=np.float32)   # dim gray
CROSSING_POINT_COLOR = "#ff3030"                                  # red
ENDPOINT_COLOR = "#ff5020"   # orange-red — matches bs_6
BRANCH_COLOR = "#3399ff"     # blue — matches bs_6
EPIDERMIS_TINT = np.array([0.60, 0.45, 0.32], dtype=np.float32)   # warm


def crop(arr):
    """Return the fixed crop used by the paper visualizations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline (identical to viz_region_grow.py)
# ─────────────────────────────────────────────────────────────────────────────
green_raw = np.array(cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB))[:, :, 1]
image = green_raw.copy()
mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
annotation = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)

roi_mask = dilate_epidermis_vertically(mask, offset_px=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
image = cv2.subtract(image, background)

clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(768, 768))
image = clahe.apply(image)

image = cv2.bitwise_and(image, image, mask=roi_mask)
image = ski.filters.sato(image, sigmas=range(3, 8), black_ridges=False)
image = (image - image.min()) / (image.max() - image.min()) * 255
image = image.astype(np.uint8)
cost_map = np.exp(1.0 - (image.astype(np.float32) / 255.0)) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Dijkstra -> component graph -> tau-pruning -> MST
# ─────────────────────────────────────────────────────────────────────────────
annotation_roi = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
annotation_bin = (annotation_roi > 127).astype(np.uint8)
annot_labeled = get_components(annotation_bin)
n_components = int(annot_labeled.max())

owner_map, dist_map, prev_y, prev_x = multi_source_dijkstra(
    cost_map=cost_map,
    annot_labeled=annot_labeled,
    connectivity=8,
    roi_mask=(roi_mask > 127),
)

connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
G = build_component_graph(connections, n_components)
G_pruned = prune_edges(G, threshold=PRUNE_THRESHOLD)
mst = minimum_spanning_forest(G_pruned)


# ─────────────────────────────────────────────────────────────────────────────
# Bridge mask -> dilation -> union -> pixel-level topology skeleton
# ─────────────────────────────────────────────────────────────────────────────
H, W = annotation_bin.shape

bridge_mask = np.zeros((H, W), dtype=np.uint8)
for _, _, data in mst.edges(data=True):
    for py, px in data.get("path", []):
        if 0 <= py < H and 0 <= px < W:
            bridge_mask[py, px] = 1

ksize = DILATE_RADIUS * 2 + 1
disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
bridge_dilated = cv2.dilate(bridge_mask, disk).astype(bool)
union = (annotation_bin > 0) | bridge_dilated

seed_graph = TopologyBuilder(segment_length=SEGMENT_LENGTH).build_seed_graph(
    union.astype(np.uint8)
)
skeleton = np.zeros((H, W), dtype=bool)
for _u, _v, data in seed_graph.edges(data=True):
    for py, px in data.get("path", []):
        if 0 <= py < H and 0 <= px < W:
            skeleton[py, px] = True
for ny, nx_ in seed_graph.nodes():
    if 0 <= ny < H and 0 <= nx_ < W:
        skeleton[ny, nx_] = True


# ─────────────────────────────────────────────────────────────────────────────
# Region labelling: epidermis where the epidermis mask M > 0, dermis otherwise
# ─────────────────────────────────────────────────────────────────────────────
epi = mask > 0


# ─────────────────────────────────────────────────────────────────────────────
# Subtree annotation coverage — for each skeleton connected component (a
# reconstructed subtree), count how many distinct annotation components it
# touches (within a small neighborhood, since skeleton pixels need not land
# exactly on annotation pixels). Subtrees covering fewer than
# MIN_TREE_COMPONENTS are excluded from the crossing count, matching
# `_exclude_small_subtrees_from_count` in core/crosses_detection/pipeline.py.
# ─────────────────────────────────────────────────────────────────────────────
skel_cc_labels = ski.measure.label(skeleton, connectivity=2)
n_cc = int(skel_cc_labels.max())

cc_coverage_count: dict[int, int] = {}
for cc_id in range(1, n_cc + 1):
    ys_cc, xs_cc = np.where(skel_cc_labels == cc_id)
    covered: set[int] = set()
    for y, x in zip(ys_cc.tolist(), xs_cc.tolist()):
        y0, y1 = max(0, y - COVER_NEIGHBORHOOD), min(H, y + COVER_NEIGHBORHOOD + 1)
        x0, x1 = max(0, x - COVER_NEIGHBORHOOD), min(W, x + COVER_NEIGHBORHOOD + 1)
        covered.update(np.unique(annot_labeled[y0:y1, x0:x1]).tolist())
    covered.discard(0)
    cc_coverage_count[cc_id] = len(covered)

qualifying_ccs = {
    cc for cc, n in cc_coverage_count.items() if n >= MIN_TREE_COMPONENTS
}

# Subtree masks reused by figures 1, 2, 3.
qualifying_mask = np.isin(
    skel_cc_labels, list(qualifying_ccs) if qualifying_ccs else [-1]
)
excluded_mask = skeleton & ~qualifying_mask

# Topology markers (matches viz_bridge_skeleton bs_6): endpoints = degree-1,
# branch points = degree>=3 in the pixel-level seed graph.
endpoint_nodes = [n for n in seed_graph.nodes() if seed_graph.degree(n) == 1]
branch_nodes = [n for n in seed_graph.nodes() if seed_graph.degree(n) >= 3]


def _crop_local_xy(nodes):
    xs, ys = [], []
    for ny, nx_ in nodes:
        if CROP_Y0 <= ny < CROP_Y0 + CROP_H and CROP_X0 <= nx_ < CROP_X0 + CROP_W:
            xs.append(nx_ - CROP_X0)
            ys.append(ny - CROP_Y0)
    return xs, ys


ep_xs, ep_ys = _crop_local_xy(endpoint_nodes)
bp_xs, bp_ys = _crop_local_xy(branch_nodes)


def _draw_topology_markers(ax):
    ax.scatter(
        bp_xs, bp_ys, s=46, c=BRANCH_COLOR, edgecolors="black",
        linewidths=0.6, zorder=4,
    )
    ax.scatter(
        ep_xs, ep_ys, s=46, c=ENDPOINT_COLOR, edgecolors="black",
        linewidths=0.6, zorder=4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Crossing points — skeleton pixels 8-adjacent to a skeleton pixel of the
# other region. (A "crossing edge" between adjacent pixels is, visually, a point.)
# ─────────────────────────────────────────────────────────────────────────────
crossing_pts = np.zeros((H, W), dtype=bool)
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        if dy == 0 and dx == 0:
            continue
        sh_skel = np.roll(np.roll(skeleton, dy, 0), dx, 1)
        sh_epi = np.roll(np.roll(epi, dy, 0), dx, 1)
        crossing_pts |= skeleton & sh_skel & (epi != sh_epi)


# ─────────────────────────────────────────────────────────────────────────────
# Segment decomposition + effective-crossing classification
#   Segments = connected runs of skeleton pixels with degree <= 2 (i.e. the
#   skeleton split at branch points, degree >= 3). A segment is an effective
#   crossing iff it spans both regions AND has >= L_MIN path length in each.
# ─────────────────────────────────────────────────────────────────────────────
skel_u8 = skeleton.astype(np.uint8)
degree = convolve(skel_u8, np.ones((3, 3), np.uint8), mode="constant") - skel_u8
segment_skel = skeleton & (degree <= 2)
seg_labels = ski.measure.label(segment_skel, connectivity=2)


def _trace_segment(coords: list) -> list:
    """Order a degree-<=2 pixel run from one end to the other."""
    coord_set = set(coords)

    def nbrs(p):
        y, x = p
        return [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0) and (y + dy, x + dx) in coord_set
        ]

    start = next((p for p in coords if len(nbrs(p)) <= 1), coords[0])
    path = [start]
    visited = {start}
    cur = start
    while True:
        nxt = next((q for q in nbrs(cur) if q not in visited), None)
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return path


def _region_lengths(path: list) -> tuple:
    """Euclidean path length inside epidermis / dermis, region from step midpoint."""
    l_epi = 0.0
    l_der = 0.0
    for (y0, x0), (y1, x1) in zip(path, path[1:]):
        my = int(round((y0 + y1) / 2.0))
        mx = int(round((x0 + x1) / 2.0))
        step = float(np.hypot(y1 - y0, x1 - x0))
        if epi[my, mx]:
            l_epi += step
        else:
            l_der += step
    return l_epi, l_der


effective_ids = set()
n_crossing_segments = 0
n_excluded_by_coverage = 0
for sid in range(1, int(seg_labels.max()) + 1):
    ys, xs = np.where(seg_labels == sid)
    regs = epi[ys, xs]
    if not (regs.any() and (~regs).any()):
        continue  # does not span both regions -> not a crossing segment
    n_crossing_segments += 1
    path = _trace_segment(list(zip(ys.tolist(), xs.tolist())))
    l_epi, l_der = _region_lengths(path)
    if l_epi < L_MIN or l_der < L_MIN:
        continue
    parent_cc = int(skel_cc_labels[ys[0], xs[0]])
    if parent_cc not in qualifying_ccs:
        n_excluded_by_coverage += 1
        continue
    effective_ids.add(sid)

effective_mask = np.isin(seg_labels, list(effective_ids)) if effective_ids else np.zeros_like(skeleton)

print(
    f"skeleton px: {int(skeleton.sum())}  crossing points: {int(crossing_pts.sum())}  "
    f"crossing segments: {n_crossing_segments}  "
    f"effective (N_cross): {len(effective_ids)}  "
    f"excluded by coverage<{MIN_TREE_COMPONENTS}: {n_excluded_by_coverage}  "
    f"subtrees: {n_cc} (qualifying: {len(qualifying_ccs)})"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared crop tiles + faded background with epidermis tint
# ─────────────────────────────────────────────────────────────────────────────
green_crop = crop(green_raw)
roi_crop = crop(roi_mask)
epi_crop = crop(epi)

faded_bg = np.stack([green_crop] * 3, axis=-1).astype(np.float32) / 255.0 * 0.5
faded_bg[epi_crop] = 0.7 * faded_bg[epi_crop] + 0.3 * EPIDERMIS_TINT
faded_bg[roi_crop == 0] = 0.0


def save_fig(disp: np.ndarray, out_name: str, extra=None) -> None:
    h, w = disp.shape[:2]
    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    ax.imshow(disp)
    ax.axis("off")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    if extra is not None:
        extra(ax)
    out_path = Path(__file__).parent / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — crossing points
# Excluded-by-coverage subtrees stay dim gray (match fig 3); crossing-point
# markers use a star so they're distinct from the orange endpoint markers.
# ─────────────────────────────────────────────────────────────────────────────
disp1 = faded_bg.copy()
disp1[crop(excluded_mask)] = EXCLUDED_COLOR
disp1[crop(qualifying_mask)] = SKELETON_COLOR
cp_ys, cp_xs = np.where(crop(crossing_pts))


def _draw_crossing_points(ax):
    _draw_topology_markers(ax)
    ax.scatter(
        cp_xs, cp_ys, s=140, c=CROSSING_POINT_COLOR, marker="*",
        edgecolors="black", linewidths=0.7, zorder=5,
    )


save_fig(disp1, "viz_crossing_points.png", extra=_draw_crossing_points)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — effective crossing segments coloured, with topology markers
# ─────────────────────────────────────────────────────────────────────────────
disp2 = faded_bg.copy()
disp2[crop(excluded_mask)] = EXCLUDED_COLOR
disp2[crop(qualifying_mask)] = SKELETON_COLOR
disp2[crop(effective_mask)] = EFFECTIVE_COLOR

save_fig(disp2, "viz_crossing_effective.png", extra=_draw_topology_markers)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — subtree annotation coverage. No epidermis tint / ROI overlay
# here; the original annotation components are drawn directly so each subtree
# can be visually associated with the components it covers. Qualifying
# subtrees (>= MIN_TREE_COMPONENTS covered annotations) are drawn in white;
# excluded subtrees are drawn dim gray. Each subtree is labelled at its
# centroid with the number of annotation components it covers.
# ─────────────────────────────────────────────────────────────────────────────
ANNOT_COLOR = np.array([0.30, 0.65, 1.00], dtype=np.float32)  # blue

plain_bg = np.stack([green_crop] * 3, axis=-1).astype(np.float32) / 255.0 * 0.5
disp3 = plain_bg.copy()
disp3[crop(annotation_bin > 0)] = ANNOT_COLOR
disp3[crop(excluded_mask)] = EXCLUDED_COLOR
disp3[crop(qualifying_mask)] = SKELETON_COLOR

cc_label_positions: list[tuple[int, int, int, bool]] = []  # (x, y, count, qualifying)
for cc_id, count in cc_coverage_count.items():
    ys_cc, xs_cc = np.where(skel_cc_labels == cc_id)
    in_crop = (
        (ys_cc >= CROP_Y0)
        & (ys_cc < CROP_Y0 + CROP_H)
        & (xs_cc >= CROP_X0)
        & (xs_cc < CROP_X0 + CROP_W)
    )
    if not in_crop.any():
        continue
    cy = int(ys_cc[in_crop].mean()) - CROP_Y0
    cx = int(xs_cc[in_crop].mean()) - CROP_X0
    cc_label_positions.append((cx, cy, count, cc_id in qualifying_ccs))


def _draw_coverage_labels(ax):
    for cx, cy, count, ok in cc_label_positions:
        ax.text(
            cx, cy, str(count),
            color="white" if ok else "#ffd0d0",
            fontsize=11, fontweight="bold",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=("#228b3a" if ok else "#992020"),
                edgecolor="black", linewidth=0.5, alpha=0.85,
            ),
            zorder=5,
        )


save_fig(disp3, "viz_crossing_coverage.png", extra=_draw_coverage_labels)
