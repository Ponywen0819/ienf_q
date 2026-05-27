"""Full-image crossing visualisation — Pure-MST reconstruction variant.

Companion to viz_crossing_full.py for A/B comparison. Everything is held fixed
between the two scripts (preprocessing, crossing/effective-crossing logic,
figures) EXCEPT the reconstruction algorithm:

  viz_crossing_full.py      — annotation-grow: multi-source Dijkstra meeting
                              points -> component graph -> tau-prune -> MST,
                              then bridges dilated + unioned with the
                              annotation and re-skeletonised.
  viz_crossing_full_mst.py  — pure MST: TopologyBuilder seed graph on the
                              annotation -> PathFinder inter-component paths
                              -> nx minimum spanning tree (PureMstLinker).

The skeleton here is taken directly from the MST graph — every edge `path`
plus every node pixel — with no dilation / union / re-skeletonisation.

  viz_crossing_full_mst_points.png    — crossing points over the full image
  viz_crossing_full_mst_effective.png — effective crossing segments coloured
  viz_crossing_full_mst_coverage.png  — subtree annotation coverage

Region of a pixel: epidermis if M > 0, dermis otherwise.
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy.ndimage import convolve

from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.algorithms.annotation_grow.dijkstra import get_components
from neural_reconstruction.algorithms.pure_mst import PureMstLinker

# ─────────────────────────────────────────────────────────────────────────────
# Config (shared with viz_crossing_full.py — no crop here, the full image is used)
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S1401-2_b"
BASE_PATH = BASE_PATH / f"data_0510/{IMAGE_ID}"

L_MIN = 5.0              # minimum per-region path length for an effective crossing
MIN_TREE_COMPONENTS = 5  # min annotation components a subtree must cover to count
COVER_NEIGHBORHOOD = 3   # half-window when checking annotation coverage

# Pure-MST reconstruction parameters (PureMstLinker._run_reconstruction).
SEGMENT_LENGTH = 100.0      # TopologyBuilder seed spacing on the annotation
SEARCH_RADIUS = 20.0        # PathFinder inter-component search radius
MIN_COMPONENT_LENGTH = 10.0 # drop reconstructed subtrees shorter than this

# Render geometry. Markers are sized for the full-image render (much smaller
# than viz_crossing.py, which is tuned for the 200x200 crop).
FIG_LONG_SIDE = 22.0          # inches on the longer axis
FIG_DPI = 200
CROSSING_MARKER_SIZE = 26
TOPOLOGY_MARKER_SIZE = 9

SKELETON_COLOR = np.array([1.00, 1.00, 1.00], dtype=np.float32)   # white
EFFECTIVE_COLOR = np.array([0.20, 0.90, 0.35], dtype=np.float32)  # green
EXCLUDED_COLOR = np.array([0.55, 0.55, 0.60], dtype=np.float32)   # dim gray
CROSSING_POINT_COLOR = "#ff3030"                                  # red
ENDPOINT_COLOR = "#ff5020"   # orange-red — matches bs_6
BRANCH_COLOR = "#3399ff"     # blue — matches bs_6
EPIDERMIS_TINT = np.array([0.60, 0.45, 0.32], dtype=np.float32)   # warm


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline (identical to viz_crossing_full.py)
# ─────────────────────────────────────────────────────────────────────────────
green_raw = np.array(cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB))[:, :, 1]
image = green_raw.copy()
mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
annotation = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)

roi_mask = dilate_epidermis_vertically(mask, offset_px=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
image = cv2.subtract(image, background)

clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(768, 768))
image = clahe.apply(image)

image = cv2.bitwise_and(image, image, mask=roi_mask)
image = ski.filters.sato(image, sigmas=range(1, 4), black_ridges=False)
image = (image - image.min()) / (image.max() - image.min()) * 255
image = image.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-MST reconstruction (PureMstLinker). Reuses the script's own preprocessed
# `image` so the only difference vs viz_crossing_full.py is the reconstruction
# algorithm. `_run_reconstruction` builds the cost map (exp(1-norm)-1, identical
# to the annotation-grow cost map), the TopologyBuilder seed graph, PathFinder
# inter-component edges, and the nx minimum spanning tree.
# ─────────────────────────────────────────────────────────────────────────────
annotation_roi = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
annotation_bin = (annotation_roi > 127).astype(np.uint8)
annot_labeled = get_components(annotation_bin)

H, W = annotation_bin.shape

linker = PureMstLinker(
    segment_length=SEGMENT_LENGTH,
    search_radius=SEARCH_RADIUS,
    min_component_length=MIN_COMPONENT_LENGTH,
    min_tree_components=MIN_TREE_COMPONENTS,
)
mst = linker._run_reconstruction(annotation_roi, image)


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton straight from the MST graph: every edge path + every node pixel.
# ─────────────────────────────────────────────────────────────────────────────
skeleton = np.zeros((H, W), dtype=bool)
for _u, _v, data in mst.edges(data=True):
    for py, px in data.get("path", []):
        if 0 <= py < H and 0 <= px < W:
            skeleton[py, px] = True
for ny, nx_ in mst.nodes():
    if 0 <= ny < H and 0 <= nx_ < W:
        skeleton[ny, nx_] = True


# ─────────────────────────────────────────────────────────────────────────────
# Region labelling: epidermis where the epidermis mask M > 0, dermis otherwise
# ─────────────────────────────────────────────────────────────────────────────
epi = mask > 0


# ─────────────────────────────────────────────────────────────────────────────
# Subtree annotation coverage — for each skeleton connected component (a
# reconstructed subtree), count how many distinct annotation components it
# touches. Subtrees covering fewer than MIN_TREE_COMPONENTS are excluded from
# the crossing count, matching `_exclude_small_subtrees_from_count` in
# core/crosses_detection/pipeline.py.
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

# Topology markers: endpoints = degree-1, branch points = degree>=3 in the MST.
endpoint_nodes = [n for n in mst.nodes() if mst.degree(n) == 1]
branch_nodes = [n for n in mst.nodes() if mst.degree(n) >= 3]


def _node_xy(nodes):
    """Split (y, x) node tuples into parallel x / y lists (full-image coords)."""
    xs = [nx_ for _ny, nx_ in nodes]
    ys = [ny for ny, _nx in nodes]
    return xs, ys


ep_xs, ep_ys = _node_xy(endpoint_nodes)
bp_xs, bp_ys = _node_xy(branch_nodes)


def _draw_topology_markers(ax):
    ax.scatter(
        bp_xs, bp_ys, s=TOPOLOGY_MARKER_SIZE, c=BRANCH_COLOR, edgecolors="black",
        linewidths=0.4, zorder=4,
    )
    ax.scatter(
        ep_xs, ep_ys, s=TOPOLOGY_MARKER_SIZE, c=ENDPOINT_COLOR, edgecolors="black",
        linewidths=0.4, zorder=4,
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
    f"[pure-mst]  skeleton px: {int(skeleton.sum())}  "
    f"crossing points: {int(crossing_pts.sum())}  "
    f"crossing segments: {n_crossing_segments}  "
    f"effective (N_cross): {len(effective_ids)}  "
    f"excluded by coverage<{MIN_TREE_COMPONENTS}: {n_excluded_by_coverage}  "
    f"subtrees: {n_cc} (qualifying: {len(qualifying_ccs)})"
)


# ─────────────────────────────────────────────────────────────────────────────
# Faded background with epidermis tint (full image)
# ─────────────────────────────────────────────────────────────────────────────
faded_bg = np.stack([green_raw] * 3, axis=-1).astype(np.float32) / 255.0 * 0.5
faded_bg[epi] = 0.7 * faded_bg[epi] + 0.3 * EPIDERMIS_TINT
faded_bg[roi_mask == 0] = 0.0


def save_fig(disp: np.ndarray, out_name: str, extra=None) -> None:
    h, w = disp.shape[:2]
    if w >= h:
        figsize = (FIG_LONG_SIDE, FIG_LONG_SIDE * h / w)
    else:
        figsize = (FIG_LONG_SIDE * w / h, FIG_LONG_SIDE)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(disp)
    ax.axis("off")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    if extra is not None:
        extra(ax)
    out_path = Path(__file__).parent / out_name
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — crossing points
# ─────────────────────────────────────────────────────────────────────────────
disp1 = faded_bg.copy()
disp1[excluded_mask] = EXCLUDED_COLOR
disp1[qualifying_mask] = SKELETON_COLOR
cp_ys, cp_xs = np.where(crossing_pts)


def _draw_crossing_points(ax):
    _draw_topology_markers(ax)
    ax.scatter(
        cp_xs, cp_ys, s=CROSSING_MARKER_SIZE, c=CROSSING_POINT_COLOR, marker="*",
        edgecolors="black", linewidths=0.4, zorder=5,
    )


save_fig(disp1, "viz_crossing_full_mst_points.png", extra=_draw_crossing_points)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — effective crossing segments coloured, with topology markers
# ─────────────────────────────────────────────────────────────────────────────
disp2 = faded_bg.copy()
disp2[excluded_mask] = EXCLUDED_COLOR
disp2[qualifying_mask] = SKELETON_COLOR
disp2[effective_mask] = EFFECTIVE_COLOR

save_fig(disp2, "viz_crossing_full_mst_effective.png", extra=_draw_topology_markers)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — subtree annotation coverage. Only qualifying subtrees are labelled
# with their coverage count (labelling every CC would clutter the full image).
# ─────────────────────────────────────────────────────────────────────────────
ANNOT_COLOR = np.array([0.30, 0.65, 1.00], dtype=np.float32)  # blue

plain_bg = np.stack([green_raw] * 3, axis=-1).astype(np.float32) / 255.0 * 0.5
disp3 = plain_bg.copy()
disp3[annotation_bin > 0] = ANNOT_COLOR
disp3[excluded_mask] = EXCLUDED_COLOR
disp3[qualifying_mask] = SKELETON_COLOR

cc_label_positions: list[tuple[int, int, int]] = []  # (x, y, count)
for cc_id in qualifying_ccs:
    ys_cc, xs_cc = np.where(skel_cc_labels == cc_id)
    cy = int(ys_cc.mean())
    cx = int(xs_cc.mean())
    cc_label_positions.append((cx, cy, cc_coverage_count[cc_id]))


def _draw_coverage_labels(ax):
    for cx, cy, count in cc_label_positions:
        ax.text(
            cx, cy, str(count),
            color="white",
            fontsize=9, fontweight="bold",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="#228b3a",
                edgecolor="black", linewidth=0.5, alpha=0.85,
            ),
            zorder=5,
        )


save_fig(disp3, "viz_crossing_full_mst_coverage.png", extra=_draw_coverage_labels)
