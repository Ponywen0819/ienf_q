"""Visualise MST-edge path backtracking, the bridge mask, and skeletonisation.

Reuses the exact crop and preprocessing pipeline of viz_region_grow.py.
Covers two paper subsections:

  \\ref{subsec:path-backtrack}  — backtracking an MST edge into a pixel path
  \\ref{subsec:bridge-skeleton} — bridge mask, dilation, union, skeleton

Six figures are produced:

  viz_bs_1_backtrack.png     — one MST edge: meeting point + backtracked path
  viz_bs_2_bridge.png        — bridge mask B: all MST edge paths rasterised
  viz_bs_3_dilated.png       — bridge mask after dilation by a radius-r_d disk
  viz_bs_4_union.png         — union U = (A > 0) ∪ (B ⊕ K_{r_d})
  viz_bs_5_skeleton_raw.png  — Zhang-Suen skeleton of U, before short-stub removal
  viz_bs_6_skeleton.png      — topology graph of U (build_seed_graph: skeleton +
                               short-stub removal), with endpoints / branch points
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
import skimage as ski

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

PRUNE_THRESHOLD = 20.0  # tau
DILATE_RADIUS = 1       # r_d — same default as build_result_graph()

BRIDGE_COLOR = np.array([1.00, 0.78, 0.10], dtype=np.float32)   # gold
ANNOT_COLOR = np.array([0.30, 0.85, 1.00], dtype=np.float32)    # light cyan
SKELETON_COLOR = np.array([1.00, 1.00, 1.00], dtype=np.float32)  # white
OTHER_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float32)


def crop(arr):
    """Return the fixed crop used by the paper visualizations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _component_palette(n_comp: int, seed: int = 0) -> np.ndarray:
    """Distinct qualitative colours for component IDs 1..n_comp (matches other figures)."""
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
# Multi-source Dijkstra expansion (prev_* needed for path backtracking)
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


# ─────────────────────────────────────────────────────────────────────────────
# Component graph → tau-pruning → MST (edges carry backtracked 'path')
# ─────────────────────────────────────────────────────────────────────────────
connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
G = build_component_graph(connections, n_components)
G_pruned = prune_edges(G, threshold=PRUNE_THRESHOLD)
mst = minimum_spanning_forest(G_pruned)
print(f"MST: {mst.number_of_edges()} edges")


# ─────────────────────────────────────────────────────────────────────────────
# Bridge mask, dilation, union, skeleton — all computed on the full image
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

# Raw Zhang-Suen skeleton — before any short-stub removal.
raw_skeleton = ski.morphology.skeletonize(union)

# Pixel-level topology graph G — the same call the annotation_grow algorithm
# uses (build_result_graph -> TopologyBuilder.build_seed_graph). Internally this
# skeletonises U and then removes short hang edges (short-stub removal).
SEGMENT_LENGTH = 100.0  # same default as build_result_graph()
seed_graph = TopologyBuilder(segment_length=SEGMENT_LENGTH).build_seed_graph(
    union.astype(np.uint8)
)

# Rasterise the graph's edge paths (+ node pixels) into a skeleton mask
skeleton = np.zeros((H, W), dtype=bool)
for _u, _v, data in seed_graph.edges(data=True):
    for py, px in data.get("path", []):
        if 0 <= py < H and 0 <= px < W:
            skeleton[py, px] = True
for ny, nx_ in seed_graph.nodes():
    if 0 <= ny < H and 0 <= nx_ < W:
        skeleton[ny, nx_] = True

# Endpoints (graph degree 1) and branch points (degree >= 3); degree-2
# seed nodes are interior "middle" nodes inserted by TopologyBuilder along
# long skeleton paths.
endpoint_nodes = [n for n in seed_graph.nodes() if seed_graph.degree(n) == 1]
branch_nodes = [n for n in seed_graph.nodes() if seed_graph.degree(n) >= 3]
middle_nodes = [n for n in seed_graph.nodes() if seed_graph.degree(n) == 2]


def _skeleton_nodes(skel: np.ndarray) -> tuple[list, list]:
    """Return (endpoints, branch_points) [(y, x), ...] from a 1-px binary skeleton.

    Endpoints are pixels with exactly one 8-neighbour. Branch points use the
    crossing number A(P) — the count of contiguous neighbour runs around the
    pixel: a real junction has A >= 3, while a plain path pixel (even a diagonal
    "staircase" one with three neighbours) has A <= 2, so this avoids the false
    branches that a raw neighbour count produces. Branch pixels still cluster
    around a junction, so each connected cluster is collapsed to one centroid.
    Used to mark the raw skeleton (which has no topology graph) like bs_6.
    """
    skel_u8 = skel.astype(np.uint8)
    on = skel_u8 == 1

    nbr_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nbr = cv2.filter2D(skel_u8, -1, nbr_kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = [(int(y), int(x)) for y, x in zip(*np.where(on & (nbr == 1)))]

    # Crossing number: walk the 8 neighbours clockwise, count 0->1 transitions.
    h, w = skel_u8.shape
    padded = np.pad(skel_u8, 1)
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    ring = np.stack(
        [padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in offsets],
        axis=0,
    )
    nxt = np.roll(ring, -1, axis=0)
    crossing = ((ring == 0) & (nxt == 1)).sum(axis=0)

    # Junction pixels within a few px belong to the same junction; dilate before
    # labelling so they merge, then take one centroid per group.
    branch_mask = on & (crossing >= 3)
    merge_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    branch_grouped = cv2.dilate(branch_mask.astype(np.uint8), merge_disk)
    branch_labeled = ski.measure.label(branch_grouped, connectivity=2)
    branches = []
    for rid in range(1, int(branch_labeled.max()) + 1):
        ys, xs = np.where((branch_labeled == rid) & branch_mask)
        if len(ys) == 0:
            continue
        branches.append((int(round(ys.mean())), int(round(xs.mean()))))
    return endpoints, branches


raw_endpoint_nodes, raw_branch_nodes = _skeleton_nodes(raw_skeleton)

print(
    f"bridge px: {int(bridge_mask.sum())}  ->  dilated: {int(bridge_dilated.sum())}  "
    f"->  union: {int(union.sum())}  ->  raw skeleton: {int(raw_skeleton.sum())}  "
    f"->  stub-removed: {int(skeleton.sum())}  "
    f"(endpoints: {len(endpoint_nodes)}, branch points: {len(branch_nodes)})"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared crop tiles + faded background
# ─────────────────────────────────────────────────────────────────────────────
green_crop = crop(green_raw)
roi_crop = crop(roi_mask)
labels_crop = crop(annot_labeled)
owner_crop = crop(owner_map)

faded_bg = np.stack([green_crop] * 3, axis=-1).astype(np.float32) / 255.0
faded_bg = faded_bg * 0.5


def save_fig(disp: np.ndarray, out_name: str, extra=None) -> None:
    """Show a HxWx3 display image with the standard styling and save it."""
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


def render_mask_layers(layers: list, out_name: str, extra=None) -> None:
    """layers: list of (full_image_bool_mask, rgb_colour), drawn in order."""
    disp = faded_bg.copy()
    for full_mask, colour in layers:
        disp[crop(full_mask)] = colour
    disp[roi_crop == 0] = 0.0
    save_fig(disp, out_name, extra=extra)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — one MST edge: meeting point + backtracked path Pi_(a,b)
#   Uses the standard crop. MST edge paths are short by nature (the MST bridges
#   nearby components), so the edge with the most path pixels visible inside the
#   crop is chosen to make the backtracking as clear as possible.
# ─────────────────────────────────────────────────────────────────────────────
palette = _component_palette(n_components)

mst_edges = []
for a, b in mst.edges():
    key = (min(a, b), max(a, b))
    info = connections.get(key)
    if info is None or not info.get("path"):
        continue
    path = np.asarray(info["path"], dtype=float)  # (N, 2) [y, x]
    py, px = path[:, 0], path[:, 1]
    inside = (
        (py >= CROP_Y0) & (py < CROP_Y0 + CROP_H)
        & (px >= CROP_X0) & (px < CROP_X0 + CROP_W)
    )
    mst_edges.append((int(inside.sum()), key, info))
if not mst_edges:
    raise RuntimeError("No MST edge with a path inside the crop.")
mst_edges.sort(key=lambda e: -e[0])
IN_CROP_LEN, (A_ID, B_ID), info = mst_edges[0]
print(f"Figure 1 MST edge: a={A_ID}, b={B_ID}  (in-crop path length={IN_CROP_LEN}px)")

A_COLOR = palette[A_ID]
B_COLOR = palette[B_ID]

# Background for figure 1 (standard crop) — annotation only, no territory.
disp1 = faded_bg.copy()
disp1[labels_crop == A_ID] = A_COLOR
disp1[labels_crop == B_ID] = B_COLOR
disp1[roi_crop == 0] = 0.0

# Backtracked path Π_(a,b) and the meeting point (local coords)
path = np.asarray(info["path"], dtype=float)
path_px = path[:, 1] - CROP_X0
path_py = path[:, 0] - CROP_Y0
mp_x = (info["x"] + info["x_b"]) / 2.0 - CROP_X0
mp_y = (info["y"] + info["y_b"]) / 2.0 - CROP_Y0


def _draw_fig1(ax):
    ax.plot(
        path_px, path_py, color="white", linewidth=2.5, zorder=3,
        path_effects=[pe.withStroke(linewidth=4.5, foreground="black")],
    )
    ax.scatter(
        [path_px[0], path_px[-1]], [path_py[0], path_py[-1]],
        marker="o", s=50, c="white", edgecolors="none", zorder=4,
    )
    ax.scatter(
        [mp_x], [mp_y], marker="o", s=50,
        c="yellow", edgecolors="none", zorder=5,
    )
    for txt, ex, ey in (
        ("a", path_px[0], path_py[0]),
        ("b", path_px[-1], path_py[-1]),
    ):
        ddx = ex - mp_x
        ddy = ey - mp_y
        norm = float(np.hypot(ddx, ddy)) or 1.0
        lx = float(np.clip(ex + 15.0 * ddx / norm, 12, CROP_W - 12))
        ly = float(np.clip(ey + 15.0 * ddy / norm, 12, CROP_H - 12))
        ax.text(
            lx, ly, txt, color="yellow", fontsize=16, fontweight="bold",
            ha="center", va="center", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
        )


save_fig(disp1, "viz_bs_1_backtrack.png", extra=_draw_fig1)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 2-5 — bridge mask, dilation, union, skeleton
# ─────────────────────────────────────────────────────────────────────────────
render_mask_layers(
    [(bridge_mask.astype(bool), BRIDGE_COLOR)],
    "viz_bs_2_bridge.png",
)
render_mask_layers(
    [(bridge_dilated, BRIDGE_COLOR)],
    "viz_bs_3_dilated.png",
)
render_mask_layers(
    [(bridge_dilated, BRIDGE_COLOR), (annotation_bin > 0, ANNOT_COLOR)],
    "viz_bs_4_union.png",
)
def _crop_local_xy(nodes):
    """(y, x) graph nodes -> (xs, ys) lists in crop-local coords, in-crop only."""
    xs, ys = [], []
    for ny, nx_ in nodes:
        if CROP_Y0 <= ny < CROP_Y0 + CROP_H and CROP_X0 <= nx_ < CROP_X0 + CROP_W:
            xs.append(nx_ - CROP_X0)
            ys.append(ny - CROP_Y0)
    return xs, ys


ep_xs, ep_ys = _crop_local_xy(endpoint_nodes)
bp_xs, bp_ys = _crop_local_xy(branch_nodes)
mid_xs, mid_ys = _crop_local_xy(middle_nodes)
raw_ep_xs, raw_ep_ys = _crop_local_xy(raw_endpoint_nodes)
raw_bp_xs, raw_bp_ys = _crop_local_xy(raw_branch_nodes)
ENDPOINT_COLOR = "#ff5020"  # orange-red
BRANCH_COLOR = "#3399ff"    # blue


def _pick_middle_node_between_branch_and_endpoint():
    """Pick the midpoint of an edge path connecting a branch point and an
    endpoint, visible inside the crop. TopologyBuilder only inserts degree-2
    seeds on paths longer than segment_length, so for short paths we synthesise
    a "middle node" position from the edge's path coordinates."""
    bp_set = set(branch_nodes)
    ep_set = set(endpoint_nodes)
    best = None
    best_len = -1
    for u, v, data in seed_graph.edges(data=True):
        if not (
            (u in bp_set and v in ep_set) or (u in ep_set and v in bp_set)
        ):
            continue
        path = data.get("path")
        if path is None or len(path) < 3:
            continue
        my, mx_ = path[len(path) // 2]
        if not (CROP_Y0 <= my < CROP_Y0 + CROP_H and CROP_X0 <= mx_ < CROP_X0 + CROP_W):
            continue
        if len(path) > best_len:
            best_len = len(path)
            best = (int(my), int(mx_))
    return best


highlight_mid = _pick_middle_node_between_branch_and_endpoint()


def _draw_skeleton_markers(ax):
    ax.scatter(
        bp_xs, bp_ys, s=110, c=BRANCH_COLOR, edgecolors="none", zorder=4,
    )
    ax.scatter(
        ep_xs, ep_ys, s=110, c=ENDPOINT_COLOR, edgecolors="none", zorder=4,
    )
    # Highlight one degree-2 "middle" seed node in yellow — a node sitting
    # along the chain between a branch point and an endpoint.
    if highlight_mid is not None:
        my, mx_ = highlight_mid
        ax.scatter(
            [mx_ - CROP_X0], [my - CROP_Y0], s=180, c="yellow",
            edgecolors="none", zorder=5,
        )


def _draw_raw_skeleton_markers(ax):
    ax.scatter(
        raw_bp_xs, raw_bp_ys, s=46, c=BRANCH_COLOR, edgecolors="black",
        linewidths=0.6, zorder=4,
    )
    ax.scatter(
        raw_ep_xs, raw_ep_ys, s=46, c=ENDPOINT_COLOR, edgecolors="black",
        linewidths=0.6, zorder=4,
    )


# Short stubs that the stub-removal step deletes = raw skeleton minus the
# stub-removed skeleton. Circle two of them on the raw-skeleton figure.
removed_stubs = raw_skeleton & ~skeleton
stub_labeled = ski.measure.label(crop(removed_stubs), connectivity=2)
stub_comps = []
for rid in range(1, int(stub_labeled.max()) + 1):
    ys, xs = np.where(stub_labeled == rid)
    n = len(ys)
    if n < 2 or n > 40:  # a removed hang-edge is short by construction
        continue
    cy, cx = float(ys.mean()), float(xs.mean())
    extent = float(np.max(np.hypot(ys - cy, xs - cx)))
    stub_comps.append((n, cy, cx, extent))
stub_comps.sort(key=lambda c: -c[0])  # largest first

# Greedily pick two well-separated stubs
highlighted_stubs = []
for comp in stub_comps:
    if all(np.hypot(comp[1] - h[1], comp[2] - h[2]) > 30 for h in highlighted_stubs):
        highlighted_stubs.append(comp)
    if len(highlighted_stubs) == 2:
        break


def _draw_raw_stub_circles(ax):
    for _n, cy, cx, extent in highlighted_stubs:
        ax.add_patch(
            Circle(
                (cx, cy), extent + 9.0, fill=False,
                edgecolor="#ff3030", linewidth=2.2, zorder=5,
            )
        )


# Figure 5 — raw skeleton, before short-stub removal: endpoint / branch-point
# markers (same style as bs_6) plus two removed stubs circled.
def _draw_raw_skeleton_fig(ax):
    _draw_raw_skeleton_markers(ax)
    _draw_raw_stub_circles(ax)


render_mask_layers(
    [(raw_skeleton, SKELETON_COLOR)],
    "viz_bs_5_skeleton_raw.png",
    extra=_draw_raw_skeleton_fig,
)
# Same raw skeleton, plain (no stub-highlight circles).
render_mask_layers(
    [(raw_skeleton, SKELETON_COLOR)],
    "viz_bs_5_skeleton_raw_plain.png",
)
# Figure 6 — topology graph skeleton after short-stub removal, with markers
render_mask_layers(
    [(skeleton, SKELETON_COLOR)],
    "viz_bs_6_skeleton.png",
    extra=_draw_skeleton_markers,
)
# Same stub-removed skeleton, plain (no endpoint / branch-point markers).
render_mask_layers(
    [(skeleton, SKELETON_COLOR)],
    "viz_bs_6_skeleton_plain.png",
)
# Same skeleton with endpoint / branch-point markers AND the same red circles
# as bs_5 marking where the stubs *were* — the circled spots are now empty,
# showing what short-stub removal cleaned up.
def _draw_skeleton_fig_circles(ax):
    _draw_skeleton_markers(ax)
    _draw_raw_stub_circles(ax)


render_mask_layers(
    [(skeleton, SKELETON_COLOR)],
    "viz_bs_6_skeleton_circles.png",
    extra=_draw_skeleton_fig_circles,
)
