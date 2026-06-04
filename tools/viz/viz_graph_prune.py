"""Visualise component-graph reduction: tau-pruning then minimum spanning forest.

Reuses the exact crop and preprocessing pipeline of viz_region_grow.py.
After multi-source Dijkstra expansion the component graph G_c is built, then
reduced in two stages:

  1. prune_edges(G_c, threshold=tau) — drop edges with weight w_ab > tau
  2. minimum_spanning_forest(...)    — keep one MST per connected component

Three figures are produced, all as abstract graphs (nodes at component
centroids, edges coloured by weight) over a faded pixel image:

  viz_graph_full.png   — G_c before pruning
  viz_graph_pruned.png — G_c after tau-pruning
  viz_graph_mst.png    — minimum spanning forest

All three share one colour scale (the full G_c weight range) so they are
directly comparable.
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# ─────────────────────────────────────────────────────────────────────────────
# Config / crop (shared with viz_region_grow.py)
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

PRUNE_THRESHOLD = 50.0  # tau — same default as run_inference.py / staged grid search
EDGE_CMAP = "cool"  # colormap for edge weights; bright at both ends → visible on dark bg


def crop(arr):
    """Return the fixed crop used by the paper visualizations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _component_palette(n_comp: int, seed: int = 0) -> np.ndarray:
    """Distinct qualitative colours for component IDs 1..n_comp.

    Identical construction to viz_region_grow.py so a given component ID
    receives the same colour across all figures.
    """
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
# Multi-source Dijkstra expansion
# ─────────────────────────────────────────────────────────────────────────────
annotation_roi = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
annotation_bin = (annotation_roi > 127).astype(np.uint8)
annot_labeled = get_components(annotation_bin)
n_components = int(annot_labeled.max())

owner_map, dist_map, _, _ = multi_source_dijkstra(
    cost_map=cost_map,
    annot_labeled=annot_labeled,
    connectivity=8,
    roi_mask=(roi_mask > 127),
)


# ─────────────────────────────────────────────────────────────────────────────
# Component graph → tau-pruning → minimum spanning forest
# ─────────────────────────────────────────────────────────────────────────────
connections = find_meeting_points(owner_map, dist_map)
G = build_component_graph(connections, n_components)
G_pruned = prune_edges(G, threshold=PRUNE_THRESHOLD)
mst = minimum_spanning_forest(G_pruned)
print(
    f"G_c: {G.number_of_edges()} edges  ->  "
    f"pruned (tau={PRUNE_THRESHOLD}): {G_pruned.number_of_edges()}  ->  "
    f"MST: {mst.number_of_edges()}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Restrict to crop: component centroids as graph nodes
# ─────────────────────────────────────────────────────────────────────────────
labels_crop = crop(annot_labeled)
crop_ids = sorted(set(int(v) for v in np.unique(labels_crop)) - {0})


def crop_centroid(cid: int):
    """Centroid (row, col) of a component's pixels that fall inside the crop."""
    ys, xs = np.where(labels_crop == cid)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


node_xy = {cid: crop_centroid(cid) for cid in crop_ids}
node_xy = {k: v for k, v in node_xy.items() if v is not None}


def crop_edges(graph) -> list:
    """Edges with both endpoints inside the crop: (weight, (x1, y1), (x2, y2))."""
    out = []
    for a, b, d in graph.edges(data=True):
        if a in node_xy and b in node_xy:
            ca, cb = node_xy[a], node_xy[b]
            out.append((float(d["weight"]), (ca[1], ca[0]), (cb[1], cb[0])))
    return out


full_edges = crop_edges(G)
pruned_edges = crop_edges(G_pruned)
mst_edges = crop_edges(mst)
print(
    f"In crop: {len(node_xy)} nodes, "
    f"{len(full_edges)} G_c edges, {len(pruned_edges)} pruned edges, "
    f"{len(mst_edges)} MST edges"
)

# Shared colour scale across all three figures. vmax is clipped to the 95th
# percentile of the full G_c edge weights so a few extreme edges don't dominate
# the scale; edges above vmax saturate at the top colour.
weight_pool = np.array([w for w, _, _ in full_edges], dtype=float)
if len(weight_pool):
    vmin = float(weight_pool.min())
    vmax = float(np.percentile(weight_pool, 95))
else:
    vmin, vmax = 0.0, 1.0
if vmin == vmax:
    vmax = vmin + 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Background: faded pixel image with component source pixels in palette colours
# ─────────────────────────────────────────────────────────────────────────────
palette = _component_palette(n_components)
green_crop = crop(green_raw)
roi_crop = crop(roi_mask)

disp = np.stack([green_crop] * 3, axis=-1).astype(np.float32) / 255.0
disp = disp * 0.5  # fade background for context only
for cid in crop_ids:
    disp[labels_crop == cid] = palette[cid]
disp[roi_crop == 0] = 0.0

node_x = [node_xy[c][1] for c in node_xy]
node_y = [node_xy[c][0] for c in node_xy]
node_c = [palette[c] for c in node_xy]


def render(edges: list, label: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    ax.imshow(disp)
    ax.axis("off")
    ax.set_xlim(-0.5, CROP_W - 0.5)
    ax.set_ylim(CROP_H - 0.5, -0.5)

    segments = [[p1, p2] for _, p1, p2 in edges]
    weights = np.array([w for w, _, _ in edges], dtype=float)
    lc = LineCollection(segments, cmap=EDGE_CMAP, linewidths=2.2, zorder=3)
    lc.set_array(weights)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)

    ax.scatter(
        node_x, node_y, s=90, c=node_c,
        edgecolors="white", linewidths=1.4, zorder=5,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cbar = fig.colorbar(lc, cax=cax)
    cbar.ax.tick_params(labelsize=16, colors="black")

    out_path = Path(__file__).parent / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


render(full_edges, r"edge weight $w_{ab}$", "viz_graph_full.png")
render(pruned_edges, r"edge weight $w_{ab}$", "viz_graph_pruned.png")
render(mst_edges, r"edge weight $w_{ab}$", "viz_graph_mst.png")
