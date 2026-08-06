"""PSD-output variant of tools/viz/viz_bridge_skeleton.py — figure 1 only.

viz_bridge_skeleton.py runs top-level (no ``main()`` guard) and produces six
figures; only figure 1 (one MST edge: meeting point + backtracked path,
"viz_bs_1_backtrack.png") is needed here, as a 3-layer PSD:

  * "background" — annotation-only crop (A/B territory colours, ROI masked),
                    with the backtracked path line + endpoint dots +
                    meeting-point dot drawn straight onto it (no separate
                    layer, no black stroke/outline on the line — plain white)
  * "label_a" / "label_b" — the "a" / "b" text, each its own layer

Since the source has no main() guard, the compute pipeline up to picking the
figure-1 MST edge is re-derived here rather than imported — only real
library functions (dilate_epidermis_vertically, get_components,
multi_source_dijkstra, find_meeting_points, build_component_graph,
prune_edges, minimum_spanning_forest) are imported, everything else is
copied from the source.

Resolution: matches the source's own save_fig (figsize=(6,6), dpi=200 ->
1200x1200, a 6x upsample of the 200x200 crop) for the same reason as
viz_meet_point_psd — otherwise markers/text end up mis-proportioned for the
canvas. Upsampling uses cv2.INTER_NEAREST (no blending/blur).

Run:
    uv run python tools/viz/viz_bridge_skeleton_psd/viz_bridge_skeleton_psd.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neural_reconstruction.algorithms.annotation_grow.dijkstra import (  # noqa: E402
    get_components,
    multi_source_dijkstra,
)
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (  # noqa: E402
    build_component_graph,
    find_meeting_points,
    minimum_spanning_forest,
    prune_edges,
)
from neural_reconstruction.core.preprocessing import (  # noqa: E402
    dilate_epidermis_vertically,
)

BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

PRUNE_THRESHOLD = 20.0  # tau

# Source renders at figsize=(6,6)/dpi=200 -> 1200x1200 even though the crop
# is only 200x200; match it or markers/text/path come out mis-proportioned
# relative to the (upsampled) background. See viz_meet_point_psd for the
# same fix.
FIG_INCHES, DPI = 6.0, 200
OUT_W = round(FIG_INCHES * DPI)
OUT_H = round(FIG_INCHES * DPI * CROP_H / CROP_W)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def crop(arr: np.ndarray) -> np.ndarray:
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _component_palette(n_comp: int, seed: int = 0) -> np.ndarray:
    """Identical construction to viz_bridge_skeleton.py / viz_meet_point.py so a
    given component ID gets the same colour across every figure."""
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


def render_overlay(draw_fn: Callable) -> np.ndarray:
    """Rasterise one matplotlib element onto a transparent OUT_H x OUT_W
    canvas, axes data coordinates in image (row, col) convention:
    xlim=[0,CROP_W], ylim=[CROP_H,0]."""
    fig = plt.figure(figsize=(OUT_W / DPI, OUT_H / DPI), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-0.5, CROP_W - 0.5)
    ax.set_ylim(CROP_H - 0.5, -0.5)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    draw_fn(ax)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    assert buf.shape[:2] == (OUT_H, OUT_W), f"canvas {buf.shape[:2]} != requested {(OUT_H, OUT_W)}"
    return buf


def main() -> None:
    # --- Preprocessing pipeline (identical to viz_bridge_skeleton.py) -----
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
    cost_map = np.exp(1.0 - (image.astype(np.float32) / 255.0)) - 1.0

    # --- Multi-source Dijkstra (prev_* needed for path backtracking) ------
    annotation_roi = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
    annotation_bin = (annotation_roi > 127).astype(np.uint8)
    annot_labeled = get_components(annotation_bin)
    n_components = int(annot_labeled.max())

    owner_map, dist_map, prev_y, prev_x = multi_source_dijkstra(
        cost_map=cost_map, annot_labeled=annot_labeled,
        connectivity=8, roi_mask=(roi_mask > 127),
    )

    # --- Component graph -> tau-pruning -> MST -----------------------------
    connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
    G = build_component_graph(connections, n_components)
    G_pruned = prune_edges(G, threshold=PRUNE_THRESHOLD)
    mst = minimum_spanning_forest(G_pruned)
    print(f"MST: {mst.number_of_edges()} edges")

    # --- Pick the MST edge with the most path pixels inside the crop ------
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
    in_crop_len, (a_id, b_id), info = mst_edges[0]
    print(f"Figure 1 MST edge: a={a_id}, b={b_id}  (in-crop path length={in_crop_len}px)")

    a_color = palette[a_id]
    b_color = palette[b_id]

    # --- Background: annotation-only crop, no territory -------------------
    green_crop = crop(green_raw)
    roi_crop = crop(roi_mask)
    labels_crop = crop(annot_labeled)

    faded_bg = np.stack([green_crop] * 3, axis=-1).astype(np.float32) / 255.0 * 0.5
    disp1 = faded_bg.copy()
    disp1[labels_crop == a_id] = a_color
    disp1[labels_crop == b_id] = b_color
    disp1[roi_crop == 0] = 0.0

    background_rgb = (np.clip(disp1, 0.0, 1.0) * 255).astype(np.uint8)
    background_rgb = cv2.resize(background_rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)

    # --- Backtracked path Pi_(a,b) and the meeting point (crop-local) -----
    path = np.asarray(info["path"], dtype=float)
    path_px = path[:, 1] - CROP_X0
    path_py = path[:, 0] - CROP_Y0
    mp_x = (info["x"] + info["x_b"]) / 2.0 - CROP_X0
    mp_y = (info["y"] + info["y_b"]) / 2.0 - CROP_Y0

    # Path + dots are drawn straight onto the background raster (no separate
    # layer, no stroke/outline) via cv2, scaled from crop-local coords into
    # the upsampled canvas. Point-based sizes (linewidth=2.5, s=50) are
    # converted through the same DPI so they still match the source's scale.
    scale = OUT_W / CROP_W
    line_px = max(1, round(2.5 * DPI / 72))
    dot_r_px = max(1, round(np.sqrt(50 / np.pi) * DPI / 72))

    pts = np.stack([path_px * scale, path_py * scale], axis=1).round().astype(np.int32)
    cv2.polylines(background_rgb, [pts.reshape(-1, 1, 2)], isClosed=False,
                  color=(255, 255, 255), thickness=line_px, lineType=cv2.LINE_AA)
    for ex, ey in ((path_px[0], path_py[0]), (path_px[-1], path_py[-1])):
        cv2.circle(background_rgb, (round(ex * scale), round(ey * scale)), dot_r_px,
                   (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(background_rgb, (round(mp_x * scale), round(mp_y * scale)), dot_r_px,
               (255, 255, 0), -1, cv2.LINE_AA)

    def label_pos(ex: float, ey: float) -> tuple[float, float]:
        ddx, ddy = ex - mp_x, ey - mp_y
        norm = float(np.hypot(ddx, ddy)) or 1.0
        lx = float(np.clip(ex + 15.0 * ddx / norm, 12, CROP_W - 12))
        ly = float(np.clip(ey + 15.0 * ddy / norm, 12, CROP_H - 12))
        return lx, ly

    def make_label_drawer(txt: str, ex: float, ey: float) -> Callable:
        lx, ly = label_pos(ex, ey)

        def draw_label(ax):
            ax.text(
                lx, ly, txt, color="yellow", fontsize=16, fontweight="bold",
                ha="center", va="center", zorder=6,
            )

        return draw_label

    # --- Write PSD -----------------------------------------------------
    psd = PSDImage.new("RGBA", (OUT_W, OUT_H))
    psd.create_pixel_layer(
        Image.fromarray(np.dstack([background_rgb, np.full((OUT_H, OUT_W), 255, np.uint8)])),
        name="background", top=0, left=0,
    )
    psd.create_pixel_layer(
        Image.fromarray(render_overlay(make_label_drawer("a", path_px[0], path_py[0]))),
        name="label_a", top=0, left=0,
    )
    psd.create_pixel_layer(
        Image.fromarray(render_overlay(make_label_drawer("b", path_px[-1], path_py[-1]))),
        name="label_b", top=0, left=0,
    )

    out_path = OUTPUT_DIR / f"{IMAGE_ID}_bs_1_backtrack.psd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(out_path)
    print(f"[save] {out_path}  layers={[l.name for l in psd]}")


if __name__ == "__main__":
    main()
