"""
Edge prune-threshold (tau) sweep — annotation-grow result visualisation.

tau is the component-graph edge prune threshold of the annotation-grow
algorithm. Unlike the preprocessing-parameter sweeps, tau only takes effect
deep in the reconstruction, so this runs the *full* AnnotationGrowLinker
pipeline once per tau and renders the resulting reconstructed network.

For each tau the linker is run end-to-end (preprocessing -> Dijkstra expansion
-> component graph -> prune@tau -> MST -> skeleton), and its result graph is
drawn over the image.

  tau too small -> edges linking real fibres are pruned, topology fragments
  tau too large -> high-cost spurious links are re-admitted

The effect of tau is mostly many small reroutes spread across the epidermis,
so a whole-image panel barely changes to the eye. The informative figure is
the *difference* between two tau values: a 3-colour overlay of one tau's
reconstruction against another, cropped to where they differ most.

Outputs — bare images (no title/axes/colorbar):
  viz_prune_result_t{tau}.png    — full-image reconstruction for each tau
  viz_prune_result_gt.png        — skeleton GT, for reference
  viz_prune_result_diff.png      — DIFF_TAU vs TAU_REF, cropped to the
                                   densest-change window
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from neural_reconstruction.algorithms.annotation_grow import AnnotationGrowLinker

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S1819-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

# Render the whole image (True) or just the shared fixed crop (False).
USE_FULL_IMAGE = True

# Crop region used when USE_FULL_IMAGE is False (same as other viz scripts).
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

# Prune thresholds tau to sweep; 20 is the pipeline default.
PRUNE_THRESHOLDS = [10, 20, 30, 40, 50]
TAU_REF = 20

# Difference panel: compare DIFF_TAU against TAU_REF.
DIFF_TAU = 50
# Crop window for the difference panel: None = auto-locate the densest-change
# window of DIFF_CROP_SIZE; or set an explicit (y0, x0, size).
DIFF_CROP = None
DIFF_CROP_SIZE = 400

# Rendering.
BG_FADE = 0.45                 # green-channel background dim factor
EDGE_COLOR = (0, 255, 255)     # reconstructed-network edges (RGB)
NODE_COLOR = (255, 50, 50)     # graph nodes (RGB)
GT_COLOR = (0, 255, 255)       # skeleton GT
EDGE_DILATE = 2                # edge half-thickness in px
NODE_RADIUS = 3
# Difference-panel colours.
DIFF_BOTH = (130, 130, 130)    # edge present at both tau values
DIFF_ONLY_REF = (60, 255, 60)  # present at TAU_REF only — rerouted away
DIFF_ONLY_NEW = (255, 40, 40)  # present at DIFF_TAU only — spurious new link
# ==========================================================================

OUT_DIR = Path(__file__).parent

# Region (y0, x0, h, w) actually rendered for the per-tau panels; resolved in
# main() once the image size is known.
_REGION = (CROP_Y0, CROP_X0, CROP_H, CROP_W)


def _crop(arr: np.ndarray, region: tuple) -> np.ndarray:
    """Return a (y0, x0, h, w) region of an array."""
    y0, x0, h, w = region
    return arr[y0 : y0 + h, x0 : x0 + w]


def _faded_background(green: np.ndarray, region: tuple) -> np.ndarray:
    """A region of the green channel as a faded RGB canvas for overlays."""
    g = _crop(green, region)
    rgb = np.stack([g] * 3, axis=-1).astype(np.float32) * BG_FADE
    return rgb.astype(np.uint8)


def _save(rgb: np.ndarray, label: str, out_name: str) -> None:
    """Save a bare RGB image — no title, axes, colorbar or any other text."""
    out_path = OUT_DIR / out_name
    Image.fromarray(rgb).save(out_path)
    print(f"Saved: {out_path}  ({label})")


def _edge_mask(graph, shape: tuple) -> np.ndarray:
    """Rasterise a result graph's edge paths into a dilated binary mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for _, _, data in graph.edges(data=True):
        path = data.get("path", [])
        if len(path) == 0:
            continue
        pts = np.asarray(path, dtype=np.int64)
        ys, xs = pts[:, 0], pts[:, 1]
        inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        mask[ys[inside], xs[inside]] = 1
    if EDGE_DILATE > 0:
        ksize = EDGE_DILATE * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, kernel)
    return mask


def _render_graph(green: np.ndarray, graph, region: tuple) -> np.ndarray:
    """Draw a pixel-level result graph over the faded background."""
    y0, x0, h, w = region
    rgb = _faded_background(green, region)
    rgb[_crop(_edge_mask(graph, green.shape), region) > 0] = EDGE_COLOR
    for node in graph.nodes:
        y, x = int(node[0]) - y0, int(node[1]) - x0
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(rgb, (x, y), NODE_RADIUS, NODE_COLOR, -1, cv2.LINE_AA)
    return rgb


def _densest_window(diff: np.ndarray, size: int) -> tuple:
    """Locate the (y0, x0) of the size×size window with the most set pixels."""
    h, w = diff.shape
    size = min(size, h, w)
    ii = np.pad(diff.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, by, bx = -1, 0, 0
    for y in range(0, h - size + 1, 20):
        for x in range(0, w - size + 1, 20):
            cnt = ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x]
            if cnt > best:
                best, by, bx = cnt, y, x
    return by, bx, size


def _render_diff(green: np.ndarray, graph_ref, graph_new) -> np.ndarray:
    """3-colour overlay of two reconstructions, cropped to where they differ."""
    m_ref = _edge_mask(graph_ref, green.shape) > 0
    m_new = _edge_mask(graph_new, green.shape) > 0

    if DIFF_CROP is not None:
        y0, x0, size = DIFF_CROP
    else:
        changed = m_ref ^ m_new
        y0, x0, size = _densest_window(changed, DIFF_CROP_SIZE)
        print(f"  diff crop auto-located at (y0={y0}, x0={x0}, size={size})")
    region = (y0, x0, size, size)

    rgb = _faded_background(green, region)
    b_ref = _crop(m_ref, region)
    b_new = _crop(m_new, region)
    rgb[b_ref & b_new] = DIFF_BOTH
    rgb[b_ref & ~b_new] = DIFF_ONLY_REF
    rgb[~b_ref & b_new] = DIFF_ONLY_NEW
    return rgb


def main() -> None:
    global _REGION
    logging.disable(logging.WARNING)  # silence the linker's per-stage logs

    image = np.array(Image.open(f"{BASE_PATH}/image.png"))
    mask = np.array(Image.open(f"{BASE_PATH}/mask.png"))
    annotation = np.array(Image.open(f"{BASE_PATH}/weka.png"))
    green = image[:, :, 1] if image.ndim == 3 else image

    if USE_FULL_IMAGE:
        _REGION = (0, 0, green.shape[0], green.shape[1])
    print(f"Render region (y0, x0, h, w) = {_REGION}")

    # ── Skeleton GT reference panel ─────────────────────────────────────────
    label = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)
    if label is not None:
        gt_rgb = _faded_background(green, _REGION)
        gt_rgb[_crop(label, _REGION) > 0] = GT_COLOR
        _save(gt_rgb, "skeleton GT", "viz_prune_result_gt.png")
    else:
        print(f"  skip GT: {BASE_PATH}/label.png not found")

    # ── Run the full pipeline once per tau ──────────────────────────────────
    graphs: dict[int, object] = {}
    for tau in PRUNE_THRESHOLDS:
        linker = AnnotationGrowLinker(
            offset_px=50,
            bg_kernel_size=5,
            clahe_grid=(768, 768),
            clahe_clip=30.0,
            sato_sigmas_start=1,
            sato_sigmas_stop=4,
            prune_threshold=float(tau),
        )
        graph = linker.run(image, mask, annotation).graph
        graphs[tau] = graph
        is_ref = tau == TAU_REF
        label_txt = (
            f"tau = {tau}" + (" (default)" if is_ref else "")
            + f"  | {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        _save(_render_graph(green, graph, _REGION), label_txt,
              f"viz_prune_result_t{tau}.png")

    # ── Difference panel: TAU_REF vs DIFF_TAU ───────────────────────────────
    if TAU_REF in graphs and DIFF_TAU in graphs:
        diff_rgb = _render_diff(green, graphs[TAU_REF], graphs[DIFF_TAU])
        _save(diff_rgb,
              f"diff tau={TAU_REF} vs tau={DIFF_TAU} "
              f"(gray=same, green=removed, red=spurious)",
              "viz_prune_result_diff.png")
    else:
        print(f"  skip diff: need tau {TAU_REF} and {DIFF_TAU} in PRUNE_THRESHOLDS")


if __name__ == "__main__":
    main()
