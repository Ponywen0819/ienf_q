"""stub_length_threshold (l_min) visualisation — a GT coverage error map.

Stub pruning removes skeleton branches that end in a boundary (degree-1) node
and are shorter than l_min pixels (see TopologyBuilder.to_simple_graph). It runs
on the final skeleton (annotation bodies + dilated MST bridges, skeletonised).
Pruning only removes edges, so skeleton(l=0) >= skeleton(l=3) >= skeleton(l=9).

The ablation table makes l_min=3 the optimum: l_min=0 keeps a few discretisation
spurs (only HD_avg marginally worse), while l_min>=5 deletes REAL short branches
and degrades every metric (fragmentation). The figure shows this asymmetry in
ONE crop swept over l = 0, 1, 3, 5, 7, 9, using the same per-pixel
prediction-vs-GT confusion map as the prune figure (pred and GT are each
dilated by COV_TOL before comparing):

    yellow  = skeleton & GT      (overlap / true positive)
    green   = GT, no skeleton    (missing / false negative — a real branch
                                  pruned away at this l_min)
    red     = skeleton, no GT    (spurious / false positive — a discretisation
                                  spur kept at this l_min)

So l=0 shows a few red spurs, l=3 is clean, and l=9 sprouts green as real
branches get deleted — the over-pruning that the table penalises.

The crop reuses the prune figure's sample (S3091-2_b) for consistency.

Run:
    uv run python tools/viz/viz_stub_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_reconstruction.algorithms.annotation_grow.dijkstra import (  # noqa: E402
    get_components,
)
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (  # noqa: E402
    minimum_spanning_forest,
    prune_edges,
)
from neural_reconstruction.algorithms.annotation_grow.skeleton import (  # noqa: E402
    build_result_graph,
)
from tools.ablation_annotation_grow import (  # noqa: E402
    FIXED_PARAMS,
    build_enhanced_image_ablated,
)
from tools.viz.analyze_prune_threshold import _component_graph  # noqa: E402

# === Edit these ===========================================================
DATA_DIR = Path("/home/pony/projects/ienf_q/data_0510")
SID = "S3091-2_b"
CROP = (404, 4293, 180)          # (y0, x0, win): a region rich in short branches

CHOSEN = 3
THRESHOLDS = [0, 1, 3, 5, 7, 9]  # l_min panels (under / chosen / over)
DILATE_RADIUS = 3                # bridge dilation, as in the pipeline
COV_TOL = 3                      # overlap tolerance, as in the prune figure

# Colours (BGR). Same confusion-map scheme as the prune figure.
COLOR_GT = (0, 235, 235)         # yellow, skeleton & GT overlap (true positive)
COLOR_MISSING = (0, 230, 0)      # green, GT only / no skeleton (false negative)
COLOR_ADDED = (60, 60, 255)      # red, skeleton only / no GT (false positive)
BG_DIM = 0.45
# ==========================================================================

OUT_DIR = Path(__file__).parent


def _skeleton_mask(mst, annotation_bin, l_min, shape) -> np.ndarray:
    """Rasterise the result-graph skeleton at this l_min to a boolean mask."""
    rg = build_result_graph(
        mst, annotation_bin, dilate_radius=DILATE_RADIUS,
        stub_length_threshold=l_min,
    )
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    for _u, _v, d in rg.edges(data=True):
        for y, x in d.get("path", []):
            if 0 <= y < H and 0 <= x < W:
                m[y, x] = True
    return m


def main() -> None:
    p = DATA_DIR / SID
    image = np.array(Image.open(p / "image.png"))
    mask = np.array(Image.open(p / "mask.png"))
    annotation = np.array(Image.open(p / "weka.png"))
    label = np.array(Image.open(p / "label.png"))

    g, roi_mask = _component_graph(image, mask, annotation)
    green = image[:, :, 1] if image.ndim == 3 else image
    enhanced = build_enhanced_image_ablated(
        green=green, roi_mask=roi_mask,
        use_wth=True, use_clahe=True, use_sato=True,
        bg_kernel_size=FIXED_PARAMS["bg_kernel_size"],
        clahe_clip=FIXED_PARAMS["clahe_clip"],
        clahe_grid=FIXED_PARAMS["clahe_grid"],
        sato_sigmas=range(FIXED_PARAMS["sato_sigmas_start"],
                          FIXED_PARAMS["sato_sigmas_stop"]),
    )
    ann2d = annotation[:, :, 0] if annotation.ndim == 3 else annotation
    annotation_bin = (cv2.bitwise_and(ann2d, ann2d, mask=roi_mask) > 127
                      ).astype(np.uint8)
    gt_bin = (label[:, :, 0] if label.ndim == 3 else label) > 127
    H, W = annotation_bin.shape

    # Skeleton at each l_min (one pipeline run each).
    mst = minimum_spanning_forest(
        prune_edges(g, threshold=FIXED_PARAMS["prune_threshold"]))
    skel = {l: _skeleton_mask(mst, annotation_bin, l, (H, W)) for l in THRESHOLDS}
    cov_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * COV_TOL + 1, 2 * COV_TOL + 1))
    gt_dil = cv2.dilate(gt_bin.astype(np.uint8), cov_kernel).astype(bool)
    skel_dil = {l: cv2.dilate(skel[l].astype(np.uint8), cov_kernel).astype(bool)
                for l in THRESHOLDS}

    y0, x0, win = CROP
    bg = (enhanced[y0:y0 + win, x0:x0 + win].astype(np.float64)
          * BG_DIM).astype(np.uint8)
    base = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    gt_win = gt_bin[y0:y0 + win, x0:x0 + win]

    # GT label panel (all yellow), the spatial reference.
    gt_panel = base.copy()
    gt_panel[gt_win] = COLOR_GT
    cv2.imwrite(str(OUT_DIR / "viz_stub_label.png"), gt_panel)
    print("Saved: viz_stub_label.png  (GT label reference)")

    # Per-pixel skeleton-vs-GT confusion map (three colours only), same as the
    # prune figure: yellow = overlap within tolerance, green = GT only
    # (missing), red = skeleton only (spurious).
    for l in THRESHOLDS:
        canvas = base.copy()
        skel_w = skel[l][y0:y0 + win, x0:x0 + win]
        skel_dil_w = skel_dil[l][y0:y0 + win, x0:x0 + win]
        gt_dil_w = gt_dil[y0:y0 + win, x0:x0 + win]
        overlap = (gt_win & skel_dil_w) | (skel_w & gt_dil_w)
        gt_only = gt_win & ~skel_dil_w
        skel_only = skel_w & ~gt_dil_w
        canvas[gt_only] = COLOR_MISSING
        canvas[skel_only] = COLOR_ADDED
        canvas[overlap] = COLOR_GT
        out = OUT_DIR / f"viz_stub_t{l}.png"
        cv2.imwrite(str(out), canvas)
        tag = " (chosen)" if l == CHOSEN else ""
        print(f"Saved: {out}  l_min={l}{tag}  "
              f"overlap/missing/spurious: "
              f"{int(overlap.sum())}/{int(gt_only.sum())}/{int(skel_only.sum())}")


if __name__ == "__main__":
    main()
