"""Crop showcase for the annotation_grow ablations.

Companion visualisation to tools/ablation_annotation_grow.py. Where that script
reports the *quantitative* effect of each switch (hd95 / avg_hd / clDice over the
dataset), this tool shows the *qualitative* effect on ONE sample: it runs the
full ablated inference for every configuration and crops each one's result to the
same windows, so the structural difference is visible side by side.

Two kinds of panel are written for each crop:

  RESULT (confusion overlay) — for every config in VIS_CONFIGS. The SAME
  confusion-matrix render as tools/evaluate_dataset.py (save_overlay_visualization):
  a black canvas with the reconstruction scored against the GT topology,
      green  = GT only           (missed)
      red    = Pred only         (spurious)
      yellow = GT ∩ Pred overlap (correct, within the per-sample tolerance).
  These are exactly what viz_crop_showcase.py would crop from
  output/ref/<config>/vis/<ID>.png. File: {ID}_crop{i}_{config}.png

  COST MAP (enhanced grayscale) — additionally, for every config in
  COSTMAP_CONFIGS (the image-enhancement group). The enhanced image that feeds
  the cost map (build_enhanced_image_ablated). Enhancement barely moves the final
  topology, so the result overlay washes it out; its effect (background
  suppression, contrast, vesselness continuity) shows on the cost-map basis.
  File: {ID}_crop{i}_{config}_costmap.png

  COST / DISTANCE (matplotlib, with colorbar) — additionally, for every config
  in COST_DIST_CONFIGS (the connection-step group: full/linear_cost/uniform_cost).
  The Dijkstra traversal cost field (build_cost_map_ablated) rendered like
  tools/viz/viz_cost_map.py (viridis_r: bright=low cost=preferred path,
  ROI-outside masked dark #1a1a1a), and the resulting accumulated-distance
  field (multi_source_dijkstra's dist_map) rendered like
  tools/viz/viz_region_grow.py (cool, 0..99th-percentile, ROI-outside masked
  black). Each config keeps its own colour scale (shown in the colorbar) since
  cost_mode changes the underlying units. Shows how cost_mode reshapes bridge
  routing. Files: {ID}_crop{i}_{config}_cost.png, {ID}_crop{i}_{config}_dist.png

Layout otherwise mirrors tools/viz/viz_crop_showcase.py:

  1. {ID}_green_boxed.png        green channel with a red box per crop (a, b, …)
  2. {ID}_crop{i}_green.png      the raw green crop

All renders share the sample's dimensions, so one crop window is valid across
every config. Edit SAMPLE_ID / CROPS / the *_CONFIGS lists below before running.

Run:
    uv run python tools/viz/viz_ablation_grow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_reconstruction.algorithms.annotation_grow.dijkstra import (  # noqa: E402
    get_components,
    multi_source_dijkstra,
)
from neural_reconstruction.core.preprocessing import (  # noqa: E402
    dilate_epidermis_vertically,
)
from neural_reconstruction.core.topology import TopologyBuilder  # noqa: E402
from tools.ablation_annotation_grow import (  # noqa: E402
    ABLATIONS,
    FIXED_PARAMS,
    build_cost_map_ablated,
    build_enhanced_image_ablated,
    run_ablated_inference,
)
from tools.evaluate_dataset import save_overlay_visualization  # noqa: E402

# === Edit these ===========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_0510"

# SAMPLE_ID = "S558-2_a"
SAMPLE_ID = "S3091-2_b"

# RESULT (confusion overlay) panels — all ablation configs. Names must exist in
# ABLATIONS (the first there is ``full``, the reference).
VIS_CONFIGS = [
    "full",          # 完整 AEL (reference)
    "no_wth",        # 移除頂帽去背景
    "no_clahe",      # 移除 CLAHE
    "no_sato",       # 移除 Sato 濾波
    "no",            # 移除全部前處理
    "linear_cost",   # 線性成本
    "uniform_cost",  # 均勻成本 (straight-line bridges)
    "no_prune",      # 不裁剪邊 (τ)
    "no_mst",        # 不取最小生成森林 (cycles allowed)
    "no_connect",    # 完全不連結 (annotation bodies only, no bridges)
]

# COST MAP panels — additionally rendered (as {config}_costmap.png) for the
# image-enhancement group, whose effect is invisible in the result overlay but
# clear on the cost-map basis. Set to [] to skip the extra cost-map panels.
COSTMAP_CONFIGS = [
    "full",          # 完整 AEL (reference)
    "no_wth",        # 移除頂帽去背景
    "no_clahe",      # 移除 CLAHE
    "no_sato",       # 移除 Sato 濾波
    "no",            # 移除全部前處理
]

# COST / DISTANCE panels — additionally rendered ({config}_cost.png /
# {config}_dist.png) for the connection-step group. cost_mode reshapes the
# Dijkstra traversal cost, which reshapes the accumulated-distance field that
# drives bridge routing; neither is visible in the result overlay. Set to []
# to skip.
COST_DIST_CONFIGS = [
    "full",          # exp cost (reference)
    "linear_cost",   # linear cost
    "uniform_cost",  # uniform cost (straight-line bridges)
]
# Colour scale: cost -> viridis_r (viz_cost_map.py), dist -> cool (viz_region_grow.py).
# Both are fixed (not read off each config's own full-frame range) so the three
# configs share one scale and are directly comparable.
COST_VMIN, COST_VMAX = 0.0, 1.7  # exp cost tops out at exp(1)-1 ≈ 1.718
# dist vmax is fixed (not the full-frame 99th percentile) because the crops only
# ever reach ~25-47; a full-frame percentile (>100) washed them out to flat cyan.
DIST_VMAX = 40.0

# Crop regions. Each entry is (x, y, size): (x, y) top-left corner, square of
# side `size`. Windows below were located by tools/viz/find_ablation_crops.py.
CROPS = [
    (4140, 320, 90),    # connection group diverges hard here (result overlay)
    (2950, 175, 90),    # enhancement most visible here (cost-map panels)
]

# # S558-2_a (original showcase crops)
# CROPS = [
#     (3950, 900, 75),
#     (6550, 825, 75),
# ]

OUTPUT_DIR = PROJECT_ROOT / "output" / "ablation_grow_showcase"

# clDice / overlap tolerance as a physical distance (matches evaluate_dataset).
# Converted to pixels per sample via px_um.json: tolerance_px = TOL_UM / px_um.
TOL_UM = 1.28

# Red box appearance (drawn on the boxed green-channel overview).
BOX_COLOR = (0, 0, 255)  # BGR -> red
LABEL_COLOR = (0, 0, 255)  # BGR -> red
LABEL_FONT = cv2.FONT_HERSHEY_TRIPLEX

# Box/label sizes are specified at the width the overview is actually *viewed*
# at (DISPLAY_WIDTH px), then scaled up to the image's real width so they look
# the same regardless of source resolution. scale = img_width / DISPLAY_WIDTH.
DISPLAY_WIDTH = 742
FONT_SIZE_PX = 12       # label height (px) as seen at DISPLAY_WIDTH
BOX_THICKNESS_PX = 2.5  # box line width (px) at DISPLAY_WIDTH
LABEL_MARGIN_PX = 5.0   # gap between box and label (px) at DISPLAY_WIDTH
# ==========================================================================

_SWITCHES_BY_NAME = dict(ABLATIONS)


def _scaled_styles(img_width: int):
    """Box thickness, label gap, cv2 font scale + stroke for this image width.

    Sizes are defined at DISPLAY_WIDTH and scaled by img_width / DISPLAY_WIDTH.
    cv2's fontScale is not pixels, so we calibrate it against the font's actual
    pixel height at scale 1.0 to hit the requested on-image height.
    """
    scale = img_width / DISPLAY_WIDTH
    ref_h = cv2.getTextSize("a", LABEL_FONT, 1.0, 1)[0][1]
    font_scale = FONT_SIZE_PX * scale / ref_h
    return {
        "box_thickness": max(1, round(BOX_THICKNESS_PX * scale)),
        "label_margin": max(1, round(LABEL_MARGIN_PX * scale)),
        "font_scale": font_scale,
        # stroke must grow with the font or large text turns into outlines.
        "label_thickness": max(1, round(font_scale * 2.2)),
    }


def load_inputs(sample_id: str):
    """Load image / mask / annotation / label for a sample."""
    p = DATA_DIR / sample_id
    image = np.array(Image.open(p / "image.png"))
    mask = np.array(Image.open(p / "mask.png"))
    annotation = np.array(Image.open(p / "weka.png"))
    label = np.array(Image.open(p / "label.png"))
    green = image[:, :, 1] if image.ndim == 3 else image
    return image, green, mask, annotation, label


def load_tolerance_px(sample_id: str) -> int:
    """Per-sample overlap tolerance in pixels, as evaluate_dataset computes it."""
    px_um_path = DATA_DIR / "px_um.json"
    ratio = None
    if px_um_path.exists():
        ratio = json.load(px_um_path.open()).get(sample_id)
    return int(TOL_UM / (ratio if ratio is not None else TOL_UM))


def render_confusion(name, image, mask, annotation, label, tolerance_px, full_dir):
    """Full-frame confusion overlay (GT vs reconstruction) for a config (BGR)."""
    pred_graph, roi_mask = run_ablated_inference(
        image, mask, annotation, switches=_SWITCHES_BY_NAME[name]
    )
    gt_label = label[:, :, 0] if label.ndim == 3 else label
    roi_label = cv2.bitwise_and(gt_label, gt_label, mask=roi_mask)
    gt_graph = TopologyBuilder().build_seed_graph(roi_label)

    # save_overlay_visualization writes <full_dir>/<name>.png; read it back so the
    # appearance is byte-identical to evaluate_dataset's vis output.
    save_overlay_visualization(
        sample_id=name,
        image_path=None,
        roi_image=image[:, :, 1] if image.ndim == 3 else image,
        pred_graph=pred_graph,
        gt_graph=gt_graph,
        vis_dir=full_dir,
        match_tolerance_px=tolerance_px,
    )
    info = f"pred nodes={pred_graph.number_of_nodes()} edges={pred_graph.number_of_edges()}"
    return cv2.imread(str(full_dir / f"{name}.png"), cv2.IMREAD_COLOR), info


def render_costmap(name, green, mask):
    """Full-frame enhanced grayscale image (cost-map basis) for a config (BGR).

    This is the input to build_cost_map_ablated; the enhancement switches change
    it directly (no_wth -> bright background survives, no_clahe -> flat contrast,
    no_sato -> fibres not tube-enhanced). Rendered as 3-channel gray so it
    crops/saves like the other panels.
    """
    sw = _SWITCHES_BY_NAME[name]
    roi_mask = dilate_epidermis_vertically(mask, offset_px=FIXED_PARAMS["offset_px"])
    enhanced = build_enhanced_image_ablated(
        green=green,
        roi_mask=roi_mask,
        use_wth=sw["use_wth"],
        use_clahe=sw["use_clahe"],
        use_sato=sw["use_sato"],
        bg_kernel_size=FIXED_PARAMS["bg_kernel_size"],
        clahe_clip=FIXED_PARAMS["clahe_clip"],
        clahe_grid=FIXED_PARAMS["clahe_grid"],
        sato_sigmas=range(
            FIXED_PARAMS["sato_sigmas_start"], FIXED_PARAMS["sato_sigmas_stop"]
        ),
    )
    info = f"enhanced range=[{int(enhanced.min())},{int(enhanced.max())}]"
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), info


def render_cost_dist(name, green, mask, annotation):
    """Raw cost_map + dist_map + roi_mask (full-frame, unrendered) for a config."""
    sw = _SWITCHES_BY_NAME[name]
    roi_mask = dilate_epidermis_vertically(mask, offset_px=FIXED_PARAMS["offset_px"])
    enhanced = build_enhanced_image_ablated(
        green=green,
        roi_mask=roi_mask,
        use_wth=sw["use_wth"],
        use_clahe=sw["use_clahe"],
        use_sato=sw["use_sato"],
        bg_kernel_size=FIXED_PARAMS["bg_kernel_size"],
        clahe_clip=FIXED_PARAMS["clahe_clip"],
        clahe_grid=FIXED_PARAMS["clahe_grid"],
        sato_sigmas=range(
            FIXED_PARAMS["sato_sigmas_start"], FIXED_PARAMS["sato_sigmas_stop"]
        ),
    )
    cost_map = build_cost_map_ablated(enhanced, mode=str(sw["cost_mode"]))

    roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
    annotation_bin = (roi_annotation > 127).astype(np.uint8)
    annot_labeled = get_components(annotation_bin)
    _owner, dist_map, _py, _px = multi_source_dijkstra(
        cost_map, annot_labeled,
        connectivity=FIXED_PARAMS["connectivity"], roi_mask=(roi_mask > 127),
    )

    finite_dist = dist_map[np.isfinite(dist_map)]
    dist_hi = f"{finite_dist.max():.1f}" if finite_dist.size else "nan"
    info = (
        f"cost range=[{cost_map.min():.3f},{cost_map.max():.3f}]"
        f"  dist range=[0,{dist_hi}]"
    )
    return cost_map, dist_map, roi_mask, info


def save_cost_panel(cost_map, roi_mask, out_path, vmin=None, vmax=None):
    """Cost-map panel: viridis_r (bright=low cost=preferred), ROI-outside dark.

    Mirrors tools/viz/viz_cost_map.py's rendering exactly (same cmap + masking).
    """
    masked = np.ma.masked_where(roi_mask == 0, cost_map)
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="#1a1a1a")
    if vmin is None:
        vmin = COST_VMIN
    if vmax is None:
        vmax = COST_VMAX
    h, w = cost_map.shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5 * h / w))
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.04)
    fig.colorbar(im, cax=cax)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_dist_panel(dist_map, roi_mask, out_path, vmax=None):
    """Distance-map panel: cool colormap 0..vmax, ROI-outside masked black.

    Mirrors tools/viz/viz_region_grow.py's rendering exactly (same cmap +
    masking + percentile ceiling for contrast against the long Dijkstra tail).
    """
    dist_view = dist_map.astype(np.float32).copy()
    finite_mask = np.isfinite(dist_view) & (roi_mask > 0)
    if vmax is None:
        vmax = DIST_VMAX
    dist_view[~finite_mask] = np.nan
    cmap = plt.get_cmap("cool").copy()
    cmap.set_bad(color="black")
    h, w = dist_map.shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5 * h / w))
    im = ax.imshow(dist_view, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.axis("off")
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.04)
    fig.colorbar(im, cax=cax)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def check_crop(x, y, size, shape, source_name):
    h, w = shape[:2]
    if x < 0 or y < 0 or x + size > w or y + size > h:
        raise ValueError(
            f"Crop ({x},{y},size={size}) is out of bounds for "
            f"'{source_name}' with size ({w},{h})"
        )


def crop_square(image, x, y, size):
    return image[y : y + size, x : x + size]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_dir = OUTPUT_DIR / "_full"  # full-frame renders, before cropping
    full_dir.mkdir(parents=True, exist_ok=True)

    for group in (VIS_CONFIGS, COSTMAP_CONFIGS, COST_DIST_CONFIGS):
        unknown = [c for c in group if c not in _SWITCHES_BY_NAME]
        if unknown:
            raise KeyError(
                f"unknown configs {unknown}; valid: {sorted(_SWITCHES_BY_NAME)}"
            )

    # --- Load inputs ------------------------------------------------------
    image, green, mask, annotation, label = load_inputs(SAMPLE_ID)
    tolerance_px = load_tolerance_px(SAMPLE_ID)
    print(f"[load] {SAMPLE_ID}  green shape={green.shape}  tol={tolerance_px}px")

    for x, y, size in CROPS:
        check_crop(x, y, size, green.shape, "green channel")

    # --- Render each config once on the full frame ------------------------
    # RESULT overlays for VIS_CONFIGS; COST MAP for COSTMAP_CONFIGS.
    confusion_full = {}
    for name in VIS_CONFIGS:
        confusion_full[name], info = render_confusion(
            name, image, mask, annotation, label, tolerance_px, full_dir
        )
        print(f"[result ] {name:<13} {info}")

    costmap_full = {}
    for name in COSTMAP_CONFIGS:
        costmap_full[name], info = render_costmap(name, green, mask)
        print(f"[costmap] {name:<13} {info}")

    # Raw fields; cost/dist both use a fixed colour scale (COST_VMIN/VMAX,
    # DIST_VMAX above) shared across configs so the three are comparable.
    cost_full, dist_full = {}, {}
    for name in COST_DIST_CONFIGS:
        cost_map, dist_map, roi_mask_cd, info = render_cost_dist(
            name, green, mask, annotation
        )
        cost_full[name], dist_full[name] = cost_map, dist_map
        print(f"[costdist] {name:<12} {info}")
    # roi_mask doesn't depend on the config's switches, so any iteration's is fine.
    roi_mask_full = roi_mask_cd

    # --- Output 1: green-channel overview with red boxes ------------------
    boxed = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
    st = _scaled_styles(green.shape[1])
    for i, (x, y, size) in enumerate(CROPS):
        cv2.rectangle(
            boxed, (x, y), (x + size, y + size), BOX_COLOR, st["box_thickness"]
        )
        label_ch = chr(ord("a") + i)
        (tw, th), _ = cv2.getTextSize(
            label_ch, LABEL_FONT, st["font_scale"], st["label_thickness"]
        )
        org = (x, y - st["label_margin"])
        if org[1] - th < 0:
            org = (x + size + st["label_margin"], y + th)
        cv2.putText(
            boxed, label_ch, org, LABEL_FONT, st["font_scale"], LABEL_COLOR,
            st["label_thickness"], cv2.LINE_AA,
        )
    boxed_path = OUTPUT_DIR / f"{SAMPLE_ID}_green_boxed.png"
    cv2.imwrite(str(boxed_path), boxed)
    print(f"[save] {boxed_path}")

    # --- Outputs 2 & 3: per-crop regions ----------------------------------
    n_files = 1
    for i, (x, y, size) in enumerate(CROPS):
        green_crop = crop_square(green, x, y, size)
        cv2.imwrite(str(OUTPUT_DIR / f"{SAMPLE_ID}_crop{i}_green.png"), green_crop)
        n_files += 1

        # RESULT (confusion) panels.
        for name in VIS_CONFIGS:
            vis_crop = crop_square(confusion_full[name], x, y, size)
            cv2.imwrite(
                str(OUTPUT_DIR / f"{SAMPLE_ID}_crop{i}_{name}.png"), vis_crop
            )
            n_files += 1

        # COST MAP panels (suffixed so they sit beside the result panels).
        for name in COSTMAP_CONFIGS:
            cm_crop = crop_square(costmap_full[name], x, y, size)
            cv2.imwrite(
                str(OUTPUT_DIR / f"{SAMPLE_ID}_crop{i}_{name}_costmap.png"), cm_crop
            )
            n_files += 1

        # COST / DISTANCE panels (connection-step group; matplotlib, with colorbar).
        roi_crop = crop_square(roi_mask_full, x, y, size)
        for name in COST_DIST_CONFIGS:
            save_cost_panel(
                crop_square(cost_full[name], x, y, size), roi_crop,
                OUTPUT_DIR / f"{SAMPLE_ID}_crop{i}_{name}_cost.png",
            )
            save_dist_panel(
                crop_square(dist_full[name], x, y, size), roi_crop,
                OUTPUT_DIR / f"{SAMPLE_ID}_crop{i}_{name}_dist.png",
                vmax=DIST_VMAX,
            )
            n_files += 2

        print(f"[save] crop{i} ({x},{y},{size}): "
              f"{len(VIS_CONFIGS)} result + {len(COSTMAP_CONFIGS)} costmap + "
              f"{len(COST_DIST_CONFIGS)} cost/dist + green")

    print(f"\nDone. {n_files} files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
