"""
CLAHE clip-limit (β) sweep visualisation.

Renders, for several CLAHE clip limits β on the same fixed crop used by the
other paper viz scripts, three views of the pipeline:

    green channel  ->  morphological background removal  ->  CLAHE(β)  ->  Sato
                                                                          |
                                                                  reconstruction

1. CLAHE output           — the raw CLAHE image; β barely moves it.
2. Sato vesselness        — Sato of the CLAHE output; β-driven contrast changes
                            are amplified here.
3. Reconstruction result  — the full AnnotationGrowLinker network reconstructed
                            with that β; this answers whether β changes the
                            *final* output at all.

For reference the bg-removed input (before CLAHE) and its Sato image are also
saved. All Sato panels share one normalisation scale, so they are directly
comparable across β.

Outputs (one panel each):
  viz_clahe_clip_bg_removed.png        — input to CLAHE
  viz_clahe_clip_bg_removed_sato.png   — Sato of the bg-removed input (no CLAHE)
  viz_clahe_clip_b{β}.png              — CLAHE output for each β in CLIP_LIMITS
  viz_clahe_clip_b{β}_sato.png         — Sato of the CLAHE output for each β
  viz_clahe_clip_b{β}_recon.png        — reconstructed network for each β
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.algorithms.annotation_grow.linker import (
    AnnotationGrowLinker,
)

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

# Shared crop region: same as every other viz script.
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200
# CROP_Y0, CROP_X0, CROP_H, CROP_W = 800,6525,200 , 200

# Background-removal and CLAHE tile size: pipeline defaults (see viz_preprocessing).
BG_KERNEL_SIZE = 5
CLAHE_TILE = 768
OFFSET_PX = 50  # epidermis ROI dilation, matches viz_region_grow.py

# Clip limits β to sweep; 30 is the pipeline default.
CLIP_LIMITS = [10, 20, 30, 40, 50]
BETA_REF = 30

# Sato vesselness scales: cost-map pipeline default (see cost_map.build_enhanced_image).
SATO_SIGMAS = (3, 4, 5, 6, 7)
# Halo padding for the Sato crop: 4·σ_max (gaussian truncate) plus a few px for
# the finite-difference Hessian, matching cost_map's strip padding.
SATO_PAD = int(np.ceil(4 * max(SATO_SIGMAS))) + 4

# Panel colour scale: plain grayscale on a fixed 0-255 scale, so the panels
# show the raw CLAHE output and stay directly comparable.
PANEL_CMAP = "gray"
# Reconstruction overlay colour (gold), drawn on a faded green background.
RECON_COLOR = np.array([1.00, 0.78, 0.10], dtype=np.float32)
# ==========================================================================

OUT_DIR = Path(__file__).parent


def _crop(arr: np.ndarray) -> np.ndarray:
    """Return the fixed crop used by the paper visualisations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _sato_crop(full: np.ndarray) -> np.ndarray:
    """Sato vesselness for the fixed crop.

    Sato is run on the crop plus a ``SATO_PAD`` halo so the result inside the
    crop matches a full-image call (no boundary artefacts), then the halo is
    discarded. Returns a float array — normalisation is left to the caller so
    multiple panels can share one scale.
    """
    H, W = full.shape
    gy0 = max(0, CROP_Y0 - SATO_PAD)
    gx0 = max(0, CROP_X0 - SATO_PAD)
    gy1 = min(H, CROP_Y0 + CROP_H + SATO_PAD)
    gx1 = min(W, CROP_X0 + CROP_W + SATO_PAD)

    region = full[gy0:gy1, gx0:gx1]
    sato = ski.filters.sato(region, sigmas=list(SATO_SIGMAS), black_ridges=False)

    cy0, cx0 = CROP_Y0 - gy0, CROP_X0 - gx0
    return sato[cy0 : cy0 + CROP_H, cx0 : cx0 + CROP_W]


def _save_panel(img: np.ndarray, label: str, out_name: str) -> None:
    """Render a sweep panel as a bare plain-grayscale image on a fixed 0-255 scale.

    No title, axes, colorbar or any other text.
    """
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(img, cmap=PANEL_CMAP, vmin=0, vmax=255)
    ax.set_axis_off()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}  ({label})")


def _save_rgb_panel(img: np.ndarray, label: str, out_name: str) -> None:
    """Render a bare HxWx3 RGB panel (no title/axes), same framing as _save_panel."""
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(img)
    ax.set_axis_off()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}  ({label})")


def _reconstruction_panel(graph, green_crop: np.ndarray,
                          roi_crop: np.ndarray) -> np.ndarray:
    """Rasterise a reconstructed network graph onto the faded green crop.

    Edge ``path`` points and node coordinates are painted in ``RECON_COLOR``;
    the network is dilated by 1px so the 1-px paths stay visible at panel size.
    Outside-ROI pixels are blacked out, matching viz_bridge_skeleton.py.
    """
    H, W = green_crop.shape[0], green_crop.shape[1]
    net = np.zeros((CROP_H, CROP_W), dtype=np.uint8)

    def _plot(py: int, px: int) -> None:
        ly, lx = py - CROP_Y0, px - CROP_X0
        if 0 <= ly < CROP_H and 0 <= lx < CROP_W:
            net[ly, lx] = 1

    for _u, _v, data in graph.edges(data=True):
        for py, px in data.get("path", []):
            _plot(int(py), int(px))
    for ny, nx_ in graph.nodes():
        _plot(int(ny), int(nx_))

    net = cv2.dilate(net, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    disp = np.stack([green_crop] * 3, axis=-1).astype(np.float32) / 255.0 * 0.5
    disp[net.astype(bool)] = RECON_COLOR
    disp[roi_crop == 0] = 0.0
    return disp


def main() -> None:
    # ── load inputs ─────────────────────────────────────────────────────────
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    green = image_rgb[:, :, 1]
    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    annotation = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)

    green_crop = _crop(green)
    roi_crop = _crop(dilate_epidermis_vertically(mask, offset_px=OFFSET_PX))

    # ── green channel → morphological background removal ────────────────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BG_KERNEL_SIZE, BG_KERNEL_SIZE)
    )
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    bg_removed = cv2.subtract(green, background)

    # Reference panel: the CLAHE input.
    _save_panel(_crop(bg_removed), "bg removed (CLAHE input)",
                "viz_clahe_clip_bg_removed.png")

    # ── CLAHE for each clip limit β, then Sato of each result ───────────────
    # CLAHE outputs are saved immediately; Sato crops are collected first so
    # every Sato panel can share one normalisation scale (fair β comparison).
    sato_crops: dict[str, np.ndarray] = {
        "viz_clahe_clip_bg_removed_sato.png": _sato_crop(bg_removed),
    }
    sato_labels: dict[str, str] = {
        "viz_clahe_clip_bg_removed_sato.png": "Sato of bg removed (no CLAHE)",
    }

    for beta in CLIP_LIMITS:
        clahe = cv2.createCLAHE(
            clipLimit=float(beta), tileGridSize=(CLAHE_TILE, CLAHE_TILE)
        )
        clahe_out = clahe.apply(bg_removed)
        is_ref = beta == BETA_REF
        suffix = " (default)" if is_ref else ""
        _save_panel(_crop(clahe_out), f"β = {beta}{suffix}",
                    f"viz_clahe_clip_b{beta}.png")

        name = f"viz_clahe_clip_b{beta}_sato.png"
        sato_crops[name] = _sato_crop(clahe_out)
        sato_labels[name] = f"Sato of β = {beta}{suffix}"

    # Shared 0-255 scale across all Sato panels.
    vmin = min(c.min() for c in sato_crops.values())
    vmax = max(c.max() for c in sato_crops.values())
    scale = 255.0 / (vmax - vmin) if vmax > vmin else 0.0

    for name, crop in sato_crops.items():
        panel = ((crop - vmin) * scale).astype(np.uint8)
        _save_panel(panel, sato_labels[name], name)

    # ── full reconstruction for each β ──────────────────────────────────────
    # Runs the real AnnotationGrowLinker per β (other params held at the values
    # the rest of the viz scripts use) and overlays the resulting network.
    for beta in CLIP_LIMITS:
        linker = AnnotationGrowLinker(
            offset_px=OFFSET_PX,
            bg_kernel_size=BG_KERNEL_SIZE,
            clahe_clip=float(beta),
            clahe_grid=(CLAHE_TILE, CLAHE_TILE),
            sato_sigmas_start=min(SATO_SIGMAS),
            sato_sigmas_stop=max(SATO_SIGMAS) + 1,
        )
        result = linker.run(image_rgb, mask, annotation)
        is_ref = beta == BETA_REF
        suffix = " (default)" if is_ref else ""
        panel = _reconstruction_panel(result.graph, green_crop, roi_crop)
        _save_rgb_panel(panel, f"reconstruction β = {beta}{suffix}",
                        f"viz_clahe_clip_b{beta}_recon.png")


if __name__ == "__main__":
    main()
