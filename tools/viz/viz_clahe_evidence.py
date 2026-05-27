"""
Visual evidence for the CLAHE clip-limit (β) trade-off.

The β sweep panels look near-identical at full crop scale, so this zooms into
two small sub-patches of the shared crop and renders each at β = 10, 30, 50.
Processing is the same as viz_clahe_clip:

    green channel  ->  morphological background removal  ->  CLAHE(β)

Two sub-patches, picked with the skeleton GT (label.png):

  * Fibre patch     — contains fibres. Shows that a low β under-enhances
                      contrast, leaving fibres dim (the cost of small β).
  * Background patch — fibre-free. Shows that a high β amplifies background
                      grain / noise (the cost of large β).

Together the two rows make β = 30 visible as the balance point: enough
contrast on fibres without over-amplifying background noise.

Outputs — 6 bare panels (no title/axes; colorbar kept, ticks but no label),
same style as the viz_clahe_clip panels:
  viz_clahe_evidence_fibre_b{β}.png
  viz_clahe_evidence_bg_b{β}.png
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

# Shared crop region: same as every other viz script.
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

# Background-removal and CLAHE tile size: pipeline defaults.
BG_KERNEL_SIZE = 5
CLAHE_TILE = 768

# β values to compare.
BETAS = [10, 30, 50]

# Sub-patches, in crop-local coordinates: (y, x, size). Picked with the GT —
# FIBRE_PATCH is the most fibre-dense 50×50 window, BG_PATCH is fibre-free.
FIBRE_PATCH = (140, 30, 50)
BG_PATCH = (0, 70, 50)

# Panel colour scale: same treatment as viz_clahe_clip / viz_bg_remove.
PANEL_CMAP = "magma"
PANEL_GAMMA = 0.45
# ==========================================================================

OUT_DIR = Path(__file__).parent


def _subpatch(full_img: np.ndarray, patch: tuple[int, int, int]) -> np.ndarray:
    """Extract a crop-local sub-patch from a full-resolution image."""
    py, px, size = patch
    y0, x0 = CROP_Y0 + py, CROP_X0 + px
    return full_img[y0 : y0 + size, x0 : x0 + size]


def _save_panel(img: np.ndarray, label: str, out_name: str) -> None:
    """Render a bare panel: gamma-lifted colormap, fixed 0-255 scale, no text.

    No title or axes; the colorbar keeps its ticks and numbers but no label.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    norm = PowerNorm(gamma=PANEL_GAMMA, vmin=0, vmax=255)
    im = ax.imshow(img, cmap=PANEL_CMAP, norm=norm, interpolation="nearest")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=12)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path}  ({label}  std={img.astype(float).std():.2f})")


def main() -> None:
    # ── green channel → morphological background removal ────────────────────
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    green = image_rgb[:, :, 1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BG_KERNEL_SIZE, BG_KERNEL_SIZE)
    )
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    bg_removed = cv2.subtract(green, background)

    # ── CLAHE for each β, extract both sub-patches ──────────────────────────
    for beta in BETAS:
        clahe = cv2.createCLAHE(
            clipLimit=float(beta), tileGridSize=(CLAHE_TILE, CLAHE_TILE)
        )
        clahe_out = clahe.apply(bg_removed)
        _save_panel(_subpatch(clahe_out, FIBRE_PATCH),
                    f"fibre β={beta}", f"viz_clahe_evidence_fibre_b{beta}.png")
        _save_panel(_subpatch(clahe_out, BG_PATCH),
                    f"background β={beta}", f"viz_clahe_evidence_bg_b{beta}.png")


if __name__ == "__main__":
    main()
