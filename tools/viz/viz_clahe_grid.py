"""
CLAHE tile-size (n_tile) sweep visualisation.

Renders the CLAHE output for several tile sizes n_tile on the same fixed crop
used by the other paper viz scripts. Processing stops after the CLAHE step:

    green channel  ->  morphological background removal  ->  CLAHE(n_tile)

so the panels isolate the effect of n_tile alone (the clip limit and the
bg-removal step are held at their pipeline defaults). For reference the
bg-removed input (before CLAHE) is also saved.

Outputs (one panel each; bare plain-grayscale images, no title/axes/colorbar):
  viz_clahe_grid_bg_removed.png      — input to CLAHE
  viz_clahe_grid_n{n}.png            — CLAHE output for each n in TILE_SIZES
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

# Shared crop region: same as every other viz script.
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

# Background-removal kernel and CLAHE clip limit: pipeline defaults.
BG_KERNEL_SIZE = 5
CLAHE_CLIP = 30.0

# Tile sizes n_tile to sweep; 768 is the pipeline default.
TILE_SIZES = [704, 736, 768, 800, 832]
TILE_REF = 768

# Panel colour scale: plain grayscale on a fixed 0-255 scale, so the panels
# show the raw CLAHE output and stay directly comparable.
PANEL_CMAP = "gray"
# ==========================================================================

OUT_DIR = Path(__file__).parent


def _crop(arr: np.ndarray) -> np.ndarray:
    """Return the fixed crop used by the paper visualisations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


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

    # Reference panel: the CLAHE input.
    _save_panel(_crop(bg_removed), "bg removed (CLAHE input)",
                "viz_clahe_grid_bg_removed.png")

    # ── CLAHE for each tile size n_tile ─────────────────────────────────────
    for n in TILE_SIZES:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(n, n))
        clahe_out = clahe.apply(bg_removed)
        is_ref = n == TILE_REF
        label = f"n_tile = {n}" + (" (default)" if is_ref else "")
        _save_panel(_crop(clahe_out), label, f"viz_clahe_grid_n{n}.png")


if __name__ == "__main__":
    main()
