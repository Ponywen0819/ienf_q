"""PSD-output variant of tools/viz/viz_cost_map.py.

viz_cost_map.py runs top-level (no ``main()`` guard), so it can't be safely
imported for reuse — importing it would execute the whole pipeline and its
own PNG writes. This script re-derives the same cost map directly and
writes two PSDs instead of two PNGs:

  * "{ID}_cost_map.psd"      — full-frame viridis_r cost map (opaque
                                "cost_map" layer) + a transparent "crop_box"
                                layer marking the zoom region, plus a
                                "colorbar" layer in a margin to the right
                                (matplotlib-rendered, vmin/vmax tick labels).
  * "{ID}_cost_map_zoom.psd" — the same colour scale, cropped to that
                                region, single layer (no box, no colorbar —
                                matches viz_cost_map.py's own zoom render).

Uses psd-tools (see tools/viz/viz_ablation_grow_psd for why: pytoshop's
output only opens in lenient readers, not real Photoshop).

Run:
    uv run python tools/viz/viz_cost_map_psd/viz_cost_map_psd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neural_reconstruction.core.preprocessing import (  # noqa: E402
    dilate_epidermis_vertically,
)

BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0510/{IMAGE_ID}"

CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

BOX_COLOR = (0, 0, 255)  # BGR -> red
# Same thickness/width ratio used by the sibling *_psd scripts' box style.
BOX_THICKNESS_RATIO = 2.5 / 742
BAD_COLOR = "#1a1a1a"  # ROI-outside colour, matches viz_cost_map.py's cmap.set_bad

COLORBAR_WIDTH = 220  # px, margin added to the right of the full-frame image
COLORBAR_DPI = 150

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def crop(arr: np.ndarray) -> np.ndarray:
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def colorize(values: np.ndarray, valid_mask: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """viridis_r colour map, bright=low cost, ROI-outside painted BAD_COLOR."""
    masked = np.ma.masked_where(~valid_mask, values)
    norm = np.ma.clip((masked - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color=BAD_COLOR)
    rgba = cmap(norm)
    return (np.asarray(rgba)[..., :3] * 255).astype(np.uint8)


def render_colorbar(height_px: int, vmin: float, vmax: float) -> np.ndarray:
    """Standalone viridis_r colorbar (vmin..vmax, tick labels), rendered as
    a transparent RGBA strip COLORBAR_WIDTH x height_px so it can sit in a
    margin beside the cost-map layer instead of baked into one flat image."""
    fig = plt.figure(figsize=(COLORBAR_WIDTH / COLORBAR_DPI, height_px / COLORBAR_DPI), dpi=COLORBAR_DPI)
    cax = fig.add_axes((0.12, 0.03, 0.30, 0.94))
    mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=plt.get_cmap("viridis_r")
    )
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=14, colors="black")
    fig.patch.set_alpha(0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    if buf.shape[:2] != (height_px, COLORBAR_WIDTH):
        # figsize-in-inches -> pixel rounding can be off by ~1px; pad/crop.
        buf = np.asarray(Image.fromarray(buf).resize((COLORBAR_WIDTH, height_px), Image.NEAREST))
    return buf


def build_box_layer(shape: tuple[int, int]) -> np.ndarray:
    """Transparent RGBA canvas with just the red crop-region rectangle."""
    h, w = shape
    thickness = max(1, round(w * BOX_THICKNESS_RATIO))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(
        canvas, (CROP_X0, CROP_Y0), (CROP_X0 + CROP_W, CROP_Y0 + CROP_H),
        BOX_COLOR, thickness, cv2.LINE_AA,
    )
    alpha = canvas[:, :, 2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255
    rgba[:, :, 3] = alpha
    return rgba


def main() -> None:
    raw = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if raw is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"mask.png not found under {BASE_PATH}")

    image = raw[:, :, 1]  # green channel
    roi_mask = dilate_epidermis_vertically(mask, offset_px=50)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.subtract(image, background)

    clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(768, 768))
    image = clahe.apply(image)

    image = ski.filters.sato(image, sigmas=range(1, 4), black_ridges=False)
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    cost_map = np.exp(1.0 - (image.astype(np.float32) / 255.0)) - 1.0

    valid = roi_mask > 0
    vmin = float(cost_map[valid].min())
    vmax = float(cost_map[valid].max())
    print(f"[cost] range=[{vmin:.3f},{vmax:.3f}]  shape={cost_map.shape}")

    h, w = cost_map.shape
    cost_rgb = colorize(cost_map, valid, vmin, vmax)

    psd_full = PSDImage.new("RGBA", (w + COLORBAR_WIDTH, h))
    psd_full.create_pixel_layer(Image.fromarray(cost_rgb), name="cost_map", top=0, left=0)
    psd_full.create_pixel_layer(
        Image.fromarray(build_box_layer((h, w))), name="crop_box", top=0, left=0
    )
    psd_full.create_pixel_layer(
        Image.fromarray(render_colorbar(h, vmin, vmax)), name="colorbar", top=0, left=w
    )
    full_path = OUTPUT_DIR / f"{IMAGE_ID}_cost_map.psd"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    psd_full.save(full_path)
    print(f"[save] {full_path}")

    zoom_rgb = colorize(crop(cost_map), crop(roi_mask) > 0, vmin, vmax)
    zh, zw = zoom_rgb.shape[:2]
    psd_zoom = PSDImage.new("RGBA", (zw, zh))
    psd_zoom.create_pixel_layer(Image.fromarray(zoom_rgb), name="cost_map_zoom", top=0, left=0)
    zoom_path = OUTPUT_DIR / f"{IMAGE_ID}_cost_map_zoom.psd"
    psd_zoom.save(zoom_path)
    print(f"[save] {zoom_path}")


if __name__ == "__main__":
    main()
