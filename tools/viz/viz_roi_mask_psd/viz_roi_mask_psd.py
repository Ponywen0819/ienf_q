"""PSD-output variant of tools/viz/viz_roi_mask.py.

viz_roi_mask.py runs top-level (no ``main()`` guard), so it can't be safely
imported for reuse — importing it would execute the whole thing, including
its own PNG writes. This script re-derives the same three masks (aux_mask /
unconstrained-dilation / roi_mask) directly against
``dilate_epidermis_vertically`` and writes each as a PSD with:

  * "background"    — cropped green channel (opaque), downsampled to
                       OUT_W wide (matching viz_roi_mask.py's own render
                       resolution: figsize=(10,...), dpi=150) and placed
                       below a margin that hosts the axis widget
  * "overlay_fill"  — the mask in question, translucent (real alpha, not
                       baked into the pixels, so opacity/color stay editable)
  * "mask_outline"  — the original epidermis mask's contour (magenta)
  * "coord_axis"    — the "(0,0)" / "x" / "y" coordinate-system widget
                       viz_roi_mask.py draws in the margin outside the image
  * "y_max_annotation" — aux_mask only: the dashed sampling line + arrow +
                       "y_max" label pointing at the mask's top edge

Resolution: downsampling uses cv2.INTER_NEAREST (no blending/blur) rather
than viz_roi_mask.py's implicit smooth resampling, per explicit request.
The background is deliberately re-rendered at viz_roi_mask.py's own on-screen
resolution (~1500px wide) rather than kept at native crop resolution,
because the axis-widget font/arrow sizes below are viz_roi_mask.py's own
point sizes — matching its resolution is what makes them line up correctly
without extra scale-factor guesswork, and it's what leaves room for the
margin the axis widget lives in.

Uses psd-tools (see tools/viz/viz_ablation_grow_psd for why: pytoshop's
output only opens in lenient readers, not real Photoshop).

Run:
    uv run python tools/viz/viz_roi_mask_psd/viz_roi_mask_psd.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neural_reconstruction.core.preprocessing import (  # noqa: E402
    dilate_epidermis_vertically,
)

BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

OFFSET_PX = 50
DILATED_COLOR = (0.20, 0.90, 0.30)   # green — dilation result (constrained/unconstrained)
ORIG_MASK_COLOR = (1.0, 0.20, 0.80)  # magenta — original mask outline
AUX_COLOR = (1.0, 0.55, 0.0)         # orange — aux mask
OUTLINE_THICKNESS = 10

# Same render target as viz_roi_mask.py's _save_with_axes: figsize=(10, ...),
# dpi=150 -> 1500px wide. Matching it lets us reuse its axis-widget/arrow
# point sizes (12pt, 20pt, lw=2/3.5, +40/+50/*0.62 offsets) verbatim.
DPI = 150
OUT_W = 1500
TOP_MARGIN = 45   # px, room for the "y" label above the image
BOTTOM_MARGIN = 68  # px, room for "(0,0)" / "x" below the image

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _compute_aux_mask(m: np.ndarray) -> np.ndarray:
    """Recompute the auxiliary mask used internally by dilate_epidermis_vertically."""
    binary = m > 0
    H = m.shape[0]
    col_has_mask = binary.any(axis=0)
    min_y = np.where(col_has_mask, np.argmax(binary, axis=0), H)
    y_indices = np.arange(H).reshape(-1, 1)
    return np.where(y_indices >= min_y[np.newaxis, :], np.uint8(255), np.uint8(0)).astype(np.uint8)


def _dilate_unconstrained(m: np.ndarray, offset_px: int) -> np.ndarray:
    """Plain elliptical-SE dilation, no downward constraint (dilate_epidermis_vertically's
    intermediate `dilated` before it's clipped to the aux mask)."""
    d = 2 * offset_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    return cv2.dilate(m, kernel, iterations=1)


def build_fill_layer(mask: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
    """Translucent RGBA fill: real alpha channel, not blended into pixels."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = int(color[0] * 255)
    rgba[:, :, 1] = int(color[1] * 255)
    rgba[:, :, 2] = int(color[2] * 255)
    rgba[:, :, 3] = np.where(mask > 0, round(alpha * 255), 0).astype(np.uint8)
    return rgba


def build_outline_layer(mask: np.ndarray, color: tuple, thickness: int) -> np.ndarray:
    """Transparent RGBA canvas with just the mask's contour line.

    Drawn once in white on black so intensity doubles as the (anti-aliased)
    alpha mask, then recolored to `color` at full RGB strength.
    """
    h, w = mask.shape
    indicator = np.zeros((h, w, 3), dtype=np.uint8)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(indicator, contours, -1, (255, 255, 255), thickness, cv2.LINE_AA)
    alpha = indicator[:, :, 0]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = int(color[0] * 255)
    rgba[:, :, 1] = int(color[1] * 255)
    rgba[:, :, 2] = int(color[2] * 255)
    rgba[:, :, 3] = alpha
    return rgba


def render_overlay(out_h: int, draw_fn: Callable) -> np.ndarray:
    """Rasterise one matplotlib element onto a transparent OUT_W x out_h
    canvas, with axes data coordinates == pixel coordinates (figsize chosen
    so 1 data unit == 1 px at DPI)."""
    fig = plt.figure(figsize=(OUT_W / DPI, out_h / DPI), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, OUT_W)
    ax.set_ylim(out_h, 0)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    draw_fn(ax)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    assert buf.shape[:2] == (out_h, OUT_W), f"canvas {buf.shape[:2]} != requested {(out_h, OUT_W)}"
    return buf


def draw_coord_axis(ax, crop_out_h: int) -> None:
    """The "(0,0)" / "x" / "y" widget viz_roi_mask.py draws in the margin
    outside the image (same 3 plain-text labels; no arrowheads — matches
    what the source code actually draws, not its docstring's description)."""
    ax.text(4, TOP_MARGIN - 6, "y", ha="left", va="bottom", fontsize=12, color="black")
    ax.text(4, TOP_MARGIN + crop_out_h + 6, "(0,0)", ha="left", va="top", fontsize=12, color="black")
    ax.text(OUT_W - 4, TOP_MARGIN + crop_out_h + 6, "x", ha="right", va="top", fontsize=12, color="black")


def draw_y_max(ax, mask_crop_small: np.ndarray, scale: float) -> None:
    """Dashed sampling line + arrow + "y_max" label, matching
    viz_roi_mask.py's _annotate_aux_mask (same offsets: chosen column at
    1/3 into the mask's x-extent, arrow tip 50px above the mask edge, label
    at 62% up the arrow, text offset +40px right). viz_roi_mask.py applies
    those 50/40px offsets in *native* crop-pixel space (matplotlib data
    coordinates track the source array, not the rendered figure's pixels);
    since we draw directly on the already-downsampled mask, they're scaled
    by `scale` (OUT_W / native crop width) here, or the arrow overshoots
    past the (now much closer) mask edge.
    """
    h, w = mask_crop_small.shape
    mask_cols = np.where(mask_crop_small.any(axis=0))[0]
    chosen_col = int(mask_cols[len(mask_cols) // 3]) if len(mask_cols) > 0 else w // 2

    ax.plot([chosen_col, chosen_col], [TOP_MARGIN, TOP_MARGIN + h],
             color="red", linestyle="--", linewidth=2, zorder=5)

    col_vals = mask_crop_small[:, chosen_col]
    if not (col_vals > 0).any():
        return
    top_row = int(np.argmax(col_vals > 0))
    head_row = max(0, top_row - round(50 * scale))
    tail_row = h
    ax.annotate(
        "", xy=(chosen_col, TOP_MARGIN + head_row), xytext=(chosen_col, TOP_MARGIN + tail_row),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=3.5, mutation_scale=26), zorder=6,
    )
    label_row = tail_row - 0.62 * (tail_row - head_row)
    ax.text(chosen_col + round(40 * scale), TOP_MARGIN + label_row, r"$y_{max}$",
             ha="left", va="center", color="red", fontsize=20, zorder=7)


def save_overlay_psd(background_rgb: np.ndarray, fill_mask: np.ndarray, fill_color: tuple,
                      fill_alpha: float, outline_mask: np.ndarray, out_path: Path,
                      *, scale: float, with_y_max: bool) -> None:
    crop_out_h, crop_out_w = background_rgb.shape[:2]
    # OUTLINE_THICKNESS is tuned for the *native* crop resolution (like
    # viz_roi_mask.py itself draws it); scale it down to match how thick it
    # would look once that native array is downsampled to OUT_W.
    thickness = max(1, round(OUTLINE_THICKNESS * scale))
    out_h = TOP_MARGIN + crop_out_h + BOTTOM_MARGIN

    psd = PSDImage.new("RGBA", (OUT_W, out_h))
    psd.create_pixel_layer(Image.fromarray(background_rgb), name="background", top=TOP_MARGIN, left=0)
    psd.create_pixel_layer(
        Image.fromarray(build_fill_layer(fill_mask, fill_color, fill_alpha)),
        name="overlay_fill", top=TOP_MARGIN, left=0,
    )
    psd.create_pixel_layer(
        Image.fromarray(build_outline_layer(outline_mask, ORIG_MASK_COLOR, thickness)),
        name="mask_outline", top=TOP_MARGIN, left=0,
    )
    psd.create_pixel_layer(
        Image.fromarray(render_overlay(out_h, lambda ax: draw_coord_axis(ax, crop_out_h))),
        name="coord_axis", top=0, left=0,
    )
    if with_y_max:
        psd.create_pixel_layer(
            Image.fromarray(render_overlay(out_h, lambda ax: draw_y_max(ax, outline_mask, scale))),
            name="y_max_annotation", top=0, left=0,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(out_path)


def main() -> None:
    image = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    bg_image = image[:, :, 1]  # green channel only

    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"mask.png not found under {BASE_PATH}")

    roi_mask = dilate_epidermis_vertically(mask, offset_px=OFFSET_PX)
    aux_mask = _compute_aux_mask(mask)
    unconstrained_dilated = _dilate_unconstrained(mask, OFFSET_PX)

    # Crop region: original mask extent + padding (matches viz_roi_mask.py).
    ys, xs = np.where(mask > 0)
    PAD_ABOVE, PAD_BELOW, PAD_LATERAL = OFFSET_PX + 40, OFFSET_PX + 30, OFFSET_PX + 30
    y0 = max(0, int(ys.min()) - PAD_ABOVE)
    y1 = min(mask.shape[0], int(ys.max()) + PAD_BELOW)
    x0 = max(0, int(xs.min()) - PAD_LATERAL)
    x1 = min(mask.shape[1], int(xs.max()) + PAD_LATERAL)

    def crop(arr):
        return arr[y0:y1, x0:x1]

    crop_h_native, crop_w_native = crop(mask).shape
    scale = OUT_W / crop_w_native
    out_crop_h = round(crop_h_native * scale)

    def crop_and_downsample(arr, interpolation=cv2.INTER_NEAREST):
        return cv2.resize(crop(arr), (OUT_W, out_crop_h), interpolation=interpolation)

    bg_crop_rgb = cv2.cvtColor(crop_and_downsample(bg_image), cv2.COLOR_GRAY2RGB)
    mask_crop_small = crop_and_downsample(mask)

    targets = [
        ("aux_mask", crop_and_downsample(aux_mask), AUX_COLOR, 0.45, True),
        ("unconstrained_dilation", crop_and_downsample(unconstrained_dilated), DILATED_COLOR, 0.5, False),
        ("roi_mask", crop_and_downsample(roi_mask), DILATED_COLOR, 0.5, False),
    ]
    for name, fill_mask, color, alpha, with_y_max in targets:
        out_path = OUTPUT_DIR / f"{IMAGE_ID}_{name}.psd"
        save_overlay_psd(bg_crop_rgb, fill_mask, color, alpha, mask_crop_small, out_path,
                          scale=scale, with_y_max=with_y_max)
        print(f"[save] {out_path}")

    print(f"\nDone. {len(targets)} PSD files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
