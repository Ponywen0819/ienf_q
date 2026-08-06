"""PSD-output variant of tools/viz/viz_crop_showcase.py — boxed overview only.

Unlike the original (one hardcoded SAMPLE_ID, edit-and-rerun per sample),
this loops over every sample listed in SAMPLE_CROPS and writes one PSD each:
the sample's green channel as an opaque background layer, plus one
transparent "box_{letter}" and one "label_{letter}" layer per crop, so the
rectangle and its letter can be toggled independently in Photoshop/GIMP.

Uses psd-tools (see tools/viz/viz_ablation_grow_psd for why: pytoshop's
output only opens in lenient readers, not real Photoshop).

Run:
    uv run python tools/viz/viz_crop_showcase_psd/viz_crop_showcase_psd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.viz.viz_crop_showcase import (  # noqa: E402
    BOX_COLOR,
    LABEL_COLOR,
    LABEL_FONT,
    PROJECT_ROOT,
    _scaled_styles,
    check_crop,
    load_green_channel,
)

# Crop regions per sample, lifted from the commented-out blocks in
# viz_crop_showcase.py. Each entry is (x, y, size): (x, y) top-left corner,
# square of side `size`.
SAMPLE_CROPS = {
    "S558-2_a": [(3950, 900, 75), (6550, 825, 75)],
    "S487-2_a": [(1480, 1300, 75), (8224, 700, 75)],
    "S1196-2_b": [(3320, 840, 75), (3766, 885, 75)],
    "S1571-2_b": [(2640, 750, 75), (6100, 565, 75)],
    "S2266-2_b": [(3869, 692, 75), (1830, 1006, 75)],
    "S1585-2_b": [(1620, 350, 75), (1330, 420, 75)],
    "S2745-2_a": [(1790, 510, 75), (1076, 850, 75)],
}

DATA_DIR = PROJECT_ROOT / "data_0510"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _red_canvas_to_rgba(canvas: np.ndarray) -> np.ndarray:
    """BGR canvas drawn in pure red (0,0,255) -> transparent RGBA: the red
    channel doubles as the (anti-aliased) alpha mask, so untouched pixels
    (still black) end up alpha=0."""
    h, w = canvas.shape[:2]
    alpha = canvas[:, :, 2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255  # solid red where alpha > 0
    rgba[:, :, 3] = alpha
    return rgba


def build_rect_layer(shape: tuple[int, int], x: int, y: int, size: int) -> np.ndarray:
    """Transparent RGBA canvas with just one crop's red box outline."""
    h, w = shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    st = _scaled_styles(w)
    cv2.rectangle(canvas, (x, y), (x + size, y + size), BOX_COLOR, st["box_thickness"])
    return _red_canvas_to_rgba(canvas)


def build_label_layer(shape: tuple[int, int], x: int, y: int, size: int, label_ch: str) -> np.ndarray:
    """Transparent RGBA canvas with just one crop's letter label."""
    h, w = shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    st = _scaled_styles(w)
    (tw, th), _ = cv2.getTextSize(label_ch, LABEL_FONT, st["font_scale"], st["label_thickness"])
    org = (x, y - st["label_margin"])
    if org[1] - th < 0:
        org = (x + size + st["label_margin"], y + th)
    cv2.putText(canvas, label_ch, org, LABEL_FONT, st["font_scale"], LABEL_COLOR,
                st["label_thickness"], cv2.LINE_AA)
    return _red_canvas_to_rgba(canvas)


def save_boxed_psd(sample_id: str, crops: list[tuple[int, int, int]]) -> Path:
    green_path = DATA_DIR / sample_id / "image.png"
    green = load_green_channel(green_path)
    print(f"[load] {sample_id}  {green_path}  shape={green.shape}")

    for x, y, size in crops:
        check_crop(x, y, size, green.shape, "green channel")

    h, w = green.shape[:2]
    background = Image.fromarray(cv2.cvtColor(green, cv2.COLOR_GRAY2RGB))

    psd = PSDImage.new("RGBA", (w, h))
    psd.create_pixel_layer(background, name=sample_id, top=0, left=0)
    for i, (x, y, size) in enumerate(crops):
        label_ch = chr(ord("a") + i)
        box_layer = Image.fromarray(build_rect_layer((h, w), x, y, size))
        psd.create_pixel_layer(box_layer, name=f"box_{label_ch}", top=0, left=0)
        label_layer = Image.fromarray(build_label_layer((h, w), x, y, size, label_ch))
        psd.create_pixel_layer(label_layer, name=f"label_{label_ch}", top=0, left=0)

    out_path = OUTPUT_DIR / f"{sample_id}_green_boxed.psd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(out_path)
    return out_path


def main():
    for sample_id, crops in SAMPLE_CROPS.items():
        out_path = save_boxed_psd(sample_id, crops)
        print(f"[save] {out_path}")
    print(f"\nDone. {len(SAMPLE_CROPS)} PSD files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
