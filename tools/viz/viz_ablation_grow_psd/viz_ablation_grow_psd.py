"""PSD-output variant of tools/viz/viz_ablation_grow.py — boxed overview only.

Reuses the green-channel-with-red-boxes overview from that script (imported,
not duplicated), writing it out as a PSD: the sample's green channel as an
opaque background layer, plus one transparent "box_{letter}" and one
"label_{letter}" layer per crop, so the rectangle and its letter can be
toggled independently.

Uses psd-tools (not pytoshop) to write the file. pytoshop's output only
opens in lenient readers (PIL, psd-tools itself) — real/online Photoshop is
a strict parser and rendered it blank. psd-tools is actively maintained and
round-trips through actual Photoshop correctly.

Run:
    uv run python tools/viz/viz_ablation_grow_psd/viz_ablation_grow_psd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.viz.viz_ablation_grow import (  # noqa: E402
    BOX_COLOR,
    CROPS,
    LABEL_COLOR,
    LABEL_FONT,
    SAMPLE_ID,
    _scaled_styles,
    check_crop,
    load_inputs,
)

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


def main():
    _, green, _, _, _ = load_inputs(SAMPLE_ID)
    print(f"[load] {SAMPLE_ID}  green shape={green.shape}")

    for x, y, size in CROPS:
        check_crop(x, y, size, green.shape, "green channel")

    h, w = green.shape[:2]
    background = Image.fromarray(cv2.cvtColor(green, cv2.COLOR_GRAY2RGB))

    psd = PSDImage.new("RGBA", (w, h))
    psd.create_pixel_layer(background, name=SAMPLE_ID, top=0, left=0)
    for i, (x, y, size) in enumerate(CROPS):
        label_ch = chr(ord("a") + i)
        box_layer = Image.fromarray(build_rect_layer((h, w), x, y, size))
        psd.create_pixel_layer(box_layer, name=f"box_{label_ch}", top=0, left=0)
        label_layer = Image.fromarray(build_label_layer((h, w), x, y, size, label_ch))
        psd.create_pixel_layer(label_layer, name=f"label_{label_ch}", top=0, left=0)

    out_path = OUTPUT_DIR / f"{SAMPLE_ID}_green_boxed.psd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(out_path)
    print(f"[save] {out_path}  layers={[l.name for l in psd]}")


if __name__ == "__main__":
    main()
