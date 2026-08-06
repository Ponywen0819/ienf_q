"""PSD-output variant of tools/viz/viz_dataset_inputs.py — boxed-full panels only.

For the same sample (IMAGE_ID, imported from viz_dataset_inputs) this writes
one 2-layer PSD per data type (image_rgb / image_green / mask / weka /
label): the full image as an opaque "background" layer, and the red crop
rectangle as a transparent "crop_box" layer on top.

Uses psd-tools (see tools/viz/viz_ablation_grow_psd for why: pytoshop's
output only opens in lenient readers, not real Photoshop).

Run:
    uv run python tools/viz/viz_dataset_inputs_psd/viz_dataset_inputs_psd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.viz.viz_dataset_inputs import (  # noqa: E402
    BASE_PATH,
    CROP_H,
    CROP_W,
    CROP_X0,
    CROP_Y0,
    IMAGE_ID,
)

BOX_COLOR = (0, 0, 255)  # BGR -> red
# Same thickness/width ratio used by the sibling *_psd scripts' box style.
BOX_THICKNESS_RATIO = 2.5 / 742

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def build_annotation_layer(shape: tuple[int, int]) -> np.ndarray:
    """Transparent RGBA canvas with just the red crop rectangle.

    Drawn once in black on a black canvas so the red channel doubles as the
    (anti-aliased) alpha mask -> everywhere nothing was drawn stays alpha=0.
    """
    h, w = shape
    thickness = max(1, round(w * BOX_THICKNESS_RATIO))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)  # BGR, draw in BOX_COLOR (red)
    cv2.rectangle(
        canvas, (CROP_X0, CROP_Y0), (CROP_X0 + CROP_W, CROP_Y0 + CROP_H),
        BOX_COLOR, thickness, cv2.LINE_AA,
    )
    alpha = canvas[:, :, 2]  # BOX_COLOR is pure red (0,0,255) in BGR
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255  # solid red where alpha > 0
    rgba[:, :, 3] = alpha
    return rgba


def save_boxed_psd(img: np.ndarray, is_rgb: bool, key: str) -> Path:
    background_rgb = img if is_rgb else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    background = Image.fromarray(background_rgb)
    annotations = Image.fromarray(build_annotation_layer((h, w)))

    psd = PSDImage.new("RGBA", (w, h))
    psd.create_pixel_layer(background, name=key, top=0, left=0)
    psd.create_pixel_layer(annotations, name="crop_box", top=0, left=0)

    out_path = OUTPUT_DIR / f"{IMAGE_ID}_{key}_boxed.psd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(out_path)
    return out_path


def main() -> None:
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    image_green = image_rgb[:, :, 1]

    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    weka = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)

    sources = [
        ("image_rgb", image_rgb, True),
        ("image_green", image_green, False),
        ("mask", mask, False),
        ("weka", weka, False),
        ("label", label, False),
    ]
    n = 0
    for key, img, is_rgb in sources:
        if img is None:
            print(f"  skip {key}: not found under {BASE_PATH}")
            continue
        out_path = save_boxed_psd(img, is_rgb, key)
        print(f"[save] {out_path}")
        n += 1

    print(f"\nDone. {n} PSD files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
