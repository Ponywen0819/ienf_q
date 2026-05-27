"""
Dataset inputs showcase — what data a single IENF-Q sample contains.

For one sample this renders, per data type, two panels:

  * the full image with a red rectangle marking the shared crop region, and
  * the cropped region itself.

Data types (5 groups -> 10 PNGs):

  1. Original image I       — IF-stained RGB image; processing uses its
                              green channel Ig as the main input   (image.png)
  2. Green channel Ig       — green channel of I, the main input    (image.png[:,:,1])
  3. Epidermis mask M       — polygon annotation of the epidermis   (mask.png)
  4. Particle annotation A  — manual fiber-particle prediction on I,
                              treated as an AEL input, NOT ground   (weka.png)
                              truth
  5. Skeleton GT Sgt        — expert-annotated fiber-skeleton binary
                              mask; ground truth for topology       (label.png)
                              quality (clDice / HD95), not an AEL input

The crop region matches every other viz script (viz_crop_location,
viz_region_grow, viz_bridge_skeleton, viz_crossing, viz_preprocessing, ...).
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

# Shared crop region: same as every other viz script.
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

RECT_COLOR = "#ff2020"
RECT_LINEWIDTH = 4

OUT_DIR = Path(__file__).parent
# ==========================================================================


def _crop(arr):
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _save_full(img, cmap, out_name: str) -> None:
    """Save the full image with a red rectangle marking the crop region."""
    h, w = img.shape[:2]
    fig_w = 12.0
    fig_h = fig_w * h / w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.imshow(img, cmap=cmap)
    ax.add_patch(Rectangle(
        (CROP_X0 - 0.5, CROP_Y0 - 0.5), CROP_W, CROP_H,
        fill=False, edgecolor=RECT_COLOR, linewidth=RECT_LINEWIDTH, zorder=5,
    ))
    ax.axis("off")
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({w}×{h})")


def _save_crop(img, cmap, out_name: str) -> None:
    """Save the cropped region only."""
    crop = _crop(img)
    h, w = crop.shape[:2]
    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    ax.imshow(crop, cmap=cmap)
    ax.axis("off")
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({w}×{h})")


def _render(img, cmap, key: str) -> None:
    _save_full(img, cmap, f"viz_dataset_inputs_{key}_full.png")
    _save_crop(img, cmap, f"viz_dataset_inputs_{key}_crop.png")


def main() -> None:
    # 1 & 2. Original RGB image I and its green channel Ig.
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    image_green = image_rgb[:, :, 1]

    # 3. Epidermis mask M.
    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    # 4. Particle annotation A (AEL input, not ground truth).
    weka = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)
    # 5. Skeleton ground truth Sgt.
    label = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)

    _render(image_rgb, None, "image_rgb")
    _render(image_green, "gray", "image_green")
    if mask is not None:
        _render(mask, "gray", "mask")
    else:
        print(f"  skip mask: {BASE_PATH}/mask.png not found")
    if weka is not None:
        _render(weka, "gray", "weka")
    else:
        print(f"  skip weka: {BASE_PATH}/weka.png not found")
    if label is not None:
        _render(label, "gray", "label")
    else:
        print(f"  skip label: {BASE_PATH}/label.png not found")


if __name__ == "__main__":
    main()
