"""
Show where the paper's fixed crop sits on each full-image source.

Saves five PNGs of the full image with a red rectangle marking the crop
region (CROP_Y0..+CROP_H, CROP_X0..+CROP_W) used by viz_region_grow,
viz_bridge_skeleton, viz_crossing, viz_preprocessing, etc.

Panels:
  viz_crop_location_image_rgb.png    — original RGB
  viz_crop_location_image_green.png  — green channel
  viz_crop_location_weka.png         — annotation
  viz_crop_location_mask.png         — epidermis mask
  viz_crop_location_label.png        — GT label
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

RECT_COLOR = "#ff2020"
RECT_LINEWIDTH = 4

OUT_DIR = Path(__file__).parent


def _save(img, cmap, out_name: str) -> None:
    h, w = img.shape[:2]
    # Keep aspect ratio of the source image.
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


def main() -> None:
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    image_green = image_rgb[:, :, 1]
    weka = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)

    _save(image_rgb, None, "viz_crop_location_image_rgb.png")
    _save(image_green, "gray", "viz_crop_location_image_green.png")
    _save(weka, "gray", "viz_crop_location_weka.png")
    _save(mask, "gray", "viz_crop_location_mask.png")
    if label is not None:
        _save(label, "gray", "viz_crop_location_label.png")
    else:
        print(f"  skip label: {BASE_PATH}/label.png not found")


if __name__ == "__main__":
    main()
