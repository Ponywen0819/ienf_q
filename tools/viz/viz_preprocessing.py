"""
Preprocessing stage visualisation.

Renders four cropped panels showing the same fixed crop used by the other paper
viz scripts: raw green channel, after morphological background removal, after
CLAHE, after Sato vesselness. The ROI mask is intentionally **not** applied so
the full pipeline response is visible everywhere in the crop.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200


def crop(arr: np.ndarray) -> np.ndarray:
    """Return the fixed crop used by the paper visualisations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _norm_u8(arr: np.ndarray) -> np.ndarray:
    """Min-max stretch to [0, 255] uint8 for display."""
    a = arr.astype(np.float32)
    vmin, vmax = float(a.min()), float(a.max())
    if vmax > vmin:
        a = (a - vmin) / (vmax - vmin) * 255.0
    return a.astype(np.uint8)


# ── Load & run preprocessing on the full image (no ROI mask applied) ────────
image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
green = image_rgb[:, :, 1]

bg_kernel_size = 5
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size)
)
background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
bg_removed = cv2.subtract(green, background)

tileGridSize = 768
clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(tileGridSize, tileGridSize))
clahe_out = clahe.apply(bg_removed)

sato_raw = ski.filters.sato(clahe_out, sigmas=range(1, 4), black_ridges=False)
sato_u8 = _norm_u8(sato_raw)

# Sato at individual sigmas (front / middle / back of the range)
sigmas_to_show = (1, 2, 3)
sato_per_sigma = {
    s: ski.filters.sato(clahe_out, sigmas=range(s, s + 1), black_ridges=False)
    for s in sigmas_to_show
}

# ── Render panels matching other viz scripts ─────────────────────────────────
out_dir = Path(__file__).parent
panel_size = (5.0, 5.0)

panels = [
    ("original", crop(green), "viz_preprocessing_original.png"),
    ("background", crop(background), "viz_preprocessing_background.png"),
    ("bg_removed", crop(_norm_u8(bg_removed)), "viz_preprocessing_bg_removed.png"),
    ("clahe", crop(clahe_out), "viz_preprocessing_clahe.png"),
    ("sato", crop(sato_u8), "viz_preprocessing_sato.png"),
]
for s, resp in sato_per_sigma.items():
    panels.append(
        (f"sato_sigma{s}", crop(_norm_u8(resp)), f"viz_preprocessing_sato_sigma{s}.png")
    )

for name, img, fname in panels:
    fig, ax = plt.subplots(figsize=panel_size, constrained_layout=True)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({name}, shape={img.shape})")
