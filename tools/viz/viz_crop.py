"""
Crop two images with the same rectangular region and save the results.
Paths and crop region are hard-coded below — edit them before running.
"""

from pathlib import Path

import cv2

SAMPLE_ID = "S2644-2_b"
# === Edit these paths and crop region ===
IMAGE1_PATH = Path(
    f"/home/pony/projects/ienf_q/output/evaluation/annotation_grow_0331/vis/{SAMPLE_ID}.png"
)
IMAGE2_PATH = Path(
    f"/home/pony/projects/ienf_q/output/evaluation/mst_0331/vis/{SAMPLE_ID}.png"
)
OUTPUT_DIR = Path("output/crops")

# Crop region: (x, y) top-left corner, plus width and height
CROP_X = 3210
CROP_Y = 210
CROP_W = 512
CROP_H = 512
# =========================================


def crop_image(image_path: Path, x: int, y: int, width: int, height: int):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    x2, y2 = x + width, y + height
    if x < 0 or y < 0 or x2 > w or y2 > h:
        raise ValueError(
            f"Crop region ({x},{y},{width},{height}) out of bounds for "
            f"image '{image_path.name}' with size ({w},{h})"
        )

    return image[y:y2, x:x2]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate((IMAGE1_PATH, IMAGE2_PATH)):
        cropped = crop_image(img_path, CROP_X, CROP_Y, CROP_W, CROP_H)
        out_path = OUTPUT_DIR / f"{img_path.stem}_{i}_crop{img_path.suffix}"
        cv2.imwrite(str(out_path), cropped)
        print(f"[OK] {img_path} -> {out_path}  shape={cropped.shape}")


if __name__ == "__main__":
    main()
