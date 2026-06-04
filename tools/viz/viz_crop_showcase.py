"""
Crop showcase tool.

For a given sample ID this produces three kinds of output:

  1. The sample's input green-channel image, with red squares drawn at every
     chosen crop location  ->  {ID}_green_boxed.png
  2. The original crop region images, taken from the green channel
     ->  {ID}_crop{i}_green.png
  3. The same crop regions taken from the result visualizations
     ->  {ID}_crop{i}_annotation_grow.png  and  {ID}_crop{i}_mst.png

Original image  : data_0510/{ID}/image.png   (green channel is extracted)
Result vis      : output/ref/{annotation_grow,mst}/vis/{ID}.png

All three source images share the same dimensions, so a crop region is valid
across all of them.

Edit SAMPLE_ID and the CROPS list below before running.
"""

from pathlib import Path

import cv2

# === Edit these ===========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SAMPLE_ID = "S558-2_a"
# SAMPLE_ID = "S487-2_a"
# SAMPLE_ID = "S1196-2_b" 
# SAMPLE_ID = "S1571-2_b"
# SAMPLE_ID = "S2266-2_b"
SAMPLE_ID = "S1585-2_b"
# SAMPLE_ID = "S2745-2_a"

# Crop regions. Each entry is (x, y, size): (x, y) is the top-left corner and
# every crop is a square of side `size` pixels.
# CROPS = [
#     (3950, 900, 75),
#     (6550, 825, 75),
# ]

# CROPS = [
#     (1480, 1300, 75),
#     (8224, 700, 75),
# ]

# CROPS = [
#     (3320,840,75),
#     (3766,885,75)
# ]

# CROPS = [
#     (2640, 750, 75),
#     (6100, 565, 75),
# ]

# # 2266-2_b
# CROPS = [
#     (3869, 692, 75),
#     (1830, 1006, 75),
# ]

# 1585-2_b
CROPS = [
    (1620, 350, 75),
    (1330, 420, 75),
]

# # 2745-2_a
# CROPS = [
#     (1790, 510, 75),
#     (1076, 850, 75),
# ]

# Result visualizations to crop from. Each name maps to
# output/ref/{name}/vis/{ID}.png
VIS_NAMES = ["annotation_grow", "mst"]

OUTPUT_DIR = PROJECT_ROOT / "output" / "crop_showcase"

# Red box appearance (drawn on the boxed green-channel overview).
BOX_COLOR = (0, 0, 255)  # BGR -> red
BOX_THICKNESS = 12

# Red label (a, b, ...) drawn next to each box.
LABEL_COLOR = (0, 0, 255)  # BGR -> red
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 18.0
LABEL_THICKNESS = 40
LABEL_MARGIN = 40  # gap between box and label
# ==========================================================================


def load_green_channel(image_path: Path):
    """Load an RGB image and return its green channel as a single-channel image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    # cv2 loads as BGR -> green is channel index 1.
    return image[:, :, 1]


def load_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def check_crop(x: int, y: int, size: int, shape, source_name: str):
    h, w = shape[:2]
    if x < 0 or y < 0 or x + size > w or y + size > h:
        raise ValueError(
            f"Crop ({x},{y},size={size}) is out of bounds for "
            f"'{source_name}' with size ({w},{h})"
        )


def crop_square(image, x: int, y: int, size: int):
    return image[y : y + size, x : x + size]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load source images -----------------------------------------------
    green_path = PROJECT_ROOT / "data_0510" / SAMPLE_ID / "image.png"
    green = load_green_channel(green_path)
    print(f"[load] green channel  {green_path}  shape={green.shape}")

    vis_images = {}
    for name in VIS_NAMES:
        vis_path = PROJECT_ROOT / "output" / "ref" / name / "vis" / f"{SAMPLE_ID}.png"
        vis_images[name] = load_image(vis_path)
        print(f"[load] vis '{name}'   {vis_path}  shape={vis_images[name].shape}")

    # --- Validate every crop against every source -------------------------
    for x, y, size in CROPS:
        check_crop(x, y, size, green.shape, "green channel")
        for name, vis in vis_images.items():
            check_crop(x, y, size, vis.shape, f"vis/{name}")

    # --- Output 1: green-channel overview with red boxes ------------------
    # Convert to BGR so the red boxes show up in color.
    boxed = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
    for i, (x, y, size) in enumerate(CROPS):
        cv2.rectangle(
            boxed, (x, y), (x + size, y + size), BOX_COLOR, BOX_THICKNESS
        )
        # Label this region with a letter (a, b, ...) beside the box.
        label = chr(ord("a") + i)
        (tw, th), _ = cv2.getTextSize(
            label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS
        )
        # Default: place the label above the box's top-left corner.
        org = (x, y - LABEL_MARGIN)
        # If there's no room above, place it to the right of the box instead.
        if org[1] - th < 0:
            org = (x + size + LABEL_MARGIN, y + th)
        cv2.putText(
            boxed,
            label,
            org,
            LABEL_FONT,
            LABEL_SCALE,
            LABEL_COLOR,
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )
    boxed_path = OUTPUT_DIR / f"{SAMPLE_ID}_green_boxed.png"
    cv2.imwrite(str(boxed_path), boxed)
    print(f"[save] {boxed_path}")

    # --- Outputs 2 & 3: per-crop regions ----------------------------------
    for i, (x, y, size) in enumerate(CROPS):
        # Output 2: original crop region from the green channel.
        green_crop = crop_square(green, x, y, size)
        green_crop_path = OUTPUT_DIR / f"{SAMPLE_ID}_crop{i}_green.png"
        cv2.imwrite(str(green_crop_path), green_crop)
        print(f"[save] {green_crop_path}  shape={green_crop.shape}")

        # Output 3: same crop region from each result visualization.
        for name, vis in vis_images.items():
            vis_crop = crop_square(vis, x, y, size)
            vis_crop_path = OUTPUT_DIR / f"{SAMPLE_ID}_crop{i}_{name}.png"
            cv2.imwrite(str(vis_crop_path), vis_crop)
            print(f"[save] {vis_crop_path}  shape={vis_crop.shape}")

    print(f"\nDone. {1 + len(CROPS) * (1 + len(VIS_NAMES))} files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
