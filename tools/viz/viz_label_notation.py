"""
Visualize label and annotation overlays for all samples in a data folder.

For each sample:
  - Yellow pixels: label.png regions
  - Red pixels:    annotation.png regions
  - Background:    green channel of image.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def process_sample(sample_dir: Path, output_dir: Path, show: bool = False) -> bool:
    image_path = sample_dir / "image.png"
    mask_path = sample_dir / "mask.png"
    annotation_path = sample_dir / "annotation.png"
    label_path = sample_dir / "label.png"

    missing = [
        p
        for p in [image_path, mask_path, annotation_path, label_path]
        if not p.exists()
    ]
    if missing:
        print(f"  [SKIP] Missing files: {[p.name for p in missing]}")
        return False

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR_RGB)[:, :, 1]  # green channel
    annotation = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
    label_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

    viz = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    viz[label_img > 0] = [255, 255, 0]  # yellow
    viz[annotation > 0] = [255, 0, 0]  # red

    out_path = output_dir / f"{sample_dir.name}_viz.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    print(f"  [OK]   Saved → {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch label/annotation visualization")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_0320"),
        help="Root data folder containing sample sub-directories (default: data_0320)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/label_notation_viz"),
        help="Directory to save output images (default: output/label_notation_viz)",
    )
    parser.add_argument(
        "--samples", nargs="*", help="Specific sample IDs to process (default: all)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display each figure interactively"
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        print(f"Error: data directory '{data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if args.samples:
        sample_dirs = [d for d in sample_dirs if d.name in args.samples]
        if not sample_dirs:
            print("No matching samples found.", file=sys.stderr)
            sys.exit(1)

    print(
        f"Processing {len(sample_dirs)} sample(s) from '{data_dir}' → '{args.output_dir}'"
    )
    ok = skip = 0
    for sample_dir in sample_dirs:
        print(f"\n{sample_dir.name}")
        if process_sample(sample_dir, args.output_dir, show=args.show):
            ok += 1
        else:
            skip += 1

    print(f"\nDone: {ok} saved, {skip} skipped.")


if __name__ == "__main__":
    main()
