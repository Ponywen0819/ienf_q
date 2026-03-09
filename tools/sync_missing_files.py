"""Sync missing image.png / annotation.png / mask.png into data_0320 from data/.

For each sample directory in TARGET_DIR:
  1. Check whether image.png, annotation.png, and mask.png exist.
  2. If missing, locate the file in SOURCE_DIR under the same sample name.
  3. Copy the file, resizing it to match the label.png dimensions in TARGET_DIR
     if the sizes differ.

Usage
-----
    uv run python tools/sync_missing_files.py
    uv run python tools/sync_missing_files.py --source data --target data_0320
    uv run python tools/sync_missing_files.py --dry-run   # preview only, no writes
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


REQUIRED_FILES = ["image.png", "annotation.png", "mask.png"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync missing image/annotation files from source into target."
    )
    p.add_argument("--source", type=Path, default=Path("data"),
                   help="Source data directory (default: data)")
    p.add_argument("--target", type=Path, default=Path("data_0320"),
                   help="Target data directory (default: data_0320)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without writing any files")
    return p.parse_args()


def image_size(path: Path) -> tuple[int, int]:
    """Return (height, width) of an image file."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img.shape[:2]


def resize_and_save(src: Path, dst: Path, target_hw: tuple[int, int]) -> None:
    """Read src, resize to (H, W), write to dst.

    Grayscale images are written as-is; colour images keep all channels.
    Uses INTER_AREA for downscaling and INTER_LINEAR for upscaling.
    """
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read source image: {src}")

    h, w = target_hw
    src_h, src_w = img.shape[:2]

    if (src_h, src_w) == (h, w):
        shutil.copy2(src, dst)
        return

    interp = cv2.INTER_AREA if (src_h > h or src_w > w) else cv2.INTER_LINEAR
    resized = cv2.resize(img, (w, h), interpolation=interp)
    cv2.imwrite(str(dst), resized)


def main() -> None:
    args = parse_args()

    if not args.target.is_dir():
        raise SystemExit(f"Target directory not found: {args.target}")
    if not args.source.is_dir():
        raise SystemExit(f"Source directory not found: {args.source}")

    sample_dirs = sorted(d for d in args.target.iterdir() if d.is_dir())
    if not sample_dirs:
        raise SystemExit(f"No sample sub-directories found in {args.target}")

    print(f"Source : {args.source.resolve()}")
    print(f"Target : {args.target.resolve()}")
    if args.dry_run:
        print("Mode   : DRY RUN (no files will be written)")
    print(f"Samples: {len(sample_dirs)}")
    print("-" * 60)

    total_copied   = 0
    total_resized  = 0
    total_skipped  = 0
    errors         = []

    for sample_dir in sample_dirs:
        name = sample_dir.name

        # Reference size comes from label.png in the target
        label_path = sample_dir / "label.png"
        if not label_path.exists():
            msg = f"[{name}] SKIP — label.png not found in target"
            print(msg)
            errors.append(msg)
            continue

        try:
            label_hw = image_size(label_path)
        except ValueError as e:
            msg = f"[{name}] ERROR reading label.png — {e}"
            print(msg)
            errors.append(msg)
            continue

        for filename in REQUIRED_FILES:
            dst = sample_dir / filename

            if dst.exists():
                # File present — verify size matches label
                try:
                    actual_hw = image_size(dst)
                except ValueError as e:
                    msg = f"[{name}/{filename}] ERROR reading existing file — {e}"
                    print(msg)
                    errors.append(msg)
                    continue

                if actual_hw == label_hw:
                    print(f"[{name}/{filename}] OK  {actual_hw[1]}×{actual_hw[0]}")
                    total_skipped += 1
                else:
                    print(
                        f"[{name}/{filename}] SIZE MISMATCH  "
                        f"file={actual_hw[1]}×{actual_hw[0]}  "
                        f"label={label_hw[1]}×{label_hw[0]}  → resize"
                    )
                    if not args.dry_run:
                        try:
                            resize_and_save(dst, dst, label_hw)
                            total_resized += 1
                        except Exception as e:
                            msg = f"[{name}/{filename}] ERROR resizing — {e}"
                            print(msg)
                            errors.append(msg)
                    else:
                        total_resized += 1
                continue

            # File missing — look in source
            src = args.source / name / filename
            if not src.exists():
                msg = f"[{name}/{filename}] ERROR — not found in source ({src})"
                print(msg)
                errors.append(msg)
                continue

            try:
                src_hw = image_size(src)
            except ValueError as e:
                msg = f"[{name}/{filename}] ERROR reading source — {e}"
                print(msg)
                errors.append(msg)
                continue

            if src_hw == label_hw:
                action = "copy"
            else:
                action = f"copy+resize  {src_hw[1]}×{src_hw[0]} → {label_hw[1]}×{label_hw[0]}"

            print(f"[{name}/{filename}] MISSING → {action}")

            if not args.dry_run:
                try:
                    resize_and_save(src, dst, label_hw)
                    total_copied += 1
                except Exception as e:
                    msg = f"[{name}/{filename}] ERROR writing — {e}"
                    print(msg)
                    errors.append(msg)
            else:
                total_copied += 1

    print("-" * 60)
    print(f"Done.  copied/added: {total_copied}  resized: {total_resized}  "
          f"already OK: {total_skipped}  errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
