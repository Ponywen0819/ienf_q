"""
Replace weka.png in every data_orig/<SAMPLE_ID>/ folder with the matching
weka.png from data/<SAMPLE_ID>/.

For each sample folder under ``--data-orig``:
  1. Locate the source weka.png at ``<data>/<SAMPLE_ID>/weka.png``.
  2. Check its size against ``<data-orig>/<SAMPLE_ID>/label.png``.
  3. If sizes differ, resize the weka image to the label size using
     nearest-neighbour interpolation (preserves discrete label values).
  4. Write the result over ``<data-orig>/<SAMPLE_ID>/weka.png``.

Samples missing the source weka.png or the reference label.png are skipped
with a warning.

Usage:
    python tools/replace_weka.py                       # apply
    python tools/replace_weka.py --dry-run             # preview only
    python tools/replace_weka.py --data-orig data_orig --data data
"""

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

# Pillow >= 9.1 moved resampling constants under Image.Resampling.
NEAREST = getattr(Image, "Resampling", Image).NEAREST


def process_sample(
    sample_id: str,
    orig_dir: Path,
    data_dir: Path,
    dry_run: bool,
) -> str:
    """
    Replace one sample's weka.png. Returns a status string:
    'replaced', 'resized', 'skipped', or 'failed'.
    """
    src_weka = data_dir / sample_id / "weka.png"
    dst_weka = orig_dir / sample_id / "weka.png"
    label = orig_dir / sample_id / "label.png"

    if not src_weka.exists():
        print(f"  [skip]   {sample_id}: no source weka at {src_weka}")
        return "skipped"
    if not label.exists():
        print(f"  [skip]   {sample_id}: no reference label at {label}")
        return "skipped"

    try:
        with Image.open(label) as label_img:
            target_size = label_img.size  # (width, height)
        with Image.open(src_weka) as weka_img:
            weka_img.load()
            src_size = weka_img.size

            if src_size == target_size:
                if not dry_run:
                    shutil.copyfile(src_weka, dst_weka)
                print(f"  [ok]     {sample_id}: {src_size} (size matches label)")
                return "replaced"

            # Sizes differ — resize weka to label size with nearest neighbour.
            resized = weka_img.resize(target_size, resample=NEAREST)
            if not dry_run:
                resized.save(dst_weka)
            print(
                f"  [resize] {sample_id}: {src_size} -> {target_size} "
                f"(nearest, matched to label)"
            )
            return "resized"
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"  [fail]   {sample_id}: {e}")
        return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace weka.png in each data_orig/<SAMPLE_ID> folder with the "
            "matching data/<SAMPLE_ID>/weka.png, resized (nearest) to label.png."
        )
    )
    parser.add_argument(
        "--data-orig",
        type=Path,
        default=Path("data_orig"),
        help="Directory whose subfolders are sample IDs (target). Default: data_orig",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help="Directory holding the source weka.png per sample. Default: data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files.",
    )
    args = parser.parse_args()

    orig_dir: Path = args.data_orig
    data_dir: Path = args.data

    if not orig_dir.is_dir():
        print(f"Error: --data-orig not found: {orig_dir}", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print(f"Error: --data not found: {data_dir}", file=sys.stderr)
        return 1

    sample_dirs = sorted(p for p in orig_dir.iterdir() if p.is_dir())
    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"Scanning {len(sample_dirs)} sample folder(s) in {orig_dir}\n"
    )

    counts = {"replaced": 0, "resized": 0, "skipped": 0, "failed": 0}
    for sample_dir in sample_dirs:
        status = process_sample(sample_dir.name, orig_dir, data_dir, args.dry_run)
        counts[status] += 1

    print(
        f"\nSummary: {counts['replaced']} copied, {counts['resized']} resized, "
        f"{counts['skipped']} skipped, {counts['failed']} failed "
        f"(total {len(sample_dirs)})"
    )
    if args.dry_run:
        print("Dry run — no files were modified.")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
