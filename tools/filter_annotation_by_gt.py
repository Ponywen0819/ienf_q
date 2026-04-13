"""
Remove annotation components that have no overlap with the GT label image.

For each sample that has a label.png, any connected component in the annotation
(weka.png) with zero pixel overlap with the GT label is considered spurious and
gets removed.

Output: annotation_filtered.png saved alongside weka.png.
Use --inplace to overwrite weka.png directly (saves a .bak backup first).
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from skimage import io
from skimage.measure import label, regionprops
from scipy import ndimage

from neural_reconstruction.dataset import DatasetLoader, SampleFiles


def get_components(binary_img, min_area=10):
    labeled = label(binary_img, connectivity=1)
    for p in regionprops(labeled):
        if p.area < min_area:
            labeled[labeled == p.label] = 0
    labeled, _ = ndimage.label(labeled > 0)
    return labeled


def filter_annotation(sample: SampleFiles, min_area=10, inplace=False, verbose=True):
    annotation = io.imread(sample.annotation_path)
    label_img = io.imread(sample.label_path)
    mask = io.imread(sample.mask_path)

    original_shape = annotation.shape

    if annotation.ndim == 3:
        annotation = annotation[..., 0]
    if label_img.ndim == 3:
        label_img = label_img[..., 0]
    if mask.ndim == 3:
        mask = mask[..., 0]

    annotation_bin = (annotation > 127).astype(np.uint8)
    label_bin = (label_img > 127).astype(np.uint8)
    # mask_bin = (mask > 127).astype(np.uint8)

    annot_labeled = get_components(annotation_bin, min_area=min_area)

    n_total = annot_labeled.max()
    n_removed = 0
    kept_mask = np.zeros_like(annot_labeled, dtype=bool)

    for a_id in range(1, n_total + 1):
        a_mask = annot_labeled == a_id
        has_gt = np.any(label_bin[a_mask] > 0)
        if has_gt:
            kept_mask |= a_mask
        else:
            n_removed += 1

    if verbose:
        print(
            f"  {sample.sample_id}: {n_total} components -> "
            f"removed {n_removed}, kept {n_total - n_removed}"
        )

    out_2d = (kept_mask * 255).astype(np.uint8)
    out = (
        np.stack([out_2d] * original_shape[2], axis=-1)
        if len(original_shape) == 3
        else out_2d
    )

    if inplace:
        bak_path = sample.annotation_path.with_suffix(".bak.png")
        shutil.copy2(sample.annotation_path, bak_path)
        io.imsave(sample.annotation_path, out, check_contrast=False)
        if verbose:
            print(
                f"    Backup -> {bak_path.name}, overwrote {sample.annotation_path.name}"
            )
    else:
        out_path = sample.annotation_path.parent / "annotation_filtered.png"
        io.imsave(out_path, out, check_contrast=False)
        if verbose:
            print(f"    Saved -> {out_path.name}")

    return n_total, n_removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data_0331",
        help="Root data directory (default: data_0331)",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Restrict to these sample IDs (default: all with label.png)",
    )
    parser.add_argument(
        "--min-area", type=int, default=0, help="Minimum component area (default: 0)"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite weka.png (saves .bak backup). "
        "Default: save as annotation_filtered.png",
    )
    args = parser.parse_args()

    loader = DatasetLoader(Path(args.data_dir))
    samples = loader.load_samples(args.sample_ids)

    # Only process samples that have a GT label
    gt_samples = [s for s in samples if s.label_path and s.label_path.exists()]
    print(
        f"Found {len(gt_samples)} samples with GT label (out of {len(samples)} total)\n"
    )

    total_components = 0
    total_removed = 0

    for sample in gt_samples:
        ok, reason = sample.is_complete()
        if not ok:
            print(f"  SKIP {sample.sample_id}: {reason}")
            continue

        n_total, n_removed = filter_annotation(
            sample, min_area=args.min_area, inplace=args.inplace
        )
        total_components += n_total
        total_removed += n_removed

    print(
        f"\nDone. Removed {total_removed}/{total_components} spurious components "
        f"({total_removed / max(total_components, 1) * 100:.1f}%)."
    )


if __name__ == "__main__":
    main()
