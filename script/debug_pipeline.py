"""
Debug script to diagnose preprocessing pipeline issues.
"""
import cv2
import numpy as np
import os
from preprocessing import (
    morphological_closing,
    morphological_opening,
    rolling_ball_background,
    otsu_threshold,
    dilate_epidermis_vertically,
    apply_mask,
    invert_mask,
    combine_masks_or
)


def check_image_stats(image, name):
    """Print statistics about an image."""
    print(f"\n{name}:")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Min: {image.min()}, Max: {image.max()}")
    print(f"  Mean: {image.mean():.2f}")
    print(f"  Unique values: {np.unique(image)[:10]}")  # First 10 unique values
    print(f"  Non-zero pixels: {np.count_nonzero(image)}")


if __name__ == "__main__":
    print("=" * 80)
    print("PREPROCESSING PIPELINE DEBUG")
    print("=" * 80)

    # Load images
    print("\n[1] Loading images...")
    label_image = cv2.imread('/Users/ponywen/projects/ienf_q/data/Label/S163-2_a.tif', cv2.IMREAD_GRAYSCALE)
    epidermis_mask = cv2.imread('/Users/ponywen/projects/ienf_q/data/Mask/S163-2_a.tif', cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread('/Users/ponywen/projects/ienf_q/data/Original/S163-2_a_green.png', cv2.IMREAD_GRAYSCALE)

    check_image_stats(label_image, "Input Label")
    check_image_stats(epidermis_mask, "Epidermis Mask")
    check_image_stats(original_image, "Original Image")

    # Step 1: Process label path
    print("\n" + "=" * 80)
    print("[2] Processing Label Path")
    print("=" * 80)

    closed = morphological_closing(label_image, kernel_size=3)
    check_image_stats(closed, "After Closing")
    print(f"  Changed from input? {not np.array_equal(label_image, closed)}")

    opened = morphological_opening(closed, kernel_size=3)
    check_image_stats(opened, "After Opening")
    print(f"  Changed from closed? {not np.array_equal(closed, opened)}")

    processed_label = opened

    # Step 2: Create dilated mask
    print("\n" + "=" * 80)
    print("[3] Creating Dilated Mask")
    print("=" * 80)

    dilated_mask = dilate_epidermis_vertically(epidermis_mask, offset_px=100)
    check_image_stats(dilated_mask, "Dilated Mask")
    print(f"  Changed from input mask? {not np.array_equal(epidermis_mask, dilated_mask)}")
    print(f"  Dilation increased area by: {np.count_nonzero(dilated_mask) - np.count_nonzero(epidermis_mask)} pixels")

    # Step 3: Process original image
    print("\n" + "=" * 80)
    print("[4] Processing Original Image")
    print("=" * 80)

    # Background correction (False = bright objects on dark background)
    corrected = rolling_ball_background(original_image, radius=12, light_background=False)
    check_image_stats(corrected, "After Background Correction")
    print(f"  Changed from original? {not np.array_equal(original_image, corrected)}")

    # Branch A: ROI using dilated mask
    roi_image = apply_mask(corrected, dilated_mask)
    check_image_stats(roi_image, "ROI Image (using dilated mask)")
    print(f"  Different from corrected? {not np.array_equal(corrected, roi_image)}")

    # Branch B: Pseudo-label
    print("\n  --- Branch B: Pseudo-label generation (CORRECTED) ---")

    # Calculate new region = dilated_mask AND (NOT epidermis_mask)
    inverted_epidermis = invert_mask(epidermis_mask)
    check_image_stats(inverted_epidermis, "Inverted Epidermis Mask")

    new_region_mask = apply_mask(dilated_mask, inverted_epidermis)
    check_image_stats(new_region_mask, "New Region Mask (dilated * inverted_epidermis)")
    print(f"  This is the extended dermis boundary region")

    masked_region = apply_mask(corrected, new_region_mask)
    check_image_stats(masked_region, "Masked Region (corrected * new_region_mask)")

    pseudo_label = otsu_threshold(masked_region, threshold_type='binary')
    check_image_stats(pseudo_label, "Pseudo Label (Otsu)")

    # Step 4: Merge labels
    print("\n" + "=" * 80)
    print("[5] Merging Labels")
    print("=" * 80)

    final_label = combine_masks_or(processed_label, pseudo_label)
    check_image_stats(final_label, "Final Label")

    print(f"\n  Input label had {np.count_nonzero(label_image)} pixels")
    print(f"  Processed label has {np.count_nonzero(processed_label)} pixels")
    print(f"  Pseudo label has {np.count_nonzero(pseudo_label)} pixels")
    print(f"  Final label has {np.count_nonzero(final_label)} pixels")
    print(f"  Net change: {np.count_nonzero(final_label) - np.count_nonzero(label_image)} pixels")

    # Save debug outputs
    print("\n" + "=" * 80)
    print("[6] Saving Debug Outputs")
    print("=" * 80)

    os.makedirs('output/preprocessing_debug', exist_ok=True)

    cv2.imwrite('output/preprocessing_debug/01_input_label.png', label_image)
    cv2.imwrite('output/preprocessing_debug/02_closed_label.png', closed)
    cv2.imwrite('output/preprocessing_debug/03_processed_label.png', processed_label)
    cv2.imwrite('output/preprocessing_debug/04_epidermis_mask.png', epidermis_mask)
    cv2.imwrite('output/preprocessing_debug/05_dilated_mask.png', dilated_mask)
    cv2.imwrite('output/preprocessing_debug/06_original_image.png', original_image)
    cv2.imwrite('output/preprocessing_debug/07_corrected_image.png', corrected)
    cv2.imwrite('output/preprocessing_debug/08_roi_image.png', roi_image)
    cv2.imwrite('output/preprocessing_debug/09_inverted_epidermis.png', inverted_epidermis)
    cv2.imwrite('output/preprocessing_debug/10_new_region_mask.png', new_region_mask)
    cv2.imwrite('output/preprocessing_debug/11_masked_region.png', masked_region)
    cv2.imwrite('output/preprocessing_debug/12_pseudo_label.png', pseudo_label)
    cv2.imwrite('output/preprocessing_debug/13_final_label.png', final_label)

    print("  All debug images saved to output/preprocessing_debug/")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
