"""
Cost map construction for annotation-grow algorithm.

Pipeline:
  green channel → background removal → CLAHE → Sato vesselness
  → normalize → invert (bright = low cost) → exp scaling
"""

import cv2
import numpy as np
import skimage as ski


def build_enhanced_image(
    green: np.ndarray,
    roi_mask: np.ndarray,
    bg_kernel_size: int = 51,
    clahe_clip: float = 20.0,
    clahe_grid: tuple[int, int] = (16, 16),
    sato_sigmas: range = range(3, 8),
) -> np.ndarray:
    """
    Preprocess green channel into a vesselness-enhanced image.

    Args:
        green:          Green channel (H, W), uint8
        roi_mask:       ROI binary mask (H, W), uint8
        bg_kernel_size: Morphological opening kernel diameter for background removal
        clahe_clip:     CLAHE clip limit
        clahe_grid:     CLAHE tile grid size
        sato_sigmas:    Scale range for Sato vesselness filter

    Returns:
        roi_image: Enhanced uint8 image (H, W), masked to ROI
    """
    if bg_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size)
        )
        background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(green, background)
    else:
        corrected = green
    roi_image = cv2.bitwise_and(corrected, corrected, mask=roi_mask)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    roi_image = clahe.apply(roi_image)

    roi_image = ski.filters.sato(roi_image, sigmas=sato_sigmas, black_ridges=False)
    vmin, vmax = roi_image.min(), roi_image.max()
    if vmax > vmin:
        roi_image = (roi_image - vmin) / (vmax - vmin) * 255
    roi_image = roi_image.astype(np.uint8)

    return roi_image


def build_cost_map(enhanced: np.ndarray) -> np.ndarray:
    """
    Convert enhanced image to traversal cost map.

    Bright pixels (nerve tissue) → low cost.
    Dark pixels (background)     → high cost.

    cost = exp(1 - normalized_intensity) - 1

    Args:
        enhanced: uint8 enhanced image (H, W)

    Returns:
        cost_map: float32 array (H, W), range [0, e-1]
    """
    norm = enhanced.astype(np.float32) / 255.0
    cost = np.exp(1.0 - norm) - 1.0
    return cost.astype(np.float32)
