"""
Thresholding and binarization operations for image preprocessing.

This module provides various thresholding methods for converting grayscale
images to binary images, including Otsu's method and adaptive thresholding.
"""

from typing import Literal
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

from .utils import (
    validate_image,
    normalize_image,
    denormalize_image,
)


ThresholdType = Literal["binary", "binary_inv"]
AdaptiveMethod = Literal["mean", "gaussian"]


def multi_otsu_threshold(
    image: np.ndarray,
    classes: int = 2,
) -> np.ndarray:
    """
    Apply multi-level Otsu thresholding and select top-level region.

    Multi-level Otsu thresholding segments the image into multiple intensity
    classes and selects the top-level (brightest) region. This is more robust
    than single Otsu for images with complex intensity distributions.

    Args:
        image: Input grayscale image (uint8 or float32)
        classes: Number of intensity classes (default: 2)

    Returns:
        Binary image with top-level region (uint8, 0 or 255)

    Raises:
        ValueError: If image is invalid or classes < 2
    """
    validate_image(image)

    if classes < 2:
        raise ValueError(f"classes must be >= 2, got {classes}")

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Apply multi-level Otsu thresholding
    thresholds = threshold_multiotsu(image_uint8, classes=classes)

    # Digitize image into regions
    regions = np.digitize(image_uint8, bins=thresholds)

    # Select top-level region (brightest pixels)
    top_level_mask = (regions == len(thresholds)).astype(np.uint8) * 255

    # Convert back to original format
    result = denormalize_image(top_level_mask, was_float, original_dtype)

    return result


def otsu_threshold(
    image: np.ndarray,
) -> np.ndarray:
    """
    Apply Otsu's automatic thresholding method to binarize an image.

    Deprecated: This function is kept for backward compatibility.
    New code should use multi_otsu_threshold() instead.

    Args:
        image: Input grayscale image (uint8 or float32)

    Returns:
        Binary image in the same format as input
    """
    return multi_otsu_threshold(image, classes=2)
