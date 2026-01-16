"""
Thresholding and binarization operations for image preprocessing.

This module provides various thresholding methods for converting grayscale
images to binary images, including Otsu's method and adaptive thresholding.
"""

from typing import Literal
import numpy as np
import cv2

from .utils import (
    validate_image,
    normalize_image,
    denormalize_image,
)


ThresholdType = Literal["binary", "binary_inv"]
AdaptiveMethod = Literal["mean", "gaussian"]


def otsu_threshold(
    image: np.ndarray,
) -> np.ndarray:
    """
    Apply Otsu's automatic thresholding method to binarize an image.

    Args:
        image: Input grayscale image (uint8 or float32)

    Returns:
        Binary image in the same format as input
    """
    validate_image(image)

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Apply Otsu's thresholding
    threshold_value, binary = cv2.threshold(
        image_uint8,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    # Convert back to original format
    result = denormalize_image(binary, was_float, original_dtype)

    return result
