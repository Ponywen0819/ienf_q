"""
Thresholding and binarization operations for image preprocessing.

This module provides various thresholding methods for converting grayscale
images to binary images, including Otsu's method and adaptive thresholding.
"""

from typing import Literal, Tuple
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
    threshold_type: ThresholdType = "binary",
    return_threshold: bool = False,
) -> np.ndarray:
    """
    Apply Otsu's automatic thresholding method to binarize an image.

    Otsu's method automatically determines an optimal threshold value by
    maximizing the between-class variance. This works well for images with
    bimodal histograms (two distinct peaks).

    Args:
        image: Input grayscale image (uint8 or float32)
        threshold_type: Type of thresholding to apply:
            - 'binary': pixels above threshold become white (255)
            - 'binary_inv': pixels above threshold become black (0)
            Default: 'binary'
        return_threshold: If True, return both the binarized image and
            the computed threshold value. Default: False

    Returns:
        If return_threshold is False:
            Binary image in the same format as input
        If return_threshold is True:
            Tuple of (binary_image, threshold_value)

    Raises:
        ValueError: If image or parameters are invalid

    Example:
        >>> import cv2
        >>> from src.preprocessing import otsu_threshold
        >>> # Load grayscale image
        >>> image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
        >>> # Apply Otsu thresholding
        >>> binary = otsu_threshold(image, threshold_type='binary')
        >>> # Get threshold value
        >>> binary, threshold = otsu_threshold(image, return_threshold=True)
        >>> print(f"Optimal threshold: {threshold}")
    """
    validate_image(image)

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Determine OpenCV threshold type
    if threshold_type == "binary":
        cv_thresh_type = cv2.THRESH_BINARY
    elif threshold_type == "binary_inv":
        cv_thresh_type = cv2.THRESH_BINARY_INV
    else:
        raise ValueError(
            f"Invalid threshold_type '{threshold_type}'. "
            f"Must be 'binary' or 'binary_inv'"
        )

    # Apply Otsu's thresholding
    threshold_value, binary = cv2.threshold(
        image_uint8,
        0,  # Threshold value is ignored when using Otsu
        255,
        cv_thresh_type | cv2.THRESH_OTSU,
    )

    # Convert back to original format
    result = denormalize_image(binary, was_float, original_dtype)

    # if return_threshold:
    #     # Return threshold in 0-1 range if original was float
    #     if was_float:
    #         threshold_value = threshold_value / 255.0
    #     return result, threshold_value
    # else:
    return result


def fixed_threshold(
    image: np.ndarray,
    threshold_value: float,
    threshold_type: ThresholdType = "binary",
    max_value: float = 255,
) -> np.ndarray:
    """
    Apply fixed threshold binarization to an image.

    Args:
        image: Input grayscale image (uint8 or float32)
        threshold_value: Threshold value. For uint8 images, use 0-255.
            For float images, use 0.0-1.0.
        threshold_type: Type of thresholding:
            - 'binary': pixels above threshold become max_value
            - 'binary_inv': pixels below threshold become max_value
            Default: 'binary'
        max_value: Maximum value to use (default: 255 for uint8)

    Returns:
        Binary image in the same format as input

    Raises:
        ValueError: If image or parameters are invalid

    Example:
        >>> import cv2
        >>> from src.preprocessing import fixed_threshold
        >>> image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
        >>> # Apply fixed threshold at 127
        >>> binary = fixed_threshold(image, threshold_value=127)
    """
    validate_image(image)

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Adjust threshold if original was float
    if was_float:
        threshold_uint8 = int(threshold_value * 255)
    else:
        threshold_uint8 = int(threshold_value)

    # Validate threshold range
    if not (0 <= threshold_uint8 <= 255):
        raise ValueError(
            f"Threshold value must be in range [0, 255] for uint8, got {threshold_uint8}"
        )

    # Determine OpenCV threshold type
    if threshold_type == "binary":
        cv_thresh_type = cv2.THRESH_BINARY
    elif threshold_type == "binary_inv":
        cv_thresh_type = cv2.THRESH_BINARY_INV
    else:
        raise ValueError(
            f"Invalid threshold_type '{threshold_type}'. "
            f"Must be 'binary' or 'binary_inv'"
        )

    # Apply fixed thresholding
    _, binary = cv2.threshold(image_uint8, threshold_uint8, max_value, cv_thresh_type)

    # Convert back to original format
    result = denormalize_image(binary, was_float, original_dtype)

    return result


def adaptive_threshold(
    image: np.ndarray,
    method: AdaptiveMethod = "gaussian",
    threshold_type: ThresholdType = "binary",
    block_size: int = 11,
    c: float = 2,
) -> np.ndarray:
    """
    Apply adaptive thresholding to an image.

    Adaptive thresholding calculates the threshold for smaller regions of
    the image, which can handle varying illumination better than global
    thresholding methods.

    Args:
        image: Input grayscale image (uint8 or float32)
        method: Adaptive method to use:
            - 'mean': Threshold is mean of neighborhood minus c
            - 'gaussian': Threshold is weighted sum (Gaussian window) minus c
            Default: 'gaussian'
        threshold_type: Type of thresholding:
            - 'binary': pixels above threshold become white
            - 'binary_inv': pixels above threshold become black
            Default: 'binary'
        block_size: Size of the neighborhood area (must be odd, >= 3).
            Default: 11
        c: Constant subtracted from the mean or weighted mean.
            Default: 2

    Returns:
        Binary image in the same format as input

    Raises:
        ValueError: If image or parameters are invalid

    Example:
        >>> import cv2
        >>> from src.preprocessing import adaptive_threshold
        >>> image = cv2.imread('document.png', cv2.IMREAD_GRAYSCALE)
        >>> # Good for documents with uneven lighting
        >>> binary = adaptive_threshold(
        ...     image,
        ...     method='gaussian',
        ...     block_size=11,
        ...     c=2
        ... )
    """
    validate_image(image)

    if block_size < 3:
        raise ValueError(f"Block size must be at least 3, got {block_size}")

    if block_size % 2 == 0:
        raise ValueError(f"Block size must be odd, got {block_size}")

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Determine adaptive method
    if method == "mean":
        cv_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method == "gaussian":
        cv_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'mean' or 'gaussian'")

    # Determine threshold type
    if threshold_type == "binary":
        cv_thresh_type = cv2.THRESH_BINARY
    elif threshold_type == "binary_inv":
        cv_thresh_type = cv2.THRESH_BINARY_INV
    else:
        raise ValueError(
            f"Invalid threshold_type '{threshold_type}'. "
            f"Must be 'binary' or 'binary_inv'"
        )

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        image_uint8, 255, cv_method, cv_thresh_type, block_size, c
    )

    # Convert back to original format
    result = denormalize_image(binary, was_float, original_dtype)

    return result


def multi_otsu_threshold(
    image: np.ndarray, n_classes: int = 3, return_thresholds: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Apply multi-level Otsu thresholding to segment image into multiple classes.

    This extends Otsu's method to find multiple thresholds that segment the
    image into n_classes regions.

    Args:
        image: Input grayscale image (uint8 or float32)
        n_classes: Number of classes to segment into (2-5).
            n_classes=2 is equivalent to standard Otsu. Default: 3
        return_thresholds: If True, return both the segmented image and
            the threshold values. Default: False

    Returns:
        If return_thresholds is False:
            Segmented image with labels 0, 1, ..., n_classes-1
        If return_thresholds is True:
            Tuple of (segmented_image, threshold_array)

    Raises:
        ValueError: If image or parameters are invalid

    Example:
        >>> import cv2
        >>> from src.preprocessing import multi_otsu_threshold
        >>> image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
        >>> # Segment into 3 classes
        >>> segmented = multi_otsu_threshold(image, n_classes=3)
        >>> # Get threshold values
        >>> segmented, thresholds = multi_otsu_threshold(
        ...     image, n_classes=3, return_thresholds=True
        ... )
    """
    validate_image(image)

    if not (2 <= n_classes <= 5):
        raise ValueError(f"n_classes must be between 2 and 5, got {n_classes}")

    # Import skimage for multi-Otsu
    try:
        from skimage.filters import threshold_multiotsu
    except ImportError:
        raise ImportError(
            "Multi-Otsu thresholding requires scikit-image. "
            "Install it with: pip install scikit-image"
        )

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Compute multi-Otsu thresholds
    thresholds = threshold_multiotsu(image_uint8, classes=n_classes)

    # Create segmented image
    segmented = np.digitize(image_uint8, bins=thresholds)

    # Convert back to original dtype
    segmented = segmented.astype(original_dtype)

    if return_thresholds:
        # Convert thresholds to 0-1 range if original was float
        if was_float:
            thresholds = np.array(thresholds) / 255.0
        return segmented, thresholds
    else:
        return segmented


def triangle_threshold(
    image: np.ndarray,
    threshold_type: ThresholdType = "binary",
    return_threshold: bool = False,
) -> np.ndarray | Tuple[np.ndarray, float]:
    """
    Apply Triangle algorithm for automatic thresholding.

    The Triangle method is good for images where the object is darker
    than the background and occupies a small portion of the image.

    Args:
        image: Input grayscale image (uint8 or float32)
        threshold_type: Type of thresholding ('binary' or 'binary_inv')
        return_threshold: If True, return both image and threshold value

    Returns:
        Binary image, optionally with threshold value

    Example:
        >>> from src.preprocessing import triangle_threshold
        >>> binary = triangle_threshold(image, threshold_type='binary')
    """
    validate_image(image)

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Determine OpenCV threshold type
    if threshold_type == "binary":
        cv_thresh_type = cv2.THRESH_BINARY
    elif threshold_type == "binary_inv":
        cv_thresh_type = cv2.THRESH_BINARY_INV
    else:
        raise ValueError(
            f"Invalid threshold_type '{threshold_type}'. "
            f"Must be 'binary' or 'binary_inv'"
        )

    # Apply Triangle thresholding
    threshold_value, binary = cv2.threshold(
        image_uint8, 0, 255, cv_thresh_type | cv2.THRESH_TRIANGLE
    )

    # Convert back to original format
    result = denormalize_image(binary, was_float, original_dtype)

    if return_threshold:
        if was_float:
            threshold_value = threshold_value / 255.0
        return result, threshold_value
    else:
        return result
