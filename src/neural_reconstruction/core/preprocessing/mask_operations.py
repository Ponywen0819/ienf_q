"""
Mask operation utilities for image preprocessing.

This module provides functions for mask manipulation including dilation,
inversion, and application to images.
"""

from typing import Union
import numpy as np
import cv2

from .utils import validate_image


def dilate_epidermis_vertically(mask: np.ndarray, offset_px: int) -> np.ndarray:
    """
    Dilate a binary mask in the vertical (y-axis) direction only.

    This is useful for extending epidermis boundaries downward to capture
    boundary-crossing neural fibers.

    Args:
        mask: Binary mask image (uint8 with values 0 or 255, or bool)
        offset_px: Number of pixels to dilate in the vertical direction

    Returns:
        Dilated mask in the same format as input

    Raises:
        ValueError: If mask is invalid or offset_px is non-positive

    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[40:60, 40:60] = 255
        >>> dilated = dilate_epidermis_vertically(mask, offset_px=10)
        >>> # The mask is now extended 10 pixels downward
    """
    validate_image(mask)

    if offset_px <= 0:
        raise ValueError(f"offset_px must be positive, got {offset_px}")

    # Remember original dtype
    original_dtype = mask.dtype
    was_bool = mask.dtype == bool

    # Convert to uint8 if needed
    if was_bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)

    # Create vertical structuring element (1 pixel wide, offset_px tall)
    # This ensures dilation only happens in the y-direction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, offset_px))

    # Set anchor to BOTTOM of kernel so dilation only extends downward
    # anchor = (x, y) where y = offset_px - 1 is the last row of kernel
    # This makes the kernel extend from current pixel downward (positive y direction)
    anchor = (0, offset_px - 1)

    # Perform dilation with bottom anchor - only extends downward (positive y direction)
    dilated = cv2.dilate(mask_uint8, kernel, anchor=anchor, iterations=1)

    # Convert back to original format
    if was_bool:
        return dilated.astype(bool)
    else:
        return dilated.astype(original_dtype)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to an image, keeping only masked regions.

    Supports both single-channel and multi-channel images. Areas where the
    mask is 0 (or False) will be set to 0 in the output.

    Args:
        image: Input image (grayscale or color, uint8 or float)
        mask: Binary mask (2D array, uint8 with 0/255, or bool)

    Returns:
        Masked image in the same format as input

    Raises:
        ValueError: If image and mask shapes are incompatible

    Example:
        >>> image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 255
        >>> masked_image = apply_mask(image, mask)
        >>> # Only the center 50x50 region is preserved
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be a numpy array, got {type(image)}")

    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Mask must be a numpy array, got {type(mask)}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

    # Check spatial dimensions match
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image spatial dimensions {image.shape[:2]} must match "
            f"mask dimensions {mask.shape}"
        )

    # Convert boolean mask to uint8 if needed
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)

    # Apply mask using bitwise AND
    # cv2.bitwise_and handles both grayscale and color images
    if image.ndim == 2:
        # Grayscale image
        result = cv2.bitwise_and(image, image, mask=mask_uint8)
    elif image.ndim == 3:
        # Color image
        result = cv2.bitwise_and(image, image, mask=mask_uint8)
    else:
        raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")

    return result


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """
    Invert a binary mask (0 becomes 255, 255 becomes 0).

    Args:
        mask: Binary mask (uint8 with values 0/255, or bool)

    Returns:
        Inverted mask in the same format as input

    Raises:
        ValueError: If mask is invalid

    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 255
        >>> inverted = invert_mask(mask)
        >>> # Now the center is 0 and the border is 255
    """
    validate_image(mask)

    # Remember original dtype
    original_dtype = mask.dtype
    was_bool = mask.dtype == bool

    # Invert based on type
    if was_bool:
        return ~mask
    else:
        # Use bitwise NOT for uint8
        mask_uint8 = mask.astype(np.uint8)
        inverted = cv2.bitwise_not(mask_uint8)
        return inverted.astype(original_dtype)


def combine_masks_or(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Combine two binary masks using OR operation.

    The result will have 255 (or True) where either mask1 OR mask2 is 255 (or True).

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        Combined mask in uint8 format (0 or 255)

    Raises:
        ValueError: If masks have incompatible shapes

    Example:
        >>> mask1 = np.zeros((100, 100), dtype=np.uint8)
        >>> mask1[0:50, :] = 255
        >>> mask2 = np.zeros((100, 100), dtype=np.uint8)
        >>> mask2[50:100, :] = 255
        >>> combined = combine_masks_or(mask1, mask2)
        >>> # The entire image is now 255
    """
    validate_image(mask1)
    validate_image(mask2)

    if mask1.shape != mask2.shape:
        raise ValueError(
            f"Mask shapes must match: mask1 {mask1.shape} vs mask2 {mask2.shape}"
        )

    # Convert to uint8 if needed
    if mask1.dtype == bool:
        mask1_uint8 = mask1.astype(np.uint8) * 255
    else:
        mask1_uint8 = mask1.astype(np.uint8)

    if mask2.dtype == bool:
        mask2_uint8 = mask2.astype(np.uint8) * 255
    else:
        mask2_uint8 = mask2.astype(np.uint8)

    # Perform OR operation
    combined = cv2.bitwise_or(mask1_uint8, mask2_uint8)

    return combined
