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
    Dilate a binary mask using a circular SE, constrained to regions above
    the bottommost mask pixel in each column.

    For each column x, the topmost (smallest y) mask pixel defines the upper
    boundary of the epidermis. An auxiliary mask is built where every pixel
    at or below that boundary (y >= min_y[x]) is 255, representing the region
    that lies within or beneath the epidermis top edge. The original mask is
    dilated with a circular SE of radius offset_px, then intersected with the
    auxiliary mask so the dilation extends downward only and cannot bleed
    upward above the epidermis top boundary.

    Args:
        mask: Binary mask image (uint8 with values 0 or 255, or bool)
        offset_px: Radius of the circular structuring element

    Returns:
        Dilated mask intersected with the auxiliary mask, same format as input

    Raises:
        ValueError: If mask is invalid or offset_px is non-positive

    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[40:60, 40:60] = 255
        >>> dilated = dilate_epidermis_vertically(mask, offset_px=10)
    """
    validate_image(mask)

    if offset_px <= 0:
        raise ValueError(f"offset_px must be positive, got {offset_px}")

    original_dtype = mask.dtype
    was_bool = mask.dtype == bool

    if was_bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)

    H, W = mask_uint8.shape
    binary = mask_uint8 > 0  # (H, W)

    # For each column, find the topmost (min y) mask pixel.
    # argmax on binary finds the first True from the top = min y with mask.
    col_has_mask = binary.any(axis=0)  # (W,)
    min_y = np.where(
        col_has_mask,
        np.argmax(binary, axis=0),
        H,  # sentinel: no mask → nothing in aux_mask for this column
    )  # (W,)

    # Auxiliary mask: pixel (y, x) is 255 if y >= min_y[x]
    # i.e. there exists a mask pixel ABOVE this pixel (at smaller y) in the same column,
    # meaning this pixel is at or below the top edge of the epidermis → allow downward dilation
    y_indices = np.arange(H).reshape(-1, 1)  # (H, 1)
    aux_mask = np.where(
        y_indices >= min_y[np.newaxis, :], np.uint8(255), np.uint8(0)
    ).astype(np.uint8)

    # Dilate original mask with circular (ellipse) SE of radius offset_px
    d = 2 * offset_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

    # Intersect: keep only dilation results that lie within the auxiliary mask
    result = cv2.bitwise_and(dilated, aux_mask)

    if was_bool:
        return result.astype(bool)
    else:
        return result.astype(original_dtype)


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
