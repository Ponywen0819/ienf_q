"""
Utility functions for image preprocessing.

This module provides common utilities for image validation, kernel generation,
and format conversion used across preprocessing operations.
"""

from typing import Tuple, Literal, Union
import numpy as np
import cv2


KernelShape = Literal['rect', 'ellipse', 'cross']


def validate_image(image: np.ndarray) -> None:
    """
    Validate that the input is a valid grayscale image.

    Args:
        image: Input image array

    Raises:
        ValueError: If image is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be a numpy array, got {type(image)}")

    if image.ndim != 2:
        raise ValueError(f"Image must be 2D grayscale, got shape {image.shape}")

    if image.size == 0:
        raise ValueError("Image is empty")


def normalize_image(image: np.ndarray) -> Tuple[np.ndarray, bool, np.dtype]:
    """
    Normalize image to uint8 format and track original format.

    Args:
        image: Input image (uint8 or float)

    Returns:
        Tuple of (normalized_image, was_float, original_dtype)
    """
    validate_image(image)

    original_dtype = image.dtype
    was_float = np.issubdtype(original_dtype, np.floating)

    if was_float:
        # Convert float [0.0, 1.0] to uint8 [0, 255]
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        # Already uint8 or convert to uint8
        image_uint8 = image.astype(np.uint8)

    return image_uint8, was_float, original_dtype


def denormalize_image(
    image: np.ndarray,
    was_float: bool,
    original_dtype: np.dtype
) -> np.ndarray:
    """
    Convert image back to its original format.

    Args:
        image: Processed image in uint8 format
        was_float: Whether original image was float type
        original_dtype: Original image dtype

    Returns:
        Image in original format
    """
    if was_float:
        # Convert uint8 [0, 255] back to float [0.0, 1.0]
        return (image.astype(np.float32) / 255.0).astype(original_dtype)
    else:
        return image.astype(original_dtype)


def create_kernel(
    size: Union[int, Tuple[int, int]],
    shape: KernelShape = 'ellipse'
) -> np.ndarray:
    """
    Create a morphological structuring element (kernel).

    Args:
        size: Kernel size. Can be:
            - int: creates square kernel of size x size
            - Tuple[int, int]: creates kernel of (width, height)
        shape: Kernel shape, one of:
            - 'rect': Rectangular kernel
            - 'ellipse': Elliptical kernel (default)
            - 'cross': Cross-shaped kernel

    Returns:
        Binary kernel as numpy array

    Raises:
        ValueError: If size or shape is invalid
    """
    # Normalize size to tuple
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"Kernel size must be positive, got {size}")
        kernel_size = (size, size)
    else:
        if len(size) != 2:
            raise ValueError(f"Kernel size tuple must have 2 elements, got {len(size)}")
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"Kernel dimensions must be positive, got {size}")
        kernel_size = size

    # Create kernel based on shape
    if shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    elif shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        raise ValueError(
            f"Invalid kernel shape '{shape}'. "
            f"Must be one of: 'rect', 'ellipse', 'cross'"
        )

    return kernel


def ensure_grayscale(image: np.ndarray, extract_channel: int = 1) -> np.ndarray:
    """
    Ensure image is grayscale, extracting a channel if needed.

    Args:
        image: Input image (grayscale or RGB/BGR)
        extract_channel: Channel to extract if image is color (default: 1 for green in BGR)

    Returns:
        Grayscale image
    """
    if image.ndim == 2:
        # Already grayscale
        return image
    elif image.ndim == 3:
        # Extract specified channel
        if image.shape[2] >= extract_channel + 1:
            return image[:, :, extract_channel]
        else:
            # Convert to grayscale using standard method
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Invalid image shape: {image.shape}")


def clip_image(image: np.ndarray) -> np.ndarray:
    """
    Clip image values to valid range based on dtype.

    Args:
        image: Input image

    Returns:
        Clipped image
    """
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image, 0.0, 1.0)
    elif np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        return np.clip(image, info.min, info.max)
    else:
        return image
