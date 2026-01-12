"""
Morphological operations for image preprocessing.

This module provides morphological operations including opening, closing,
and other morphological transformations.
"""

from typing import Union, Tuple
import numpy as np
import cv2

from .utils import (
    validate_image,
    normalize_image,
    denormalize_image,
    create_kernel,
    KernelShape,
)


def morphological_opening(
    image: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_shape: KernelShape = "ellipse",
    iterations: int = 1,
) -> np.ndarray:
    """
    Perform morphological opening operation on an image.

    Opening is erosion followed by dilation. It is useful for:
    - Removing small objects (noise) from the foreground
    - Separating objects connected by thin bridges
    - Smoothing object boundaries

    Args:
        image: Input grayscale image (uint8 or float32)
        kernel_size: Size of the structuring element. Can be:
            - int: square kernel of size x size
            - Tuple[int, int]: kernel of (width, height)
            Default: 3
        kernel_shape: Shape of the structuring element:
            - 'rect': Rectangular
            - 'ellipse': Elliptical (default)
            - 'cross': Cross-shaped
        iterations: Number of times to apply the operation. Default: 1

    Returns:
        Processed image in the same format as input

    Raises:
        ValueError: If image or parameters are invalid

    Example:
        >>> import numpy as np
        >>> from src.preprocessing import morphological_opening
        >>> # Remove small noise from binary image
        >>> noisy_image = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]], dtype=np.uint8)
        >>> cleaned = morphological_opening(noisy_image, kernel_size=3, kernel_shape='ellipse')
    """
    validate_image(image)

    if iterations < 1:
        raise ValueError(f"Iterations must be at least 1, got {iterations}")

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Create structuring element
    kernel = create_kernel(kernel_size, kernel_shape)

    # Apply morphological opening
    opened = cv2.morphologyEx(
        image_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations
    )

    # Convert back to original format
    result = denormalize_image(opened, was_float, original_dtype)

    return result


def morphological_closing(
    image: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_shape: KernelShape = "ellipse",
    iterations: int = 1,
) -> np.ndarray:
    """
    Perform morphological closing operation on an image.

    Closing is dilation followed by erosion. It is useful for:
    - Closing small holes in the foreground
    - Connecting nearby objects
    - Smoothing object boundaries

    Args:
        image: Input grayscale image (uint8 or float32)
        kernel_size: Size of the structuring element. Can be:
            - int: square kernel of size x size
            - Tuple[int, int]: kernel of (width, height)
            Default: 3
        kernel_shape: Shape of the structuring element:
            - 'rect': Rectangular
            - 'ellipse': Elliptical (default)
            - 'cross': Cross-shaped
        iterations: Number of times to apply the operation. Default: 1

    Returns:
        Processed image in the same format as input

    Raises:
        ValueError: If image or parameters are invalid

    Example:
        >>> import numpy as np
        >>> from src.preprocessing import morphological_closing
        >>> # Fill small holes in binary image
        >>> image_with_holes = np.array([[255, 255, 255], [255, 0, 255], [255, 255, 255]], dtype=np.uint8)
        >>> filled = morphological_closing(image_with_holes, kernel_size=3, kernel_shape='ellipse')
    """
    validate_image(image)

    if iterations < 1:
        raise ValueError(f"Iterations must be at least 1, got {iterations}")

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Create structuring element
    kernel = create_kernel(kernel_size, kernel_shape)

    # Apply morphological closing
    closed = cv2.morphologyEx(
        image_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )

    # Convert back to original format
    result = denormalize_image(closed, was_float, original_dtype)

    return result


def morphological_gradient(
    image: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_shape: KernelShape = "ellipse",
) -> np.ndarray:
    """
    Compute morphological gradient (dilation - erosion).

    Useful for edge detection and boundary extraction.

    Args:
        image: Input grayscale image (uint8 or float32)
        kernel_size: Size of the structuring element
        kernel_shape: Shape of the structuring element

    Returns:
        Gradient image in the same format as input
    """
    validate_image(image)

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Create structuring element
    kernel = create_kernel(kernel_size, kernel_shape)

    # Apply morphological gradient
    gradient = cv2.morphologyEx(image_uint8, cv2.MORPH_GRADIENT, kernel)

    # Convert back to original format
    result = denormalize_image(gradient, was_float, original_dtype)

    return result


def top_hat(
    image: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_shape: KernelShape = "ellipse",
) -> np.ndarray:
    """
    Compute top-hat transform (image - opening).

    Useful for extracting small bright objects on dark background.

    Args:
        image: Input grayscale image (uint8 or float32)
        kernel_size: Size of the structuring element
        kernel_shape: Shape of the structuring element

    Returns:
        Top-hat transformed image in the same format as input
    """
    validate_image(image)

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Create structuring element
    kernel = create_kernel(kernel_size, kernel_shape)

    # Apply top-hat transform
    tophat = cv2.morphologyEx(image_uint8, cv2.MORPH_TOPHAT, kernel)

    # Convert back to original format
    result = denormalize_image(tophat, was_float, original_dtype)

    return result


def black_hat(
    image: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_shape: KernelShape = "ellipse",
) -> np.ndarray:
    """
    Compute black-hat transform (closing - image).

    Useful for extracting small dark objects on bright background.

    Args:
        image: Input grayscale image (uint8 or float32)
        kernel_size: Size of the structuring element
        kernel_shape: Shape of the structuring element

    Returns:
        Black-hat transformed image in the same format as input
    """
    validate_image(image)

    # Normalize to uint8 and track original format
    image_uint8, was_float, original_dtype = normalize_image(image)

    # Create structuring element
    kernel = create_kernel(kernel_size, kernel_shape)

    # Apply black-hat transform
    blackhat = cv2.morphologyEx(image_uint8, cv2.MORPH_BLACKHAT, kernel)

    # Convert back to original format
    result = denormalize_image(blackhat, was_float, original_dtype)

    return result
