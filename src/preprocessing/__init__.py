"""
Preprocessing module for image preprocessing operations.

This module provides standalone preprocessing functions for image processing,
including morphological operations, background correction, and thresholding.

Main Functions:
    Morphological Operations:
        - morphological_opening: Remove small objects and noise
        - morphological_closing: Fill small holes and connect objects
        - morphological_gradient: Extract edges and boundaries
        - top_hat: Extract small bright objects on dark background
        - black_hat: Extract small dark objects on bright background

    Background Correction:
        - rolling_ball_background: Remove uneven background illumination
        - simple_background_subtraction: Subtract a known background image

    Thresholding/Binarization:
        - otsu_threshold: Automatic Otsu thresholding
        - fixed_threshold: Fixed threshold binarization
        - adaptive_threshold: Adaptive local thresholding
        - multi_otsu_threshold: Multi-level Otsu segmentation
        - triangle_threshold: Triangle algorithm thresholding

    Mask Operations:
        - dilate_epidermis_vertically: Dilate mask in vertical direction
        - apply_mask: Apply binary mask to image
        - invert_mask: Invert binary mask
        - combine_masks_or: Combine two masks with OR operation

    Pipeline:
        - SkinAnalysisPipeline: Complete pipeline for skin analysis

Utilities:
    - create_kernel: Create morphological structuring elements
    - ensure_grayscale: Convert images to grayscale

Example Usage:
    >>> import cv2
    >>> from src.preprocessing import (
    ...     morphological_opening,
    ...     rolling_ball_background,
    ...     otsu_threshold
    ... )
    >>>
    >>> # Load image
    >>> image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    >>>
    >>> # Remove small noise
    >>> cleaned = morphological_opening(image, kernel_size=3, kernel_shape='ellipse')
    >>>
    >>> # Correct uneven background
    >>> corrected = rolling_ball_background(image, radius=50, light_background=True)
    >>>
    >>> # Apply Otsu thresholding
    >>> binary = otsu_threshold(corrected, threshold_type='binary')
"""

# Morphological operations
from .morphology import (
    morphological_opening,
    morphological_closing,
    morphological_gradient,
    top_hat,
    black_hat,
)

# Background correction
from .background_correction import (
    rolling_ball_background,
    simple_background_subtraction,
    create_ball_kernel,
)

# Thresholding and binarization
from .thresholding import (
    otsu_threshold,
    fixed_threshold,
    adaptive_threshold,
    multi_otsu_threshold,
    triangle_threshold,
)

# Mask operations
from .mask_operations import (
    dilate_epidermis_vertically,
    apply_mask,
    invert_mask,
    combine_masks_or,
)

# Pipeline
from .pipeline import (
    SkinAnalysisPipeline,
)

# Utilities
from .utils import (
    create_kernel,
    ensure_grayscale,
    validate_image,
)


__all__ = [
    # Morphological operations
    'morphological_opening',
    'morphological_closing',
    'morphological_gradient',
    'top_hat',
    'black_hat',
    # Background correction
    'rolling_ball_background',
    'simple_background_subtraction',
    'create_ball_kernel',
    # Thresholding and binarization
    'otsu_threshold',
    'fixed_threshold',
    'adaptive_threshold',
    'multi_otsu_threshold',
    'triangle_threshold',
    # Mask operations
    'dilate_epidermis_vertically',
    'apply_mask',
    'invert_mask',
    'combine_masks_or',
    # Pipeline
    'SkinAnalysisPipeline',
    # Utilities
    'create_kernel',
    'ensure_grayscale',
    'validate_image',
]

__version__ = '1.0.0'
