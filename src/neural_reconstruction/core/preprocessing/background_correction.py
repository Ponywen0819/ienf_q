"""
Background correction for image preprocessing.

This module provides background correction methods including rolling ball
algorithm for removing uneven illumination.
"""

from typing import Literal
import numpy as np
import cv2
from skimage.restoration import rolling_ball
from skimage.filters import sato

from .utils import (
    validate_image,
    normalize_image,
    denormalize_image,
)


class BackgroundCorrection:
    """
    Background correction class for removing uneven illumination from images.

    This class provides methods for background correction using rolling ball
    and morphology techniques, with optional Sato vesselness filtering.

    Attributes:
        method: Background correction method ('rolling_ball' or 'morphology')
        radius: Radius for rolling ball/morphology operations
        sato_weight: Blend weight for Sato filter (0=disabled, >0=enabled)
        sato_sigmas: Iterable of floats for Sato filter scales

    Args:
        method: Background correction method ('rolling_ball' or 'morphology'). Default: 'morphology'
        radius: Radius for rolling ball/morphology operations. Default: 50
        sato_weight: Blend weight for Sato filter. Default: 0.0 (disabled)
        sato_sigmas: Iterable of floats for Sato filter scales. Default: (1.0, 2.0, 3.0)

    Example:
        >>> corrector = BackgroundCorrection(method='rolling_ball', radius=50, sato_weight=0.3)
        >>> corrected_image = corrector.correct(image)
    """

    def __init__(
        self,
        method: Literal["rolling_ball", "morphology"] = "morphology",
        radius: int = 50,
        sato_weight: float = 0.0,
        sato_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
    ):
        """
        Initialize BackgroundCorrection.

        Args:
            method: Background correction method ('rolling_ball' or 'morphology')
            radius: Radius for rolling ball/morphology operations
            sato_weight: Blend weight for Sato filter (0=disabled, >0=enabled, max 1)
            sato_sigmas: Iterable of floats for Sato filter scales

        Raises:
            ValueError: If method is not 'rolling_ball' or 'morphology'
            ValueError: If radius is not positive
            ValueError: If sato_weight is not in [0, 1]
        """
        if method not in ["rolling_ball", "morphology"]:
            raise ValueError(
                f"Invalid method '{method}'. Must be 'rolling_ball' or 'morphology'"
            )
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        if not 0 <= sato_weight <= 1:
            raise ValueError(f"sato_weight must be in [0, 1], got {sato_weight}")

        self.method = method
        self.radius = radius
        self.sato_weight = sato_weight
        self.sato_sigmas = sato_sigmas

    def correct(self, image: np.ndarray) -> np.ndarray:
        """
        Apply background correction to an image.

        This is the main function for background correction. It removes
        uneven illumination from the image using the specified method.
        Optionally applies Sato vesselness filter to enhance tubular structures.

        Args:
            image: Input grayscale image (uint8 or float32)

        Returns:
            Background-corrected image in the same format as input

        Raises:
            ValueError: If image is invalid
        """
        validate_image(image)
        image_uint8, was_float, original_dtype = normalize_image(image)

        background = self._estimate_background(image_uint8)

        corrected = cv2.subtract(image_uint8, background)

        # Apply Sato filter if weight > 0
        if self.sato_weight > 0:
            sato_filtered = self._apply_sato_filter(corrected)
            # Blend corrected image with Sato result
            corrected = (1 - self.sato_weight) * corrected.astype(
                np.float32
            ) + self.sato_weight * sato_filtered.astype(np.float32)
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        result = denormalize_image(corrected, was_float, original_dtype)

        return result

    def _create_filter_kernel(self, radius: int) -> np.ndarray:
        """
        Create an elliptical structuring element for morphological operations.

        Args:
            radius: Radius of the elliptical kernel in pixels

        Returns:
            Elliptical structuring element as numpy array (uint8)

        Raises:
            ValueError: If radius is not positive
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")

        diameter = 2 * radius + 1

        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))

    def _apply_sato_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Sato vesselness filter to enhance tubular structures.

        The Sato filter enhances tubular/fiber-like structures in the image,
        making nerve fibers more prominent.

        Args:
            image: Background-corrected image (uint8)

        Returns:
            Sato-filtered image normalized to 0-255 range (uint8)
        """
        # Apply Sato filter on float32
        sato_result = sato(
            image.astype(np.float32),
            sigmas=self.sato_sigmas,
            black_ridges=False,  # Detect bright structures (nerve fibers)
            mode="reflect",
        )

        # Normalize to 0-1 range
        sato_min = sato_result.min()
        sato_max = sato_result.max()
        if sato_max - sato_min > 1e-8:
            sato_normalized = (sato_result - sato_min) / (sato_max - sato_min)
        else:
            sato_normalized = np.zeros_like(sato_result)

        return (sato_normalized * 255).astype(np.uint8)

    def _estimate_background(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate background of the image using the specified method.

        Args:
            image: Input grayscale image (uint8)

        Returns:
            Estimated background image (uint8)
        """
        if self.method == "rolling_ball":
            bg_small = rolling_ball(image, radius=20).astype(np.uint8)
            bg_tiny = rolling_ball(image, radius=self.radius).astype(np.uint8)
            background = np.minimum(bg_small, bg_tiny)
        elif self.method == "morphology":
            kernel = self._create_filter_kernel(self.radius)
            background = cv2.morphologyEx(
                image.astype(np.uint8),
                cv2.MORPH_OPEN,
                kernel,
            )
        else:
            raise ValueError(f"Invalid method '{self.method}'")

        return background

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Allow the class instance to be called directly.

        Args:
            image: Input grayscale image

        Returns:
            Background-corrected image
        """
        return self.correct(image)

    def __repr__(self) -> str:
        """String representation of the BackgroundCorrection instance."""
        return (
            f"BackgroundCorrection(method='{self.method}', radius={self.radius}, "
            f"sato_weight={self.sato_weight}, "
            f"sato_sigmas={self.sato_sigmas})"
        )
