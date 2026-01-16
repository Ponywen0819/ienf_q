"""
Background correction for image preprocessing.

This module provides background correction methods including rolling ball
algorithm for removing uneven illumination.
"""

from typing import Literal
import numpy as np
import cv2
from skimage.restoration import rolling_ball

from .utils import (
    validate_image,
    normalize_image,
    denormalize_image,
)


class BackgroundCorrection:
    """
    Background correction class for removing uneven illumination from images.

    This class provides methods for background correction using rolling ball
    and morphology techniques.

    Attributes:
        method: Background correction method ('rolling_ball' or 'morphology')
        radius: Radius for rolling ball/morphology operations
        smoothing: Whether to apply Gaussian smoothing to background estimate
        smoothing_sigma: Sigma for smoothing Gaussian filter

    Args:
        method: Background correction method ('rolling_ball' or 'morphology'). Default: 'morphology'
        radius: Radius for rolling ball/morphology operations. Default: 50
        smoothing: Whether to apply Gaussian smoothing to background estimate. Default: False
        smoothing_sigma: Sigma for smoothing Gaussian filter. Default: 2.0

    Example:
        >>> corrector = BackgroundCorrection(method='rolling_ball', radius=50)
        >>> corrected_image = corrector.correct(image)
    """

    def __init__(
        self,
        method: Literal["rolling_ball", "morphology"] = "morphology",
        radius: int = 50,
        smoothing: bool = False,
        smoothing_sigma: float = 2.0,
    ):
        """
        Initialize BackgroundCorrection.

        Args:
            method: Background correction method ('rolling_ball' or 'morphology')
            radius: Radius for rolling ball/morphology operations
            light_background: True if background is brighter than foreground
            smoothing: Whether to apply Gaussian smoothing to background estimate
            smoothing_sigma: Sigma for smoothing Gaussian filter

        Raises:
            ValueError: If method is not 'rolling_ball' or 'morphology'
            ValueError: If radius is not positive
        """
        if method not in ["rolling_ball", "morphology"]:
            raise ValueError(
                f"Invalid method '{method}'. Must be 'rolling_ball' or 'morphology'"
            )
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        if smoothing and smoothing_sigma <= 0:
            raise ValueError(f"Smoothing sigma must be positive, got {smoothing_sigma}")

        self.method = method
        self.radius = radius
        self.smoothing = smoothing
        self.smoothing_sigma = smoothing_sigma

    def correct(self, image: np.ndarray) -> np.ndarray:
        """
        Apply background correction to an image.

        This is the main function for background correction. It removes
        uneven illumination from the image using the specified method.

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

    def _estimate_background(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate background of the image using the specified method.

        Args:
            image: Input grayscale image (uint8)

        Returns:
            Estimated background image (uint8)
        """
        if self.method == "rolling_ball":
            background = rolling_ball(image, radius=self.radius).astype(np.uint8)
        elif self.method == "morphology":
            kernel = self._create_filter_kernel(self.radius)
            background = cv2.morphologyEx(
                image.astype(np.uint8),
                cv2.MORPH_OPEN,
                kernel,
            )
        else:
            raise ValueError(f"Invalid method '{self.method}'")

        if self.smoothing:
            background = cv2.GaussianBlur(
                background.astype(np.float32), (0, 0), sigmaX=self.smoothing_sigma
            )
            background = background.astype(np.uint8)

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
            f"smoothing={self.smoothing}, "
            f"smoothing_sigma={self.smoothing_sigma})"
        )
