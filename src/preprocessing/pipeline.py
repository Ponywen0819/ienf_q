"""
Skin analysis pipeline for neural fiber reconstruction.

This module provides a complete pipeline for processing skin images with
epidermis masks and labels to generate high-quality neural fiber reconstructions.
"""

from typing import Tuple, Dict, Any
import numpy as np

from .morphology import morphological_closing, morphological_opening
from .background_correction import rolling_ball_background
from .thresholding import otsu_threshold
from .mask_operations import (
    dilate_epidermis_vertically,
    apply_mask,
    invert_mask,
    combine_masks_or
)


class SkinAnalysisPipeline:
    """
    Pipeline for processing skin images with epidermis masks and labels.

    This pipeline performs three parallel processing paths:
    1. Label path: Morphological operations on input labels
    2. Mask path: Vertical dilation of epidermis mask
    3. Original image path: Background correction and pseudo-label generation

    The results are combined to produce a final label and ROI image.

    Example:
        >>> config = {
        ...     'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
        ...     'mask': {'dilate_offset': 50},
        ...     'background': {'radius': 12, 'light_background': True},
        ...     'threshold': {'method': 'binary'}
        ... }
        >>> pipeline = SkinAnalysisPipeline(config)
        >>> final_label, roi_image = pipeline.run(
        ...     label_image, epidermis_mask, original_image
        ... )
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the skin analysis pipeline with configuration.

        Args:
            config: Configuration dictionary with the following structure:
                {
                    'morphology': {
                        'closing_kernel': int,  # Kernel size for closing (default: 3)
                        'opening_kernel': int   # Kernel size for opening (default: 3)
                    },
                    'mask': {
                        'dilate_offset': int    # Vertical dilation offset in pixels (default: 50)
                    },
                    'background': {
                        'radius': int,          # Rolling ball radius (default: 12)
                        'light_background': bool  # Whether background is light (default: True)
                    },
                    'threshold': {
                        'method': str           # Threshold type: 'binary' or 'binary_inv' (default: 'binary')
                    }
                }

        Raises:
            ValueError: If required configuration keys are missing
        """
        self.config = config
        self._validate_config()

        # Extract configuration parameters with defaults
        self.closing_kernel = config.get('morphology', {}).get('closing_kernel', 3)
        self.opening_kernel = config.get('morphology', {}).get('opening_kernel', 3)
        self.dilate_offset = config.get('mask', {}).get('dilate_offset', 50)
        self.bg_radius = config.get('background', {}).get('radius', 12)
        self.light_background = config.get('background', {}).get('light_background', False)
        self.threshold_method = config.get('threshold', {}).get('method', 'binary')

    def _validate_config(self) -> None:
        """
        Validate that the configuration has required structure.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        # Check for required top-level keys
        required_keys = ['morphology', 'mask', 'background', 'threshold']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: '{key}'")

    def run(
        self,
        label_image: np.ndarray,
        epidermis_mask: np.ndarray,
        original_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute the complete skin analysis pipeline.

        This method processes the input images through three paths:
        1. Process the label image with morphological operations
        2. Create a dilated epidermis mask
        3. Process the original image to extract ROI and generate pseudo-labels

        The processed label and pseudo-label are then merged using OR operation.

        Args:
            label_image: Binary label image of neural fibers (2D array, uint8)
            epidermis_mask: Binary mask of epidermis region (2D array, uint8)
            original_image: Original grayscale image (2D array, uint8 or float)

        Returns:
            Tuple of (final_label, roi_image):
                - final_label: Combined label image (uint8, 0 or 255)
                - roi_image: Original image masked to epidermis region

        Raises:
            ValueError: If input images have incompatible shapes

        Example:
            >>> pipeline = SkinAnalysisPipeline(config)
            >>> final_label, roi = pipeline.run(labels, mask, image)
            >>> print(f"Final label shape: {final_label.shape}")
            >>> print(f"ROI image shape: {roi.shape}")
        """
        # Validate input shapes
        if label_image.shape != epidermis_mask.shape:
            raise ValueError(
                f"Label image shape {label_image.shape} must match "
                f"mask shape {epidermis_mask.shape}"
            )

        if original_image.shape[:2] != epidermis_mask.shape:
            raise ValueError(
                f"Original image spatial dimensions {original_image.shape[:2]} "
                f"must match mask shape {epidermis_mask.shape}"
            )

        # Step 1: Process label path (can be parallelized)
        processed_label = self._process_label_path(label_image)

        # Step 2: Create dilated mask (can be parallelized with step 1)
        dilated_mask = self._create_dilated_mask(epidermis_mask)

        # Step 3: Process original image (depends on dilated_mask)
        roi_image, pseudo_label = self._process_original_with_masks(
            original_image,
            epidermis_mask,
            dilated_mask
        )

        # Step 4: Merge labels using OR operation
        final_label = self._merge_labels(processed_label, pseudo_label)

        return final_label, roi_image

    def _process_label_path(self, label_image: np.ndarray) -> np.ndarray:
        """
        Process label image with morphological operations.

        Applies closing followed by opening to remove noise and fill small gaps.

        Args:
            label_image: Input binary label image

        Returns:
            Processed label image
        """
        # Apply closing operation to fill small gaps
        closed = morphological_closing(
            label_image,
            kernel_size=self.closing_kernel,
            kernel_shape='ellipse',
            iterations=1
        )

        # Apply opening operation to remove small noise
        opened = morphological_opening(
            closed,
            kernel_size=self.opening_kernel,
            kernel_shape='ellipse',
            iterations=1
        )

        return opened

    def _create_dilated_mask(self, epidermis_mask: np.ndarray) -> np.ndarray:
        """
        Create vertically dilated epidermis mask.

        Dilates the mask downward to capture boundary-crossing fibers.

        Args:
            epidermis_mask: Input epidermis mask

        Returns:
            Dilated mask
        """
        dilated = dilate_epidermis_vertically(
            epidermis_mask,
            offset_px=self.dilate_offset
        )

        return dilated

    def _process_original_with_masks(
        self,
        original_image: np.ndarray,
        epidermis_mask: np.ndarray,
        dilated_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process original image to generate ROI and pseudo-label.

        This method:
        1. Applies rolling ball background correction
        2. Creates ROI by masking with epidermis mask
        3. Generates pseudo-label through selective masking and thresholding

        Args:
            original_image: Original grayscale image
            epidermis_mask: Epidermis region mask
            dilated_mask: Vertically dilated epidermis mask

        Returns:
            Tuple of (roi_image, pseudo_label)
        """
        # Step 1: Apply rolling ball background correction
        corrected = rolling_ball_background(
            original_image,
            radius=self.bg_radius,
            light_background=self.light_background,
            smoothing=False
        )

        # Branch A: Create ROI image using dilated mask
        roi_image = apply_mask(corrected, dilated_mask)

        # Branch B: Generate pseudo-label for boundary-crossing region
        # 1. Calculate new region = dilated_mask AND (NOT epidermis_mask)
        #    This gives us the extended region (dermis boundary area)
        inverted_epidermis = invert_mask(epidermis_mask)
        new_region_mask = apply_mask(dilated_mask, inverted_epidermis)

        # 2. Apply new region mask to corrected image
        masked_region = apply_mask(corrected, new_region_mask)

        # 3. Apply Otsu thresholding to generate pseudo-label
        pseudo_label = otsu_threshold(
            masked_region,
            threshold_type=self.threshold_method
        )

        return roi_image, pseudo_label

    def _merge_labels(
        self,
        label1: np.ndarray,
        label2: np.ndarray
    ) -> np.ndarray:
        """
        Merge two binary labels using OR operation.

        Args:
            label1: First label (processed original label)
            label2: Second label (pseudo-label)

        Returns:
            Combined label
        """
        final_label = combine_masks_or(label1, label2)

        return final_label


