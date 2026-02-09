"""
Skin analysis pipeline for neural fiber reconstruction.

This module provides a complete pipeline for processing skin images with
epidermis masks and labels to generate high-quality neural fiber reconstructions.
"""

from typing import Tuple
import numpy as np

from .config import PipelineConfig
from .background_correction import BackgroundCorrection
from .morphology import morphological_closing, morphological_opening
from .thresholding import multi_otsu_threshold
from .mask_operations import (
    dilate_epidermis_vertically,
    apply_mask,
    invert_mask,
    combine_masks_or,
)
from .utils import regional_minmax_normalize


class SkinAnalysisPipeline:
    """
    Pipeline for processing skin images with epidermis masks and labels.

    This pipeline performs three parallel processing paths:
    1. Label path: Morphological operations on input labels
    2. Mask path: Vertical dilation of epidermis mask
    3. Original image path: Background correction and pseudo-label generation

    The results are combined to produce a final label and ROI image.

    Args:
        config: Either a PipelineConfig instance or a dictionary for backward compatibility
    """

    def __init__(self, config: PipelineConfig | dict | None = None):
        """
        Initialize the skin analysis pipeline with configuration.

        Args:
            config: PipelineConfig instance or dict for backward compatibility
        Raises:
            ValueError: If required configuration keys are missing
        """

        if config is None:
            self.config = PipelineConfig()
        elif isinstance(config, dict):
            # Backward compatibility: convert dict to PipelineConfig
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config

        self.background_corrector = BackgroundCorrection(
            method=self.config.background.method,
            radius=self.config.background.radius,
            smoothing=False,
            smoothing_sigma=self.config.background.sigma,
            sato_weight=self.config.background.sato_weight,
            sato_sigmas=self.config.background.sato_sigmas,
        )

    def run(
        self,
        label_image: np.ndarray,
        epidermis_mask: np.ndarray,
        original_image: np.ndarray,
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
            debug: 如果為 True，額外返回包含各階段中間輸出的 DebugOutput

        Returns:
            如果 debug=False:
                Tuple of (final_label, roi_image):
                    - final_label: Combined label image (uint8, 0 or 255)
                    - roi_image: Original image masked to epidermis region
            如果 debug=True:
                Tuple of (final_label, roi_image, debug_output):
                    - final_label: Combined label image (uint8, 0 or 255)
                    - roi_image: Original image masked to epidermis region
                    - debug_output: DebugOutput 包含各階段的中間輸出

        Raises:
            ValueError: If input images have incompatible shapes

        Example:
            >>> pipeline = SkinAnalysisPipeline(config)
            >>> final_label, roi = pipeline.run(labels, mask, image)
            >>> print(f"Final label shape: {final_label.shape}")
            >>> print(f"ROI image shape: {roi.shape}")

            # 使用 debug 模式
            >>> final_label, roi, dbg = pipeline.run(labels, mask, image, debug=True)
            >>> print(f"Background corrected shape: {dbg.background_corrected.shape}")
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
            original_image, epidermis_mask, dilated_mask
        )

        # Step 4: Merge labels using OR operation
        if self.config.threshold.use_full_roi:
            final_label = pseudo_label
        else:
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
        if self.config.morphology.closing_kernel > 0:
            closed = morphological_closing(
                label_image,
                kernel_size=self.config.morphology.closing_kernel,
                kernel_shape="ellipse",
                iterations=1,
            )
        else:
            closed = label_image.copy()

        # Apply opening operation to remove small noise
        if self.config.morphology.opening_kernel > 0:
            opened = morphological_opening(
                closed,
                kernel_size=self.config.morphology.opening_kernel,
                kernel_shape="ellipse",
                iterations=1,
            )
        else:
            opened = closed.copy()

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
            epidermis_mask, offset_px=self.config.mask.dilate_offset
        )

        return dilated

    def _process_original_with_masks(
        self,
        original_image: np.ndarray,
        epidermis_mask: np.ndarray,
        dilated_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process original image to generate ROI and pseudo-label.

        This method:
        1. Applies background correction (rolling ball or gaussian)
        2. Creates ROI by masking with epidermis mask
        3. Generates pseudo-label through selective masking and thresholding

        Args:
            original_image: Original grayscale image
            epidermis_mask: Epidermis region mask
            dilated_mask: Vertically dilated epidermis mask

        Returns:
            Tuple of (roi_image, pseudo_label)
        """
        # Step 1: Apply background correction
        should_correct = False
        if (
            self.config.background.method in ["morphology", "rolling_ball"]
            and self.config.background.radius > 0
        ):
            should_correct = True
        elif (
            self.config.background.method == "gaussian"
            and self.config.background.sigma > 0
        ):
            should_correct = True

        if should_correct:
            corrected = self.background_corrector(original_image)
        else:
            corrected = original_image.copy()

        # Step 2: Calculate dermis ROI mask (moved before normalization)
        # dermis_roi = dilated_mask AND (NOT epidermis_mask)
        inverted_epidermis = invert_mask(epidermis_mask)
        dermis_roi_mask = apply_mask(dilated_mask, inverted_epidermis)

        # Step 3: Apply regional normalization (optional)
        if self.config.normalization.enabled:
            corrected = regional_minmax_normalize(
                corrected, epidermis_mask=epidermis_mask, dermis_mask=dermis_roi_mask
            )

        # Step 4: Create ROI image using dilated mask
        roi_image = apply_mask(corrected, dilated_mask)

        # Step 5 & 6: Generate pseudo-label
        if self.config.threshold.use_full_roi:
            # 使用整個 ROI image 進行 threshold
            pseudo_label = multi_otsu_threshold(roi_image)
        else:
            # 原本做法：只對 masked region (dermis_roi_mask) 進行 threshold
            masked_region = apply_mask(corrected, dermis_roi_mask)
            pseudo_label = multi_otsu_threshold(masked_region)
        return roi_image, pseudo_label

    def _merge_labels(self, label1: np.ndarray, label2: np.ndarray) -> np.ndarray:
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
