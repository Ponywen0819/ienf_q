"""
Shared preprocessing pipeline for neural fiber reconstruction linkers.

Both ``AnnotationGrowLinker`` and ``PureMstLinker`` start from an identical
preprocessing flow. ``PreprocessingPipeline`` encapsulates it so the linkers
share one implementation instead of duplicating it.

Flow:
    1. Extract the green channel (strongest nerve-fiber signal) and squeeze
       any 3-D mask / annotation inputs to 2-D.
    2. Vertically dilate the epidermis mask into an ROI mask.
    3. Build a vesselness-enhanced ROI image
       (background removal → CLAHE → Sato).
    4. Clip the manual annotation to the ROI mask.
    5. Convert the enhanced image into a traversal cost map.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .cost_map import build_cost_map, build_enhanced_image
from .mask_operations import dilate_epidermis_vertically


@dataclass
class PreprocessingResult:
    """
    Output of :meth:`PreprocessingPipeline.run`.

    Attributes:
        roi_mask:       Vertically dilated epidermis ROI mask (H, W), uint8
        roi_image:      Vesselness-enhanced ROI image (H, W), uint8
        roi_annotation: Manual annotation clipped to the ROI (H, W)
        cost_map:       Traversal cost map derived from ``roi_image``
                        (H, W), float32
    """

    roi_mask: np.ndarray
    roi_image: np.ndarray
    roi_annotation: np.ndarray
    cost_map: np.ndarray


class PreprocessingPipeline:
    """
    Shared preprocessing flow for reconstruction linkers.

    Args:
        offset_px:         Pixels to dilate the epidermis mask downward.
        bg_kernel_size:    Morphological opening kernel for background removal.
        clahe_clip:        CLAHE clip limit.
        clahe_grid:        CLAHE tile grid size.
        sato_sigmas_start: Sato vesselness scale range start (inclusive).
        sato_sigmas_stop:  Sato vesselness scale range stop (exclusive).

    Examples:
        >>> pipeline = PreprocessingPipeline(offset_px=50, bg_kernel_size=51)
        >>> result = pipeline.run(image, mask, annotation)
        >>> result.roi_image, result.cost_map
    """

    def __init__(
        self,
        offset_px: int = 50,
        bg_kernel_size: int = 51,
        clahe_clip: float = 20.0,
        clahe_grid: tuple[int, int] = (16, 16),
        sato_sigmas_start: int = 3,
        sato_sigmas_stop: int = 8,
    ):
        self.offset_px = offset_px
        self.bg_kernel_size = bg_kernel_size
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.sato_sigmas_start = sato_sigmas_start
        self.sato_sigmas_stop = sato_sigmas_stop

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        annotation: np.ndarray,
    ) -> PreprocessingResult:
        """
        Run the shared preprocessing flow.

        Args:
            image:      RGB or grayscale original image (H, W, 3) or (H, W).
            mask:       Epidermis mask (H, W) or (H, W, C).
            annotation: Binary manual annotation / weka output
                        (H, W) or (H, W, C).

        Returns:
            PreprocessingResult with roi_mask, roi_image, roi_annotation
            and cost_map.
        """
        # 1. Normalize input dimensions
        if image.ndim == 3:
            green = image[:, :, 1]  # green channel carries the strongest signal
        else:
            green = image
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if annotation.ndim == 3:
            annotation = annotation[:, :, 0]

        # 2. ROI mask via vertical dilation
        roi_mask = dilate_epidermis_vertically(mask, offset_px=self.offset_px)

        # 3. Vesselness-enhanced ROI image
        roi_image = build_enhanced_image(
            green=green,
            roi_mask=roi_mask,
            bg_kernel_size=self.bg_kernel_size,
            clahe_clip=self.clahe_clip,
            clahe_grid=self.clahe_grid,
            sato_sigmas=range(self.sato_sigmas_start, self.sato_sigmas_stop),
        )

        # 4. Clip annotation to ROI
        roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)

        # 5. Traversal cost map
        cost_map = build_cost_map(roi_image)

        return PreprocessingResult(
            roi_mask=roi_mask,
            roi_image=roi_image,
            roi_annotation=roi_annotation,
            cost_map=cost_map,
        )
