"""
Cost map / image-enhancement construction (shared, canonical location).

This is the canonical home of ``build_enhanced_image`` / ``build_cost_map`` so
they can be shared by all reconstruction algorithms without ``core`` having to
depend on ``algorithms``. ``algorithms/annotation_grow/cost_map.py`` keeps a
standalone copy for backward compatibility; if you change the algorithm here,
keep that copy in sync.

Pipeline:
  green channel → background removal → CLAHE → Sato vesselness
  → normalize → invert (bright = low cost) → exp scaling

Global filters (morphology, Sato) are run on vertical strips cropped to each
strip's ROI y-extent, so the U-shaped epidermis mask doesn't pay for huge empty
regions. Outputs are byte-identical inside the ROI when padding ≥ each
operator's halo radius (morphology: kernel/2; Sato: 4·sigma_max).
"""

from typing import Callable, Iterable

import cv2
import numpy as np
import skimage as ski


DEFAULT_STRIP_WIDTH = 256


def apply_within_mask_strips(
    img: np.ndarray,
    roi_mask: np.ndarray,
    op: Callable[[np.ndarray], np.ndarray],
    pad: int,
    strip_w: int = DEFAULT_STRIP_WIDTH,
) -> np.ndarray:
    """
    Run ``op`` only over regions of ``img`` that intersect ``roi_mask``,
    processed as vertical strips. Each strip is cropped to its mask y-extent
    plus ``pad`` pixels of halo (horizontal and vertical) so a translation-
    invariant ``op`` produces identical inside-ROI output to a full-image call.

    Outside-ROI pixels in the returned array stay zero. Strips that touch no
    mask pixels are skipped entirely.

    Args:
        img:      Full-size input (H, W).
        roi_mask: ROI mask (H, W); non-zero pixels are inside ROI.
        op:       Translation-invariant operator: patch → same-shape patch.
        pad:      Halo radius; must be ≥ ``op``'s boundary influence in pixels.
        strip_w:  Strip width before halo padding.

    Returns:
        Full-size array, dtype of ``op``'s output (or ``img.dtype`` if no
        strip ran).
    """
    H, W = roi_mask.shape
    bin_mask = roi_mask > 0
    out: np.ndarray | None = None

    x = 0
    while x < W:
        x1 = min(x + strip_w, W)
        gx0 = max(0, x - pad)
        gx1 = min(W, x1 + pad)

        ys = np.where(bin_mask[:, gx0:gx1].any(axis=1))[0]
        if ys.size == 0:
            x = x1
            continue

        ymin, ymax = int(ys.min()), int(ys.max())
        gy0 = max(0, ymin - pad)
        gy1 = min(H, ymax + 1 + pad)

        out_patch = op(img[gy0:gy1, gx0:gx1])

        if out is None:
            out = np.zeros((H, W), dtype=out_patch.dtype)

        # Paste only the strip's "core" columns [x:x1] — neighbouring strips
        # cover the horizontal halo, and vertical halo lies outside the mask
        # so it's safe to write back as-is for the core columns.
        px0 = x - gx0
        out[gy0:gy1, x:x1] = out_patch[:, px0 : px0 + (x1 - x)]

        x = x1

    if out is None:
        out = np.zeros((H, W), dtype=img.dtype)
    return out


def _morph_open_subtract(green: np.ndarray, bg_kernel_size: int) -> np.ndarray:
    """Background removal via morphological opening + subtraction."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size)
    )
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(green, background)


def build_enhanced_image(
    green: np.ndarray,
    roi_mask: np.ndarray,
    bg_kernel_size: int = 51,
    clahe_clip: float = 20.0,
    clahe_grid: tuple[int, int] = (16, 16),
    sato_sigmas: Iterable[int] = range(3, 8),
    strip_w: int = DEFAULT_STRIP_WIDTH,
) -> np.ndarray:
    """
    Preprocess green channel into a vesselness-enhanced image.

    Args:
        green:          Green channel (H, W), uint8
        roi_mask:       ROI binary mask (H, W), uint8
        bg_kernel_size: Morphological opening kernel diameter for background removal
        clahe_clip:     CLAHE clip limit
        clahe_grid:     CLAHE tile grid size
        sato_sigmas:    Scale range for Sato vesselness filter
        strip_w:        Vertical-strip width for masked Sato/morphology

    Returns:
        roi_image: Enhanced uint8 image (H, W), masked to ROI
    """
    sigmas = list(sato_sigmas)

    if bg_kernel_size > 0:
        # Opening = erosion → dilation, so the halo is 2× the kernel radius.
        bg_pad = bg_kernel_size
        corrected = apply_within_mask_strips(
            green,
            roi_mask,
            lambda patch: _morph_open_subtract(patch, bg_kernel_size),
            pad=bg_pad,
            strip_w=strip_w,
        )
    else:
        corrected = green
    roi_image = cv2.bitwise_and(corrected, corrected, mask=roi_mask)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    roi_image = clahe.apply(roi_image)

    # skimage gaussian uses truncate=4; Sato chains Hessian via finite
    # differences, so add a few px of safety beyond 4·σ.
    sato_pad = int(np.ceil(4 * max(sigmas))) + 4 if sigmas else 0
    roi_image = apply_within_mask_strips(
        roi_image,
        roi_mask,
        lambda patch: ski.filters.sato(patch, sigmas=sigmas, black_ridges=False),
        pad=sato_pad,
        strip_w=strip_w,
    )

    vmin, vmax = roi_image.min(), roi_image.max()
    if vmax > vmin:
        roi_image = (roi_image - vmin) / (vmax - vmin) * 255
    return roi_image.astype(np.uint8)


def build_cost_map(enhanced: np.ndarray) -> np.ndarray:
    """
    Convert enhanced image to traversal cost map.

    Bright pixels (nerve tissue) → low cost.
    Dark pixels (background)     → high cost.

    cost = exp(1 - normalized_intensity) - 1

    Args:
        enhanced: uint8 enhanced image (H, W)

    Returns:
        cost_map: float32 array (H, W), range [0, e-1]
    """
    norm = enhanced.astype(np.float32) / 255.0
    cost = np.exp(1.0 - norm) - 1.0
    return cost.astype(np.float32)
