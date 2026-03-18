"""Augmentations for segmentation training.

Geometric transforms apply the same random operation to the image and every
mask simultaneously so spatial correspondence is preserved.

Intensity transforms (brightness, contrast, blur, noise) apply to the image
only — masks are left unchanged.

Usage::

    aug = SegmentationAugment()
    x, lbl, skel = aug(x, lbl, skel)   # any number of extra masks
"""

import random

import torch
import torchvision.transforms.functional as TF


class SegmentationAugment:
    """Randomly apply geometric and intensity augmentations.

    Geometric transforms are applied identically to the image and all masks.
    Intensity transforms are applied to the image only.

    Supported transforms (each independently gated by its probability):

    Geometric:
      - Horizontal flip
      - Vertical flip
      - Random 90° rotation (90 / 180 / 270°)

    Intensity (image only):
      - Brightness jitter  — multiplicative scale in [1-delta, 1+delta]
      - Contrast jitter    — blend toward mean intensity
      - Gaussian blur      — kernel size 3 or 5
      - Gaussian noise     — additive zero-mean noise

    Args:
        p_hflip:     Probability of horizontal flip        (default 0.5).
        p_vflip:     Probability of vertical flip          (default 0.5).
        p_rot90:     Probability of 90° rotation           (default 0.5).
        p_brightness: Probability of brightness jitter     (default 0.5).
        brightness_delta: Max brightness change            (default 0.3).
        p_contrast:  Probability of contrast jitter        (default 0.3).
        contrast_delta: Max contrast change                (default 0.3).
        p_blur:      Probability of Gaussian blur          (default 0.3).
        p_noise:     Probability of Gaussian noise         (default 0.3).
        noise_std:   Std of additive noise (image in [0,1]) (default 0.02).
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rot90: float = 0.5,
        p_brightness: float = 0.5,
        brightness_delta: float = 0.3,
        p_contrast: float = 0.3,
        contrast_delta: float = 0.3,
        p_blur: float = 0.3,
        p_noise: float = 0.3,
        noise_std: float = 0.02,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.p_brightness = p_brightness
        self.brightness_delta = brightness_delta
        self.p_contrast = p_contrast
        self.contrast_delta = contrast_delta
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.noise_std = noise_std

    # ------------------------------------------------------------------
    # Geometric (image + all masks)
    # ------------------------------------------------------------------

    def _apply_geometric(
        self,
        image: torch.Tensor,
        masks: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if random.random() < self.p_hflip:
            image = TF.hflip(image)
            masks = tuple(TF.hflip(m.unsqueeze(0)).squeeze(0) for m in masks)

        if random.random() < self.p_vflip:
            image = TF.vflip(image)
            masks = tuple(TF.vflip(m.unsqueeze(0)).squeeze(0) for m in masks)

        if random.random() < self.p_rot90:
            k = random.randint(1, 3)
            image = torch.rot90(image, k, dims=[1, 2])
            masks = tuple(torch.rot90(m, k, dims=[0, 1]) for m in masks)

        return image, masks

    # ------------------------------------------------------------------
    # Intensity (image only)
    # ------------------------------------------------------------------

    def _apply_intensity(self, image: torch.Tensor) -> torch.Tensor:
        # Brightness: multiply by a random scale factor
        if random.random() < self.p_brightness:
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            image = torch.clamp(image * (1.0 + delta), 0.0, 1.0)

        # Contrast: blend toward the mean intensity of the image
        if random.random() < self.p_contrast:
            delta = random.uniform(-self.contrast_delta, self.contrast_delta)
            mean = image.mean()
            image = torch.clamp(image + (image - mean) * delta, 0.0, 1.0)

        # Gaussian blur: randomly choose kernel size 3 or 5
        if random.random() < self.p_blur:
            kernel_size = random.choice([3, 5])
            image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])

        # Gaussian noise: additive zero-mean noise
        if random.random() < self.p_noise:
            noise = torch.randn_like(image) * self.noise_std
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image

    # ------------------------------------------------------------------

    def __call__(
        self,
        image: torch.Tensor,
        *masks: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Apply augmentations.

        Args:
            image:  (C, H, W) float32 tensor in [0, 1] — the input image.
            *masks: One or more (H, W) int64 mask tensors (label, skeleton, …).

        Returns:
            Tuple of (augmented_image, *augmented_masks).
        """
        image, masks = self._apply_geometric(image, masks)
        image = self._apply_intensity(image)
        return (image, *masks)
