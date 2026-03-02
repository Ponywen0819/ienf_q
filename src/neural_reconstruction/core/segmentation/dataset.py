"""Patch-based dataset for UNet training.

Utilities:
  - load_sample        : load (green channel, annotation, label) from a sample dir
  - get_patch_starts   : compute sliding-window start positions
  - extract_patches    : extract all PATCH_SIZE patches from one sample
  - PatchDataset       : PyTorch Dataset with optional augmentation
"""

import random
from pathlib import Path

import cv2
import numpy as np
import skimage.restoration
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def load_sample(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load image, annotation, and label from a sample directory.

    Applies rolling-ball background subtraction (radius=50) to the green
    channel to improve fiber/background contrast before training.

    Args:
        sample_dir: Directory containing image.png, annotation.png, label.png.

    Returns:
        (image, annotation, label) — all uint8 numpy arrays (H, W).
        image      : background-corrected green channel.
        annotation : binary {0, 255}.
        label      : binary {0, 255} ground truth.
    """
    bgr        = cv2.imread(str(sample_dir / "image.png"))
    image      = bgr[:, :, 1]   # green channel (strongest nerve signal)
    background = skimage.restoration.rolling_ball(image, radius=50)
    image      = cv2.subtract(image, background.astype(np.uint8))
    annotation = (cv2.imread(str(sample_dir / "annotation.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
    label      = (cv2.imread(str(sample_dir / "label.png"),      cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
    return image, annotation, label


def get_patch_starts(total: int, patch: int, stride: int) -> list[int]:
    """Compute patch start positions that tile [0, total) completely.

    Args:
        total:  Dimension length (height or width).
        patch:  Patch size.
        stride: Step between patches.

    Returns:
        Sorted list of unique start positions.
    """
    if total <= patch:
        return [0]
    starts = list(range(0, total - patch + 1, stride))
    if starts[-1] + patch < total:
        starts.append(total - patch)
    return sorted(set(starts))


def extract_patches(
    image: np.ndarray,
    annotation: np.ndarray,
    label: np.ndarray,
    patch_size: int = 512,
    stride: int = 480,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract patch_size × patch_size patches with the given stride.

    Edge patches are reflect-padded to exactly patch_size × patch_size.

    Args:
        image:       (H, W) uint8 green channel.
        annotation:  (H, W) uint8 binary annotation.
        label:       (H, W) uint8 binary ground truth.
        patch_size:  Patch size in pixels (default 512).
        stride:      Step between patches (default 480, i.e. 32 px overlap).

    Returns:
        List of (image_patch, annotation_patch, label_patch) tuples.
    """
    h, w = label.shape[:2]
    patches = []

    for y0 in get_patch_starts(h, patch_size, stride):
        for x0 in get_patch_starts(w, patch_size, stride):
            y1 = min(y0 + patch_size, h)
            x1 = min(x0 + patch_size, w)

            lbl_p = label[y0:y1, x0:x1]
            if lbl_p.max() == 0:
                continue   # no label content → skip to avoid training on empty patches

            img_p = image[y0:y1, x0:x1]
            ann_p = annotation[y0:y1, x0:x1]

            ph, pw = img_p.shape[:2]
            if ph < patch_size or pw < patch_size:
                pad_h, pad_w = patch_size - ph, patch_size - pw
                img_p = np.pad(img_p, ((0, pad_h), (0, pad_w)), mode="reflect")
                ann_p = np.pad(ann_p, ((0, pad_h), (0, pad_w)), mode="reflect")
                lbl_p = np.pad(lbl_p, ((0, pad_h), (0, pad_w)), mode="constant")

            patches.append((img_p, ann_p, lbl_p))

    return patches


class PatchDataset(Dataset):
    """Patch-based dataset for UNet training.

    Each item:
      x : torch.Tensor (2, H, W) float32 — [green / 255, annotation / 255]
      y : torch.Tensor (H, W)   int64    — binary label {0=BG, 1=FG}

    Augmentation (training only): random horizontal/vertical flip and 90° rotation.

    Args:
        patches: List of (image_patch, annotation_patch, label_patch) tuples.
        augment: Enable random augmentation (default False).
    """

    def __init__(self, patches: list, augment: bool = False):
        self.patches = patches
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, ann, lbl = self.patches[idx]

        img_t = torch.from_numpy(img[None]).float() / 255.0   # (1, H, W)
        ann_t = torch.from_numpy(ann[None]).float() / 255.0   # (1, H, W)
        lbl_t = (torch.from_numpy(lbl.astype(np.int64)) > 127).long()  # (H, W)

        x = torch.cat([img_t, ann_t], dim=0)   # (2, H, W)

        if self.augment:
            if random.random() > 0.5:
                x     = TF.hflip(x)
                lbl_t = TF.hflip(lbl_t.unsqueeze(0)).squeeze(0)
            if random.random() > 0.5:
                x     = TF.vflip(x)
                lbl_t = TF.vflip(lbl_t.unsqueeze(0)).squeeze(0)
            k = random.randint(0, 3)
            if k > 0:
                x     = torch.rot90(x,     k, dims=[1, 2])
                lbl_t = torch.rot90(lbl_t, k, dims=[0, 1])

        return x, lbl_t
