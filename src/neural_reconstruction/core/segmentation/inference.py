"""Full-image inference with Gaussian-weighted sliding window.

Splits a full-resolution image into overlapping patches, runs UNet on each,
and blends the probability maps back using a 2-D Gaussian weight centred on
each patch — this smoothly eliminates seam artefacts at patch boundaries.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .dataset import get_patch_starts
from .model import UNet


def make_gaussian_weight(patch_size: int) -> np.ndarray:
    """2-D Gaussian weight map (peak = 1 at centre, floor = 0.01 at edges).

    Args:
        patch_size: Side length of the square patch.

    Returns:
        (patch_size, patch_size) float32 array.
    """
    w = np.zeros((patch_size, patch_size), dtype=np.float32)
    w[patch_size // 2, patch_size // 2] = 1.0
    w = gaussian_filter(w, sigma=patch_size / 8.0)
    w /= w.max()
    return np.clip(w, 0.01, 1.0)


def predict_full_image(
    model: UNet,
    image: np.ndarray,
    annotation: np.ndarray,
    patch_size: int = 512,
    stride: int = 480,
    device: str = "cpu",
) -> np.ndarray:
    """Sliding-window inference with Gaussian weight blending.

    Args:
        model:       Trained UNet (will be set to eval mode internally).
        image:       (H, W) uint8 — green channel of the microscopy image.
        annotation:  (H, W) uint8 — sparse manual annotation.
        patch_size:  Patch size used during training (default 512).
        stride:      Sliding-window stride (default 480, i.e. 32 px overlap).
        device:      Torch device string (default "cpu").

    Returns:
        (H, W) float32 foreground probability map in [0, 1].
    """
    model.eval()
    h, w = image.shape[:2]
    gaussian = make_gaussian_weight(patch_size)

    pred_sum   = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    with torch.no_grad():
        for y0 in get_patch_starts(h, patch_size, stride):
            for x0 in get_patch_starts(w, patch_size, stride):
                y1 = min(y0 + patch_size, h)
                x1 = min(x0 + patch_size, w)
                ph, pw = y1 - y0, x1 - x0

                img_p = image[y0:y1, x0:x1]
                ann_p = annotation[y0:y1, x0:x1]

                if ph < patch_size or pw < patch_size:
                    img_p = np.pad(img_p, ((0, patch_size - ph), (0, patch_size - pw)), "reflect")
                    ann_p = np.pad(ann_p, ((0, patch_size - ph), (0, patch_size - pw)), "reflect")

                img_t = torch.from_numpy(img_p[None]).float() / 255.0
                ann_t = torch.from_numpy(ann_p[None]).float() / 255.0
                x_in  = torch.cat([img_t, ann_t], dim=0).unsqueeze(0).to(device)

                logits = model(x_in)
                prob   = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

                pred_sum[y0:y0 + ph, x0:x0 + pw]   += prob[:ph, :pw] * gaussian[:ph, :pw]
                weight_sum[y0:y0 + ph, x0:x0 + pw]  += gaussian[:ph, :pw]

    return (pred_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)
