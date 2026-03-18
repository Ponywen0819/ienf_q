"""Training utilities for UNet segmentation.

Provides:
  - train_epoch : run one training epoch, return mean loss
  - val_epoch   : run one validation epoch, return (mean loss, mean Dice)
  - dice_coeff  : batch-level Dice score (foreground class)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def dice_coeff(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Dice score for the foreground class.

    Args:
        logits:  (B, 2, H, W) — raw model output.
        targets: (B, H, W) int64 — binary ground truth {0, 1}.

    Returns:
        Scalar Dice score in [0, 1].
    """
    pred_fg   = (logits.argmax(dim=1) == 1).float()
    target_fg = (targets == 1).float()
    intersection = (pred_fg * target_fg).sum()
    return (2.0 * intersection / (pred_fg.sum() + target_fg.sum() + 1e-8)).item()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Run one training epoch.

    Args:
        model:     UNet in train mode (set internally).
        loader:    Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device:    Torch device string.

    Returns:
        Mean loss per sample.
    """
    model.train()
    total_loss = 0.0
    for x, y, skel in loader:
        x, y, skel = x.to(device), y.to(device), skel.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y, skel)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Run one validation epoch.

    Args:
        model:     UNet (set to eval mode internally).
        loader:    Validation DataLoader.
        criterion: Loss function.
        device:    Torch device string.

    Returns:
        (mean_loss, mean_dice) per sample.
    """
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    with torch.no_grad():
        for x, y, skel in loader:
            x, y, skel = x.to(device), y.to(device), skel.to(device)
            logits = model(x)
            total_loss += criterion(logits, y, skel).item() * x.size(0)
            total_dice += dice_coeff(logits, y) * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_dice / n
