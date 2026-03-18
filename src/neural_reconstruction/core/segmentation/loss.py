"""Loss functions for UNet binary segmentation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import distance_transform_edt

from .topoloss_pytorch import getTopoLoss


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Works with 2-channel logits (BG / FG).  Applies softmax and uses the
    foreground channel probability to compute:

        Dice = 1 - (2 * sum(p * g) + smooth) / (sum(p) + sum(g) + smooth)

    Args:
        smooth: Laplace smoothing to avoid division by zero (default: 1.0).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, 1]  # (B, H, W)
        gt = targets.float()
        intersection = (prob * gt).sum(dim=(1, 2))
        dice = 1.0 - (2.0 * intersection + self.smooth) / (
            prob.sum(dim=(1, 2)) + gt.sum(dim=(1, 2)) + self.smooth
        )
        return dice.mean()


class HDLoss(nn.Module):
    """Hausdorff Distance Loss (Karimi & Salcudean, 2019).

    Uses boundary distance transforms of the binarised prediction and ground
    truth as weighting maps.  The DT is computed as the distance of every pixel
    to the nearest mask boundary (inside + outside), so no skeletonisation is
    needed.  The gradient flows through the soft foreground probability.

    Loss = mean( (prob - gt)^2 * (dt_pred^2 + dt_gt^2) )
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def _boundary_dt(mask: np.ndarray) -> np.ndarray:
        dt_in = distance_transform_edt(mask).astype(np.float32)
        dt_out = distance_transform_edt(~mask).astype(np.float32)
        dt = dt_in + dt_out
        max_val = dt.max()
        if max_val > 0:
            dt = dt / max_val
        return dt

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, 1]
        target_f = targets.float()
        pred_bin = (prob.detach() > self.threshold).cpu().numpy().astype(bool)
        gt_bin = targets.cpu().numpy().astype(bool)
        dt_pred = torch.from_numpy(
            np.stack([self._boundary_dt(p) for p in pred_bin])
        ).to(logits.device)
        dt_gt = torch.from_numpy(np.stack([self._boundary_dt(g) for g in gt_bin])).to(
            logits.device
        )
        return ((prob - target_f) ** 2 * (dt_pred**2 + dt_gt**2)).mean()


class SegmentationLoss(nn.Module):
    """Unified segmentation loss with per-term weight switches.

    Terms are enabled when their weight > 0 and skipped otherwise:
      - BCE   : BCEWithLogitsLoss on fg logit  (pos_weight for class imbalance)
      - Dice  : Soft Dice on softmax fg prob
      - HD    : Boundary distance-transform Hausdorff loss
      - Topo  : Persistent-homology topology loss (GUDHI)

    Args:
        bce_weight   : BCE term weight   (default 0.5, set 0 to disable)
        dice_weight  : Dice term weight  (default 0.5, set 0 to disable)
        hd_weight    : HD term weight    (default 0.0, set >0 to enable)
        topo_weight  : Topo term weight  (default 0.0, set >0 to enable)
        pos_weight   : Foreground class weight for BCE (default 1.0)
        hd_threshold : Binarisation threshold for HD loss (default 0.5)
        topo_size    : Patch size inside getTopoLoss (default 100)
        topo_workers : Parallel workers for topo loss (default 4)
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        hd_weight: float = 0.0,
        topo_weight: float = 0.0,
        pos_weight: float = 1.0,
        hd_threshold: float = 0.5,
        topo_size: int = 100,
        topo_workers: int = 24,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.hd_weight = hd_weight
        self.topo_weight = topo_weight
        self.topo_size = topo_size
        self.topo_workers = topo_workers

        if bce_weight > 0:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        if dice_weight > 0:
            self.dice = SoftDiceLoss()
        if hd_weight > 0:
            self.hd = HDLoss(threshold=hd_threshold)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        skeleton: torch.Tensor
        | None = None,  # kept for trainer interface compatibility
    ) -> torch.Tensor:
        """Args:
        logits:  (B, 2, H, W) — raw model output.
        targets: (B, H, W) int64 — binary ground truth {0, 1}.
        """
        loss = logits.new_zeros(1).squeeze()

        if self.bce_weight > 0:
            assert isinstance(self.pos_weight, torch.Tensor)
            bce = F.binary_cross_entropy_with_logits(
                logits[:, 1], targets.float(), pos_weight=self.pos_weight
            )
            loss = loss + self.bce_weight * bce

        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)

        if self.hd_weight > 0:
            loss = loss + self.hd_weight * self.hd(logits, targets)

        if self.topo_weight > 0:
            B = logits.shape[0]
            fg_prob = torch.softmax(logits, dim=1)[:, 1]
            target_f = targets.float()
            with ThreadPoolExecutor(max_workers=min(B, self.topo_workers)) as pool:
                futures = [
                    pool.submit(getTopoLoss, fg_prob[i], target_f[i], self.topo_size)
                    for i in range(B)
                ]
                topo = sum(f.result() for f in futures) / B
            loss = loss + self.topo_weight * topo

        return loss
