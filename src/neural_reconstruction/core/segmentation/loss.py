"""Loss functions for UNet binary segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss for binary segmentation.

    Works with 2-channel UNet output (BG / FG logits).
    Extracts the foreground channel logit and computes:
      - BCEWithLogitsLoss  (pos_weight handles class imbalance)
      - Soft Dice loss     (on sigmoid probabilities)

    Args:
        bce_weight:  Contribution of BCE term  (default 0.5).
        dice_weight: Contribution of Dice term (default 0.5).
        pos_weight:  Weight on positive (foreground) class in BCE.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits:  (B, 2, H, W) — raw model output.
            targets: (B, H, W) int64 — binary ground truth {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        fg_logit = logits[:, 1, :, :]   # foreground channel  (B, H, W)
        target_f = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            fg_logit, target_f,
            pos_weight=self.pos_weight.to(logits.device),
        )

        prob         = torch.sigmoid(fg_logit)
        intersection = (prob * target_f).sum(dim=(1, 2))
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            prob.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2)) + 1.0
        )
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice
