"""UNet-based binary segmentation for IENF-Q.

Public API
----------
Model
  UNet              - encoder-decoder network (2-ch input, 2-ch output)

Loss
  BCEDiceLoss       - combined BCE + soft-Dice loss

Dataset
  PatchDataset      - PyTorch Dataset of 512×512 patches
  load_sample       - load (green, annotation, label) from a sample directory
  extract_patches   - sliding-window patch extraction
  get_patch_starts  - helper: compute sliding-window start positions

Inference
  predict_full_image  - full-image sliding-window inference with Gaussian blending
  make_gaussian_weight - 2-D Gaussian weight map

Training
  train_epoch  - one training epoch
  val_epoch    - one validation epoch (returns loss + Dice)
  dice_coeff   - batch Dice score
"""

from .model import UNet, DoubleConv, Down, Up
from .loss import SoftDiceLoss, HDLoss, SegmentationLoss
from .dataset import PatchDataset, load_sample, extract_patches, get_patch_starts
from .inference import predict_full_image, make_gaussian_weight
from .trainer import train_epoch, val_epoch, dice_coeff

__all__ = [
    # model
    "UNet", "DoubleConv", "Down", "Up",
    # loss
    "SoftDiceLoss", "HDLoss", "SegmentationLoss",
    # dataset
    "PatchDataset", "load_sample", "extract_patches", "get_patch_starts",
    # inference
    "predict_full_image", "make_gaussian_weight",
    # training
    "train_epoch", "val_epoch", "dice_coeff",
]
