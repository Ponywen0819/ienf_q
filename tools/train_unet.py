"""UNet training script for IENF-Q nerve-fiber segmentation.

Scans DATA_DIR for samples with image.png / annotation.png / label.png,
performs a sample-level train/val split, extracts 512×512 patches, and
trains a UNet with combined BCE+Dice loss.

Usage
-----
    uv run python tools/train_unet.py
    uv run python tools/train_unet.py --data-dir data --output-dir output/unet --epochs 200
    uv run python tools/train_unet.py --base-channels 64 --batch-size 2 --lr 3e-4

    # Resume from a previous run (adds --epochs more epochs on top)
    uv run python tools/train_unet.py --resume output/unet/unet_best.pth --epochs 100

Key options
-----------
    --data-dir        Root directory containing per-sample sub-directories (default: data)
    --output-dir      Where to save checkpoints and curves (default: output/unet)
    --epochs          Total training epochs (default: 200)
    --lr              Initial learning rate (default: 1e-4)
    --batch-size      Training batch size (default: 1)
    --patch-size      Patch size in pixels (default: 512)
    --overlap         Overlap between patches in pixels (default: 64)
    --train-ratio     Fraction of samples for training (default: 0.8)
    --base-channels   Feature-map width at first encoder level (default: 32)
    --seed            Random seed (default: 42)
    --device          'cuda', 'cpu', or 'auto' (default: auto)
    --resume          Path to a checkpoint .pth to resume from (optional)
    --load-workers    Threads for parallel sample loading (default: 4)
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from neural_reconstruction.core.segmentation import (
    SegmentationLoss,
    PatchDataset,
    UNet,
    extract_patches,
    load_sample,
    train_epoch,
    val_epoch,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train UNet for binary nerve-fiber segmentation."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/unet"),
        help="Output directory for checkpoints and plots (default: output/unet)",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Feature-map width at first encoder level (default: 32)",
    )
    p.add_argument(
        "--loss",
        choices=["bce_dice", "hd", "bce_dice_hd", "bce_dice_topo"],
        default="bce_dice_topo",
        help="Loss function (default: bce_dice_topo)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'cuda', 'cpu', or 'auto' (default: auto)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint .pth to resume training from",
    )
    p.add_argument(
        "--load-workers",
        type=int,
        default=4,
        help="Threads for parallel sample loading / rolling-ball (default: 4)",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def find_valid_samples(data_dir: Path) -> list[Path]:
    required = ["image.png", "annotation.png", "label.png"]
    return [
        d
        for d in sorted(data_dir.iterdir())
        if d.is_dir() and all((d / f).exists() for f in required)
    ]


def compute_class_weights(patches: list) -> tuple[int, int]:
    """Return (bg_pixels, fg_pixels) from a list of (img, ann, lbl) patches."""
    total_px = sum(p[2].size for p in patches)
    fg_px = sum(int((p[2] > 127).sum()) for p in patches)
    return total_px - fg_px, fg_px


def _load_one(sample_dir: Path, patch_size: int, stride: int) -> tuple[Path, list]:
    """Load one sample and extract patches (runs in a thread)."""
    patches = extract_patches(
        *load_sample(sample_dir), patch_size=patch_size, stride=stride
    )
    return sample_dir, patches


def extract_from_dirs(
    sample_dirs: list[Path],
    patch_size: int,
    stride: int,
    tag: str,
    num_workers: int = 4,
) -> list:
    results: dict[Path, list] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_load_one, d, patch_size, stride): d for d in sample_dirs
        }
        for fut in as_completed(futures):
            sample_dir, patches = fut.result()
            results[sample_dir] = patches
            print(f"  [{tag}] {sample_dir.name}: {len(patches)} patches")

    # Return in original order so dataset ordering is deterministic
    all_patches = []
    for d in sample_dirs:
        all_patches.extend(results[d])
    return all_patches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    stride = args.patch_size - args.overlap

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover samples ───────────────────────────────────────────────────
    valid_samples = find_valid_samples(args.data_dir)
    if not valid_samples:
        raise RuntimeError(f"No valid samples found in {args.data_dir}")
    print(f"Found {len(valid_samples)} valid samples in {args.data_dir}")

    # ── Sample-level split ─────────────────────────────────────────────────
    shuffled = valid_samples.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * args.train_ratio)
    train_sample_dirs = shuffled[:n_train]
    val_sample_dirs = shuffled[n_train:]
    print(f"Train ({len(train_sample_dirs)}): {[s.name for s in train_sample_dirs]}")
    print(f"Val   ({len(val_sample_dirs)}):   {[s.name for s in val_sample_dirs]}")

    # ── Patch extraction ───────────────────────────────────────────────────
    print(f"\nExtracting patches (load_workers={args.load_workers})...")
    train_patches = extract_from_dirs(
        train_sample_dirs, args.patch_size, stride, "train", args.load_workers
    )
    val_patches = extract_from_dirs(
        val_sample_dirs, args.patch_size, stride, "val", args.load_workers
    )
    print(f"\nTotal  train: {len(train_patches)}  |  val: {len(val_patches)}")

    # ── Class imbalance ────────────────────────────────────────────────────
    bg_px, fg_px = compute_class_weights(train_patches)
    total_px = bg_px + fg_px
    print(
        f"Class distribution  BG: {bg_px / total_px:.1%}  |  FG: {fg_px / total_px:.1%}"
    )
    pos_weight = min(bg_px / max(fg_px, 1), 20.0)  # cap at 20× for stability

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_ds = PatchDataset(train_patches, augment=True)
    val_ds = PatchDataset(val_patches, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=(device == "cuda"),
    )

    # ── Model, loss, optimiser ─────────────────────────────────────────────
    model = UNet(in_channels=1, out_channels=2, base_channels=args.base_channels).to(
        device
    )
    loss_kwargs = dict(pos_weight=pos_weight)
    if args.loss == "bce_dice":
        loss_kwargs.update(bce_weight=0.5, dice_weight=0.5)
    elif args.loss == "hd":
        loss_kwargs.update(bce_weight=0.0, dice_weight=0.0, hd_weight=1.0)
    elif args.loss == "bce_dice_hd":
        loss_kwargs.update(bce_weight=0.4, dice_weight=0.4, hd_weight=0.2)
    else:  # bce_dice_topo
        loss_kwargs.update(bce_weight=0.4, dice_weight=0.4, topo_weight=0.2)

    criterion = SegmentationLoss(**loss_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nUNet  base_channels={args.base_channels}  "
        f"trainable params: {n_params / 2**20:.2f} M"
    )
    print(f"Device: {device}  |  pos_weight: {pos_weight:.1f}")

    # ── Resume from checkpoint (optional) ────────────────────────────────
    checkpoint_path = args.output_dir / "unet_best.pth"
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_loss = np.inf
    start_epoch = 0  # 0-based; training runs [start_epoch+1 .. start_epoch+epochs]

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            best_loss = ckpt.get("best_dice", 0.0)
            history = ckpt.get("history", history)
            print(
                f"Resumed from {args.resume}  "
                f"(epoch {start_epoch}, best Dice {best_loss:.4f})"
            )
        else:
            # Legacy checkpoint: plain state_dict
            model.load_state_dict(ckpt)
            print(
                f"Loaded weights from {args.resume}  (legacy format, optimizer not restored)"
            )

    print(
        f"\nTraining {args.epochs} epochs on {device}  "
        f"(epochs {start_epoch + 1}–{start_epoch + args.epochs})"
    )
    print("-" * 65)

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_dice = val_epoch(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_dice"].append(v_dice)

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_dice": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history,
                },
                checkpoint_path,
            )
            marker = " ← best"
        else:
            marker = ""

        total_epochs = start_epoch + args.epochs
        if epoch % 10 == 0 or epoch == start_epoch + 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"Train {t_loss:.4f} | Val {v_loss:.4f} | "
                f"Dice {v_dice:.4f} | LR {lr:.2e}{marker}"
            )

    print(f"\nBest Val Dice : {best_loss:.4f}")
    print(f"Checkpoint    : {checkpoint_path}")

    # ── Save training history ──────────────────────────────────────────────
    history_path = args.output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History       : {history_path}")

    # ── Per-sample evaluation with best checkpoint ─────────────────────────
    from neural_reconstruction.core.segmentation import predict_full_image

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt)
    model.eval()

    def dice_np(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
        intersection = (pred_bin & gt_bin).sum()
        total = pred_bin.sum() + gt_bin.sum()
        return float(2 * intersection / max(total, 1))

    print("\nPer-sample evaluation:")
    print(f"  {'Sample':<20}  {'Split':>5}  {'Dice':>6}")
    print("  " + "-" * 36)

    all_dice, train_dice_vals, val_dice_vals = [], [], []
    for sample_dir in valid_samples:
        img, _, lbl = load_sample(sample_dir)
        prob_map = predict_full_image(
            model, img, patch_size=args.patch_size, stride=stride, device=device
        )
        pred_bin = prob_map > 0.5
        target_bin = lbl > 127
        d = dice_np(pred_bin, target_bin)
        split = "train" if sample_dir in train_sample_dirs else "val"

        all_dice.append(d)
        (train_dice_vals if split == "train" else val_dice_vals).append(d)
        print(f"  {sample_dir.name:<20}  {split:>5}  {d:.4f}")

    print(f"\n  Mean Dice (all)   : {np.mean(all_dice):.4f}")
    print(f"  Mean Dice (train) : {np.mean(train_dice_vals):.4f}")
    print(f"  Mean Dice (val)   : {np.mean(val_dice_vals):.4f}")


if __name__ == "__main__":
    main()
