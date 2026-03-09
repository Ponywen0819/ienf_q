"""UNet full-image inference tool.

Scans a data directory for samples (requires image.png + annotation.png),
runs sliding-window inference with the trained UNet, thresholds at 0.5,
and saves a binary uint8 prediction mask for each sample.

Usage
-----
    uv run python tools/predict_unet.py --checkpoint output/unet_0320/unet_best.pth
    uv run python tools/predict_unet.py \\
        --checkpoint output/unet_0320/unet_best.pth \\
        --data-dir   data_0320 \\
        --output-dir output/predictions \\
        --threshold  0.5

Key options
-----------
    --checkpoint    Path to trained .pth checkpoint (required)
    --data-dir      Root data directory to scan (default: data_0320)
    --output-dir    Directory to write prediction PNGs (default: output/predictions)
    --base-channels base_channels used when training the model (default: 32)
    --patch-size    Patch size used during training (default: 512)
    --overlap       Overlap between patches in pixels (default: 64)
    --threshold     Foreground probability threshold (default: 0.5)
    --device        'cuda', 'cpu', or 'auto' (default: auto)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from neural_reconstruction.core.segmentation import UNet, predict_full_image
from neural_reconstruction.core.segmentation.dataset import load_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run UNet inference on all samples in a directory."
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to trained .pth checkpoint"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_0320"),
        help="Root data directory (default: data_0320)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/predictions"),
        help="Output directory for prediction masks (default: output/predictions)",
    )
    p.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="base_channels used when training (default: 32)",
    )
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Foreground probability threshold (default: 0.5)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'cuda', 'cpu', or 'auto' (default: auto)",
    )
    return p.parse_args()


def find_samples(data_dir: Path) -> list[Path]:
    """Return sample dirs that have both image.png and annotation.png."""
    return [
        d
        for d in sorted(data_dir.iterdir())
        if d.is_dir() and (d / "image.png").exists() and (d / "annotation.png").exists()
    ]


def main() -> None:
    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    stride = args.patch_size - args.overlap

    # ── Load model ──────────────────────────────────────────────────────────
    model = UNet(in_channels=1, out_channels=2, base_channels=args.base_channels).to(
        device
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt)
    model.eval()

    epoch_info = (
        f" (epoch {ckpt['epoch']})"
        if isinstance(ckpt, dict) and "epoch" in ckpt
        else ""
    )
    print(f"Loaded checkpoint : {args.checkpoint}{epoch_info}")
    print(f"Device            : {device}")
    print(f"Threshold         : {args.threshold}")

    # ── Discover samples ────────────────────────────────────────────────────
    samples = find_samples(args.data_dir)
    if not samples:
        raise SystemExit(f"No valid samples found in {args.data_dir}")
    print(f"Samples           : {len(samples)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output            : {args.output_dir.resolve()}")
    print("-" * 55)

    # ── Inference loop ──────────────────────────────────────────────────────
    for sample_dir in samples:
        img = load_image(sample_dir)  # only image channel needed

        prob_map = predict_full_image(
            model,
            img,
            patch_size=args.patch_size,
            stride=stride,
            device=device,
        )

        # Threshold → binary uint8  (0 / 255)
        pred_bin = ((prob_map > args.threshold).astype(np.uint8)) * 255

        out_path = args.output_dir / f"{sample_dir.name}_pred.png"
        cv2.imwrite(str(out_path), pred_bin)
        print(
            f"  [{sample_dir.name}]  fg_ratio={pred_bin.mean() / 255:.3%}  → {out_path.name}"
        )

    print(f"\nDone. {len(samples)} predictions saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
