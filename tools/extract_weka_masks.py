"""
從 Weka 分類結果 TIF 檔中萃取紅色（索引 0）區域，儲存為二元遮罩。

輸入資料夾結構：
    <input-dir>/<Sample_id>.tif

輸出資料夾結構：
    <output-dir>/<Sample_id>/weka.png

Weka TIF 為 palette 索引圖：
    - 索引 0 = 紅色 (255, 0, 0) → 神經纖維（foreground）
    - 索引 1 = 綠色 → 背景

Usage:
    uv run python tools/extract_weka_masks.py \
        --input-dir nas/neuroimages/20250214/第一批/Weka3_1_1 \
        --output-dir output/weka_masks

    # 僅處理特定 sample
    uv run python tools/extract_weka_masks.py \
        --input-dir nas/neuroimages/20250214/第一批/Weka3_1_1 \
        --output-dir output/weka_masks \
        --sample-ids S1037-2_a S1037-2_b
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def extract_red_mask(tif_path: Path) -> np.ndarray:
    """
    讀取 Weka palette TIF，萃取紅色（索引 0）區域為二元遮罩。

    Returns:
        uint8 array, 255 = fiber, 0 = background
    """
    img = Image.open(tif_path)

    if img.mode == "P":
        # palette 索引圖：索引 0 對應紅色
        arr = np.array(img, dtype=np.uint8)
        mask = (arr == 0).astype(np.uint8) * 255
    elif img.mode in ("RGB", "RGBA"):
        # 直接比對紅色像素
        arr = np.array(img)
        red_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
        mask = red_mask.astype(np.uint8) * 255
    elif img.mode == "L":
        # 灰階：假設 0 為 foreground
        arr = np.array(img, dtype=np.uint8)
        mask = (arr == 0).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unsupported image mode: {img.mode} in {tif_path}")

    return mask


def process_directory(
    input_dir: Path,
    output_dir: Path,
    sample_ids: list[str] | None = None,
    verbose: bool = False,
) -> dict:
    """
    批次處理資料夾中所有 TIF 檔。

    Returns:
        dict with keys 'success', 'skipped', 'failed'
    """
    tif_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.TIF"))

    if not tif_files:
        print(f"[WARNING] No .tif files found in {input_dir}")
        return {"success": [], "skipped": [], "failed": []}

    # 若有指定 sample_ids，過濾
    if sample_ids:
        id_set = set(sample_ids)
        tif_files = [f for f in tif_files if f.stem in id_set]
        if not tif_files:
            print(f"[WARNING] None of the specified sample IDs found in {input_dir}")
            return {"success": [], "skipped": [], "failed": []}

    results = {"success": [], "skipped": [], "failed": []}

    for tif_path in tif_files:
        sample_id = tif_path.stem
        sample_output_dir = output_dir / sample_id
        output_path = sample_output_dir / "weka.png"

        if output_path.exists():
            if verbose:
                print(f"[SKIP] {sample_id} (already exists)")
            results["skipped"].append(sample_id)
            continue

        try:
            mask = extract_red_mask(tif_path)
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(output_path)

            fiber_pixels = (mask > 0).sum()
            total_pixels = mask.size
            ratio = fiber_pixels / total_pixels * 100

            if verbose:
                print(
                    f"[OK] {sample_id} → {output_path} "
                    f"(fiber: {fiber_pixels:,} px, {ratio:.2f}%)"
                )
            else:
                print(f"[OK] {sample_id}")

            results["success"].append(sample_id)

        except Exception as e:
            print(f"[FAIL] {sample_id}: {e}", file=sys.stderr)
            results["failed"].append(sample_id)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract red (class 0) regions from Weka TIF files as binary masks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("nas/neuroimages/20250214/第一批/Weka3_1_1"),
        help="Directory containing <Sample_id>.tif files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/weka_masks"),
        help="Output root directory; each sample gets its own subdirectory",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        metavar="SAMPLE_ID",
        help="Only process these sample IDs (e.g. S1037-2_a S1037-2_b)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-pixel statistics for each sample",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Input : {args.input_dir}")
    print(f"Output: {args.output_dir}")
    if args.sample_ids:
        print(f"Filter: {args.sample_ids}")
    print()

    results = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_ids=args.sample_ids,
        verbose=args.verbose,
    )

    print()
    print(f"Done — success: {len(results['success'])}, "
          f"skipped: {len(results['skipped'])}, "
          f"failed: {len(results['failed'])}")

    if results["failed"]:
        print(f"Failed samples: {results['failed']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
