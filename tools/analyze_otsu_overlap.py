#!/usr/bin/env python3
"""
Otsu Thresholding and Label Overlap Analysis Tool

此腳本使用前處理獲取的 ROI image 進行以下分析：
1. 對 ROI image 執行 Otsu 二值化
2. 將二值化結果套用表皮 mask
3. Label 也套用相同的 mask
4. 計算二值化結果與 label 做 AND 運算後，剩餘的二值化遮罩面積

Usage:
    python tools/analyze_otsu_overlap.py -i <input_image> -l <label_image> -m <mask_image> -o <output_dir>
"""

import sys
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

# Add src to python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline
from neural_reconstruction.core.preprocessing.thresholding import otsu_threshold

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def apply_mask_to_binary(binary_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    將 mask 套用到二值化影像上。

    Args:
        binary_image: 二值化影像 (0 或 255)
        mask: 表皮 mask (0 或 255)

    Returns:
        套用 mask 後的影像
    """
    # 確保兩者都是 0/255 格式
    binary_normalized = (binary_image > 0).astype(np.uint8) * 255
    mask_normalized = (mask > 0).astype(np.uint8) * 255

    # 套用 mask (AND 運算)
    return cv2.bitwise_and(binary_normalized, mask_normalized)


def calculate_overlap_stats(otsu_masked: np.ndarray, label_masked: np.ndarray) -> dict:
    """
    計算 Otsu 二值化結果與 label 的重疊統計。

    Args:
        otsu_masked: 套用 mask 後的 Otsu 二值化結果
        label_masked: 套用 mask 後的 label

    Returns:
        包含統計資訊的字典
    """
    # 轉換為 0/1 格式進行計算
    otsu_binary = (otsu_masked > 0).astype(np.uint8)
    label_binary = (label_masked > 0).astype(np.uint8)

    # 計算各種面積
    otsu_area = np.sum(otsu_binary)
    label_area = np.sum(label_binary)

    # AND 運算 - 重疊區域
    overlap = cv2.bitwise_and(otsu_binary, label_binary)
    overlap_area = np.sum(overlap)

    # 剩餘的 Otsu 區域 (Otsu - overlap)
    remaining_otsu = otsu_binary - overlap
    remaining_area = np.sum(remaining_otsu)

    # 計算百分比
    if otsu_area > 0:
        overlap_percentage = (overlap_area / otsu_area) * 100
        remaining_percentage = (remaining_area / otsu_area) * 100
    else:
        overlap_percentage = 0.0
        remaining_percentage = 0.0

    return {
        "otsu_total_area": int(otsu_area),
        "label_area": int(label_area),
        "overlap_area": int(overlap_area),
        "remaining_otsu_area": int(remaining_area),
        "overlap_percentage": overlap_percentage,
        "remaining_percentage": remaining_percentage,
        "overlap_mask": overlap * 255,
        "remaining_mask": remaining_otsu * 255,
    }


def save_visualization(
    roi_image: np.ndarray,
    otsu_masked: np.ndarray,
    label_masked: np.ndarray,
    overlap_mask: np.ndarray,
    remaining_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """
    儲存視覺化結果。

    Args:
        roi_image: ROI 影像
        otsu_masked: 套用 mask 後的 Otsu 二值化結果
        label_masked: 套用 mask 後的 label
        overlap_mask: 重疊區域遮罩
        remaining_mask: 剩餘的 Otsu 區域遮罩
        output_dir: 輸出目錄
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 儲存各個遮罩
    cv2.imwrite(str(output_dir / "01_roi_image.png"), roi_image)
    cv2.imwrite(str(output_dir / "02_otsu_masked.png"), otsu_masked)
    cv2.imwrite(str(output_dir / "03_label_masked.png"), label_masked)
    cv2.imwrite(str(output_dir / "04_overlap.png"), overlap_mask)
    cv2.imwrite(str(output_dir / "05_remaining_otsu.png"), remaining_mask)

    # 創建彩色疊加視覺化
    # 將 ROI 轉為彩色
    if len(roi_image.shape) == 2:
        roi_color = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    else:
        roi_color = roi_image.copy()

    # 疊加不同顏色
    overlay = roi_color.copy()

    # Label 顯示為綠色
    label_coords = np.where(label_masked > 0)
    overlay[label_coords[0], label_coords[1]] = [0, 255, 0]

    # 重疊區域顯示為黃色
    overlap_coords = np.where(overlap_mask > 0)
    overlay[overlap_coords[0], overlap_coords[1]] = [0, 255, 255]

    # 剩餘 Otsu 區域顯示為紅色
    remaining_coords = np.where(remaining_mask > 0)
    overlay[remaining_coords[0], remaining_coords[1]] = [0, 0, 255]

    # 混合原圖與疊加
    result = cv2.addWeighted(roi_color, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(output_dir / "06_overlay_visualization.png"), result)

    # 創建圖例說明
    legend_height = 150
    legend_width = roi_color.shape[1]
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

    # 添加圖例文字
    cv2.putText(
        legend,
        "Green: Label (manual annotation)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        legend,
        "Yellow: Overlap (Otsu AND Label)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        legend,
        "Red: Remaining Otsu (Otsu - Label)",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        legend,
        "Gray: ROI background",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (128, 128, 128),
        2,
    )

    # 合併圖例和結果
    final_result = np.vstack([result, legend])
    cv2.imwrite(str(output_dir / "07_final_result_with_legend.png"), final_result)

    logger.info(f"Saved visualization images to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Otsu thresholding and label overlap."
    )
    parser.add_argument("-i", "--image", required=True, help="Input raw image path")
    parser.add_argument("-l", "--label", required=True, help="Input label image path")
    parser.add_argument("-m", "--mask", required=True, help="Input mask image path")
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for results"
    )

    args = parser.parse_args()

    # Check inputs
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return
    if not Path(args.label).exists():
        logger.error(f"Label not found: {args.label}")
        return
    if not Path(args.mask).exists():
        logger.error(f"Mask not found: {args.mask}")
        return

    # Load images
    logger.info("Loading images...")
    orig_img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    label_img = cv2.imread(args.label, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    if orig_img is None:
        logger.error(f"Failed to load image: {args.image}")
        return
    if label_img is None:
        logger.error(f"Failed to load label: {args.label}")
        return
    if mask_img is None:
        logger.error(f"Failed to load mask: {args.mask}")
        return
    original_green = orig_img[:, :, 1]

    # Initialize preprocessing pipeline
    logger.info("Initializing preprocessing pipeline...")
    preprocessing_config = {
        "morphology": {
            "closing_kernel": 0,
            "opening_kernel": 3,
        },
        "mask": {
            "dilate_offset": 50,
        },
        "background": {
            "method": "rolling_ball",
            "radius": 2,
            "light_background": True,
        },
        "threshold": {"method": "binary"},
        "normalization": {
            "enabled": True,
        },
    }

    pipeline = SkinAnalysisPipeline(preprocessing_config)

    # Run preprocessing to get ROI image and processed label
    logger.info("Running preprocessing to get ROI image...")
    try:
        final_label, roi_image = pipeline.run(label_img, mask_img, original_green)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return

    logger.info(f"ROI image shape: {roi_image.shape}, Label shape: {final_label.shape}")

    # Step 1: 對整張 ROI image 執行 Otsu 二值化
    logger.info("Applying Otsu thresholding to ROI image...")
    otsu_binary = otsu_threshold(roi_image)

    # Step 2: 將 Otsu 二值化結果套用表皮 mask
    logger.info("Applying mask to Otsu result...")
    otsu_masked = apply_mask_to_binary(otsu_binary, mask_img)

    # Step 3: Label 也套用相同的 mask
    logger.info("Applying mask to label...")
    label_masked = apply_mask_to_binary(final_label, mask_img)

    # Step 4: 計算統計資訊
    logger.info("Calculating overlap statistics...")
    stats = calculate_overlap_stats(otsu_masked, label_masked)

    # 輸出統計結果
    logger.info("\n" + "=" * 60)
    logger.info("統計結果:")
    logger.info("=" * 60)
    logger.info(f"Otsu 二值化總面積 (pixels): {stats['otsu_total_area']:,}")
    logger.info(f"Label 總面積 (pixels): {stats['label_area']:,}")
    logger.info(f"重疊區域面積 (pixels): {stats['overlap_area']:,}")
    logger.info(f"剩餘 Otsu 區域面積 (pixels): {stats['remaining_otsu_area']:,}")
    logger.info("-" * 60)
    logger.info(f"重疊百分比 (相對於 Otsu): {stats['overlap_percentage']:.2f}%")
    logger.info(f"剩餘百分比 (相對於 Otsu): {stats['remaining_percentage']:.2f}%")
    logger.info("=" * 60)

    # 儲存視覺化結果
    output_dir = Path(args.output)
    logger.info(f"\nSaving visualization to: {output_dir}")
    save_visualization(
        roi_image,
        otsu_masked,
        label_masked,
        stats["overlap_mask"],
        stats["remaining_mask"],
        output_dir,
    )

    # 儲存統計資訊到文字檔
    stats_file = output_dir / "statistics.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Otsu 二值化與 Label 重疊分析統計\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Otsu 二值化總面積 (pixels): {stats['otsu_total_area']:,}\n")
        f.write(f"Label 總面積 (pixels): {stats['label_area']:,}\n")
        f.write(f"重疊區域面積 (pixels): {stats['overlap_area']:,}\n")
        f.write(f"剩餘 Otsu 區域面積 (pixels): {stats['remaining_otsu_area']:,}\n")
        f.write("-" * 60 + "\n")
        f.write(f"重疊百分比 (相對於 Otsu): {stats['overlap_percentage']:.2f}%\n")
        f.write(f"剩餘百分比 (相對於 Otsu): {stats['remaining_percentage']:.2f}%\n")
        f.write("=" * 60 + "\n")

    logger.info(f"Statistics saved to: {stats_file}")
    logger.info("\n處理完成！")


if __name__ == "__main__":
    main()
