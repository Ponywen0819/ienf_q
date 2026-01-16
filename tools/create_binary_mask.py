#!/usr/bin/env python3
"""
圖片轉二值化遮罩腳本

從 PNG 圖片的 alpha channel 創建二值化遮罩。
將所有可見區域（alpha > 0）轉換為白色，透明區域（alpha = 0）轉換為黑色。
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_binary_mask_from_alpha(image_path, output_path=None, alpha_threshold=0):
    """
    從 PNG 圖片的 alpha channel 創建二值化遮罩

    Args:
        image_path: 輸入圖片路徑
        output_path: 輸出遮罩路徑 (如果為 None，則自動生成)
        alpha_threshold: alpha 值閾值，大於此值的像素將被視為可見 (預設: 0)

    Returns:
        mask: 二值化遮罩 (numpy array)
        output_path: 實際輸出路徑
    """

    # 載入圖片
    try:
        img = Image.open(image_path)
        logger.info(f"載入圖片: {image_path} (模式: {img.mode}, 大小: {img.size})")
    except Exception as e:
        logger.error(f"無法載入圖片 {image_path}: {e}")
        return None, None

    # 確保圖片有 alpha channel
    if img.mode != "RGBA":
        if img.mode == "RGB":
            logger.warning(
                f"圖片 {image_path} 沒有 alpha channel，將所有非黑色像素視為可見區域"
            )
            img_array = np.array(img)
            # 對於 RGB 圖片，將任何非黑色像素視為可見
            mask = np.any(img_array > 0, axis=2).astype(np.uint8) * 255
        else:
            logger.info(f"將圖片從 {img.mode} 轉換為 RGBA")
            img = img.convert("RGBA")
            img_array = np.array(img)
            alpha_channel = img_array[:, :, 3]
            mask = (alpha_channel > alpha_threshold).astype(np.uint8) * 255
    else:
        # 轉換為 numpy array
        img_array = np.array(img)

        # 提取 alpha channel
        alpha_channel = img_array[:, :, 3]

        # 創建二值化遮罩：alpha > threshold 的區域為白色 (255)，其他為黑色 (0)
        mask = (alpha_channel > alpha_threshold).astype(np.uint8) * 255

    # 計算遮罩中的像素數量
    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    percentage = (white_pixels / total_pixels) * 100

    logger.info(f"遮罩中的白色像素: {white_pixels}/{total_pixels} ({percentage:.2f}%)")

    # 生成輸出路徑
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_mask.png"
    else:
        output_path = Path(output_path)

    # 儲存遮罩
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img = Image.fromarray(mask)
        mask_img.save(output_path)
        logger.info(f"已儲存遮罩: {output_path}")
    except Exception as e:
        logger.error(f"無法儲存遮罩 {output_path}: {e}")
        return mask, None

    return mask, output_path


def batch_convert(input_dir, output_dir=None, pattern="*.png", **kwargs):
    """
    批量轉換資料夾中的圖片

    Args:
        input_dir: 輸入資料夾
        output_dir: 輸出資料夾 (如果為 None，則在原位置生成)
        pattern: 檔案匹配模式
        **kwargs: 其他參數傳遞給 create_binary_mask_from_alpha
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        logger.error(f"輸入資料夾不存在: {input_dir}")
        return

    # 獲取所有匹配的檔案
    files = list(input_dir.glob(pattern))

    if not files:
        logger.warning(f"在 {input_dir} 中找不到符合模式 {pattern} 的檔案")
        return

    logger.info(f"找到 {len(files)} 個檔案")

    success_count = 0
    fail_count = 0

    for file_path in files:
        # 生成輸出路徑
        if output_dir:
            output_dir_path = Path(output_dir)
            output_path = output_dir_path / f"{file_path.stem}_mask.png"
        else:
            output_path = None

        # 轉換
        mask, saved_path = create_binary_mask_from_alpha(
            file_path, output_path, **kwargs
        )

        if mask is not None and saved_path is not None:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"\n批量轉換完成!")
    logger.info(f"成功: {success_count}, 失敗: {fail_count}")


def main():
    """命令列介面"""
    parser = argparse.ArgumentParser(
        description="從 PNG 圖片的 alpha channel 創建二值化遮罩"
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True, help="輸入圖片或資料夾路徑"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="輸出遮罩路徑或資料夾路徑 (預設: 在原位置生成 *_mask.png)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=0,
        help="Alpha 值閾值 0-255 (預設: 0, 表示任何非透明像素都會被包含)",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*.png",
        help="批量處理時的檔案匹配模式 (預設: *.png)",
    )

    parser.add_argument("-b", "--batch", action="store_true", help="批量處理模式")

    args = parser.parse_args()

    input_path = Path(args.input)

    # 批量處理模式
    if args.batch or input_path.is_dir():
        batch_convert(
            input_path,
            args.output,
            pattern=args.pattern,
            alpha_threshold=args.threshold,
        )
    # 單檔案處理模式
    else:
        if not input_path.exists():
            logger.error(f"輸入檔案不存在: {input_path}")
            return

        create_binary_mask_from_alpha(
            input_path, args.output, alpha_threshold=args.threshold
        )


if __name__ == "__main__":
    main()
