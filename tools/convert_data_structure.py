#!/usr/bin/env python3
"""
資料結構轉換腳本

將資料從當前結構轉換為新結構：
- 原始結構：data/Original/[ID]_[type].tif, data/Mask/[ID]_[type].tif, data/Label/[ID]_[type].tif
- 新結構：converted_data/[ID]/image.png, mask.png, annotation.png
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from PIL import Image
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_id(filename):
    """
    從檔名提取 ID
    例如: S1585-2_a.tif -> S1585-2_a
    """
    # 移除副檔名，得到完整 ID
    id_name = Path(filename).stem
    return id_name


def group_files_by_id(folder_path):
    """
    按 ID 分組檔案
    返回字典: {ID: [file1, file2, ...]}
    """
    files_by_id = defaultdict(list)

    if not os.path.exists(folder_path):
        logger.warning(f"資料夾不存在: {folder_path}")
        return files_by_id

    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            file_id = extract_id(filename)
            files_by_id[file_id].append(filename)

    return files_by_id


def load_image(image_path):
    """載入影像"""
    try:
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"無法載入影像 {image_path}: {e}")
        return None


def save_image_as_png(image, output_path):
    """將影像儲存為 PNG"""
    try:
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        logger.info(f"已儲存: {output_path}")
        return True
    except Exception as e:
        logger.error(f"無法儲存影像 {output_path}: {e}")
        return False


def convert_data_structure(source_dir, output_dir):
    """
    主轉換函數

    Args:
        source_dir: 源資料夾路徑 (通常是 data/)
        output_dir: 輸出資料夾路徑 (通常是 converted_data/)
    """

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # 確保源資料夾存在
    if not source_dir.exists():
        logger.error(f"源資料夾不存在: {source_dir}")
        return False

    # 建立輸出資料夾
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分別載入三個資料夾中的檔案
    original_folder = source_dir / "Original"
    mask_folder = source_dir / "Mask"
    label_folder = source_dir / "Label"

    original_files = group_files_by_id(original_folder)
    mask_files = group_files_by_id(mask_folder)
    label_files = group_files_by_id(label_folder)

    # 獲取所有唯一的 ID
    all_ids = (
        set(original_files.keys()) | set(mask_files.keys()) | set(label_files.keys())
    )

    logger.info(f"找到 {len(all_ids)} 個唯一的 ID")

    # 為每個 ID 建立資料夾並轉換檔案
    success_count = 0
    fail_count = 0

    for id_name in sorted(all_ids):
        # 為該 ID 建立輸出資料夾
        id_output_dir = output_dir / id_name
        id_output_dir.mkdir(parents=True, exist_ok=True)

        # 處理 Original 檔案
        if id_name in original_files:
            for filename in original_files[id_name]:
                input_path = original_folder / filename
                image = load_image(input_path)
                if image:
                    output_path = id_output_dir / "image.png"
                    if save_image_as_png(image, output_path):
                        success_count += 1
                    else:
                        fail_count += 1

        # 處理 Mask 檔案
        if id_name in mask_files:
            for filename in mask_files[id_name]:
                input_path = mask_folder / filename
                image = load_image(input_path)
                if image:
                    output_path = id_output_dir / "mask.png"
                    if save_image_as_png(image, output_path):
                        success_count += 1
                    else:
                        fail_count += 1

        # 處理 Label 檔案 -> annotation
        if id_name in label_files:
            for filename in label_files[id_name]:
                input_path = label_folder / filename
                image = load_image(input_path)
                if image:
                    output_path = id_output_dir / "annotation.png"
                    if save_image_as_png(image, output_path):
                        success_count += 1
                    else:
                        fail_count += 1

    logger.info(f"\n轉換完成!")
    logger.info(f"成功: {success_count}, 失敗: {fail_count}")
    logger.info(f"輸出位置: {output_dir}")

    return True


def main():
    """
    命令列介面
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="將資料結構從分類資料夾轉換為 ID 資料夾"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data",
        help="源資料夾路徑 (包含 Original/, Mask/, Label/)",
    )
    parser.add_argument(
        "--output", type=str, default="converted_data", help="輸出資料夾路徑"
    )

    args = parser.parse_args()

    convert_data_structure(args.source, args.output)


if __name__ == "__main__":
    main()
