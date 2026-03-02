"""
將尺寸不一致的 label.png 縮放至與 image.png 相同大小。

使用 NEAREST 插值，確保二值標注影像不產生中間值。
原始檔案會備份為 label_original.png。

使用方式:
    uv run python tools/resize_labels.py --data-dir data/
    uv run python tools/resize_labels.py --data-dir data/ --dry-run  # 只預覽不修改
"""

import argparse
from pathlib import Path

from PIL import Image


def resize_label(sample_dir: Path, dry_run: bool) -> bool:
    """
    將 label.png resize 成與 image.png 相同大小。

    Returns:
        True 表示有執行 resize，False 表示跳過
    """
    image_path = sample_dir / "image.png"
    label_path = sample_dir / "label.png"

    if not image_path.exists() or not label_path.exists():
        return False

    with Image.open(image_path) as img:
        target_size = img.size  # (width, height)

    with Image.open(label_path) as lbl:
        label_size = lbl.size

    if label_size == target_size:
        return False

    print(f"[{sample_dir.name}] {label_size[0]}x{label_size[1]} -> {target_size[0]}x{target_size[1]}")

    if dry_run:
        return True

    # 備份原始檔案
    backup_path = sample_dir / "label_original.png"
    if not backup_path.exists():
        label_path.rename(backup_path)
        print(f"  備份: label_original.png")
    else:
        print(f"  備份已存在，跳過備份")
        label_path.unlink()

    # 用 NEAREST 插值 resize（保持二值特性）
    with Image.open(backup_path) as lbl:
        resized = lbl.resize(target_size, Image.NEAREST)
        resized.save(label_path)

    print(f"  已儲存: label.png")
    return True


def main():
    parser = argparse.ArgumentParser(description="Resize label.png 至與 image.png 相同大小")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="資料集根目錄")
    parser.add_argument("--dry-run", action="store_true", help="只列出需要處理的樣本，不實際修改")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"ERROR: 目錄不存在: {data_dir}")
        return

    if args.dry_run:
        print("=== DRY RUN 模式，不會修改任何檔案 ===\n")

    sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    print(f"掃描 {len(sample_dirs)} 個樣本...\n")

    resized_count = 0
    for sample_dir in sample_dirs:
        if resize_label(sample_dir, args.dry_run):
            resized_count += 1

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"共 {resized_count} 個樣本需要 resize（dry run，未修改）")
    else:
        print(f"共 {resized_count} 個樣本已完成 resize")


if __name__ == "__main__":
    main()
