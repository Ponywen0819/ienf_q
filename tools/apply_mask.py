"""
影像遮罩應用工具

將遮罩應用到輸入影像上，提取感興趣區域。

使用範例:
python tools/apply_mask.py \
    --image data/Image/S163-2_a.tif \
    --mask data/Mask/S163-2_a.tif \
    --output output/masked_image.png

python tools/apply_mask.py \
    --image data/Image/S163-2_a.tif \
    --mask data/Mask/S163-2_a.tif \
    --output output/masked_image.png \
    --background black
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Literal


def load_image(image_path: str) -> np.ndarray:
    """
    載入影像

    Args:
        image_path: 影像檔案路徑

    Returns:
        影像陣列
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"無法讀取影像: {image_path}")

    return image


def load_mask(mask_path: str, threshold: int = 1) -> np.ndarray:
    """
    載入並二值化遮罩

    Args:
        mask_path: 遮罩檔案路徑
        threshold: 二值化閾值

    Returns:
        二值遮罩 (0 或 255)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"無法讀取遮罩: {mask_path}")

    # 二值化：大於閾值的設為 255，其他為 0
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    return binary_mask


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background: Literal['black', 'white', 'transparent'] = 'black'
) -> np.ndarray:
    """
    將遮罩應用到影像上

    Args:
        image: 輸入影像
        mask: 二值遮罩 (0 或 255)
        background: 背景類型 ('black', 'white', 'transparent')

    Returns:
        應用遮罩後的影像
    """
    # 確保遮罩與影像尺寸相同
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"影像尺寸 {image.shape[:2]} 與遮罩尺寸 {mask.shape} 不匹配"
        )

    # 將遮罩轉換為布林陣列 (True 表示保留的區域)
    mask_bool = mask > 0

    if background == 'transparent':
        # 建立帶 Alpha 通道的影像
        if len(image.shape) == 2:
            # 灰階影像轉為 BGRA
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            # BGR 轉為 BGRA
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4:
            # 已經是 BGRA
            result = image.copy()
        else:
            raise ValueError(f"不支援的影像格式: {image.shape}")

        # 設定 Alpha 通道：遮罩內為 255（不透明），遮罩外為 0（透明）
        result[:, :, 3] = mask

    else:
        # 建立輸出影像
        result = image.copy()

        # 決定背景顏色
        if background == 'white':
            bg_value = 255
        else:  # black
            bg_value = 0

        # 應用遮罩：遮罩外的區域設為背景顏色
        if len(image.shape) == 2:
            # 灰階影像
            result[~mask_bool] = bg_value
        else:
            # 彩色影像
            result[~mask_bool] = bg_value

    return result


def calculate_statistics(
    image: np.ndarray,
    mask: np.ndarray,
    masked_image: np.ndarray
) -> dict:
    """
    計算遮罩應用統計資訊

    Args:
        image: 原始影像
        mask: 遮罩
        masked_image: 應用遮罩後的影像

    Returns:
        統計資訊字典
    """
    mask_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    coverage = (mask_pixels / total_pixels) * 100

    # 計算遮罩區域內的影像統計
    mask_bool = mask > 0
    if len(image.shape) == 2:
        masked_values = image[mask_bool]
    else:
        # 對於彩色影像，計算所有通道的平均
        masked_values = image[mask_bool].flatten()

    return {
        'image_size': image.shape,
        'mask_pixels': int(mask_pixels),
        'total_pixels': int(total_pixels),
        'mask_coverage': float(coverage),
        'masked_region_mean': float(np.mean(masked_values)),
        'masked_region_std': float(np.std(masked_values)),
        'masked_region_min': float(np.min(masked_values)),
        'masked_region_max': float(np.max(masked_values)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='將遮罩應用到影像上，提取感興趣區域',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必要參數
    parser.add_argument(
        '--image', '-i',
        required=True,
        help='輸入影像路徑'
    )

    parser.add_argument(
        '--mask', '-m',
        required=True,
        help='輸入遮罩路徑（二值遮罩）'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='輸出影像路徑'
    )

    # 可選參數
    parser.add_argument(
        '--background', '-b',
        choices=['black', 'white', 'transparent'],
        default='black',
        help='遮罩外的背景類型（預設: black）'
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=1,
        help='遮罩二值化閾值（預設: 1）'
    )

    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='不顯示統計資訊'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    if not Path(args.image).exists():
        print(f"✗ 錯誤: 影像檔案不存在: {args.image}")
        sys.exit(1)

    if not Path(args.mask).exists():
        print(f"✗ 錯誤: 遮罩檔案不存在: {args.mask}")
        sys.exit(1)

    # 建立輸出目錄
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("影像遮罩應用工具")
    print("=" * 60)

    try:
        # 步驟 1: 載入影像和遮罩
        print(f"\n[1/3] 載入影像和遮罩...")
        image = load_image(args.image)
        print(f"  ✓ 影像尺寸: {image.shape}, dtype: {image.dtype}")

        mask = load_mask(args.mask, threshold=args.threshold)
        print(f"  ✓ 遮罩尺寸: {mask.shape}")

        # 步驟 2: 應用遮罩
        print(f"\n[2/3] 應用遮罩...")
        print(f"  背景類型: {args.background}")

        masked_image = apply_mask(image, mask, background=args.background)
        print(f"  ✓ 遮罩已應用")

        # 計算統計資訊
        if not args.no_stats:
            stats = calculate_statistics(image, mask, masked_image)
            print(f"\n  統計資訊:")
            print(f"    影像尺寸: {stats['image_size']}")
            print(f"    遮罩像素數: {stats['mask_pixels']:,}")
            print(f"    總像素數: {stats['total_pixels']:,}")
            print(f"    遮罩覆蓋率: {stats['mask_coverage']:.2f}%")
            print(f"\n  遮罩區域影像統計:")
            print(f"    平均值: {stats['masked_region_mean']:.2f}")
            print(f"    標準差: {stats['masked_region_std']:.2f}")
            print(f"    最小值: {stats['masked_region_min']:.0f}")
            print(f"    最大值: {stats['masked_region_max']:.0f}")

        # 步驟 3: 保存結果
        print(f"\n[3/3] 保存結果...")
        cv2.imwrite(str(output_path), masked_image)
        print(f"  ✓ 已保存應用遮罩後的影像: {output_path}")

        print("\n" + "=" * 60)
        print("✓ 遮罩應用完成！")
        print("=" * 60)

        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
