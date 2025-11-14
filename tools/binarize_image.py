"""
影像二值化工具

使用指定閾值將影像轉換為二值影像

使用方式:
python binarize_image.py --input image.tif --output binary.tif --threshold 128
python binarize_image.py --input image.tif --output binary.tif --threshold 128 --invert
python binarize_image.py --input image.tif --output binary.tif --auto otsu
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    載入影像

    Args:
        image_path: 影像路徑

    Returns:
        影像陣列
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"無法讀取影像: {image_path}")

    print(f"  載入影像: {image.shape}, dtype: {image.dtype}")
    return image


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    提取綠色通道

    Args:
        image: 輸入影像

    Returns:
        綠色通道灰階影像
    """
    if len(image.shape) == 2:
        return image

    if len(image.shape) == 3 and image.shape[2] >= 3:
        return image[:, :, 1]  # G 通道 (BGR)

    raise ValueError(f"不支援的影像格式: {image.shape}")


def calculate_otsu_threshold(image: np.ndarray) -> int:
    """
    使用 Otsu 方法計算最佳閾值

    Args:
        image: 輸入灰階影像

    Returns:
        最佳閾值
    """
    threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(threshold)


def calculate_triangle_threshold(image: np.ndarray) -> int:
    """
    使用 Triangle 方法計算最佳閾值

    Args:
        image: 輸入灰階影像

    Returns:
        最佳閾值
    """
    threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return int(threshold)


def calculate_mean_threshold(image: np.ndarray) -> int:
    """
    使用平均值作為閾值

    Args:
        image: 輸入灰階影像

    Returns:
        平均值閾值
    """
    return int(np.mean(image))


def calculate_median_threshold(image: np.ndarray) -> int:
    """
    使用中位數作為閾值

    Args:
        image: 輸入灰階影像

    Returns:
        中位數閾值
    """
    return int(np.median(image))


def binarize_image(
    image: np.ndarray,
    threshold: int,
    invert: bool = False,
    exclude_zero: bool = False
) -> np.ndarray:
    """
    二值化影像

    Args:
        image: 輸入灰階影像
        threshold: 閾值
        invert: 是否反轉（True: 大於閾值為黑，False: 大於閾值為白）
        exclude_zero: 是否保留原本為 0 的像素為 0

    Returns:
        二值化影像
    """
    if exclude_zero:
        # 保留原本為 0 的像素
        mask_zero = image == 0
        if invert:
            binary = np.where(image <= threshold, 255, 0).astype(np.uint8)
        else:
            binary = np.where(image > threshold, 255, 0).astype(np.uint8)
        binary[mask_zero] = 0
    else:
        if invert:
            binary = np.where(image <= threshold, 255, 0).astype(np.uint8)
        else:
            binary = np.where(image > threshold, 255, 0).astype(np.uint8)

    return binary


def save_image(image: np.ndarray, output_path: str):
    """
    保存影像

    Args:
        image: 影像陣列
        output_path: 輸出路徑
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise ValueError(f"無法保存影像: {output_path}")

    print(f"  ✓ 已保存二值化影像: {output_path}")


def calculate_statistics(image: np.ndarray, binary: np.ndarray) -> dict:
    """
    計算二值化統計資訊

    Args:
        image: 原始影像
        binary: 二值化影像

    Returns:
        統計資訊字典
    """
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    total_pixels = image.size

    stats = {
        'white_pixels': white_pixels,
        'black_pixels': black_pixels,
        'total_pixels': total_pixels,
        'white_percentage': (white_pixels / total_pixels) * 100,
        'black_percentage': (black_pixels / total_pixels) * 100,
        'original_mean': np.mean(image),
        'original_std': np.std(image),
        'original_min': np.min(image),
        'original_max': np.max(image),
    }

    return stats


def print_statistics(stats: dict, threshold: int):
    """
    印出統計資訊

    Args:
        stats: 統計資訊字典
        threshold: 使用的閾值
    """
    print("\n" + "=" * 60)
    print("二值化統計資訊")
    print("=" * 60)
    print(f"\n【閾值】")
    print(f"  使用閾值: {threshold}")
    print(f"\n【原始影像統計】")
    print(f"  平均值: {stats['original_mean']:.2f}")
    print(f"  標準差: {stats['original_std']:.2f}")
    print(f"  最小值: {stats['original_min']:.0f}")
    print(f"  最大值: {stats['original_max']:.0f}")
    print(f"\n【二值化結果】")
    print(f"  白色像素: {stats['white_pixels']:,} ({stats['white_percentage']:.2f}%)")
    print(f"  黑色像素: {stats['black_pixels']:,} ({stats['black_percentage']:.2f}%)")
    print(f"  總像素數: {stats['total_pixels']:,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='影像二值化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
      
    )

    # 必要參數
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='輸入影像路徑'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='輸出影像路徑'
    )

    # 閾值參數（二選一）
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument(
        '--threshold', '-t',
        type=int,
        help='固定閾值 (0-255)'
    )
    threshold_group.add_argument(
        '--auto', '-a',
        choices=['otsu', 'triangle', 'mean', 'median'],
        help='自動計算閾值方法 (otsu, triangle, mean, median)'
    )

    # 可選參數
    parser.add_argument(
        '--invert',
        action='store_true',
        help='反轉二值化結果（大於閾值為黑，小於等於閾值為白）'
    )

    parser.add_argument(
        '--exclude-zero',
        action='store_true',
        help='保留原本強度值為 0 的像素為 0'
    )

    parser.add_argument(
        '--green-channel', '-g',
        action='store_true',
        help='只使用綠色通道'
    )

    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='不顯示統計資訊'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 錯誤: 輸入檔案不存在: {args.input}")
        sys.exit(1)

    # 驗證固定閾值範圍
    if args.threshold is not None:
        if args.threshold < 0 or args.threshold > 255:
            print(f"✗ 錯誤: 閾值必須在 0-255 之間")
            sys.exit(1)

    print("=" * 60)
    print("影像二值化工具")
    print("=" * 60)

    try:
        # 步驟 1: 載入影像
        print(f"\n[1/4] 載入影像...")
        print(f"  輸入: {args.input}")
        image = load_image(args.input)

        # 提取綠色通道（如果需要）
        if args.green_channel or len(image.shape) == 3:
            print("  提取綠色通道...")
            image = extract_green_channel(image)

        print(f"  影像尺寸: {image.shape}")
        print(f"  強度範圍: {image.min()} - {image.max()}")

        # 步驟 2: 計算或使用閾值
        print(f"\n[2/4] 計算閾值...")
        if args.threshold is not None:
            threshold = args.threshold
            print(f"  使用固定閾值: {threshold}")
        else:
            if args.auto == 'otsu':
                threshold = calculate_otsu_threshold(image)
                print(f"  使用 Otsu 自動閾值: {threshold}")
            elif args.auto == 'triangle':
                threshold = calculate_triangle_threshold(image)
                print(f"  使用 Triangle 自動閾值: {threshold}")
            elif args.auto == 'mean':
                threshold = calculate_mean_threshold(image)
                print(f"  使用平均值閾值: {threshold}")
            elif args.auto == 'median':
                threshold = calculate_median_threshold(image)
                print(f"  使用中位數閾值: {threshold}")

        # 步驟 3: 二值化
        print(f"\n[3/4] 執行二值化...")
        if args.invert:
            print("  模式: 反轉 (大於閾值 → 黑, 小於等於閾值 → 白)")
        else:
            print("  模式: 正常 (大於閾值 → 白, 小於等於閾值 → 黑)")

        if args.exclude_zero:
            print("  保留原本為 0 的像素")

        binary = binarize_image(
            image=image,
            threshold=threshold,
            invert=args.invert,
            exclude_zero=args.exclude_zero
        )

        # 步驟 4: 保存影像
        print(f"\n[4/4] 保存影像...")
        print(f"  輸出: {args.output}")
        save_image(binary, args.output)

        # 顯示統計資訊
        if not args.no_stats:
            stats = calculate_statistics(image, binary)
            print_statistics(stats, threshold)

        print("\n✓ 二值化完成！")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 二值化失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
