"""
綠色通道提取工具

從彩色影像中提取綠色通道，輸出為灰階影像。

使用範例:
python tools/extract_green_channel.py --input image.tif --output green.png
python tools/extract_green_channel.py -i image.tif -o green.png
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np


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


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    提取綠色通道

    Args:
        image: 輸入影像

    Returns:
        綠色通道灰階影像
    """
    if len(image.shape) == 2:
        # 已經是灰階影像
        print("  ⚠ 警告: 輸入已是灰階影像，直接輸出")
        return image

    if len(image.shape) == 3 and image.shape[2] >= 3:
        # BGR 或 BGRA 格式，提取 G 通道（索引 1）
        return image[:, :, 1]

    raise ValueError(f"不支援的影像格式: {image.shape}")


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


def calculate_statistics(green_channel: np.ndarray) -> dict:
    """
    計算綠色通道統計資訊

    Args:
        green_channel: 綠色通道影像

    Returns:
        統計資訊字典
    """
    return {
        'size': green_channel.shape,
        'mean': float(np.mean(green_channel)),
        'std': float(np.std(green_channel)),
        'min': float(np.min(green_channel)),
        'max': float(np.max(green_channel)),
        'median': float(np.median(green_channel)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='從彩色影像中提取綠色通道',
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

    # 可選參數
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='不顯示統計資訊'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    if not Path(args.input).exists():
        print(f"✗ 錯誤: 輸入檔案不存在: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("綠色通道提取工具")
    print("=" * 60)

    try:
        # 步驟 1: 載入影像
        print(f"\n[1/3] 載入影像...")
        print(f"  輸入: {args.input}")
        image = load_image(args.input)
        print(f"  ✓ 影像尺寸: {image.shape}, dtype: {image.dtype}")

        # 步驟 2: 提取綠色通道
        print(f"\n[2/3] 提取綠色通道...")
        green_channel = extract_green_channel(image)
        print(f"  ✓ 綠色通道尺寸: {green_channel.shape}")

        # 顯示統計資訊
        if not args.no_stats:
            stats = calculate_statistics(green_channel)
            print(f"\n  綠色通道統計:")
            print(f"    平均值: {stats['mean']:.2f}")
            print(f"    標準差: {stats['std']:.2f}")
            print(f"    最小值: {stats['min']:.0f}")
            print(f"    最大值: {stats['max']:.0f}")
            print(f"    中位數: {stats['median']:.2f}")

        # 步驟 3: 保存影像
        print(f"\n[3/3] 保存影像...")
        print(f"  輸出: {args.output}")
        save_image(green_channel, args.output)
        print(f"  ✓ 已保存綠色通道影像")

        print("\n" + "=" * 60)
        print("✓ 綠色通道提取完成！")
        print("=" * 60)

        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
