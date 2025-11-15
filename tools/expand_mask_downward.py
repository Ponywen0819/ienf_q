"""
遮罩向下擴展工具

將二值遮罩向下（Y 軸正方向）擴展特定寬度。

主要用途：
- 將表皮遮罩向下擴展，涵蓋表皮-真皮邊界區域
- 為邊界分析提供擴展的感興趣區域

使用範例：
python tools/expand_mask_downward.py \
    --mask data/Mask/S163-2_a.tif \
    --output-dir output/expanded_mask \
    --expansion-width 50
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np


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


def expand_mask_downward(
    mask: np.ndarray,
    expansion_width: int,
    kernel_shape: str = 'rect'
) -> np.ndarray:
    """
    將遮罩向下（Y 軸正方向）擴展

    Args:
        mask: 輸入二值遮罩 (0 或 255)
        expansion_width: 向下擴展的像素數
        kernel_shape: 結構元素形狀 ('rect' 或 'ellipse')

    Returns:
        擴展後的二值遮罩
    """
    if expansion_width <= 0:
        return mask.copy()

    # 建立垂直方向的結構元素
    # 寬度為 1，高度為 expansion_width
    if kernel_shape == 'ellipse':
        # 橢圓形：稍微有點水平寬度，更平滑
        kernel_width = max(3, expansion_width // 10)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_width, expansion_width)
        )
    else:
        # 矩形：純垂直擴展
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, expansion_width)
        )

    # 進行膨脹操作
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)

    return expanded_mask


def calculate_statistics(
    original_mask: np.ndarray,
    expanded_mask: np.ndarray
) -> dict:
    """
    計算擴展統計資訊

    Args:
        original_mask: 原始遮罩
        expanded_mask: 擴展後遮罩

    Returns:
        統計資訊字典
    """
    original_pixels = np.sum(original_mask > 0)
    expanded_pixels = np.sum(expanded_mask > 0)
    new_pixels = expanded_pixels - original_pixels

    total_pixels = original_mask.shape[0] * original_mask.shape[1]

    return {
        'image_size': original_mask.shape,
        'original_mask_pixels': int(original_pixels),
        'expanded_mask_pixels': int(expanded_pixels),
        'new_pixels_added': int(new_pixels),
        'original_coverage': float(original_pixels / total_pixels * 100),
        'expanded_coverage': float(expanded_pixels / total_pixels * 100),
        'coverage_increase': float(new_pixels / total_pixels * 100)
    }


def main():
    parser = argparse.ArgumentParser(
        description='將二值遮罩向下擴展特定寬度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必要參數
    parser.add_argument(
        '--mask', '-m',
        required=True,
        help='輸入遮罩路徑（二值遮罩）'
    )

    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='輸出目錄'
    )

    # 可選參數
    parser.add_argument(
        '--expansion-width', "-w",
        type=int,
        default=50,
        help='向下擴展的像素數（預設: 50）'
    )

    parser.add_argument(
        '--kernel-shape',
        choices=['rect', 'ellipse'],
        default='rect',
        help='結構元素形狀（預設: rect）'
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=1,
        help='遮罩二值化閾值（預設: 1）'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    if not Path(args.mask).exists():
        print(f"✗ 錯誤: 遮罩檔案不存在: {args.mask}")
        sys.exit(1)

    # 建立輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("遮罩向下擴展工具")
    print("=" * 60)

    try:
        # 步驟 1: 載入遮罩
        print(f"\n[1/3] 載入遮罩...")
        mask = load_mask(args.mask, threshold=args.threshold)
        print(f"  ✓ 遮罩尺寸: {mask.shape}")

        # 步驟 2: 擴展遮罩
        print(f"\n[2/3] 向下擴展遮罩...")
        print(f"  擴展寬度: {args.expansion_width} 像素")
        print(f"  結構元素: {args.kernel_shape}")

        expanded_mask = expand_mask_downward(
            mask,
            expansion_width=args.expansion_width,
            kernel_shape=args.kernel_shape
        )

        # 計算統計資訊
        stats = calculate_statistics(mask, expanded_mask)
        print(f"\n  統計資訊:")
        print(f"    原始遮罩像素數: {stats['original_mask_pixels']:,}")
        print(f"    擴展遮罩像素數: {stats['expanded_mask_pixels']:,}")
        print(f"    新增像素數: {stats['new_pixels_added']:,}")
        print(f"    原始覆蓋率: {stats['original_coverage']:.2f}%")
        print(f"    擴展覆蓋率: {stats['expanded_coverage']:.2f}%")
        print(f"    覆蓋率增加: {stats['coverage_increase']:.2f}%")

        # 步驟 3: 保存結果
        print(f"\n[3/3] 保存結果...")

        # 保存擴展後的遮罩
        expanded_mask_path = output_dir / 'expanded_mask.png'
        cv2.imwrite(str(expanded_mask_path), expanded_mask)
        print(f"  ✓ 擴展遮罩已保存: {expanded_mask_path}")

        print("\n" + "=" * 60)
        print("✓ 遮罩擴展完成！")
        print("=" * 60)

        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
