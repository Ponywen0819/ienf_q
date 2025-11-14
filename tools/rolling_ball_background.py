"""
Rolling Ball 背景光去除工具

使用 Rolling Ball 演算法去除影像的不均勻背景光

使用方式:
python rolling_ball_background.py --input image.tif --output corrected.tif --radius 50
python rolling_ball_background.py --input image.tif --output corrected.tif --radius 50 --show-background
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


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


def create_ball_kernel(radius: int) -> np.ndarray:
    """
    創建球形結構元素

    Args:
        radius: 球的半徑

    Returns:
        球形結構元素
    """
    diameter = 2 * radius + 1
    kernel = np.zeros((diameter, diameter), dtype=np.float32)

    center = radius
    radius_sq = radius * radius

    for y in range(diameter):
        for x in range(diameter):
            dy = y - center
            dx = x - center
            dist_sq = dx * dx + dy * dy

            if dist_sq <= radius_sq:
                # 計算球面高度
                kernel[y, x] = np.sqrt(radius_sq - dist_sq)

    return kernel


def rolling_ball_background(
    image: np.ndarray,
    radius: int,
    light_background: bool = False,
    smoothing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Rolling Ball 演算法估計背景

    Args:
        image: 輸入灰階影像
        radius: Rolling Ball 半徑
        light_background: 是否為亮背景（True）或暗背景（False）
        smoothing: 是否對背景進行平滑

    Returns:
        (背景影像, 去除背景後的影像)
    """
    # 轉換為 float32
    img_float = image.astype(np.float32)

    # 如果是亮背景，反轉影像
    if light_background:
        img_float = 255.0 - img_float

    # 使用形態學開運算（先侵蝕後膨脹）來估計背景
    # 這相當於 rolling ball 在影像下方滾動
    kernel = create_ball_kernel(radius)

    # 使用 grey opening (形態學開運算)
    background = ndimage.grey_opening(img_float, footprint=kernel > 0, size=2*radius+1)

    # 可選：對背景進行高斯平滑
    if smoothing:
        sigma = radius / 3.0
        background = ndimage.gaussian_filter(background, sigma=sigma)

    # 如果是亮背景，反轉回來
    if light_background:
        background = 255.0 - background
        img_float = 255.0 - img_float

    # 去除背景
    corrected = img_float - background

    # 裁剪到有效範圍
    corrected = np.clip(corrected, 0, 255)

    return background.astype(np.uint8), corrected.astype(np.uint8)


def rolling_ball_background_morphology(
    image: np.ndarray,
    radius: int,
    light_background: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用形態學方法進行 Rolling Ball 背景去除（更快速的版本）

    Args:
        image: 輸入灰階影像
        radius: Rolling Ball 半徑
        light_background: 是否為亮背景

    Returns:
        (背景影像, 去除背景後的影像)
    """
    # 創建圓形結構元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

    # 轉換為 float32
    img_float = image.astype(np.float32)

    # 如果是亮背景，反轉影像
    if light_background:
        img_float = 255.0 - img_float

    # 使用形態學開運算估計背景
    background = cv2.morphologyEx(img_float, cv2.MORPH_OPEN, kernel)

    # 如果是亮背景，反轉回來
    if light_background:
        background = 255.0 - background
        img_float = 255.0 - img_float

    # 去除背景
    corrected = img_float - background

    # 裁剪到有效範圍
    corrected = np.clip(corrected, 0, 255)

    return background.astype(np.uint8), corrected.astype(np.uint8)


def save_image(image: np.ndarray, output_path: str, description: str = "影像"):
    """
    保存影像

    Args:
        image: 影像陣列
        output_path: 輸出路徑
        description: 影像描述
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise ValueError(f"無法保存{description}: {output_path}")

    print(f"  ✓ 已保存{description}: {output_path}")


def calculate_statistics(image: np.ndarray) -> dict:
    """
    計算影像統計資訊

    Args:
        image: 輸入灰階影像

    Returns:
        統計資訊字典
    """
    stats = {
        'mean': np.mean(image),
        'std': np.std(image),
        'median': np.median(image),
        'min': np.min(image),
        'max': np.max(image),
    }
    return stats


def print_statistics(original_stats: dict, corrected_stats: dict):
    """
    印出統計資訊

    Args:
        original_stats: 原始影像統計
        corrected_stats: 校正後影像統計
    """
    print("\n" + "=" * 60)
    print("影像統計比較")
    print("=" * 60)
    print(f"\n{'統計量':<15} {'原始影像':>12} {'校正後影像':>12} {'變化':>12}")
    print("-" * 60)
    print(f"{'平均值':<15} {original_stats['mean']:>12.2f} {corrected_stats['mean']:>12.2f} {corrected_stats['mean']-original_stats['mean']:>+12.2f}")
    print(f"{'標準差':<15} {original_stats['std']:>12.2f} {corrected_stats['std']:>12.2f} {corrected_stats['std']-original_stats['std']:>+12.2f}")
    print(f"{'中位數':<15} {original_stats['median']:>12.2f} {corrected_stats['median']:>12.2f} {corrected_stats['median']-original_stats['median']:>+12.2f}")
    print(f"{'最小值':<15} {original_stats['min']:>12.0f} {corrected_stats['min']:>12.0f} {corrected_stats['min']-original_stats['min']:>+12.0f}")
    print(f"{'最大值':<15} {original_stats['max']:>12.0f} {corrected_stats['max']:>12.0f} {corrected_stats['max']-original_stats['max']:>+12.0f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Rolling Ball 背景光去除工具',
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
        help='輸出影像路徑（去除背景後）'
    )

    parser.add_argument(
        '--radius', '-r',
        type=int,
        required=True,
        help='Rolling Ball 半徑（像素）。建議值：50-100'
    )

    # 可選參數
    parser.add_argument(
        '--save-background', '-b',
        default=None,
        help='保存估計的背景影像路徑（可選）'
    )

    parser.add_argument(
        '--light-background',
        action='store_true',
        help='亮背景模式（適用於亮背景的影像，如螢光影像）'
    )

    parser.add_argument(
        '--no-smoothing',
        action='store_true',
        help='不對背景進行平滑處理'
    )

    parser.add_argument(
        '--method', '-m',
        choices=['accurate', 'fast'],
        default='accurate',
        help='處理方法：accurate (精確但慢) 或 fast (快速但簡化)'
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

    # 驗證半徑
    if args.radius < 1:
        print(f"✗ 錯誤: 半徑必須大於 0")
        sys.exit(1)

    print("=" * 60)
    print("Rolling Ball 背景光去除工具")
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

        # 步驟 2: 計算原始統計
        if not args.no_stats:
            print(f"\n[2/4] 計算原始影像統計...")
            original_stats = calculate_statistics(image)
            print(f"  平均值: {original_stats['mean']:.2f}")
            print(f"  標準差: {original_stats['std']:.2f}")

        # 步驟 3: 執行 Rolling Ball 背景去除
        print(f"\n[3/4] 執行 Rolling Ball 背景去除...")
        print(f"  半徑: {args.radius} 像素")
        print(f"  方法: {'精確' if args.method == 'accurate' else '快速'}")
        print(f"  背景類型: {'亮背景' if args.light_background else '暗背景'}")
        print(f"  平滑: {'否' if args.no_smoothing else '是'}")

        if args.method == 'accurate':
            background, corrected = rolling_ball_background(
                image=image,
                radius=args.radius,
                light_background=args.light_background,
                smoothing=not args.no_smoothing
            )
        else:  # fast
            background, corrected = rolling_ball_background_morphology(
                image=image,
                radius=args.radius,
                light_background=args.light_background
            )

        # 步驟 4: 保存結果
        print(f"\n[4/4] 保存結果...")
        print(f"  輸出: {args.output}")
        save_image(corrected, args.output, "校正後影像")

        if args.save_background:
            print(f"  背景: {args.save_background}")
            save_image(background, args.save_background, "背景影像")

        # 顯示統計資訊
        if not args.no_stats:
            corrected_stats = calculate_statistics(corrected)
            print_statistics(original_stats, corrected_stats)

        print("\n✓ 背景去除完成！")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 背景去除失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
