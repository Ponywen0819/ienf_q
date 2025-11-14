"""
表皮邊緣區域提取工具

提取表皮遮罩下邊緣特定寬度的空間區域

使用方式:
python extract_epidermis_border.py \\
    --image data/Original/S163-2_a.tif \\
    --mask data/Mask/S163-2_a.tif \\
    --output border.tif \\
    --width 50
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def create_binary_mask(mask_image: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    創建二值化 mask

    Args:
        mask_image: 輸入 mask 影像
        threshold: 二值化閾值

    Returns:
        二值化 mask (0 或 255)
    """
    # 如果是彩色影像，轉為灰階
    if len(mask_image.shape) == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image

    # 二值化
    _, binary_mask = cv2.threshold(mask_gray, threshold, 255, cv2.THRESH_BINARY)

    return binary_mask


def find_bottom_border(mask: np.ndarray) -> np.ndarray:
    """
    找到表皮遮罩的下邊緣

    Args:
        mask: 二值化 mask (255 為表皮內，0 為表皮外)

    Returns:
        下邊緣 mask (僅邊緣像素為 255)
    """
    # 確保 mask 為二值化
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # 使用形態學侵蝕找到邊緣
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_binary, kernel, iterations=1)

    # 邊緣 = 原始 - 侵蝕
    border = mask_binary - eroded

    # 只保留下邊緣
    # 對每一列，找到最下方的邊緣點
    bottom_border = np.zeros_like(border)

    for col in range(border.shape[1]):
        # 找到這一列所有邊緣點的 y 座標
        edge_rows = np.where(border[:, col] > 0)[0]
        if len(edge_rows) > 0:
            # 取最大的 y 座標（最下方）
            bottom_row = edge_rows[-1]
            bottom_border[bottom_row, col] = 255

    return bottom_border


def create_border_region_mask(
    bottom_border: np.ndarray,
    width: int,
    direction: str = 'down'
) -> np.ndarray:
    """
    從下邊緣創建特定寬度的區域 mask

    Args:
        bottom_border: 下邊緣 mask
        width: 區域寬度（像素）
        direction: 方向 ('down' 向下，'up' 向上，'both' 雙向)

    Returns:
        區域 mask
    """
    region_mask = np.zeros_like(bottom_border)

    if direction == 'down':
        # 向下擴展
        kernel = np.ones((width * 2 + 1, 1), np.uint8)
        dilated = cv2.dilate(bottom_border, kernel, iterations=1)
        region_mask = dilated

    elif direction == 'up':
        # 向上擴展
        kernel = np.ones((width * 2 + 1, 1), np.uint8)
        dilated = cv2.dilate(bottom_border, kernel, iterations=1)
        # 翻轉以向上擴展
        for col in range(bottom_border.shape[1]):
            edge_rows = np.where(bottom_border[:, col] > 0)[0]
            if len(edge_rows) > 0:
                bottom_row = edge_rows[0]
                start_row = max(0, bottom_row - width)
                region_mask[start_row:bottom_row + 1, col] = 255

    elif direction == 'both':
        # 雙向擴展
        kernel = np.ones((width * 2 + 1, 1), np.uint8)
        region_mask = cv2.dilate(bottom_border, kernel, iterations=1)

    else:
        raise ValueError(f"不支援的方向: {direction}")

    return region_mask


def extract_border_region(
    image: np.ndarray,
    mask: np.ndarray,
    width: int,
    direction: str = 'down',
    background_value: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取表皮下邊緣區域

    Args:
        image: 原始影像
        mask: 表皮 mask
        width: 邊緣寬度（像素）
        direction: 擴展方向
        background_value: 背景值

    Returns:
        (下邊緣 mask, 邊緣區域 mask, 提取的邊緣區域影像)
    """
    # 找到下邊緣
    print(f"  尋找表皮下邊緣...")
    bottom_border = find_bottom_border(mask)

    border_points = np.sum(bottom_border > 0)
    print(f"  下邊緣點數: {border_points}")

    # 創建邊緣區域 mask
    print(f"  創建寬度 {width} 像素的邊緣區域...")
    region_mask = create_border_region_mask(bottom_border, width, direction)

    region_pixels = np.sum(region_mask > 0)
    print(f"  邊緣區域像素: {region_pixels} ({region_pixels/image.size*100:.2f}%)")

    # 提取區域
    border_image = image.copy()
    border_image[region_mask == 0] = background_value

    return bottom_border, region_mask, border_image


def visualize_border_extraction(
    original: np.ndarray,
    mask: np.ndarray,
    bottom_border: np.ndarray,
    region_mask: np.ndarray,
    border_image: np.ndarray,
    output_path: str,
    title: str = None,
    dpi: int = 150
):
    """
    視覺化邊緣提取結果

    Args:
        original: 原始影像
        mask: 表皮 mask
        bottom_border: 下邊緣 mask
        region_mask: 邊緣區域 mask
        border_image: 提取的邊緣區域影像
        output_path: 輸出路徑
        title: 圖表標題
        dpi: 輸出解析度
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=dpi)

    # 原始影像
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, weight='bold')
    axes[0, 0].axis('off')

    # 表皮 mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Epidermis Mask', fontsize=12, weight='bold')
    axes[0, 1].axis('off')

    # 下邊緣
    axes[0, 2].imshow(bottom_border, cmap='gray')
    axes[0, 2].set_title('Bottom Border', fontsize=12, weight='bold')
    axes[0, 2].axis('off')

    # 邊緣區域 mask
    axes[1, 0].imshow(region_mask, cmap='gray')
    axes[1, 0].set_title('Border Region Mask', fontsize=12, weight='bold')
    axes[1, 0].axis('off')

    # 提取的邊緣區域
    axes[1, 1].imshow(border_image, cmap='gray')
    axes[1, 1].set_title('Extracted Border Region', fontsize=12, weight='bold')
    axes[1, 1].axis('off')

    # 疊加顯示
    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB) if len(original.shape) == 2 else original.copy()
    overlay_colored = overlay.copy()
    overlay_colored[region_mask > 0] = [0, 255, 255]  # 黃色標記邊緣區域
    overlay_blend = cv2.addWeighted(overlay, 0.6, overlay_colored, 0.4, 0)

    axes[1, 2].imshow(overlay_blend)
    axes[1, 2].set_title('Overlay (Border Region in Yellow)', fontsize=12, weight='bold')
    axes[1, 2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
    else:
        fig.suptitle('Epidermis Border Region Extraction', fontsize=14, weight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存視覺化結果: {output_path}")


def visualize_border_viridis(
    original: np.ndarray,
    mask: np.ndarray,
    bottom_border: np.ndarray,
    region_mask: np.ndarray,
    border_image: np.ndarray,
    output_path: str,
    title: str = None,
    dpi: int = 150
):
    """
    使用 Viridis 色彩映射視覺化邊緣提取結果

    Args:
        original: 原始影像
        mask: 表皮 mask
        bottom_border: 下邊緣 mask
        region_mask: 邊緣區域 mask
        border_image: 提取的邊緣區域影像
        output_path: 輸出路徑
        title: 圖表標題
        dpi: 輸出解析度
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=dpi)

    # 原始影像
    im0 = axes[0, 0].imshow(original, cmap='viridis')
    axes[0, 0].set_title('Original Image', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], label='Intensity', shrink=0.8)

    # 表皮 mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Epidermis Mask', fontsize=12, weight='bold')
    axes[0, 1].axis('off')

    # 邊緣區域 mask
    axes[1, 0].imshow(region_mask, cmap='gray')
    axes[1, 0].set_title('Border Region Mask', fontsize=12, weight='bold')
    axes[1, 0].axis('off')

    # 提取的邊緣區域
    im3 = axes[1, 1].imshow(border_image, cmap='viridis')
    axes[1, 1].set_title('Extracted Border Region', fontsize=12, weight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], label='Intensity', shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
    else:
        fig.suptitle('Epidermis Border Region Extraction (Viridis)', fontsize=14, weight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存 Viridis 視覺化結果: {output_path}")


def calculate_statistics(image: np.ndarray, border_image: np.ndarray, background_value: int = 0) -> dict:
    """
    計算邊緣區域統計資訊

    Args:
        image: 原始影像
        border_image: 邊緣區域影像
        background_value: 背景值

    Returns:
        統計資訊字典
    """
    border_pixels = border_image[border_image != background_value]

    if len(border_pixels) == 0:
        return {
            'mean': 0,
            'std': 0,
            'median': 0,
            'min': 0,
            'max': 0,
            'num_pixels': 0,
        }

    stats = {
        'mean': np.mean(border_pixels),
        'std': np.std(border_pixels),
        'median': np.median(border_pixels),
        'min': np.min(border_pixels),
        'max': np.max(border_pixels),
        'num_pixels': len(border_pixels),
    }

    return stats


def print_statistics(stats: dict):
    """
    印出統計資訊

    Args:
        stats: 統計資訊字典
    """
    print("\n" + "=" * 60)
    print("邊緣區域統計資訊")
    print("=" * 60)
    print(f"\n【像素統計】")
    print(f"  邊緣區域像素數: {stats['num_pixels']:,}")
    print(f"\n【強度統計】")
    print(f"  平均值: {stats['mean']:.2f}")
    print(f"  標準差: {stats['std']:.2f}")
    print(f"  中位數: {stats['median']:.2f}")
    print(f"  最小值: {stats['min']:.0f}")
    print(f"  最大值: {stats['max']:.0f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='提取表皮遮罩下邊緣特定寬度的空間區域',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必要參數
    parser.add_argument(
        '--image', '-i',
        required=True,
        help='原始影像路徑'
    )

    parser.add_argument(
        '--mask', '-m',
        required=True,
        help='表皮 mask 路徑'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='輸出影像路徑'
    )

    parser.add_argument(
        '--width', '-w',
        type=int,
        required=True,
        help='邊緣區域寬度（像素）'
    )

    # 可選參數
    parser.add_argument(
        '--direction', '-d',
        choices=['down', 'up', 'both'],
        default='down',
        help='擴展方向：down (向下), up (向上), both (雙向)。預設: down'
    )

    parser.add_argument(
        '--background', '-b',
        type=int,
        default=0,
        help='背景填充值（預設: 0）'
    )

    parser.add_argument(
        '--threshold', '-t',
        type=int,
        default=1,
        help='Mask 二值化閾值（預設: 1）'
    )

    parser.add_argument(
        '--green-channel', '-g',
        action='store_true',
        help='只使用綠色通道'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='生成視覺化比較圖'
    )

    parser.add_argument(
        '--save-masks',
        action='store_true',
        help='保存中間 mask 結果'
    )

    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='不顯示統計資訊'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='視覺化輸出解析度（預設: 150）'
    )

    parser.add_argument(
        '--title',
        default=None,
        help='圖表標題（可選）'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    for file_path, name in [(args.image, '影像'), (args.mask, 'Mask')]:
        if not Path(file_path).exists():
            print(f"✗ 錯誤: {name}檔案不存在: {file_path}")
            sys.exit(1)

    # 驗證寬度
    if args.width < 1:
        print(f"✗ 錯誤: 寬度必須大於 0")
        sys.exit(1)

    print("=" * 60)
    print("表皮邊緣區域提取")
    print("=" * 60)

    try:
        # 步驟 1: 載入資料
        print(f"\n[1/5] 載入資料...")
        print(f"  影像: {args.image}")
        image = load_image(args.image)

        print(f"  Mask: {args.mask}")
        mask_image = load_image(args.mask)

        # 提取綠色通道（如果需要）
        if args.green_channel or len(image.shape) == 3:
            print("  提取綠色通道...")
            image = extract_green_channel(image)

        # 步驟 2: 處理 mask
        print(f"\n[2/5] 處理 mask...")
        binary_mask = create_binary_mask(mask_image, args.threshold)
        print(f"  Mask 尺寸: {binary_mask.shape}")

        # 驗證尺寸匹配
        if image.shape[:2] != binary_mask.shape[:2]:
            raise ValueError(f"影像尺寸 {image.shape[:2]} 與 mask 尺寸 {binary_mask.shape[:2]} 不匹配")

        # 步驟 3: 提取邊緣區域
        print(f"\n[3/5] 提取邊緣區域...")
        print(f"  邊緣寬度: {args.width} 像素")
        print(f"  擴展方向: {args.direction}")

        bottom_border, region_mask, border_image = extract_border_region(
            image=image,
            mask=binary_mask,
            width=args.width,
            direction=args.direction,
            background_value=args.background
        )

        # 步驟 4: 保存結果
        print(f"\n[4/5] 保存結果...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), border_image)
        print(f"  ✓ 已保存邊緣區域影像: {output_path}")

        # 保存 mask
        if args.save_masks:
            base_name = output_path.stem

            bottom_border_path = output_path.parent / f"{base_name}_bottom_border.tif"
            cv2.imwrite(str(bottom_border_path), bottom_border)
            print(f"  ✓ 已保存下邊緣 mask: {bottom_border_path}")

            region_mask_path = output_path.parent / f"{base_name}_region_mask.tif"
            cv2.imwrite(str(region_mask_path), region_mask)
            print(f"  ✓ 已保存邊緣區域 mask: {region_mask_path}")

        # 步驟 5: 視覺化
        if args.visualize:
            print(f"\n[5/5] 生成視覺化...")
            base_name = output_path.stem

            # 灰階視覺化
            vis_gray_path = output_path.parent / f"{base_name}_visualization_gray.png"
            visualize_border_extraction(
                original=image,
                mask=binary_mask,
                bottom_border=bottom_border,
                region_mask=region_mask,
                border_image=border_image,
                output_path=str(vis_gray_path),
                title=args.title,
                dpi=args.dpi
            )

            # Viridis 視覺化
            vis_viridis_path = output_path.parent / f"{base_name}_visualization_viridis.png"
            visualize_border_viridis(
                original=image,
                mask=binary_mask,
                bottom_border=bottom_border,
                region_mask=region_mask,
                border_image=border_image,
                output_path=str(vis_viridis_path),
                title=args.title,
                dpi=args.dpi
            )

        # 統計資訊
        if not args.no_stats:
            stats = calculate_statistics(image, border_image, args.background)
            print_statistics(stats)

        print("\n✓ 提取完成！")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 提取失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
