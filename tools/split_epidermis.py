"""
表皮影像分割工具

根據表皮 mask 將原始影像切分成兩個部分：
1. 表皮內 (Epidermis) - mask 區域內的影像
2. 表皮外 (Dermis) - mask 區域外的影像

使用方式:
python split_epidermis.py \\
    --image data/Original/S163-2_a.tif \\
    --mask data/Mask/S163-2_a.tif \\
    --output-dir output/split/
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple


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


def split_by_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_value: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根據 mask 將影像分割成內部和外部

    Args:
        image: 原始影像
        mask: 二值化 mask (255 為表皮內，0 為表皮外)
        background_value: 背景填充值

    Returns:
        (表皮內影像, 表皮外影像)
    """
    # 確保 mask 為二值化
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # 表皮內：保留 mask 內的區域
    epidermis = image.copy()
    epidermis[mask_binary == 0] = background_value

    # 表皮外：保留 mask 外的區域
    dermis = image.copy()
    dermis[mask_binary > 0] = background_value

    # 計算統計
    num_epidermis_pixels = np.count_nonzero(mask_binary)
    num_dermis_pixels = np.count_nonzero(mask_binary == 0)
    total_pixels = mask_binary.size

    print(f"  表皮內像素: {num_epidermis_pixels} ({num_epidermis_pixels/total_pixels*100:.2f}%)")
    print(f"  表皮外像素: {num_dermis_pixels} ({num_dermis_pixels/total_pixels*100:.2f}%)")

    return epidermis, dermis


def visualize_split_gray(
    original: np.ndarray,
    mask: np.ndarray,
    epidermis: np.ndarray,
    dermis: np.ndarray,
    output_path: str,
    title: str = None,
    dpi: int = 150
):
    """
    視覺化分割結果（灰階）

    Args:
        original: 原始影像
        mask: 二值化 mask
        epidermis: 表皮內影像
        dermis: 表皮外影像
        output_path: 輸出路徑
        title: 圖表標題
        dpi: 輸出解析度
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=dpi)

    # 原始影像
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, weight='bold')
    axes[0, 0].axis('off')

    # Mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Epidermis Mask', fontsize=12, weight='bold')
    axes[0, 1].axis('off')

    # 表皮內
    axes[1, 0].imshow(epidermis, cmap='gray')
    axes[1, 0].set_title('Epidermis (Inside Mask)', fontsize=12, weight='bold')
    axes[1, 0].axis('off')

    # 表皮外
    axes[1, 1].imshow(dermis, cmap='gray')
    axes[1, 1].set_title('Dermis (Outside Mask)', fontsize=12, weight='bold')
    axes[1, 1].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
    else:
        fig.suptitle('Epidermis Image Splitting', fontsize=14, weight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存灰階比較圖: {output_path}")


def visualize_split_viridis(
    original: np.ndarray,
    mask: np.ndarray,
    epidermis: np.ndarray,
    dermis: np.ndarray,
    output_path: str,
    title: str = None,
    dpi: int = 150
):
    """
    視覺化分割結果（Viridis）

    Args:
        original: 原始影像
        mask: 二值化 mask
        epidermis: 表皮內影像
        dermis: 表皮外影像
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

    # Mask (使用 gray)
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Epidermis Mask', fontsize=12, weight='bold')
    axes[0, 1].axis('off')

    # 表皮內
    im2 = axes[1, 0].imshow(epidermis, cmap='viridis')
    axes[1, 0].set_title('Epidermis (Inside Mask)', fontsize=12, weight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='Intensity', shrink=0.8)

    # 表皮外
    im3 = axes[1, 1].imshow(dermis, cmap='viridis')
    axes[1, 1].set_title('Dermis (Outside Mask)', fontsize=12, weight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], label='Intensity', shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
    else:
        fig.suptitle('Epidermis Image Splitting (Viridis)', fontsize=14, weight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存 Viridis 比較圖: {output_path}")


def visualize_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    overlay_alpha: float = 0.4,
    mask_color: Tuple[int, int, int] = (0, 255, 255),  # BGR: 黃色
    dpi: int = 150
):
    """
    視覺化 mask 疊加在原始影像上

    Args:
        original: 原始影像
        mask: 二值化 mask
        output_path: 輸出路徑
        overlay_alpha: 疊加透明度
        mask_color: mask 顏色 (B, G, R)
        dpi: 輸出解析度
    """
    # 轉換為 RGB 用於顯示
    if len(original.shape) == 2:
        # 灰階轉 RGB
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # 創建 mask 疊加層
    overlay = original_rgb.copy()
    mask_colored = np.zeros_like(original_rgb)
    mask_colored[mask > 0] = mask_color[::-1]  # BGR to RGB

    # 混合
    overlay = cv2.addWeighted(original_rgb, 1, mask_colored, overlay_alpha, 0)

    # 繪製
    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
    ax.imshow(overlay)
    ax.set_title('Epidermis Mask Overlay', fontsize=14, weight='bold')
    ax.set_xlabel('X coordinate (pixels)', fontsize=10)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存 mask 疊加圖: {output_path}")


def create_statistics_report(
    original: np.ndarray,
    mask: np.ndarray,
    epidermis: np.ndarray,
    dermis: np.ndarray,
    output_path: str
):
    """
    創建統計報告

    Args:
        original: 原始影像
        mask: 二值化 mask
        epidermis: 表皮內影像
        dermis: 表皮外影像
        output_path: 輸出路徑
    """
    # 計算統計
    mask_binary = (mask > 0).astype(bool)

    epidermis_pixels = original[mask_binary]
    dermis_pixels = original[~mask_binary]

    report = []
    report.append("=" * 60)
    report.append("表皮影像分割統計報告")
    report.append("=" * 60)
    report.append("")

    report.append("【影像資訊】")
    report.append(f"  影像尺寸: {original.shape}")
    report.append(f"  總像素數: {original.size}")
    report.append("")

    report.append("【區域劃分】")
    report.append(f"  表皮內像素數: {np.count_nonzero(mask_binary)}")
    report.append(f"  表皮內佔比: {np.count_nonzero(mask_binary)/original.size*100:.2f}%")
    report.append(f"  表皮外像素數: {np.count_nonzero(~mask_binary)}")
    report.append(f"  表皮外佔比: {np.count_nonzero(~mask_binary)/original.size*100:.2f}%")
    report.append("")

    report.append("【表皮內強度統計】")
    report.append(f"  平均值: {epidermis_pixels.mean():.2f}")
    report.append(f"  標準差: {epidermis_pixels.std():.2f}")
    report.append(f"  最小值: {epidermis_pixels.min()}")
    report.append(f"  最大值: {epidermis_pixels.max()}")
    report.append(f"  中位數: {np.median(epidermis_pixels):.2f}")
    report.append("")

    report.append("【表皮外強度統計】")
    report.append(f"  平均值: {dermis_pixels.mean():.2f}")
    report.append(f"  標準差: {dermis_pixels.std():.2f}")
    report.append(f"  最小值: {dermis_pixels.min()}")
    report.append(f"  最大值: {dermis_pixels.max()}")
    report.append(f"  中位數: {np.median(dermis_pixels):.2f}")
    report.append("")

    report.append("【強度差異】")
    mean_diff = epidermis_pixels.mean() - dermis_pixels.mean()
    report.append(f"  平均值差異: {mean_diff:.2f}")
    report.append(f"  標準差比: {epidermis_pixels.std() / dermis_pixels.std():.2f}")
    report.append("")

    report.append("=" * 60)

    # 保存報告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # 也打印到控制台
    print('\n'.join(report))
    print(f"\n  ✓ 已保存統計報告: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='根據表皮 mask 將影像分割成表皮內和表皮外',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用 - 輸出分割影像
  python split_epidermis.py \\
      --image data/Original/S163-2_a.tif \\
      --mask data/Mask/S163-2_a.tif \\
      --output-dir output/split/

  # 完整模式 - 包含所有視覺化
  python split_epidermis.py \\
      --image data/Original/S163-2_a.tif \\
      --mask data/Mask/S163-2_a.tif \\
      --output-dir output/split/ \\
      --visualize \\
      --statistics

  # 自訂背景值
  python split_epidermis.py \\
      --image data/Original/S163-2_a.tif \\
      --mask data/Mask/S163-2_a.tif \\
      --output-dir output/split/ \\
      --background 255

  # 使用綠色通道
  python split_epidermis.py \\
      --image data/Original/S163-2_a.tif \\
      --mask data/Mask/S163-2_a.tif \\
      --output-dir output/split/ \\
      --green-channel
        """
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
        '--output-dir', '-o',
        required=True,
        help='輸出目錄'
    )

    # 可選參數
    parser.add_argument(
        '--background', '-b',
        type=int,
        default=0,
        help='背景填充值（預設: 0，黑色）'
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
        '--statistics', '-s',
        action='store_true',
        help='生成統計報告'
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

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("表皮影像分割")
    print("=" * 60)

    try:
        # 步驟 1: 載入資料
        print(f"\n[1/4] 載入資料...")
        print(f"  影像: {args.image}")
        image = load_image(args.image)

        print(f"  Mask: {args.mask}")
        mask_image = load_image(args.mask)

        # 提取綠色通道（如果需要）
        if args.green_channel or len(image.shape) == 3:
            print("  提取綠色通道...")
            image = extract_green_channel(image)

        # 步驟 2: 處理 mask
        print(f"\n[2/4] 處理 mask...")
        binary_mask = create_binary_mask(mask_image, args.threshold)
        print(f"  Mask 尺寸: {binary_mask.shape}")

        # 驗證尺寸匹配
        if image.shape[:2] != binary_mask.shape[:2]:
            raise ValueError(f"影像尺寸 {image.shape[:2]} 與 mask 尺寸 {binary_mask.shape[:2]} 不匹配")

        # 步驟 3: 執行分割
        print(f"\n[3/4] 執行影像分割...")
        epidermis, dermis = split_by_mask(image, binary_mask, args.background)

        # 步驟 4: 保存結果
        print(f"\n[4/4] 保存結果...")

        base_name = Path(args.image).stem

        # 保存分割影像
        epidermis_path = output_dir / f"{base_name}_epidermis.tif"
        cv2.imwrite(str(epidermis_path), epidermis)
        print(f"  ✓ 已保存表皮內影像: {epidermis_path}")

        dermis_path = output_dir / f"{base_name}_dermis.tif"
        cv2.imwrite(str(dermis_path), dermis)
        print(f"  ✓ 已保存表皮外影像: {dermis_path}")

        # 保存 mask
        mask_path = output_dir / f"{base_name}_mask_binary.tif"
        cv2.imwrite(str(mask_path), binary_mask)
        print(f"  ✓ 已保存二值化 mask: {mask_path}")

        # 視覺化
        if args.visualize:
            print(f"\n  生成視覺化...")

            # 灰階比較
            comparison_gray_path = output_dir / f"{base_name}_split_comparison_gray.png"
            visualize_split_gray(
                original=image,
                mask=binary_mask,
                epidermis=epidermis,
                dermis=dermis,
                output_path=str(comparison_gray_path),
                title=args.title,
                dpi=args.dpi
            )

            # Viridis 比較
            comparison_viridis_path = output_dir / f"{base_name}_split_comparison_viridis.png"
            visualize_split_viridis(
                original=image,
                mask=binary_mask,
                epidermis=epidermis,
                dermis=dermis,
                output_path=str(comparison_viridis_path),
                title=args.title,
                dpi=args.dpi
            )

            # Mask 疊加
            overlay_path = output_dir / f"{base_name}_mask_overlay.png"
            visualize_overlay(
                original=image,
                mask=binary_mask,
                output_path=str(overlay_path),
                dpi=args.dpi
            )

        # 統計報告
        if args.statistics:
            print(f"\n  生成統計報告...")
            stats_path = output_dir / f"{base_name}_statistics.txt"
            create_statistics_report(
                original=image,
                mask=binary_mask,
                epidermis=epidermis,
                dermis=dermis,
                output_path=str(stats_path)
            )

        print("\n" + "=" * 60)
        print("✓ 分割完成！")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 分割失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
