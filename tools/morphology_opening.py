"""
形態學開操作工具

對輸入影像進行開操作（Opening = 腐蝕 + 膨脹），並輸出結果和 Viridis 視覺化

開操作的用途：
- 去除小的噪點和細小突起
- 分離接近但不相連的物體
- 平滑物體輪廓
- 不會顯著改變物體的大小

使用方式:
python morphology_opening.py --input image.tif --output output.tif --kernel-size 5
python morphology_opening.py --input image.tif --output-dir output/ --visualize
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple


def extract_green_channel(image_path: str) -> np.ndarray:
    """
    提取影像的綠色通道

    Args:
        image_path: 影像路徑

    Returns:
        綠色通道灰階影像
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"無法讀取影像: {image_path}")

    # 如果是灰階影像，直接返回
    if len(image.shape) == 2:
        return image

    # 如果是彩色影像，提取綠色通道
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return image[:, :, 1]  # G 通道 (BGR)

    raise ValueError(f"不支援的影像格式: {image.shape}")


def morphology_opening(
    image: np.ndarray,
    kernel_size: int = 5,
    kernel_shape: str = 'ellipse',
    iterations: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    對影像進行形態學開操作

    Args:
        image: 輸入灰階影像
        kernel_size: 結構元素大小（奇數）
        kernel_shape: 結構元素形狀 ('rect', 'ellipse', 'cross')
        iterations: 迭代次數

    Returns:
        (開操作結果, 結構元素)
    """
    print(f"  執行開操作...")
    print(f"    結構元素: {kernel_shape}, 大小: {kernel_size}x{kernel_size}")
    print(f"    迭代次數: {iterations}")

    # 確保 kernel_size 為奇數
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"    調整 kernel_size 為奇數: {kernel_size}")

    # 創建結構元素
    if kernel_shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:
        raise ValueError(f"不支援的結構元素形狀: {kernel_shape}")

    # 執行開操作
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return result, kernel


def visualize_comparison(
    original: np.ndarray,
    opened: np.ndarray,
    output_path: str,
    title: str = None,
    dpi: int = 150
):
    """
    視覺化原始影像和開操作結果的比較（灰階）

    Args:
        original: 原始影像
        opened: 開操作結果
        output_path: 輸出路徑
        title: 圖表標題
        dpi: 輸出解析度
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)

    # 原始影像
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original', fontsize=12, weight='bold')
    axes[0].axis('off')

    # 開操作結果
    axes[1].imshow(opened, cmap='gray')
    axes[1].set_title('After Opening', fontsize=12, weight='bold')
    axes[1].axis('off')

    # 差異圖
    diff = cv2.absdiff(original, opened)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference (Hot colormap)', fontsize=12, weight='bold')
    axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, weight='bold')
    else:
        fig.suptitle('Morphological Opening Comparison', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存比較圖: {output_path}")


def visualize_viridis(
    image: np.ndarray,
    output_path: str,
    title: str = None,
    show_colorbar: bool = True,
    dpi: int = 150
):
    """
    使用 Viridis 色彩映射視覺化影像

    Args:
        image: 輸入灰階影像
        output_path: 輸出路徑
        title: 圖表標題
        show_colorbar: 是否顯示 colorbar
        dpi: 輸出解析度
    """
    # 計算合適的圖表尺寸
    height, width = image.shape
    figsize = (width / 100, height / 100)
    max_size = 20
    if figsize[0] > max_size or figsize[1] > max_size:
        scale = max_size / max(figsize)
        figsize = (figsize[0] * scale, figsize[1] * scale)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 繪製影像
    im = ax.imshow(image, cmap='viridis', interpolation='nearest')

    # 添加 colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, label='Intensity',
                           pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

    # 設定標題
    if title:
        ax.set_title(title, fontsize=14, weight='bold', pad=10)

    ax.set_xlabel('X coordinate (pixels)', fontsize=10)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存 Viridis 視覺化: {output_path}")


def visualize_viridis_direct(
    image: np.ndarray,
    output_path: str
):
    """
    直接將 Viridis 色彩映射應用到影像並保存（無邊框）

    Args:
        image: 輸入灰階影像
        output_path: 輸出路徑
    """
    # 標準化到 0-1
    normalized = (image - image.min()) / (image.max() - image.min())
    normalized = np.clip(normalized, 0, 1)

    # 應用 Viridis colormap
    viridis_cmap = cm.get_cmap('viridis')
    colored = viridis_cmap(normalized)

    # 轉換為 8-bit RGB (去掉 alpha 通道)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    # 轉換為 BGR 供 OpenCV 保存
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)

    # 保存
    cv2.imwrite(output_path, colored_bgr)

    print(f"  ✓ 已保存 Viridis 影像（無邊框）: {output_path}")


def visualize_side_by_side_viridis(
    original: np.ndarray,
    opened: np.ndarray,
    output_path: str,
    title: str = None,
    dpi: int = 150
):
    """
    並排視覺化原始影像和開操作結果（Viridis）

    Args:
        original: 原始影像
        opened: 開操作結果
        output_path: 輸出路徑
        title: 圖表標題
        dpi: 輸出解析度
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)

    # 原始影像
    im1 = axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original', fontsize=12, weight='bold')
    axes[0].set_xlabel('X coordinate (pixels)', fontsize=10)
    axes[0].set_ylabel('Y coordinate (pixels)', fontsize=10)
    plt.colorbar(im1, ax=axes[0], label='Intensity', pad=0.02, shrink=0.8)

    # 開操作結果
    im2 = axes[1].imshow(opened, cmap='viridis')
    axes[1].set_title('After Opening', fontsize=12, weight='bold')
    axes[1].set_xlabel('X coordinate (pixels)', fontsize=10)
    axes[1].set_ylabel('Y coordinate (pixels)', fontsize=10)
    plt.colorbar(im2, ax=axes[1], label='Intensity', pad=0.02, shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=14, weight='bold')
    else:
        fig.suptitle('Morphological Opening - Viridis Visualization', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存並排 Viridis 視覺化: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='對影像進行形態學開操作並視覺化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用 - 輸出開操作結果
  python morphology_opening.py \\
      --input image.tif \\
      --output output.tif \\
      --kernel-size 5

  # 輸出到目錄 - 自動生成多個視覺化
  python morphology_opening.py \\
      --input image.tif \\
      --output-dir output/opening/ \\
      --kernel-size 7 \\
      --visualize

  # 調整結構元素形狀和迭代次數
  python morphology_opening.py \\
      --input image.tif \\
      --output-dir output/opening/ \\
      --kernel-size 9 \\
      --kernel-shape ellipse \\
      --iterations 2 \\
      --visualize

  # 只輸出 Viridis 視覺化（無邊框）
  python morphology_opening.py \\
      --input image.tif \\
      --output output_viridis.png \\
      --kernel-size 5 \\
      --viridis-only
        """
    )

    # 必要參數
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='輸入影像路徑'
    )

    # 輸出參數（二選一）
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--output', '-o',
        help='輸出影像路徑（單一檔案）'
    )
    output_group.add_argument(
        '--output-dir', '-d',
        help='輸出目錄（生成多個檔案）'
    )

    # 開操作參數
    parser.add_argument(
        '--kernel-size', '-k',
        type=int,
        default=5,
        help='結構元素大小（預設: 5）'
    )

    parser.add_argument(
        '--kernel-shape',
        choices=['rect', 'ellipse', 'cross'],
        default='ellipse',
        help='結構元素形狀（預設: ellipse）'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='開操作迭代次數（預設: 1）'
    )

    # 視覺化參數
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='生成視覺化圖表（需要 --output-dir）'
    )

    parser.add_argument(
        '--viridis-only',
        action='store_true',
        help='只輸出 Viridis 視覺化影像（無邊框）'
    )

    parser.add_argument(
        '--colorbar',
        action='store_true',
        help='在 Viridis 視覺化中顯示 colorbar'
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
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 錯誤: 輸入檔案不存在: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("形態學開操作")
    print("=" * 60)

    try:
        # 步驟 1: 讀取影像
        print(f"\n[1/3] 讀取影像...")
        print(f"  輸入: {args.input}")
        image = extract_green_channel(args.input)
        print(f"  影像尺寸: {image.shape}")
        print(f"  強度範圍: {image.min()} - {image.max()}")

        # 步驟 2: 執行開操作
        print(f"\n[2/3] 執行形態學開操作...")
        opened, kernel = morphology_opening(
            image=image,
            kernel_size=args.kernel_size,
            kernel_shape=args.kernel_shape,
            iterations=args.iterations
        )

        # 計算統計
        diff = cv2.absdiff(image, opened)
        num_changed = np.count_nonzero(diff)
        percent_changed = (num_changed / diff.size) * 100

        print(f"  ✓ 開操作完成")
        print(f"    改變的像素數: {num_changed} ({percent_changed:.2f}%)")
        print(f"    強度範圍: {opened.min()} - {opened.max()}")

        # 步驟 3: 保存結果
        print(f"\n[3/3] 保存結果...")

        if args.output_dir:
            # 輸出到目錄模式
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = input_path.stem

            # 保存開操作結果
            opened_path = output_dir / f"{base_name}_opened.tif"
            cv2.imwrite(str(opened_path), opened)
            print(f"  ✓ 已保存開操作結果: {opened_path}")

            # 保存差異圖
            diff_path = output_dir / f"{base_name}_diff.tif"
            cv2.imwrite(str(diff_path), diff)
            print(f"  ✓ 已保存差異圖: {diff_path}")

            if args.visualize:
                # 灰階比較圖
                comparison_path = output_dir / f"{base_name}_comparison.png"
                visualize_comparison(
                    original=image,
                    opened=opened,
                    output_path=str(comparison_path),
                    title=args.title,
                    dpi=args.dpi
                )

                # Viridis 原始影像
                viridis_orig_path = output_dir / f"{base_name}_original_viridis.png"
                visualize_viridis(
                    image=image,
                    output_path=str(viridis_orig_path),
                    title="Original (Viridis)",
                    show_colorbar=args.colorbar,
                    dpi=args.dpi
                )

                # Viridis 開操作結果
                viridis_opened_path = output_dir / f"{base_name}_opened_viridis.png"
                visualize_viridis(
                    image=opened,
                    output_path=str(viridis_opened_path),
                    title="After Opening (Viridis)",
                    show_colorbar=args.colorbar,
                    dpi=args.dpi
                )

                # Viridis 並排比較
                viridis_side_path = output_dir / f"{base_name}_viridis_comparison.png"
                visualize_side_by_side_viridis(
                    original=image,
                    opened=opened,
                    output_path=str(viridis_side_path),
                    title=args.title,
                    dpi=args.dpi
                )

                # Viridis 無邊框版本
                viridis_direct_orig = output_dir / f"{base_name}_original_viridis_direct.png"
                visualize_viridis_direct(image, str(viridis_direct_orig))

                viridis_direct_opened = output_dir / f"{base_name}_opened_viridis_direct.png"
                visualize_viridis_direct(opened, str(viridis_direct_opened))

        else:
            # 單一輸出檔案模式
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if args.viridis_only:
                # 只輸出 Viridis 視覺化（無邊框）
                visualize_viridis_direct(opened, args.output)
            else:
                # 輸出開操作結果
                cv2.imwrite(args.output, opened)
                print(f"  ✓ 已保存開操作結果: {args.output}")

        print("\n" + "=" * 60)
        print("✓ 處理完成！")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
