#!/usr/bin/env python3
"""
標記覆蓋視覺化工具

視覺化 pipeline 處理後的最終標記,將表皮標記和真皮標記以不同顏色覆蓋在原始影像上。

功能:
1. 載入原始影像、表皮遮罩和最終標記
2. 區分表皮區域標記和真皮區域標記
3. 使用不同顏色覆蓋標記在原始影像上
4. 生成並排比較圖和統計資訊

使用範例:
    from visualization.visualize_label_overlay import visualize_label_overlay

    # 視覺化最終標記
    visualize_label_overlay(
        original_image_path='data/original.png',
        epidermis_mask_path='data/epidermis_mask.png',
        final_label_path='output/final_label.png',
        output_dir='output/label_overlay'
    )

作者: Generated with Claude Code
日期: 2025-12-09
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 設定 logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    載入影像

    Args:
        image_path: 影像路徑
        grayscale: 是否轉為灰階

    Returns:
        影像陣列

    Raises:
        FileNotFoundError: 檔案不存在
        ValueError: 無法讀取影像
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"影像檔案不存在: {image_path}")

    # 讀取影像
    if grayscale:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"無法讀取影像: {image_path}")

    return image


def separate_labels(
    final_label: np.ndarray,
    epidermis_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    分離表皮標記和真皮標記

    Args:
        final_label: 最終標記 (binary, 0 或 255)
        epidermis_mask: 表皮遮罩 (binary, 0 或 255)

    Returns:
        (epidermis_label, dermis_label) 元組
    """
    # 確保都是二值影像
    _, final_label_bin = cv2.threshold(final_label, 127, 255, cv2.THRESH_BINARY)
    _, epidermis_mask_bin = cv2.threshold(epidermis_mask, 127, 255, cv2.THRESH_BINARY)

    # 表皮標記 = 最終標記 AND 表皮遮罩
    epidermis_label = cv2.bitwise_and(final_label_bin, epidermis_mask_bin)

    # 真皮標記 = 最終標記 AND (NOT 表皮遮罩)
    epidermis_mask_inv = cv2.bitwise_not(epidermis_mask_bin)
    dermis_label = cv2.bitwise_and(final_label_bin, epidermis_mask_inv)

    return epidermis_label, dermis_label


def create_colored_overlay(
    original_image: np.ndarray,
    epidermis_label: np.ndarray,
    dermis_label: np.ndarray,
    epidermis_color: Tuple[int, int, int] = (0, 255, 0),  # 綠色
    dermis_color: Tuple[int, int, int] = (255, 0, 0),     # 藍色
    alpha: float = 0.5
) -> np.ndarray:
    """
    創建彩色覆蓋影像

    Args:
        original_image: 原始影像 (可以是灰階或彩色)
        epidermis_label: 表皮標記
        dermis_label: 真皮標記
        epidermis_color: 表皮標記顏色 (B, G, R)
        dermis_color: 真皮標記顏色 (B, G, R)
        alpha: 透明度 (0-1)

    Returns:
        覆蓋後的彩色影像 (BGR)
    """
    # 如果原始影像是灰階,轉為彩色
    if len(original_image.shape) == 2:
        overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay_image = original_image.copy()

    # 創建彩色遮罩
    color_mask = np.zeros_like(overlay_image)

    # 添加表皮標記 (綠色)
    color_mask[epidermis_label > 0] = epidermis_color

    # 添加真皮標記 (紅色)
    color_mask[dermis_label > 0] = dermis_color

    # 混合原始影像和彩色遮罩
    overlay_image = cv2.addWeighted(overlay_image, 1.0, color_mask, alpha, 0)

    return overlay_image


def calculate_statistics(
    final_label: np.ndarray,
    epidermis_label: np.ndarray,
    dermis_label: np.ndarray,
    epidermis_mask: np.ndarray
) -> Dict[str, Any]:
    """
    計算標記統計資訊

    Args:
        final_label: 最終標記
        epidermis_label: 表皮標記
        dermis_label: 真皮標記
        epidermis_mask: 表皮遮罩

    Returns:
        統計字典
    """
    # 計算像素數量
    total_label_pixels = np.count_nonzero(final_label > 0)
    epidermis_label_pixels = np.count_nonzero(epidermis_label > 0)
    dermis_label_pixels = np.count_nonzero(dermis_label > 0)
    epidermis_mask_pixels = np.count_nonzero(epidermis_mask > 0)
    total_pixels = final_label.size

    # 計算百分比
    label_coverage = (total_label_pixels / total_pixels) * 100
    epidermis_coverage = (epidermis_label_pixels / epidermis_mask_pixels * 100) if epidermis_mask_pixels > 0 else 0

    stats = {
        'total_pixels': total_pixels,
        'total_label_pixels': total_label_pixels,
        'epidermis_label_pixels': epidermis_label_pixels,
        'dermis_label_pixels': dermis_label_pixels,
        'epidermis_mask_pixels': epidermis_mask_pixels,
        'label_coverage_percent': label_coverage,
        'epidermis_coverage_percent': epidermis_coverage,
        'epidermis_ratio': (epidermis_label_pixels / total_label_pixels * 100) if total_label_pixels > 0 else 0,
        'dermis_ratio': (dermis_label_pixels / total_label_pixels * 100) if total_label_pixels > 0 else 0,
    }

    return stats


def create_comparison_visualization(
    original_image: np.ndarray,
    overlay_image: np.ndarray,
    epidermis_label: np.ndarray,
    dermis_label: np.ndarray,
    epidermis_mask: np.ndarray,
    stats: Dict[str, Any],
    output_path: str,
    dpi: int = 150
) -> None:
    """
    創建並排比較圖

    Args:
        original_image: 原始影像
        overlay_image: 覆蓋後的影像
        epidermis_label: 表皮標記
        dermis_label: 真皮標記
        epidermis_mask: 表皮遮罩
        stats: 統計資訊
        output_path: 輸出路徑
        dpi: 解析度
    """
    fig = plt.figure(figsize=(20, 12), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 設定整體標題
    fig.suptitle('Label Overlay Visualization - Epidermis vs Dermis',
                 fontsize=16, fontweight='bold', y=0.98)

    # 第一行：原始影像、覆蓋影像、表皮遮罩
    # 1. 原始影像
    ax1 = fig.add_subplot(gs[0, 0])
    if len(original_image.shape) == 2:
        ax1.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    else:
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. 覆蓋影像
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    ax2.set_title('Overlay (Green: Epidermis, Red: Dermis)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 添加圖例
    green_patch = mpatches.Patch(color='green', label='Epidermis Label', alpha=0.7)
    blue_patch = mpatches.Patch(color='blue', label='Dermis Label', alpha=0.7)
    ax2.legend(handles=[green_patch, blue_patch], loc='upper right', fontsize=10)

    # 3. 表皮遮罩
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(epidermis_mask, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Epidermis Mask', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 第二行：表皮標記、真皮標記、統計資訊
    # 4. 表皮標記
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(epidermis_label, cmap='Greens', vmin=0, vmax=255)
    ax4.set_title(f'Epidermis Label ({stats["epidermis_label_pixels"]:,} pixels)',
                  fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5. 真皮標記
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(dermis_label, cmap='Reds', vmin=0, vmax=255)
    ax5.set_title(f'Dermis Label ({stats["dermis_label_pixels"]:,} pixels)',
                  fontsize=12, fontweight='bold')
    ax5.axis('off')

    # 6. 統計資訊
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    stats_text = f"""
    Label Statistics
    {'=' * 50}

    Image Information
    {'-' * 50}
    Total pixels:              {stats['total_pixels']:,}
    Epidermis mask pixels:     {stats['epidermis_mask_pixels']:,}

    Label Pixels
    {'-' * 50}
    Total label pixels:        {stats['total_label_pixels']:,}
    Epidermis label pixels:    {stats['epidermis_label_pixels']:,}
    Dermis label pixels:       {stats['dermis_label_pixels']:,}

    Coverage
    {'-' * 50}
    Overall label coverage:    {stats['label_coverage_percent']:.2f}%
    Epidermis coverage:        {stats['epidermis_coverage_percent']:.2f}%

    Label Distribution
    {'-' * 50}
    Epidermis ratio:           {stats['epidermis_ratio']:.2f}%
    Dermis ratio:              {stats['dermis_ratio']:.2f}%
    """

    ax6.text(0.05, 0.5, stats_text, fontfamily='monospace', fontsize=10,
             verticalalignment='center', transform=ax6.transAxes)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 比較圖已儲存: {output_path}")


def create_overlay_only_visualization(
    overlay_image: np.ndarray,
    output_path: str,
    dpi: int = 150,
    show_legend: bool = True
) -> None:
    """
    創建單獨的覆蓋層視覺化圖

    Args:
        overlay_image: 覆蓋後的影像
        output_path: 輸出路徑
        dpi: 解析度
        show_legend: 是否顯示圖例
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=dpi)

    # 顯示覆蓋影像
    ax.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    ax.set_title('Label Overlay (Green: Epidermis, Red: Dermis)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    # 添加圖例
    if show_legend:
        green_patch = mpatches.Patch(color='green', label='Epidermis Label', alpha=0.7)
        blue_patch = mpatches.Patch(color='blue', label='Dermis Label', alpha=0.7)
        ax.legend(handles=[green_patch, blue_patch], loc='upper right',
                 fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    logger.info(f"✓ 單獨覆蓋圖已儲存: {output_path}")


def create_side_by_side_comparison(
    original_image: np.ndarray,
    overlay_image: np.ndarray,
    output_path: str,
    dpi: int = 150
) -> None:
    """
    創建簡化版的並排比較圖

    Args:
        original_image: 原始影像
        overlay_image: 覆蓋後的影像
        output_path: 輸出路徑
        dpi: 解析度
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)

    fig.suptitle('Original vs Overlay', fontsize=16, fontweight='bold')

    # 原始影像
    if len(original_image.shape) == 2:
        axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    else:
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 覆蓋影像
    axes[1].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Label Overlay', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 添加圖例
    green_patch = mpatches.Patch(color='green', label='Epidermis Label', alpha=0.7)
    red_patch = mpatches.Patch(color='red', label='Dermis Label', alpha=0.7)
    axes[1].legend(handles=[green_patch, red_patch], loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 並排比較圖已儲存: {output_path}")


def visualize_label_overlay(
    original_image_path: str,
    epidermis_mask_path: str,
    final_label_path: str,
    output_dir: str,
    epidermis_color: Tuple[int, int, int] = (0, 255, 0),  # 綠色 (BGR)
    dermis_color: Tuple[int, int, int] = (255, 0, 0),     # 紅色 (BGR)
    alpha: float = 0.5,
    dpi: int = 150
) -> Dict[str, Any]:
    """
    視覺化標記覆蓋,區分表皮和真皮標記

    Args:
        original_image_path: 原始影像路徑
        epidermis_mask_path: 表皮遮罩路徑
        final_label_path: 最終標記路徑
        output_dir: 輸出目錄
        epidermis_color: 表皮標記顏色 (B, G, R)
        dermis_color: 真皮標記顏色 (B, G, R)
        alpha: 透明度 (0-1)
        dpi: 輸出解析度

    Returns:
        包含統計資訊的字典
    """
    logger.info("=" * 80)
    logger.info("標記覆蓋視覺化")
    logger.info("=" * 80)

    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 步驟 1: 載入影像
    logger.info("\n步驟 1: 載入影像")
    logger.info(f"  原始影像: {original_image_path}")
    logger.info(f"  表皮遮罩: {epidermis_mask_path}")
    logger.info(f"  最終標記: {final_label_path}")

    original_image = load_image(original_image_path, grayscale=True)
    epidermis_mask = load_image(epidermis_mask_path, grayscale=True)
    final_label = load_image(final_label_path, grayscale=True)

    logger.info(f"  影像尺寸: {original_image.shape}")

    # 步驟 2: 分離表皮和真皮標記
    logger.info("\n步驟 2: 分離表皮和真皮標記")
    epidermis_label, dermis_label = separate_labels(final_label, epidermis_mask)

    epidermis_pixels = np.count_nonzero(epidermis_label > 0)
    dermis_pixels = np.count_nonzero(dermis_label > 0)

    logger.info(f"  表皮標記像素: {epidermis_pixels:,}")
    logger.info(f"  真皮標記像素: {dermis_pixels:,}")

    # 步驟 3: 創建彩色覆蓋影像
    logger.info("\n步驟 3: 創建彩色覆蓋影像")
    logger.info(f"  表皮顏色 (BGR): {epidermis_color}")
    logger.info(f"  真皮顏色 (BGR): {dermis_color}")
    logger.info(f"  透明度: {alpha}")

    overlay_image = create_colored_overlay(
        original_image,
        epidermis_label,
        dermis_label,
        epidermis_color=epidermis_color,
        dermis_color=dermis_color,
        alpha=alpha
    )

    logger.info("  ✓ 覆蓋影像創建完成")

    # 步驟 4: 計算統計資訊
    logger.info("\n步驟 4: 計算統計資訊")
    stats = calculate_statistics(final_label, epidermis_label, dermis_label, epidermis_mask)

    logger.info(f"  總標記像素: {stats['total_label_pixels']:,}")
    logger.info(f"  表皮標記: {stats['epidermis_label_pixels']:,} ({stats['epidermis_ratio']:.2f}%)")
    logger.info(f"  真皮標記: {stats['dermis_label_pixels']:,} ({stats['dermis_ratio']:.2f}%)")
    logger.info(f"  整體覆蓋率: {stats['label_coverage_percent']:.2f}%")
    logger.info(f"  表皮覆蓋率: {stats['epidermis_coverage_percent']:.2f}%")

    # 步驟 5: 創建詳細比較圖
    logger.info("\n步驟 5: 創建詳細比較圖")
    comparison_output = output_path / "01_detailed_comparison.png"
    create_comparison_visualization(
        original_image,
        overlay_image,
        epidermis_label,
        dermis_label,
        epidermis_mask,
        stats,
        str(comparison_output),
        dpi
    )

    # 步驟 6: 創建單獨覆蓋層圖
    logger.info("\n步驟 6: 創建單獨覆蓋層圖")
    overlay_only_output = output_path / "02_overlay_only.png"
    create_overlay_only_visualization(
        overlay_image,
        str(overlay_only_output),
        dpi,
        show_legend=True
    )

    # 步驟 7: 創建簡化並排比較圖
    logger.info("\n步驟 7: 創建簡化並排比較圖")
    sidebyside_output = output_path / "03_side_by_side.png"
    create_side_by_side_comparison(
        original_image,
        overlay_image,
        str(sidebyside_output),
        dpi
    )

    # 步驟 8: 保存處理後的影像
    logger.info("\n步驟 8: 保存處理後的影像")
    overlay_save = output_path / "overlay.png"
    epidermis_label_save = output_path / "epidermis_label.png"
    dermis_label_save = output_path / "dermis_label.png"

    cv2.imwrite(str(overlay_save), overlay_image)
    cv2.imwrite(str(epidermis_label_save), epidermis_label)
    cv2.imwrite(str(dermis_label_save), dermis_label)

    logger.info(f"  ✓ 已儲存覆蓋影像: {overlay_save}")
    logger.info(f"  ✓ 已儲存表皮標記: {epidermis_label_save}")
    logger.info(f"  ✓ 已儲存真皮標記: {dermis_label_save}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ 視覺化完成！")
    logger.info("=" * 80)
    logger.info(f"\n輸出目錄: {output_path}")
    logger.info("生成的檔案:")
    logger.info("  1. 01_detailed_comparison.png - 詳細比較圖 (3x2 網格)")
    logger.info("  2. 02_overlay_only.png        - 單獨覆蓋層圖 (含圖例)")
    logger.info("  3. 03_side_by_side.png        - 簡化並排比較")
    logger.info("  4. overlay.png                - 純覆蓋影像 (無邊框、標題)")
    logger.info("  5. epidermis_label.png        - 表皮標記")
    logger.info("  6. dermis_label.png           - 真皮標記")

    return {
        'original_image': original_image,
        'overlay_image': overlay_image,
        'epidermis_label': epidermis_label,
        'dermis_label': dermis_label,
        'statistics': stats,
        'output_dir': str(output_path)
    }


# ============================================================================
# 主程式與使用範例
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：視覺化標記覆蓋
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='視覺化標記覆蓋,區分表皮和真皮標記',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用
  python visualize_label_overlay.py \\
      --original data/original_image.png \\
      --mask data/epidermis_mask.png \\
      --label output/final_label.png \\
      --output-dir output/label_overlay

  # 自訂顏色 (BGR 格式)
  python visualize_label_overlay.py \\
      --original data/original_image.png \\
      --mask data/epidermis_mask.png \\
      --label output/final_label.png \\
      --output-dir output/label_overlay \\
      --epidermis-color 0 255 255 \\
      --dermis-color 255 0 255

  # 調整透明度
  python visualize_label_overlay.py \\
      --original data/original_image.png \\
      --mask data/epidermis_mask.png \\
      --label output/final_label.png \\
      --output-dir output/label_overlay \\
      --alpha 0.7

生成的檔案:
  1. 01_detailed_comparison.png - 詳細比較圖,包含原始影像、覆蓋影像、
                                   表皮遮罩、表皮標記、真皮標記和統計資訊
  2. 02_overlay_only.png        - 單獨覆蓋層圖 (含標題和圖例)
  3. 03_side_by_side.png        - 簡化版並排比較 (原始 vs 覆蓋)
  4. overlay.png                - 純覆蓋影像 (無邊框、無標題,可直接使用)
  5. epidermis_label.png        - 分離出的表皮標記
  6. dermis_label.png           - 分離出的真皮標記

顏色說明:
  - 預設表皮標記為綠色 (0, 255, 0)
  - 預設真皮標記為藍色 (255, 0, 0)
  - 顏色格式為 BGR (Blue, Green, Red)
        """
    )

    parser.add_argument(
        '--original', '-o',
        required=True,
        help='原始影像路徑'
    )

    parser.add_argument(
        '--mask', '-m',
        required=True,
        help='表皮遮罩路徑'
    )

    parser.add_argument(
        '--label', '-l',
        required=True,
        help='最終標記路徑'
    )

    parser.add_argument(
        '--output-dir', '-d',
        required=True,
        help='輸出目錄'
    )

    parser.add_argument(
        '--epidermis-color',
        type=int,
        nargs=3,
        default=[0, 255, 0],
        metavar=('B', 'G', 'R'),
        help='表皮標記顏色 BGR 格式 (預設: 0 255 0 綠色)'
    )

    parser.add_argument(
        '--dermis-color',
        type=int,
        nargs=3,
        default=[255, 0, 0],
        metavar=('B', 'G', 'R'),
        help='真皮標記顏色 BGR 格式 (預設: 255 0 0 紅色)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='透明度 (0-1, 預設: 0.5)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='輸出解析度 (預設: 150)'
    )

    args = parser.parse_args()

    try:
        result = visualize_label_overlay(
            original_image_path=args.original,
            epidermis_mask_path=args.mask,
            final_label_path=args.label,
            output_dir=args.output_dir,
            epidermis_color=tuple(args.epidermis_color),
            dermis_color=tuple(args.dermis_color),
            alpha=args.alpha,
            dpi=args.dpi
        )

        print("\n" + "=" * 80)
        print("✓ 處理完成！")
        print("=" * 80)
        print(f"\n輸出目錄: {result['output_dir']}")
        print("\n統計摘要:")
        print(f"  總標記像素:     {result['statistics']['total_label_pixels']:,}")
        print(f"  表皮標記像素:   {result['statistics']['epidermis_label_pixels']:,}")
        print(f"  真皮標記像素:   {result['statistics']['dermis_label_pixels']:,}")
        print(f"\n  整體覆蓋率:     {result['statistics']['label_coverage_percent']:.2f}%")
        print(f"  表皮覆蓋率:     {result['statistics']['epidermis_coverage_percent']:.2f}%")
        print(f"\n  標記分布:")
        print(f"    表皮比例:     {result['statistics']['epidermis_ratio']:.2f}%")
        print(f"    真皮比例:     {result['statistics']['dermis_ratio']:.2f}%")

    except Exception as e:
        logger.error(f"處理失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
