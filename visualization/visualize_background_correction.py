#!/usr/bin/env python3
"""
背景光修正視覺化工具

視覺化 rolling ball background correction 前後的直方圖變化：
- 原始影像的直方圖
- 背景估計的直方圖
- 背景修正後的直方圖
- 統計資訊（亮度分布、對比度變化等）

功能：
1. 自動執行 rolling ball background correction
2. 視覺化修正前後的直方圖
3. 生成並排比較圖
4. 提供詳細統計資訊

使用範例:
    from visualization.visualize_background_correction import visualize_background_correction

    # 執行背景光修正並視覺化
    visualize_background_correction(
        input_path='original_image.png',
        output_dir='output/background_viz',
        radius=12,
        light_background=True
    )

作者: Generated with Claude Code
日期: 2025-12-08
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.background_correction import rolling_ball_background

# 設定 logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_grayscale_image(image_path: str) -> np.ndarray:
    """
    載入灰階影像

    Args:
        image_path: 影像路徑

    Returns:
        灰階影像 (uint8)

    Raises:
        FileNotFoundError: 檔案不存在
        ValueError: 無法讀取影像
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"影像檔案不存在: {image_path}")

    # 讀取影像
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"無法讀取影像: {image_path}")

    return image


def compute_histogram_statistics(image: np.ndarray) -> Dict[str, float]:
    """
    計算影像的直方圖統計資訊

    Args:
        image: 輸入灰階影像

    Returns:
        統計字典包含：
        - mean: 平均亮度
        - std: 標準差
        - min: 最小值
        - max: 最大值
        - median: 中位數
        - q25: 第一四分位數
        - q75: 第三四分位數
        - contrast: 對比度 (std / mean)
        - dynamic_range: 動態範圍 (max - min)
    """
    flat = image.flatten()

    stats = {
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
        'median': float(np.median(flat)),
        'q25': float(np.percentile(flat, 25)),
        'q75': float(np.percentile(flat, 75)),
    }

    # 計算對比度（避免除以零）
    if stats['mean'] > 0:
        stats['contrast'] = stats['std'] / stats['mean']
    else:
        stats['contrast'] = 0.0

    stats['dynamic_range'] = stats['max'] - stats['min']

    return stats


def plot_histogram_with_stats(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    color: str = 'blue',
    show_stats: bool = True,
    y_limit_scale: float = 1.5
) -> None:
    """
    在給定的 axes 上繪製直方圖和統計資訊

    Args:
        ax: matplotlib axes
        image: 輸入影像
        title: 圖表標題
        color: 直方圖顏色
        show_stats: 是否顯示統計資訊
        y_limit_scale: y 軸最大值為數據最大值的倍數（預設: 1.5）
    """
    # 計算直方圖
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # 繪製直方圖
    ax.fill_between(bins[:-1], hist, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Pixel Intensity', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 255)

    # 設定 y 軸最大值為數據最大值的 y_limit_scale 倍
    ax.set_ylim(0, 250000)

    if show_stats:
        # 計算統計資訊
        stats = compute_histogram_statistics(image)

        # 標記平均值
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.1f}")

        # 標記中位數
        ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.1f}")

        # 標記四分位數
        ax.axvline(stats['q25'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f"Q25: {stats['q25']:.1f}")
        ax.axvline(stats['q75'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f"Q75: {stats['q75']:.1f}")

        ax.legend(loc='upper right', fontsize=9)


def create_comparison_visualization(
    original: np.ndarray,
    background: np.ndarray,
    corrected: np.ndarray,
    output_path: str,
    dpi: int = 150
) -> None:
    """
    創建並排影像和直方圖比較圖

    Args:
        original: 原始影像
        background: 估計的背景
        corrected: 背景修正後的影像
        output_path: 輸出路徑
        dpi: 解析度
    """
    fig = plt.figure(figsize=(18, 10), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 設定整體標題
    fig.suptitle('Background Correction - Image and Histogram Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    # 第一行：影像
    # 原始影像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 背景估計
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(background, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Estimated Background', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 背景修正後
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(corrected, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Background Corrected', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 第二行：直方圖
    # 原始影像直方圖
    ax4 = fig.add_subplot(gs[1, 0])
    plot_histogram_with_stats(ax4, original, 'Original Histogram', color='blue')

    # 背景估計直方圖
    ax5 = fig.add_subplot(gs[1, 1])
    plot_histogram_with_stats(ax5, background, 'Background Histogram', color='gray')

    # 背景修正後直方圖
    ax6 = fig.add_subplot(gs[1, 2])
    plot_histogram_with_stats(ax6, corrected, 'Corrected Histogram', color='green')

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 比較圖已儲存: {output_path}")


def create_overlay_histogram(
    original: np.ndarray,
    corrected: np.ndarray,
    output_path: str,
    dpi: int = 150
) -> None:
    """
    創建疊加直方圖比較圖

    Args:
        original: 原始影像
        corrected: 背景修正後的影像
        output_path: 輸出路徑
        dpi: 解析度
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=dpi)

    # 計算直方圖
    hist_original, bins = np.histogram(original.flatten(), bins=256, range=(0, 256))
    hist_corrected, _ = np.histogram(corrected.flatten(), bins=256, range=(0, 256))

    # 繪製疊加直方圖
    ax.fill_between(bins[:-1], hist_original, alpha=0.5, color='blue',
                     edgecolor='darkblue', linewidth=1, label='Original')
    ax.fill_between(bins[:-1], hist_corrected, alpha=0.5, color='green',
                     edgecolor='darkgreen', linewidth=1, label='Corrected')

    # 計算統計
    stats_orig = compute_histogram_statistics(original)
    stats_corr = compute_histogram_statistics(corrected)

    # 標記平均值
    ax.axvline(stats_orig['mean'], color='blue', linestyle='--', linewidth=2,
               label=f"Original Mean: {stats_orig['mean']:.1f}")
    ax.axvline(stats_corr['mean'], color='green', linestyle='--', linewidth=2,
               label=f"Corrected Mean: {stats_corr['mean']:.1f}")

    ax.set_xlabel('Pixel Intensity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Histogram Overlay: Original vs Corrected', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 255)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 250000)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 疊加直方圖已儲存: {output_path}")


def create_statistics_summary(
    original: np.ndarray,
    background: np.ndarray,
    corrected: np.ndarray,
    output_path: str,
    config: Dict[str, Any],
    dpi: int = 150
) -> None:
    """
    創建統計摘要圖表

    Args:
        original: 原始影像
        background: 估計的背景
        corrected: 背景修正後的影像
        output_path: 輸出路徑
        config: 背景修正配置
        dpi: 解析度
    """
    # 計算統計
    stats_orig = compute_histogram_statistics(original)
    stats_bg = compute_histogram_statistics(background)
    stats_corr = compute_histogram_statistics(corrected)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=dpi)
    fig.suptitle('Background Correction - Statistical Summary', fontsize=16, fontweight='bold')

    # 圖 1: 平均亮度比較
    ax1 = axes[0, 0]
    categories = ['Original', 'Background', 'Corrected']
    means = [stats_orig['mean'], stats_bg['mean'], stats_corr['mean']]
    colors = ['blue', 'gray', 'green']

    bars = ax1.bar(categories, means, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Intensity', fontsize=12)
    ax1.set_title('Mean Intensity Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加數值標籤
    for bar, val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    # 圖 2: 動態範圍比較
    ax2 = axes[0, 1]
    dynamic_ranges = [stats_orig['dynamic_range'], stats_bg['dynamic_range'],
                      stats_corr['dynamic_range']]

    bars = ax2.bar(categories, dynamic_ranges, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Dynamic Range', fontsize=12)
    ax2.set_title('Dynamic Range Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, dynamic_ranges):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    # 圖 3: 標準差比較
    ax3 = axes[1, 0]
    stds = [stats_orig['std'], stats_bg['std'], stats_corr['std']]

    bars = ax3.bar(categories, stds, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Standard Deviation', fontsize=12)
    ax3.set_title('Standard Deviation Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    # 圖 4: 詳細統計資訊
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Background Correction Configuration
    {'=' * 55}

    Radius:              {config['radius']}
    Light Background:    {config['light_background']}
    Method:              {config['method']}
    Smoothing:           {config['smoothing']}


    Original Image Statistics
    {'=' * 55}
    Mean:                {stats_orig['mean']:.2f}
    Median:              {stats_orig['median']:.2f}
    Std Dev:             {stats_orig['std']:.2f}
    Min:                 {stats_orig['min']:.2f}
    Max:                 {stats_orig['max']:.2f}
    Q25:                 {stats_orig['q25']:.2f}
    Q75:                 {stats_orig['q75']:.2f}
    Contrast:            {stats_orig['contrast']:.4f}
    Dynamic Range:       {stats_orig['dynamic_range']:.2f}


    Corrected Image Statistics
    {'=' * 55}
    Mean:                {stats_corr['mean']:.2f}
    Median:              {stats_corr['median']:.2f}
    Std Dev:             {stats_corr['std']:.2f}
    Min:                 {stats_corr['min']:.2f}
    Max:                 {stats_corr['max']:.2f}
    Q25:                 {stats_corr['q25']:.2f}
    Q75:                 {stats_corr['q75']:.2f}
    Contrast:            {stats_corr['contrast']:.4f}
    Dynamic Range:       {stats_corr['dynamic_range']:.2f}


    Changes
    {'=' * 55}
    Mean change:         {stats_corr['mean'] - stats_orig['mean']:+.2f}
    Std Dev change:      {stats_corr['std'] - stats_orig['std']:+.2f}
    Contrast change:     {stats_corr['contrast'] - stats_orig['contrast']:+.4f}
    """

    ax4.text(0.05, 0.5, stats_text, fontfamily='monospace', fontsize=9,
             verticalalignment='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 統計摘要已儲存: {output_path}")


def visualize_background_correction(
    input_path: str,
    output_dir: str,
    radius: int = 12,
    light_background: bool = True,
    smoothing: bool = False,
    smoothing_sigma: float = 2.0,
    method: str = 'morphology',
    mask_path: Optional[str] = None,
    dpi: int = 150
) -> Dict[str, Any]:
    """
    執行背景光修正並視覺化直方圖變化

    Args:
        input_path: 輸入影像路徑
        output_dir: 輸出目錄
        radius: Rolling ball 半徑
        light_background: 是否為亮背景
        smoothing: 是否進行平滑化
        smoothing_sigma: 平滑化 sigma 值
        method: 背景估計方法 ('morphology' 或 'rolling_ball')
        mask_path: 可選的遮罩路徑（只分析遮罩區域）
        dpi: 輸出解析度

    Returns:
        包含統計資訊的字典
    """
    logger.info("=" * 80)
    logger.info("背景光修正視覺化")
    logger.info("=" * 80)

    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 步驟 1: 載入原始影像
    logger.info("\n步驟 1: 載入原始影像")
    logger.info(f"  輸入: {input_path}")
    original = load_grayscale_image(input_path)
    logger.info(f"  影像尺寸: {original.shape}")

    # 載入遮罩（如果有）
    mask = None
    if mask_path:
        logger.info(f"  遮罩: {mask_path}")
        mask = load_grayscale_image(mask_path)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        logger.info(f"  遮罩區域像素: {np.count_nonzero(mask == 255):,}")

    # 步驟 2: 執行背景光修正
    logger.info(f"\n步驟 2: 執行背景光修正")
    logger.info(f"  Radius: {radius}")
    logger.info(f"  Light background: {light_background}")
    logger.info(f"  Method: {method}")
    logger.info(f"  Smoothing: {smoothing}")

    config = {
        'radius': radius,
        'light_background': light_background,
        'method': method,
        'smoothing': smoothing,
        'smoothing_sigma': smoothing_sigma
    }

    corrected = rolling_ball_background(
        original,
        radius=radius,
        light_background=light_background,
        smoothing=smoothing,
        smoothing_sigma=smoothing_sigma,
        method=method
    )
    logger.info(f"  ✓ 背景修正完成")

    # 步驟 3: 估計背景（用於視覺化）
    logger.info("\n步驟 3: 重新估計背景用於視覺化")
    # 使用相同參數估計背景
    from src.preprocessing.background_correction import (
        _estimate_background_morphology,
        _estimate_background_rolling_ball
    )
    from src.preprocessing.utils import normalize_image

    original_uint8, _, _ = normalize_image(original)

    if method == 'morphology':
        background = _estimate_background_morphology(original_uint8, radius, light_background)
    else:
        background = _estimate_background_rolling_ball(original_uint8, radius, light_background)

    logger.info(f"  ✓ 背景估計完成")

    # 如果有遮罩，只分析遮罩區域
    if mask is not None:
        logger.info("\n應用遮罩...")
        original_masked = cv2.bitwise_and(original, original, mask=mask)
        corrected_masked = cv2.bitwise_and(corrected, corrected, mask=mask)
        background_masked = cv2.bitwise_and(background, background, mask=mask)

        # 只計算遮罩內的統計
        analysis_original = original[mask == 255]
        analysis_corrected = corrected[mask == 255]
        analysis_background = background[mask == 255]
    else:
        analysis_original = original
        analysis_corrected = corrected
        analysis_background = background

    # 步驟 4: 計算統計資訊
    logger.info("\n步驟 4: 計算統計資訊")
    stats_orig = compute_histogram_statistics(analysis_original)
    stats_corr = compute_histogram_statistics(analysis_corrected)

    logger.info(f"  原始影像:")
    logger.info(f"    平均亮度: {stats_orig['mean']:.2f}")
    logger.info(f"    標準差:   {stats_orig['std']:.2f}")
    logger.info(f"    對比度:   {stats_orig['contrast']:.4f}")

    logger.info(f"  修正後影像:")
    logger.info(f"    平均亮度: {stats_corr['mean']:.2f}")
    logger.info(f"    標準差:   {stats_corr['std']:.2f}")
    logger.info(f"    對比度:   {stats_corr['contrast']:.4f}")

    logger.info(f"  變化:")
    logger.info(f"    亮度變化: {stats_corr['mean'] - stats_orig['mean']:+.2f}")
    logger.info(f"    對比度變化: {stats_corr['contrast'] - stats_orig['contrast']:+.4f}")

    # 步驟 5: 創建並排比較圖
    logger.info("\n步驟 5: 創建並排比較圖")
    comparison_output = output_path / "01_comparison.png"
    create_comparison_visualization(
        original, background, corrected,
        str(comparison_output), dpi
    )

    # 步驟 6: 創建疊加直方圖
    logger.info("\n步驟 6: 創建疊加直方圖")
    overlay_output = output_path / "02_histogram_overlay.png"
    create_overlay_histogram(
        analysis_original if mask is None else original[mask == 255].reshape(-1, 1),
        analysis_corrected if mask is None else corrected[mask == 255].reshape(-1, 1),
        str(overlay_output), dpi
    )

    # 步驟 7: 創建統計摘要
    logger.info("\n步驟 7: 創建統計摘要")
    stats_output = output_path / "03_statistics.png"
    create_statistics_summary(
        analysis_original if mask is None else original,
        analysis_background if mask is None else background,
        analysis_corrected if mask is None else corrected,
        str(stats_output), config, dpi
    )

    # 步驟 8: 保存處理後的影像
    logger.info("\n步驟 8: 保存處理後的影像")
    original_save = output_path / "original.png"
    background_save = output_path / "background.png"
    corrected_save = output_path / "corrected.png"

    cv2.imwrite(str(original_save), original)
    cv2.imwrite(str(background_save), background)
    cv2.imwrite(str(corrected_save), corrected)

    logger.info(f"  ✓ 已儲存原始影像: {original_save}")
    logger.info(f"  ✓ 已儲存背景估計: {background_save}")
    logger.info(f"  ✓ 已儲存修正結果: {corrected_save}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ 視覺化完成！")
    logger.info("=" * 80)
    logger.info(f"\n輸出目錄: {output_path}")
    logger.info("生成的檔案:")
    logger.info("  1. 01_comparison.png        - 影像和直方圖並排比較")
    logger.info("  2. 02_histogram_overlay.png - 疊加直方圖")
    logger.info("  3. 03_statistics.png        - 統計摘要")
    logger.info("  4. original.png             - 原始影像")
    logger.info("  5. background.png           - 背景估計")
    logger.info("  6. corrected.png            - 背景修正結果")

    return {
        'original': original,
        'background': background,
        'corrected': corrected,
        'statistics': {
            'original': stats_orig,
            'corrected': stats_corr,
            'change': {
                'mean': stats_corr['mean'] - stats_orig['mean'],
                'std': stats_corr['std'] - stats_orig['std'],
                'contrast': stats_corr['contrast'] - stats_orig['contrast']
            }
        },
        'config': config,
        'output_dir': str(output_path)
    }


# ============================================================================
# 主程式與使用範例
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：執行背景光修正並視覺化直方圖
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='執行 rolling ball background correction 並視覺化直方圖變化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用 - 亮背景（如明場顯微鏡）
  python visualize_background_correction.py \\
      --input original_image.png \\
      --output-dir output/background_viz \\
      --radius 12 \\
      --light-background

  # 暗背景（如螢光顯微鏡）
  python visualize_background_correction.py \\
      --input fluorescence.png \\
      --output-dir output/background_viz \\
      --radius 30 \\
      --no-light-background

  # 使用遮罩只分析特定區域
  python visualize_background_correction.py \\
      --input image.png \\
      --output-dir output/background_viz \\
      --mask epidermis_mask.png \\
      --radius 12

  # 使用平滑化和不同方法
  python visualize_background_correction.py \\
      --input image.png \\
      --output-dir output/background_viz \\
      --radius 50 \\
      --method rolling_ball \\
      --smoothing \\
      --smoothing-sigma 3.0

生成的檔案:
  1. 01_comparison.png        - 影像和直方圖並排比較（3x2 網格）
  2. 02_histogram_overlay.png - 原始 vs 修正後的疊加直方圖
  3. 03_statistics.png        - 詳細統計摘要和配置資訊
  4. original.png             - 原始影像
  5. background.png           - 估計的背景影像
  6. corrected.png            - 背景修正後的影像
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='輸入影像路徑'
    )

    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='輸出目錄'
    )

    parser.add_argument(
        '--radius', '-r',
        type=int,
        default=12,
        help='Rolling ball 半徑（預設: 12）'
    )

    parser.add_argument(
        '--light-background',
        action='store_true',
        default=True,
        help='影像有亮背景（預設: True）'
    )

    parser.add_argument(
        '--no-light-background',
        action='store_false',
        dest='light_background',
        help='影像有暗背景（設定此選項以關閉 light_background）'
    )

    parser.add_argument(
        '--method', '-m',
        choices=['morphology', 'rolling_ball'],
        default='morphology',
        help='背景估計方法（預設: morphology）'
    )

    parser.add_argument(
        '--smoothing',
        action='store_true',
        help='對背景估計進行平滑化'
    )

    parser.add_argument(
        '--smoothing-sigma',
        type=float,
        default=2.0,
        help='平滑化 sigma 值（預設: 2.0）'
    )

    parser.add_argument(
        '--mask',
        help='遮罩影像路徑（可選，只分析遮罩區域）'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='輸出解析度（預設: 150）'
    )

    args = parser.parse_args()

    try:
        result = visualize_background_correction(
            input_path=args.input,
            output_dir=args.output_dir,
            radius=args.radius,
            light_background=args.light_background,
            smoothing=args.smoothing,
            smoothing_sigma=args.smoothing_sigma,
            method=args.method,
            mask_path=args.mask,
            dpi=args.dpi
        )

        print("\n" + "=" * 80)
        print("✓ 處理完成！")
        print("=" * 80)
        print(f"\n輸出目錄: {result['output_dir']}")
        print("\n統計摘要:")
        print(f"  原始影像:")
        print(f"    平均亮度:   {result['statistics']['original']['mean']:.2f}")
        print(f"    標準差:     {result['statistics']['original']['std']:.2f}")
        print(f"    對比度:     {result['statistics']['original']['contrast']:.4f}")
        print(f"\n  修正後影像:")
        print(f"    平均亮度:   {result['statistics']['corrected']['mean']:.2f}")
        print(f"    標準差:     {result['statistics']['corrected']['std']:.2f}")
        print(f"    對比度:     {result['statistics']['corrected']['contrast']:.4f}")
        print(f"\n  變化:")
        print(f"    亮度變化:   {result['statistics']['change']['mean']:+.2f}")
        print(f"    標準差變化: {result['statistics']['change']['std']:+.2f}")
        print(f"    對比度變化: {result['statistics']['change']['contrast']:+.4f}")

    except Exception as e:
        logger.error(f"處理失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
