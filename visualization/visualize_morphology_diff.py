#!/usr/bin/env python3
"""
形態學操作前後差異視覺化工具

視覺化形態學操作（closing + opening）前後的變化：
- 被消除的區域使用紅色標注
- 擴增的區域使用黃色標注
- 保持不變的區域保持原樣

功能：
1. 自動執行 closing + opening 操作
2. 視覺化每個步驟的差異
3. 生成彩色疊加視覺化
4. 提供統計資訊（消除/擴增的像素數量）

使用範例:
    from visualization.visualize_morphology_diff import visualize_morphology_pipeline

    # 執行 closing + opening 並視覺化
    visualize_morphology_pipeline(
        input_path='annotation.png',
        output_dir='output/morphology_viz',
        closing_kernel=5,
        opening_kernel=3
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

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.morphology import morphological_closing, morphological_opening

# 設定 logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_binary_image(image_path: str) -> np.ndarray:
    """
    載入並二值化影像

    Args:
        image_path: 影像路徑

    Returns:
        二值影像 (0 或 255)

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

    # 二值化（確保只有 0 和 255）
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    return binary


def compute_morphology_diff(
    before: np.ndarray,
    after: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    計算形態學操作前後的差異

    Args:
        before: 操作前的二值影像
        after: 操作後的二值影像

    Returns:
        (removed_mask, added_mask, statistics)
        - removed_mask: 被消除的區域 (255 表示被消除)
        - added_mask: 擴增的區域 (255 表示擴增)
        - statistics: 統計字典

    Raises:
        ValueError: 影像尺寸不匹配
    """
    if before.shape != after.shape:
        raise ValueError(
            f"影像尺寸不匹配: before={before.shape}, after={after.shape}"
        )

    # 確保是二值影像
    before_binary = (before > 127).astype(np.uint8) * 255
    after_binary = (after > 127).astype(np.uint8) * 255

    # 計算差異
    # 被消除: 原本是白色(255)，現在變黑色(0)
    removed_mask = np.logical_and(before_binary == 255, after_binary == 0).astype(np.uint8) * 255

    # 擴增: 原本是黑色(0)，現在變白色(255)
    added_mask = np.logical_and(before_binary == 0, after_binary == 255).astype(np.uint8) * 255

    # 統計
    total_pixels = before.shape[0] * before.shape[1]
    before_white = np.count_nonzero(before_binary == 255)
    after_white = np.count_nonzero(after_binary == 255)
    removed_count = np.count_nonzero(removed_mask == 255)
    added_count = np.count_nonzero(added_mask == 255)

    statistics = {
        'total_pixels': total_pixels,
        'before_white_pixels': before_white,
        'after_white_pixels': after_white,
        'removed_pixels': removed_count,
        'added_pixels': added_count,
        'removed_percent': (removed_count / total_pixels) * 100,
        'added_percent': (added_count / total_pixels) * 100,
        'net_change': after_white - before_white,
        'net_change_percent': ((after_white - before_white) / total_pixels) * 100
    }

    return removed_mask, added_mask, statistics


def create_diff_overlay(
    before: np.ndarray,
    after: np.ndarray,
    removed_mask: np.ndarray,
    added_mask: np.ndarray,
    background: Optional[np.ndarray] = None,
    removed_color: Tuple[int, int, int] = (0, 0, 255),    # BGR: 紅色
    added_color: Tuple[int, int, int] = (0, 255, 255),    # BGR: 黃色
    unchanged_color: Tuple[int, int, int] = (200, 200, 200),  # BGR: 淺灰色
    alpha: float = 0.6
) -> np.ndarray:
    """
    創建差異疊加視覺化影像

    Args:
        before: 操作前的二值影像
        after: 操作後的二值影像
        removed_mask: 被消除的區域
        added_mask: 擴增的區域
        background: 背景影像（可選）
        removed_color: 被消除區域的顏色 (BGR)
        added_color: 擴增區域的顏色 (BGR)
        unchanged_color: 不變區域的顏色 (BGR)
        alpha: 透明度 (0-1)

    Returns:
        彩色疊加影像 (BGR)
    """
    h, w = before.shape

    # 準備背景影像
    if background is not None:
        if len(background.shape) == 2:
            base_image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        else:
            base_image = background.copy()
    else:
        # 使用深灰色背景
        base_image = np.full((h, w, 3), 50, dtype=np.uint8)

    # 創建疊加層
    overlay = base_image.copy().astype(np.float32)

    # 標記不變的白色區域（淺灰色）
    unchanged_mask = np.logical_and(before == 255, after == 255)
    overlay[unchanged_mask] = unchanged_color

    # 標記被消除的區域（紅色）
    removed_pixels = (removed_mask == 255)
    overlay[removed_pixels] = removed_color

    # 標記擴增的區域（黃色）
    added_pixels = (added_mask == 255)
    overlay[added_pixels] = added_color

    # 混合
    result = cv2.addWeighted(
        base_image.astype(np.float32),
        1 - alpha,
        overlay,
        alpha,
        0
    )

    return result.astype(np.uint8)


def create_statistics_plot(
    statistics: Dict[str, Any],
    output_path: str,
    operation_name: str = "Morphological Operation"
) -> None:
    """
    創建統計圖表

    Args:
        statistics: 統計字典
        output_path: 輸出路徑
        operation_name: 操作名稱
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{operation_name} - Statistics', fontsize=16, fontweight='bold')

    # 左圖：像素數量比較
    ax1 = axes[0]
    categories = ['Before', 'After', 'Removed', 'Added']
    values = [
        statistics['before_white_pixels'],
        statistics['after_white_pixels'],
        statistics['removed_pixels'],
        statistics['added_pixels']
    ]
    colors = ['lightgray', 'gray', 'red', 'yellow']

    bars = ax1.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Number of Pixels', fontsize=12)
    ax1.set_title('Pixel Count Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加數值標籤
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{val:,}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # 右圖：統計摘要文字
    ax2 = axes[1]
    ax2.axis('off')

    net_change_sign = '+' if statistics['net_change'] >= 0 else ''
    stats_text = f"""
    Operation Summary
    {'=' * 50}

    Total Pixels:              {statistics['total_pixels']:,}

    Before Operation:
      • White pixels:          {statistics['before_white_pixels']:,}
      • Coverage:              {(statistics['before_white_pixels']/statistics['total_pixels']*100):.2f}%

    After Operation:
      • White pixels:          {statistics['after_white_pixels']:,}
      • Coverage:              {(statistics['after_white_pixels']/statistics['total_pixels']*100):.2f}%

    Changes:
      • Removed pixels:        {statistics['removed_pixels']:,}
      • Removed percentage:    {statistics['removed_percent']:.3f}%

      • Added pixels:          {statistics['added_pixels']:,}
      • Added percentage:      {statistics['added_percent']:.3f}%

      • Net change:            {net_change_sign}{statistics['net_change']:,}
      • Net change %:          {net_change_sign}{statistics['net_change_percent']:.3f}%
    """

    ax2.text(
        0.1, 0.5,
        stats_text,
        fontfamily='monospace',
        fontsize=10,
        verticalalignment='center',
        transform=ax2.transAxes
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 統計圖表已儲存: {output_path}")


def visualize_morphology_pipeline(
    input_path: str,
    output_dir: str,
    closing_kernel: int = 5,
    opening_kernel: int = 3,
    background_path: Optional[str] = None,
    removed_color: Tuple[int, int, int] = (0, 0, 255),    # BGR: 紅色
    added_color: Tuple[int, int, int] = (0, 255, 255),    # BGR: 黃色
    alpha: float = 0.6,
    dpi: int = 150
) -> Dict[str, Any]:
    """
    執行完整的形態學操作 pipeline 並視覺化每個步驟

    Pipeline 步驟：
    1. 原始影像
    2. Closing 操作（填補小孔洞）
    3. Opening 操作（移除小噪點）

    Args:
        input_path: 輸入影像路徑
        output_dir: 輸出目錄
        closing_kernel: Closing 操作的 kernel 大小
        opening_kernel: Opening 操作的 kernel 大小
        background_path: 背景影像路徑（可選）
        removed_color: 被消除區域顏色 (BGR)
        added_color: 擴增區域顏色 (BGR)
        alpha: 透明度 (0-1)
        dpi: 輸出解析度

    Returns:
        包含所有步驟統計資訊的字典
    """
    logger.info("=" * 80)
    logger.info("形態學操作 Pipeline 視覺化")
    logger.info("=" * 80)

    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 步驟 1: 載入原始影像
    logger.info("\n步驟 1: 載入原始影像")
    logger.info(f"  輸入: {input_path}")
    original = load_binary_image(input_path)
    logger.info(f"  影像尺寸: {original.shape}")
    logger.info(f"  白色像素: {np.count_nonzero(original == 255):,}")

    # 載入背景影像（如果有）並轉換成 viridis colormap
    background = None
    if background_path:
        logger.info(f"  背景影像: {background_path}")
        background_gray = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        if background_gray is None:
            logger.warning(f"  無法載入背景影像，將使用純色背景")
        else:
            # 轉換成 viridis colormap
            normalized = (background_gray - background_gray.min()) / (background_gray.max() - background_gray.min())
            normalized = np.clip(normalized, 0, 1)
            viridis_cmap = plt.cm.get_cmap('viridis')
            background_colored = viridis_cmap(normalized)[:, :, :3]  # 移除 alpha 通道
            background = (background_colored * 255).astype(np.uint8)
            # 轉換為 BGR 供 OpenCV 使用
            background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
            logger.info(f"  ✓ 已將背景影像轉換成 viridis colormap")
    
    

    # 步驟 2: 執行 Closing 操作
    logger.info(f"\n步驟 2: 執行 Closing 操作 (kernel={closing_kernel})")
    logger.info("  目的：填補小孔洞，連接鄰近物體")
    closed = morphological_closing(
        original,
        kernel_size=closing_kernel,
        kernel_shape='ellipse',
        iterations=1
    )
    logger.info(f"  ✓ Closing 完成")
    logger.info(f"  白色像素: {np.count_nonzero(closed == 255):,}")

    # 步驟 3: 執行 Opening 操作
    logger.info(f"\n步驟 3: 執行 Opening 操作 (kernel={opening_kernel})")
    logger.info("  目的：移除小噪點，平滑輪廓")
    opened = morphological_opening(
        closed,
        kernel_size=opening_kernel,
        kernel_shape='ellipse',
        iterations=1
    )
    logger.info(f"  ✓ Opening 完成")
    logger.info(f"  白色像素: {np.count_nonzero(opened == 255):,}")

    # 步驟 4: 視覺化 Closing 的效果
    logger.info("\n步驟 4: 視覺化 Closing 效果")
    removed_closing, added_closing, stats_closing = compute_morphology_diff(original, closed)

    diff_closing = create_diff_overlay(
        before=original,
        after=closed,
        removed_mask=removed_closing,
        added_mask=added_closing,
        background=background,
        removed_color=removed_color,
        added_color=added_color,
        alpha=alpha
    )

    closing_output = output_path / "01_closing_diff.png"
    cv2.imwrite(str(closing_output), diff_closing)
    logger.info(f"  ✓ 已儲存: {closing_output}")
    logger.info(f"    擴增像素: {stats_closing['added_pixels']:,} ({stats_closing['added_percent']:.3f}%)")
    logger.info(f"    消除像素: {stats_closing['removed_pixels']:,} ({stats_closing['removed_percent']:.3f}%)")

    # 步驟 5: 視覺化 Opening 的效果
    logger.info("\n步驟 5: 視覺化 Opening 效果")
    removed_opening, added_opening, stats_opening = compute_morphology_diff(closed, opened)

    diff_opening = create_diff_overlay(
        before=closed,
        after=opened,
        removed_mask=removed_opening,
        added_mask=added_opening,
        background=background,
        removed_color=removed_color,
        added_color=added_color,
        alpha=alpha
    )

    opening_output = output_path / "02_opening_diff.png"
    cv2.imwrite(str(opening_output), diff_opening)
    logger.info(f"  ✓ 已儲存: {opening_output}")
    logger.info(f"    擴增像素: {stats_opening['added_pixels']:,} ({stats_opening['added_percent']:.3f}%)")
    logger.info(f"    消除像素: {stats_opening['removed_pixels']:,} ({stats_opening['removed_percent']:.3f}%)")

    # 步驟 6: 視覺化整體效果（原始 vs 最終）
    logger.info("\n步驟 6: 視覺化整體效果（原始 vs 最終）")
    removed_overall, added_overall, stats_overall = compute_morphology_diff(original, opened)

    diff_overall = create_diff_overlay(
        before=original,
        after=opened,
        removed_mask=removed_overall,
        added_mask=added_overall,
        background=background,
        removed_color=removed_color,
        added_color=added_color,
        alpha=alpha
    )

    overall_output = output_path / "03_overall_diff.png"
    cv2.imwrite(str(overall_output), diff_overall)
    logger.info(f"  ✓ 已儲存: {overall_output}")
    logger.info(f"    擴增像素: {stats_overall['added_pixels']:,} ({stats_overall['added_percent']:.3f}%)")
    logger.info(f"    消除像素: {stats_overall['removed_pixels']:,} ({stats_overall['removed_percent']:.3f}%)")

    # 步驟 7: 創建完整的並排比較圖
    logger.info("\n步驟 7: 創建完整並排比較圖")
    comparison_output = output_path / "04_full_comparison.png"

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=dpi)
    fig.suptitle(
        f'Morphological Pipeline: Closing (k={closing_kernel}) + Opening (k={opening_kernel})',
        fontsize=16,
        fontweight='bold'
    )

    # 第一行：原始影像和操作結果
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(closed, cmap='gray')
    axes[0, 1].set_title(f'After Closing (k={closing_kernel})', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(opened, cmap='gray')
    axes[0, 2].set_title(f'After Opening (k={opening_kernel})', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # 第二行：差異視覺化
    diff_closing_rgb = cv2.cvtColor(diff_closing, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(diff_closing_rgb)
    axes[1, 0].set_title('Closing Effect', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    diff_opening_rgb = cv2.cvtColor(diff_opening, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(diff_opening_rgb)
    axes[1, 1].set_title('Opening Effect', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    diff_overall_rgb = cv2.cvtColor(diff_overall, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(diff_overall_rgb)
    axes[1, 2].set_title('Overall Effect', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    # 添加圖例
    removed_patch = mpatches.Patch(color=(1, 0, 0), label='Removed')
    added_patch = mpatches.Patch(color=(1, 1, 0), label='Added')
    unchanged_patch = mpatches.Patch(color=(0.78, 0.78, 0.78), label='Unchanged')
    fig.legend(
        handles=[removed_patch, added_patch, unchanged_patch],
        loc='lower center',
        ncol=3,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    plt.savefig(str(comparison_output), dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已儲存: {comparison_output}")

    # 步驟 8: 創建統計摘要
    logger.info("\n步驟 8: 創建統計摘要")
    stats_output = output_path / "05_statistics.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Morphological Pipeline Statistics', fontsize=16, fontweight='bold')

    # 圖 1: 像素數量變化
    ax1 = axes[0, 0]
    stages = ['Original', 'After\nClosing', 'After\nOpening']
    white_pixels = [
        np.count_nonzero(original == 255),
        np.count_nonzero(closed == 255),
        np.count_nonzero(opened == 255)
    ]
    bars = ax1.bar(stages, white_pixels, color=['lightgray', 'skyblue', 'lightgreen'],
                   edgecolor='black', alpha=0.8)
    ax1.set_ylabel('White Pixels', fontsize=12)
    ax1.set_title('White Pixel Count by Stage', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, white_pixels):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height,
                f'{val:,}', ha='center', va='bottom', fontsize=10)

    # 圖 2: Closing 統計
    ax2 = axes[0, 1]
    ax2.axis('off')
    closing_text = f"""
    Closing Operation (kernel={closing_kernel})
    {'=' * 45}

    Purpose: Fill small gaps

    Added pixels:     {stats_closing['added_pixels']:,}
    Added %:          {stats_closing['added_percent']:.3f}%

    Removed pixels:   {stats_closing['removed_pixels']:,}
    Removed %:        {stats_closing['removed_percent']:.3f}%

    Net change:       {stats_closing['net_change']:+,}
    Net change %:     {stats_closing['net_change_percent']:+.3f}%
    """
    ax2.text(0.1, 0.5, closing_text, fontfamily='monospace', fontsize=11,
             verticalalignment='center', transform=ax2.transAxes)

    # 圖 3: Opening 統計
    ax3 = axes[1, 0]
    ax3.axis('off')
    opening_text = f"""
    Opening Operation (kernel={opening_kernel})
    {'=' * 45}

    Purpose: Remove small noise

    Added pixels:     {stats_opening['added_pixels']:,}
    Added %:          {stats_opening['added_percent']:.3f}%

    Removed pixels:   {stats_opening['removed_pixels']:,}
    Removed %:        {stats_opening['removed_percent']:.3f}%

    Net change:       {stats_opening['net_change']:+,}
    Net change %:     {stats_opening['net_change_percent']:+.3f}%
    """
    ax3.text(0.1, 0.5, opening_text, fontfamily='monospace', fontsize=11,
             verticalalignment='center', transform=ax3.transAxes)

    # 圖 4: 整體統計
    ax4 = axes[1, 1]
    ax4.axis('off')
    overall_text = f"""
    Overall Effect (Original → Final)
    {'=' * 45}

    Pipeline: Closing + Opening

    Added pixels:     {stats_overall['added_pixels']:,}
    Added %:          {stats_overall['added_percent']:.3f}%

    Removed pixels:   {stats_overall['removed_pixels']:,}
    Removed %:        {stats_overall['removed_percent']:.3f}%

    Net change:       {stats_overall['net_change']:+,}
    Net change %:     {stats_overall['net_change_percent']:+.3f}%
    """
    ax4.text(0.1, 0.5, overall_text, fontfamily='monospace', fontsize=11,
             verticalalignment='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(str(stats_output), dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已儲存: {stats_output}")

    # 步驟 9: 保存處理後的影像
    logger.info("\n步驟 9: 保存處理後的影像")
    closed_save = output_path / "closed_result.png"
    opened_save = output_path / "opened_result.png"
    cv2.imwrite(str(closed_save), closed)
    cv2.imwrite(str(opened_save), opened)
    logger.info(f"  ✓ 已儲存 Closing 結果: {closed_save}")
    logger.info(f"  ✓ 已儲存 Opening 結果: {opened_save}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Pipeline 視覺化完成！")
    logger.info("=" * 80)
    logger.info(f"\n輸出目錄: {output_path}")
    logger.info("生成的檔案:")
    logger.info("  1. 01_closing_diff.png      - Closing 操作差異")
    logger.info("  2. 02_opening_diff.png      - Opening 操作差異")
    logger.info("  3. 03_overall_diff.png      - 整體差異")
    logger.info("  4. 04_full_comparison.png   - 完整並排比較")
    logger.info("  5. 05_statistics.png        - 統計摘要")
    logger.info("  6. closed_result.png        - Closing 結果影像")
    logger.info("  7. opened_result.png        - Opening 結果影像")

    return {
        'original': original,
        'closed': closed,
        'opened': opened,
        'statistics': {
            'closing': stats_closing,
            'opening': stats_opening,
            'overall': stats_overall
        },
        'output_dir': str(output_path)
    }


def visualize_morphology_diff(
    before_path: str,
    after_path: str,
    output_path: str,
    background_path: Optional[str] = None,
    operation_name: str = "Morphological Operation",
    removed_color: Tuple[int, int, int] = (0, 0, 255),    # BGR: 紅色
    added_color: Tuple[int, int, int] = (0, 255, 255),    # BGR: 黃色
    alpha: float = 0.6,
    create_stats: bool = True,
    create_sidebyside: bool = True,
    dpi: int = 150
) -> Dict[str, Any]:
    """
    完整的形態學操作差異視覺化流程

    Args:
        before_path: 操作前影像路徑
        after_path: 操作後影像路徑
        output_path: 輸出路徑（主視覺化影像）
        background_path: 背景影像路徑（可選）
        operation_name: 操作名稱（用於標題）
        removed_color: 被消除區域顏色 (BGR)
        added_color: 擴增區域顏色 (BGR)
        alpha: 透明度 (0-1)
        create_stats: 是否生成統計圖表
        create_sidebyside: 是否生成並排比較圖
        dpi: 輸出解析度

    Returns:
        包含統計資訊的字典
    """
    logger.info("=" * 60)
    logger.info(f"開始 {operation_name} 差異視覺化")
    logger.info("=" * 60)

    # 創建輸出目錄
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 步驟 1: 載入影像
    logger.info("\n步驟 1: 載入影像")
    logger.info(f"  操作前: {before_path}")
    logger.info(f"  操作後: {after_path}")

    before = load_binary_image(before_path)
    after = load_binary_image(after_path)
    logger.info(f"  影像尺寸: {before.shape}")

    background = None
    if background_path:
        logger.info(f"  背景影像: {background_path}")
        background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        if background is None:
            logger.warning(f"  無法載入背景影像，將使用純色背景")

    # 步驟 2: 計算差異
    logger.info("\n步驟 2: 計算差異")
    removed_mask, added_mask, statistics = compute_morphology_diff(before, after)

    logger.info(f"  ✓ 差異計算完成")
    logger.info(f"    被消除像素: {statistics['removed_pixels']:,} ({statistics['removed_percent']:.3f}%)")
    logger.info(f"    擴增像素: {statistics['added_pixels']:,} ({statistics['added_percent']:.3f}%)")
    logger.info(f"    淨變化: {statistics['net_change']:+,} ({statistics['net_change_percent']:+.3f}%)")

    # 步驟 3: 創建疊加視覺化
    logger.info("\n步驟 3: 創建差異疊加視覺化")
    diff_overlay = create_diff_overlay(
        before=before,
        after=after,
        removed_mask=removed_mask,
        added_mask=added_mask,
        background=background,
        removed_color=removed_color,
        added_color=added_color,
        alpha=alpha
    )

    # 保存主視覺化
    cv2.imwrite(str(output_path), diff_overlay)
    logger.info(f"✓ 主視覺化已儲存: {output_path}")

    # 步驟 4: 創建並排比較圖（可選）
    if create_sidebyside:
        logger.info("\n步驟 4: 創建並排比較圖")
        sidebyside_path = output_path.parent / f"{output_path.stem}_comparison.png"

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
        fig.suptitle(f'{operation_name} - Comparison', fontsize=16, fontweight='bold')

        # 操作前
        axes[0].imshow(before, cmap='gray')
        axes[0].set_title('Before', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 操作後
        axes[1].imshow(after, cmap='gray')
        axes[1].set_title('After', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 差異疊加
        diff_rgb = cv2.cvtColor(diff_overlay, cv2.COLOR_BGR2RGB)
        axes[2].imshow(diff_rgb)
        axes[2].set_title('Difference Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # 添加圖例
        removed_patch = mpatches.Patch(color=(1, 0, 0), label='Removed')
        added_patch = mpatches.Patch(color=(1, 1, 0), label='Added')
        unchanged_patch = mpatches.Patch(color=(0.78, 0.78, 0.78), label='Unchanged')
        axes[2].legend(
            handles=[removed_patch, added_patch, unchanged_patch],
            loc='upper right',
            fontsize=10
        )

        plt.tight_layout()
        plt.savefig(str(sidebyside_path), dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ 並排比較圖已儲存: {sidebyside_path}")

    # 步驟 5: 創建統計圖表（可選）
    if create_stats:
        logger.info("\n步驟 5: 創建統計圖表")
        stats_path = output_path.parent / f"{output_path.stem}_statistics.png"
        create_statistics_plot(
            statistics=statistics,
            output_path=str(stats_path),
            operation_name=operation_name
        )

    logger.info("\n" + "=" * 60)
    logger.info("✓ 視覺化完成！")
    logger.info("=" * 60)

    return {
        'statistics': statistics,
        'output_path': str(output_path),
        'removed_mask': removed_mask,
        'added_mask': added_mask
    }


# ============================================================================
# 主程式與使用範例
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：執行形態學操作 pipeline 並視覺化
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='執行形態學操作 pipeline (closing + opening) 並視覺化每個步驟',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用 - 執行 closing + opening pipeline
  python visualize_morphology_diff.py \\
      --input annotation.png \\
      --output-dir output/morphology_viz \\
      --closing-kernel 5 \\
      --opening-kernel 3

  # 使用背景影像（如 green channel）
  python visualize_morphology_diff.py \\
      --input annotation.png \\
      --output-dir output/morphology_viz \\
      --background green_channel.png \\
      --closing-kernel 7 \\
      --opening-kernel 5

  # 自訂顏色和透明度
  python visualize_morphology_diff.py \\
      --input annotation.png \\
      --output-dir output/morphology_viz \\
      --removed-color 255 0 0 \\
      --added-color 0 255 0 \\
      --alpha 0.8

生成的檔案:
  1. 01_closing_diff.png      - Closing 操作差異視覺化
  2. 02_opening_diff.png      - Opening 操作差異視覺化
  3. 03_overall_diff.png      - 整體效果差異視覺化
  4. 04_full_comparison.png   - 完整並排比較圖
  5. 05_statistics.png        - 統計摘要圖表
  6. closed_result.png        - Closing 操作結果影像
  7. opened_result.png        - Opening 操作結果影像（最終結果）
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
        '--closing-kernel', '-c',
        type=int,
        default=3,
        help='Closing 操作的 kernel 大小（預設: 5）'
    )

    parser.add_argument(
        '--opening-kernel', '-k',
        type=int,
        default=3,
        help='Opening 操作的 kernel 大小（預設: 3）'
    )

    parser.add_argument(
        '--background', '-b',
        help='背景影像路徑（可選，如 green channel）'
    )

    parser.add_argument(
        '--removed-color',
        nargs=3,
        type=int,
        default=[0, 0, 255],
        metavar=('B', 'G', 'R'),
        help='被消除區域顏色 (BGR, 預設: 0 0 255 紅色)'
    )

    parser.add_argument(
        '--added-color',
        nargs=3,
        type=int,
        default=[0, 255, 255],
        metavar=('B', 'G', 'R'),
        help='擴增區域顏色 (BGR, 預設: 0 255 255 黃色)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='透明度 (0-1, 預設: 0.6)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='輸出解析度 (預設: 150)'
    )

    args = parser.parse_args()

    try:
        result = visualize_morphology_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            closing_kernel=args.closing_kernel,
            opening_kernel=args.opening_kernel,
            background_path=args.background,
            removed_color=tuple(args.removed_color),
            added_color=tuple(args.added_color),
            alpha=args.alpha,
            dpi=args.dpi
        )

        print("\n" + "=" * 80)
        print("✓ 處理完成！")
        print("=" * 80)
        print(f"\n輸出目錄: {result['output_dir']}")
        print("\n統計摘要:")
        print(f"  原始白色像素:   {np.count_nonzero(result['original'] == 255):,}")
        print(f"  Closing 後:      {np.count_nonzero(result['closed'] == 255):,}")
        print(f"  Opening 後:      {np.count_nonzero(result['opened'] == 255):,}")
        print(f"\n  整體變化:")
        print(f"    擴增像素:      {result['statistics']['overall']['added_pixels']:,}")
        print(f"    消除像素:      {result['statistics']['overall']['removed_pixels']:,}")
        print(f"    淨變化:        {result['statistics']['overall']['net_change']:+,}")

    except Exception as e:
        logger.error(f"處理失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
