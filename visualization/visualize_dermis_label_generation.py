#!/usr/bin/env python3
"""
真皮標記生成視覺化工具

視覺化 pipeline 中真皮標記生成的過程，包括：
1. 在輸入影像上顯示表皮範圍（透明色塊覆蓋）
2. 標記處理的 ROI 區域（表皮遮罩往下延伸的部分）
3. 單獨提取 ROI 影像（灰階和 viridis colormap）
4. 顯示 Otsu 閾值處理結果並覆蓋在 ROI 上

功能:
1. 載入原始影像、表皮遮罩
2. 計算 ROI 區域（表皮遮罩 + 垂直擴展區域）
3. 視覺化表皮和真皮邊界區域
4. 顯示背景修正和 Otsu 閾值處理結果
5. 生成多種視覺化比較圖

使用範例:
    from visualization.visualize_dermis_label_generation import visualize_dermis_label_generation

    # 視覺化真皮標記生成過程
    visualize_dermis_label_generation(
        original_image_path='data/original.png',
        epidermis_mask_path='data/epidermis_mask.png',
        output_dir='output/dermis_viz',
        dilate_offset=50,
        bg_radius=12,
        light_background=True
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

from src.preprocessing.background_correction import rolling_ball_background
from src.preprocessing.thresholding import otsu_threshold
from src.preprocessing.mask_operations import (
    dilate_epidermis_vertically,
    apply_mask,
    invert_mask
)

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


def create_epidermis_overlay(
    original_image: np.ndarray,
    epidermis_mask: np.ndarray,
    epidermis_color: Tuple[int, int, int] = (0, 255, 0),  # 綠色
    alpha: float = 0.3
) -> np.ndarray:
    """
    在原始影像上覆蓋表皮範圍

    Args:
        original_image: 原始影像 (灰階或彩色)
        epidermis_mask: 表皮遮罩
        epidermis_color: 表皮覆蓋顏色 (B, G, R)
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
    color_mask[epidermis_mask > 0] = epidermis_color

    # 混合原始影像和彩色遮罩
    overlay_image = cv2.addWeighted(overlay_image, 1.0, color_mask, alpha, 0)

    return overlay_image


def create_roi_overlay(
    original_image: np.ndarray,
    epidermis_mask: np.ndarray,
    dilated_mask: np.ndarray,
    epidermis_color: Tuple[int, int, int] = (0, 255, 0),  # 綠色
    roi_color: Tuple[int, int, int] = (0, 165, 255),      # 橙色
    alpha: float = 0.4
) -> np.ndarray:
    """
    在原始影像上覆蓋表皮和 ROI 範圍

    Args:
        original_image: 原始影像 (灰階或彩色)
        epidermis_mask: 表皮遮罩
        dilated_mask: 擴展後的遮罩 (表皮 + 延伸區域)
        epidermis_color: 表皮覆蓋顏色 (B, G, R)
        roi_color: ROI 覆蓋顏色 (B, G, R)
        alpha: 透明度 (0-1)

    Returns:
        覆蓋後的彩色影像 (BGR)
    """
    # 如果原始影像是灰階,轉為彩色
    if len(original_image.shape) == 2:
        overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay_image = original_image.copy()

    # 計算真皮延伸區域 = dilated_mask AND (NOT epidermis_mask)
    inverted_epidermis = invert_mask(epidermis_mask)
    _, inverted_epidermis = cv2.threshold(inverted_epidermis, 127, 255, cv2.THRESH_BINARY)
    _, dilated_mask_bin = cv2.threshold(dilated_mask, 127, 255, cv2.THRESH_BINARY)
    dermis_extension = cv2.bitwise_and(dilated_mask_bin, inverted_epidermis)

    # 創建彩色遮罩
    color_mask = np.zeros_like(overlay_image)

    # 添加表皮區域 (綠色)
    color_mask[epidermis_mask > 0] = epidermis_color

    # 添加真皮延伸區域 (橙色)
    color_mask[dermis_extension > 0] = roi_color

    # 混合原始影像和彩色遮罩
    overlay_image = cv2.addWeighted(overlay_image, 1.0, color_mask, alpha, 0)

    return overlay_image


def extract_roi_image(
    original_image: np.ndarray,
    dilated_mask: np.ndarray
) -> np.ndarray:
    """
    提取 ROI 區域的影像

    Args:
        original_image: 原始影像
        dilated_mask: 擴展後的遮罩

    Returns:
        ROI 影像（非 ROI 區域為黑色）
    """
    roi_image = apply_mask(original_image, dilated_mask)
    return roi_image


def apply_viridis_colormap(
    grayscale_image: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    應用 viridis colormap 到灰階影像

    Args:
        grayscale_image: 灰階影像
        mask: 可選的遮罩，只對遮罩區域上色

    Returns:
        彩色影像 (BGR 格式)
    """
    # 應用 viridis colormap
    colored = cv2.applyColorMap(grayscale_image, cv2.COLORMAP_VIRIDIS)

    # 如果有遮罩，只保留遮罩區域的顏色
    if mask is not None:
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # 創建黑色背景
        result = np.zeros_like(colored)
        # 只複製遮罩區域
        result[mask_bin > 0] = colored[mask_bin > 0]
        return result

    return colored


def create_epidermis_roi_visualization(
    original_image: np.ndarray,
    epidermis_overlay: np.ndarray,
    roi_overlay: np.ndarray,
    epidermis_mask: np.ndarray,
    dilated_mask: np.ndarray,
    stats: Dict[str, Any],
    output_path: str,
    dpi: int = 150
) -> None:
    """
    創建表皮和 ROI 區域的比較圖

    Args:
        original_image: 原始影像
        epidermis_overlay: 表皮覆蓋影像
        roi_overlay: ROI 覆蓋影像
        epidermis_mask: 表皮遮罩
        dilated_mask: 擴展遮罩
        stats: 統計資訊
        output_path: 輸出路徑
        dpi: 解析度
    """
    fig = plt.figure(figsize=(20, 12), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 設定整體標題
    fig.suptitle('Dermis Label Generation - Step 1: Region Definition',
                 fontsize=16, fontweight='bold', y=0.98)

    # 第一行：原始影像、表皮覆蓋、ROI 覆蓋
    # 1. 原始影像
    ax1 = fig.add_subplot(gs[0, 0])
    if len(original_image.shape) == 2:
        ax1.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    else:
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. 表皮覆蓋
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(epidermis_overlay, cv2.COLOR_BGR2RGB))
    ax2.set_title('Epidermis Region (Green)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 添加圖例
    green_patch = mpatches.Patch(color='green', label='Epidermis', alpha=0.7)
    ax2.legend(handles=[green_patch], loc='upper right', fontsize=10)

    # 3. ROI 覆蓋
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(cv2.cvtColor(roi_overlay, cv2.COLOR_BGR2RGB))
    ax3.set_title('ROI: Epidermis + Extension (Orange)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 添加圖例
    green_patch = mpatches.Patch(color='green', label='Epidermis', alpha=0.7)
    orange_patch = mpatches.Patch(color='orange', label='Dermis Extension (ROI)', alpha=0.7)
    ax3.legend(handles=[green_patch, orange_patch], loc='upper right', fontsize=10)

    # 第二行：表皮遮罩、擴展遮罩、統計資訊
    # 4. 表皮遮罩
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(epidermis_mask, cmap='Greens', vmin=0, vmax=255)
    ax4.set_title(f'Epidermis Mask ({stats["epidermis_pixels"]:,} pixels)',
                  fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5. 擴展遮罩
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(dilated_mask, cmap='Oranges', vmin=0, vmax=255)
    ax5.set_title(f'Dilated Mask ({stats["dilated_pixels"]:,} pixels)',
                  fontsize=12, fontweight='bold')
    ax5.axis('off')

    # 6. 統計資訊
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    stats_text = f"""
    Region Statistics
    {'=' * 50}

    Mask Information
    {'-' * 50}
    Epidermis mask pixels:      {stats['epidermis_pixels']:,}
    Dilated mask pixels:        {stats['dilated_pixels']:,}
    Dermis extension pixels:    {stats['extension_pixels']:,}

    Configuration
    {'-' * 50}
    Dilate offset (px):         {stats['dilate_offset']}
    Extension percentage:       {stats['extension_percentage']:.2f}%

    Image Dimensions
    {'-' * 50}
    Width:                      {stats['width']} px
    Height:                     {stats['height']} px
    Total pixels:               {stats['total_pixels']:,}
    """

    ax6.text(0.05, 0.5, stats_text, fontfamily='monospace', fontsize=10,
             verticalalignment='center', transform=ax6.transAxes)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 區域定義圖已儲存: {output_path}")


def create_roi_extraction_visualization(
    roi_grayscale: np.ndarray,
    roi_viridis: np.ndarray,
    bg_corrected_roi: np.ndarray,
    bg_corrected_viridis: np.ndarray,
    stats: Dict[str, Any],
    output_path: str,
    dpi: int = 150
) -> None:
    """
    創建 ROI 提取和背景修正的視覺化

    Args:
        roi_grayscale: ROI 灰階影像
        roi_viridis: ROI viridis colormap 影像
        bg_corrected_roi: 背景修正後的 ROI
        bg_corrected_viridis: 背景修正後的 viridis
        stats: 統計資訊
        output_path: 輸出路徑
        dpi: 解析度
    """
    fig = plt.figure(figsize=(20, 10), dpi=dpi)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 設定整體標題
    fig.suptitle('Dermis Label Generation - Step 2: ROI Extraction & Background Correction',
                 fontsize=16, fontweight='bold', y=0.98)

    # 第一行：原始 ROI
    # 1. ROI 灰階
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(roi_grayscale, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('ROI - Grayscale', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. ROI viridis
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(roi_viridis, cv2.COLOR_BGR2RGB))
    ax2.set_title('ROI - Viridis Colormap', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 第二行：背景修正後
    # 3. 背景修正 ROI 灰階
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(bg_corrected_roi, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Background Corrected - Grayscale', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. 背景修正 ROI viridis
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(cv2.cvtColor(bg_corrected_viridis, cv2.COLOR_BGR2RGB))
    ax4.set_title('Background Corrected - Viridis Colormap', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 添加文字說明
    info_text = f"""Background Correction Config: Radius={stats['bg_radius']}, Light={stats['light_background']}
ROI Statistics: Mean={stats['roi_mean']:.2f}, Std={stats['roi_std']:.2f}
Corrected Statistics: Mean={stats['corrected_mean']:.2f}, Std={stats['corrected_std']:.2f}"""

    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ ROI 提取圖已儲存: {output_path}")


def create_otsu_threshold_visualization(
    bg_corrected_roi: np.ndarray,
    bg_corrected_viridis: np.ndarray,
    otsu_label: np.ndarray,
    otsu_overlay_gray: np.ndarray,
    otsu_overlay_viridis: np.ndarray,
    dermis_extension_mask: np.ndarray,
    stats: Dict[str, Any],
    output_path: str,
    dpi: int = 150
) -> None:
    """
    創建 Otsu 閾值處理結果的視覺化

    Args:
        bg_corrected_roi: 背景修正後的 ROI (灰階)
        bg_corrected_viridis: 背景修正後的 ROI (viridis colormap)
        otsu_label: Otsu 閾值處理結果
        otsu_overlay_gray: Otsu 結果覆蓋在灰階 ROI 上
        otsu_overlay_viridis: Otsu 結果覆蓋在 viridis ROI 上
        dermis_extension_mask: 真皮延伸區域遮罩
        stats: 統計資訊
        output_path: 輸出路徑
        dpi: 解析度
    """
    fig = plt.figure(figsize=(24, 12), dpi=dpi)
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # 設定整體標題
    fig.suptitle('Dermis Label Generation - Step 3: Otsu Thresholding',
                 fontsize=16, fontweight='bold', y=0.98)

    # 第一行：背景修正 ROI（灰階和 viridis）、真皮延伸區域、Otsu 結果
    # 1. 背景修正 ROI 灰階
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(bg_corrected_roi, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Background Corrected ROI\n(Grayscale)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. 背景修正 ROI viridis
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(bg_corrected_viridis, cv2.COLOR_BGR2RGB))
    ax2.set_title('Background Corrected ROI\n(Viridis)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. 真皮延伸區域遮罩
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(dermis_extension_mask, cmap='Oranges', vmin=0, vmax=255)
    ax3.set_title(f'Dermis Extension Mask\n({stats["extension_pixels"]:,} pixels)',
                  fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. Otsu 閾值結果
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(otsu_label, cmap='Reds', vmin=0, vmax=255)
    ax4.set_title(f'Otsu Threshold Result\n({stats["otsu_pixels"]:,} pixels)',
                  fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 第二行：Otsu 覆蓋（灰階和 viridis）、直方圖、統計資訊
    # 5. Otsu 覆蓋在灰階上
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(cv2.cvtColor(otsu_overlay_gray, cv2.COLOR_BGR2RGB))
    ax5.set_title('Otsu Overlay on Grayscale ROI', fontsize=12, fontweight='bold')
    ax5.axis('off')

    # 添加圖例
    red_patch = mpatches.Patch(color='red', label='Dermis Label (Pseudo)', alpha=0.7)
    ax5.legend(handles=[red_patch], loc='upper right', fontsize=10)

    # 6. Otsu 覆蓋在 viridis 上
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(cv2.cvtColor(otsu_overlay_viridis, cv2.COLOR_BGR2RGB))
    ax6.set_title('Otsu Overlay on Viridis ROI', fontsize=12, fontweight='bold')
    ax6.axis('off')

    # 添加圖例
    red_patch = mpatches.Patch(color='red', label='Dermis Label (Pseudo)', alpha=0.7)
    ax6.legend(handles=[red_patch], loc='upper right', fontsize=10)

    # 7. 直方圖
    ax7 = fig.add_subplot(gs[1, 2])

    # 計算真皮延伸區域的直方圖
    masked_region = bg_corrected_roi[dermis_extension_mask > 0]
    if len(masked_region) > 0:
        hist, bins = np.histogram(masked_region, bins=256, range=(0, 256))
        ax7.fill_between(bins[:-1], hist, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)

        # 標記 Otsu 閾值
        if 'otsu_threshold' in stats and stats['otsu_threshold'] is not None:
            ax7.axvline(stats['otsu_threshold'], color='red', linestyle='--',
                       linewidth=2, label=f"Otsu Threshold: {stats['otsu_threshold']:.1f}")
            ax7.legend(loc='upper right', fontsize=10)

    ax7.set_xlabel('Pixel Intensity', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('Dermis Extension Region Histogram', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.set_xlim(0, 255)

    # 8. 統計資訊
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')

    stats_text = f"""
    Otsu Thresholding Statistics
    {'=' * 50}

    Threshold Information
    {'-' * 50}
    Otsu threshold value:       {stats.get('otsu_threshold', 'N/A')}
    Threshold method:           {stats.get('threshold_method', 'binary')}

    Pixel Counts
    {'-' * 50}
    Dermis extension pixels:    {stats['extension_pixels']:,}
    Otsu label pixels:          {stats['otsu_pixels']:,}
    Label percentage:           {stats['label_percentage']:.2f}%

    Intensity Statistics
    {'-' * 50}
    Mean intensity:             {stats.get('mean_intensity', 0):.2f}
    Std deviation:              {stats.get('std_intensity', 0):.2f}
    Min intensity:              {stats.get('min_intensity', 0):.2f}
    Max intensity:              {stats.get('max_intensity', 0):.2f}
    """

    ax8.text(0.05, 0.5, stats_text, fontfamily='monospace', fontsize=10,
             verticalalignment='center', transform=ax8.transAxes)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Otsu 閾值處理圖已儲存: {output_path}")


def visualize_dermis_label_generation(
    original_image_path: str,
    epidermis_mask_path: str,
    output_dir: str,
    dilate_offset: int = 50,
    bg_radius: int = 12,
    light_background: bool = True,
    threshold_method: str = 'binary',
    epidermis_color: Tuple[int, int, int] = (0, 255, 0),  # 綠色
    roi_color: Tuple[int, int, int] = (0, 165, 255),      # 橙色
    label_color: Tuple[int, int, int] = (255, 0, 0),      # 紅色
    alpha: float = 0.4,
    dpi: int = 150
) -> Dict[str, Any]:
    """
    視覺化真皮標記生成的完整流程

    Args:
        original_image_path: 原始影像路徑
        epidermis_mask_path: 表皮遮罩路徑
        output_dir: 輸出目錄
        dilate_offset: 垂直擴展像素數（預設：50）
        bg_radius: Rolling ball 半徑（預設：12）
        light_background: 是否為亮背景（預設：True）
        threshold_method: 閾值方法（預設：'binary'）
        epidermis_color: 表皮顏色 (B, G, R)
        roi_color: ROI 顏色 (B, G, R)
        label_color: 標記顏色 (B, G, R)
        alpha: 透明度 (0-1)
        dpi: 輸出解析度

    Returns:
        包含所有結果和統計資訊的字典
    """
    logger.info("=" * 80)
    logger.info("真皮標記生成視覺化")
    logger.info("=" * 80)

    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 步驟 1: 載入影像
    logger.info("\n步驟 1: 載入影像")
    logger.info(f"  原始影像: {original_image_path}")
    logger.info(f"  表皮遮罩: {epidermis_mask_path}")

    original_image = load_image(original_image_path, grayscale=True)
    epidermis_mask = load_image(epidermis_mask_path, grayscale=True)

    # 確保遮罩是二值化的
    _, epidermis_mask = cv2.threshold(epidermis_mask, 127, 255, cv2.THRESH_BINARY)

    logger.info(f"  影像尺寸: {original_image.shape}")

    # 步驟 2: 創建擴展遮罩
    logger.info(f"\n步驟 2: 創建擴展遮罩（往下延伸 {dilate_offset} px）")
    dilated_mask = dilate_epidermis_vertically(epidermis_mask, offset_px=dilate_offset)

    # 計算真皮延伸區域
    inverted_epidermis = invert_mask(epidermis_mask)
    _, inverted_epidermis = cv2.threshold(inverted_epidermis, 127, 255, cv2.THRESH_BINARY)
    dermis_extension_mask = cv2.bitwise_and(dilated_mask, inverted_epidermis)

    epidermis_pixels = np.count_nonzero(epidermis_mask > 0)
    dilated_pixels = np.count_nonzero(dilated_mask > 0)
    extension_pixels = np.count_nonzero(dermis_extension_mask > 0)

    logger.info(f"  表皮遮罩像素: {epidermis_pixels:,}")
    logger.info(f"  擴展遮罩像素: {dilated_pixels:,}")
    logger.info(f"  真皮延伸像素: {extension_pixels:,}")

    # 步驟 3: 創建表皮和 ROI 覆蓋
    logger.info("\n步驟 3: 創建表皮和 ROI 覆蓋影像")
    epidermis_overlay = create_epidermis_overlay(
        original_image, epidermis_mask, epidermis_color, alpha=0.3
    )
    roi_overlay = create_roi_overlay(
        original_image, epidermis_mask, dilated_mask,
        epidermis_color, roi_color, alpha
    )

    # 計算統計資訊
    height, width = original_image.shape
    total_pixels = height * width
    extension_percentage = (extension_pixels / epidermis_pixels * 100) if epidermis_pixels > 0 else 0

    region_stats = {
        'epidermis_pixels': epidermis_pixels,
        'dilated_pixels': dilated_pixels,
        'extension_pixels': extension_pixels,
        'dilate_offset': dilate_offset,
        'extension_percentage': extension_percentage,
        'width': width,
        'height': height,
        'total_pixels': total_pixels
    }

    # 生成區域定義視覺化
    logger.info("\n步驟 4: 生成區域定義視覺化")
    region_output = output_path / "01_region_definition.png"
    create_epidermis_roi_visualization(
        original_image, epidermis_overlay, roi_overlay,
        epidermis_mask, dilated_mask, region_stats,
        str(region_output), dpi
    )

    # 步驟 5: 提取 ROI 並應用 viridis colormap
    logger.info("\n步驟 5: 提取 ROI 影像")
    roi_grayscale = extract_roi_image(original_image, dilated_mask)
    roi_viridis = apply_viridis_colormap(roi_grayscale, dilated_mask)

    logger.info("  ✓ ROI 灰階影像提取完成")
    logger.info("  ✓ Viridis colormap 應用完成")

    # 步驟 6: 背景修正
    logger.info(f"\n步驟 6: 背景修正（Radius={bg_radius}, Light={light_background}）")
    corrected = rolling_ball_background(
        original_image,
        radius=bg_radius,
        light_background=light_background,
        smoothing=False
    )

    # 提取背景修正後的 ROI
    bg_corrected_roi = extract_roi_image(corrected, dilated_mask)
    bg_corrected_viridis = apply_viridis_colormap(bg_corrected_roi, dilated_mask)

    # 計算統計
    roi_mean = np.mean(roi_grayscale[dilated_mask > 0]) if np.any(dilated_mask > 0) else 0
    roi_std = np.std(roi_grayscale[dilated_mask > 0]) if np.any(dilated_mask > 0) else 0
    corrected_mean = np.mean(bg_corrected_roi[dilated_mask > 0]) if np.any(dilated_mask > 0) else 0
    corrected_std = np.std(bg_corrected_roi[dilated_mask > 0]) if np.any(dilated_mask > 0) else 0

    roi_stats = {
        'bg_radius': bg_radius,
        'light_background': light_background,
        'roi_mean': roi_mean,
        'roi_std': roi_std,
        'corrected_mean': corrected_mean,
        'corrected_std': corrected_std
    }

    logger.info(f"  原始 ROI: Mean={roi_mean:.2f}, Std={roi_std:.2f}")
    logger.info(f"  修正後:   Mean={corrected_mean:.2f}, Std={corrected_std:.2f}")

    # 生成 ROI 提取視覺化
    logger.info("\n步驟 7: 生成 ROI 提取視覺化")
    roi_output = output_path / "02_roi_extraction.png"
    create_roi_extraction_visualization(
        roi_grayscale, roi_viridis,
        bg_corrected_roi, bg_corrected_viridis,
        roi_stats, str(roi_output), dpi
    )

    # 步驟 8: Otsu 閾值處理
    logger.info(f"\n步驟 8: Otsu 閾值處理（方法={threshold_method}）")

    # 只對真皮延伸區域應用遮罩
    masked_region = apply_mask(corrected, dermis_extension_mask)

    # 應用 Otsu 閾值
    otsu_label = otsu_threshold(masked_region, threshold_type=threshold_method)

    # 計算 Otsu 閾值（用於顯示）
    extension_region_pixels = masked_region[dermis_extension_mask > 0]
    otsu_threshold_value = None
    if len(extension_region_pixels) > 0 and np.any(extension_region_pixels > 0):
        otsu_threshold_value, _ = cv2.threshold(
            extension_region_pixels.astype(np.uint8),
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    otsu_pixels = np.count_nonzero(otsu_label > 0)
    label_percentage = (otsu_pixels / extension_pixels * 100) if extension_pixels > 0 else 0

    # 計算強度統計
    mean_intensity = np.mean(extension_region_pixels) if len(extension_region_pixels) > 0 else 0
    std_intensity = np.std(extension_region_pixels) if len(extension_region_pixels) > 0 else 0
    min_intensity = np.min(extension_region_pixels) if len(extension_region_pixels) > 0 else 0
    max_intensity = np.max(extension_region_pixels) if len(extension_region_pixels) > 0 else 0

    logger.info(f"  Otsu 閾值: {otsu_threshold_value}")
    logger.info(f"  標記像素: {otsu_pixels:,} ({label_percentage:.2f}%)")

    # 創建 Otsu 覆蓋影像 - 灰階版本
    if len(bg_corrected_roi.shape) == 2:
        otsu_overlay_gray = cv2.cvtColor(bg_corrected_roi, cv2.COLOR_GRAY2BGR)
    else:
        otsu_overlay_gray = bg_corrected_roi.copy()

    # 創建彩色遮罩
    color_mask = np.zeros_like(otsu_overlay_gray)
    color_mask[otsu_label > 0] = label_color

    # 混合灰階版本
    otsu_overlay_gray = cv2.addWeighted(otsu_overlay_gray, 1.0, color_mask, 0.5, 0)

    # 創建 Otsu 覆蓋影像 - viridis 版本
    otsu_overlay_viridis = bg_corrected_viridis.copy()

    # 在 viridis 版本上覆蓋標記
    color_mask_viridis = np.zeros_like(otsu_overlay_viridis)
    color_mask_viridis[otsu_label > 0] = label_color

    # 混合 viridis 版本
    otsu_overlay_viridis = cv2.addWeighted(otsu_overlay_viridis, 1.0, color_mask_viridis, 0.7, 0)

    logger.info("  ✓ Otsu 覆蓋影像創建完成（灰階和 viridis）")

    otsu_stats = {
        'extension_pixels': extension_pixels,
        'otsu_pixels': otsu_pixels,
        'label_percentage': label_percentage,
        'otsu_threshold': otsu_threshold_value,
        'threshold_method': threshold_method,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'min_intensity': min_intensity,
        'max_intensity': max_intensity
    }

    # 生成 Otsu 閾值處理視覺化
    logger.info("\n步驟 9: 生成 Otsu 閾值處理視覺化")
    otsu_output = output_path / "03_otsu_thresholding.png"
    create_otsu_threshold_visualization(
        bg_corrected_roi, bg_corrected_viridis, otsu_label,
        otsu_overlay_gray, otsu_overlay_viridis,
        dermis_extension_mask, otsu_stats,
        str(otsu_output), dpi
    )

    # 步驟 10: 保存所有中間結果
    logger.info("\n步驟 10: 保存中間結果影像")

    # 保存影像
    save_files = [
        ('epidermis_overlay.png', epidermis_overlay),
        ('roi_overlay.png', roi_overlay),
        ('epidermis_mask.png', epidermis_mask),
        ('dilated_mask.png', dilated_mask),
        ('dermis_extension_mask.png', dermis_extension_mask),
        ('roi_grayscale.png', roi_grayscale),
        ('roi_viridis.png', roi_viridis),
        ('bg_corrected_roi_grayscale.png', bg_corrected_roi),
        ('bg_corrected_roi_viridis.png', bg_corrected_viridis),
        ('otsu_label.png', otsu_label),
        ('otsu_overlay_gray.png', otsu_overlay_gray),
        ('otsu_overlay_viridis.png', otsu_overlay_viridis),
    ]

    for filename, image in save_files:
        save_path = output_path / filename
        cv2.imwrite(str(save_path), image)
        logger.info(f"  ✓ 已儲存: {filename}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ 視覺化完成！")
    logger.info("=" * 80)
    logger.info(f"\n輸出目錄: {output_path}")
    logger.info("\n生成的檔案:")
    logger.info("  視覺化圖:")
    logger.info("    1. 01_region_definition.png    - 表皮和 ROI 區域定義")
    logger.info("    2. 02_roi_extraction.png        - ROI 提取和背景修正")
    logger.info("    3. 03_otsu_thresholding.png    - Otsu 閾值處理結果")
    logger.info("\n  中間結果影像:")
    logger.info("    - epidermis_overlay.png         - 表皮覆蓋")
    logger.info("    - roi_overlay.png               - ROI 覆蓋")
    logger.info("    - epidermis_mask.png            - 表皮遮罩")
    logger.info("    - dilated_mask.png              - 擴展遮罩")
    logger.info("    - dermis_extension_mask.png     - 真皮延伸區域遮罩")
    logger.info("    - roi_grayscale.png             - ROI 灰階")
    logger.info("    - roi_viridis.png               - ROI Viridis")
    logger.info("    - bg_corrected_roi_grayscale.png - 背景修正 ROI 灰階")
    logger.info("    - bg_corrected_roi_viridis.png   - 背景修正 ROI Viridis")
    logger.info("    - otsu_label.png                - Otsu 標記")
    logger.info("    - otsu_overlay_gray.png         - Otsu 覆蓋（灰階）")
    logger.info("    - otsu_overlay_viridis.png      - Otsu 覆蓋（Viridis）")

    return {
        'original_image': original_image,
        'epidermis_mask': epidermis_mask,
        'dilated_mask': dilated_mask,
        'dermis_extension_mask': dermis_extension_mask,
        'roi_grayscale': roi_grayscale,
        'roi_viridis': roi_viridis,
        'bg_corrected_roi': bg_corrected_roi,
        'bg_corrected_viridis': bg_corrected_viridis,
        'otsu_label': otsu_label,
        'otsu_overlay_gray': otsu_overlay_gray,
        'otsu_overlay_viridis': otsu_overlay_viridis,
        'region_stats': region_stats,
        'roi_stats': roi_stats,
        'otsu_stats': otsu_stats,
        'output_dir': str(output_path)
    }


# ============================================================================
# 主程式與使用範例
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：視覺化真皮標記生成過程
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='視覺化真皮標記生成的完整流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用
  python visualize_dermis_label_generation.py \\
      --original data/original_image.png \\
      --mask data/epidermis_mask.png \\
      --output-dir output/dermis_viz

  # 自訂參數
  python visualize_dermis_label_generation.py \\
      --original data/original_image.png \\
      --mask data/epidermis_mask.png \\
      --output-dir output/dermis_viz \\
      --dilate-offset 50 \\
      --bg-radius 12 \\
      --light-background

  # 自訂顏色
  python visualize_dermis_label_generation.py \\
      --original data/original_image.png \\
      --mask data/epidermis_mask.png \\
      --output-dir output/dermis_viz \\
      --epidermis-color 0 255 255 \\
      --roi-color 255 0 255 \\
      --label-color 0 255 0

生成的檔案:
  視覺化圖:
    1. 01_region_definition.png    - 表皮和 ROI 區域定義 (3x2 網格)
    2. 02_roi_extraction.png        - ROI 提取和背景修正比較 (2x2 網格)
    3. 03_otsu_thresholding.png    - Otsu 閾值處理和統計 (2x4 網格)

  中間結果影像 (12 張):
    - epidermis_overlay.png         - 表皮區域覆蓋在原始影像上
    - roi_overlay.png               - ROI 區域覆蓋（表皮+延伸）
    - epidermis_mask.png            - 表皮遮罩
    - dilated_mask.png              - 擴展遮罩
    - dermis_extension_mask.png     - 真皮延伸區域遮罩
    - roi_grayscale.png             - ROI 灰階影像
    - roi_viridis.png               - ROI Viridis colormap
    - bg_corrected_roi_grayscale.png - 背景修正後 ROI 灰階
    - bg_corrected_roi_viridis.png   - 背景修正後 ROI Viridis
    - otsu_label.png                - Otsu 閾值處理標記
    - otsu_overlay_gray.png         - Otsu 標記覆蓋在灰階 ROI 上
    - otsu_overlay_viridis.png      - Otsu 標記覆蓋在 Viridis ROI 上

處理流程:
  1. 定義表皮和 ROI 區域
  2. 提取 ROI 影像（灰階和 viridis colormap）
  3. 背景修正
  4. Otsu 閾值處理生成真皮標記
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
        '--output-dir', '-d',
        required=True,
        help='輸出目錄'
    )

    parser.add_argument(
        '--dilate-offset',
        type=int,
        default=50,
        help='垂直擴展像素數（預設: 50）'
    )

    parser.add_argument(
        '--bg-radius',
        type=int,
        default=12,
        help='Rolling ball 半徑（預設: 12）'
    )

    parser.add_argument(
        '--light-background',
        action='store_true',
        default=False,
        help='影像有亮背景（預設: False）'
    )

    parser.add_argument(
        '--no-light-background',
        action='store_false',
        default=True,
        dest='light_background',
        help='影像有暗背景'
    )

    parser.add_argument(
        '--threshold-method',
        choices=['binary', 'binary_inv'],
        default='binary',
        help='閾值方法（預設: binary）'
    )

    parser.add_argument(
        '--epidermis-color',
        type=int,
        nargs=3,
        default=[0, 255, 0],
        metavar=('B', 'G', 'R'),
        help='表皮顏色 BGR 格式（預設: 0 255 0 綠色）'
    )

    parser.add_argument(
        '--roi-color',
        type=int,
        nargs=3,
        default=[0, 165, 255],
        metavar=('B', 'G', 'R'),
        help='ROI 顏色 BGR 格式（預設: 0 165 255 橙色）'
    )

    parser.add_argument(
        '--label-color',
        type=int,
        nargs=3,
        default=[255, 0, 0],
        metavar=('B', 'G', 'R'),
        help='標記顏色 BGR 格式（預設: 255 0 0 紅色）'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='透明度 (0-1, 預設: 0.4)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='輸出解析度（預設: 150）'
    )

    args = parser.parse_args()

    try:
        result = visualize_dermis_label_generation(
            original_image_path=args.original,
            epidermis_mask_path=args.mask,
            output_dir=args.output_dir,
            dilate_offset=args.dilate_offset,
            bg_radius=args.bg_radius,
            light_background=args.light_background,
            threshold_method=args.threshold_method,
            epidermis_color=tuple(args.epidermis_color),
            roi_color=tuple(args.roi_color),
            label_color=tuple(args.label_color),
            alpha=args.alpha,
            dpi=args.dpi
        )

        print("\n" + "=" * 80)
        print("✓ 處理完成！")
        print("=" * 80)
        print(f"\n輸出目錄: {result['output_dir']}")
        print("\n統計摘要:")
        print(f"  區域統計:")
        print(f"    表皮像素:       {result['region_stats']['epidermis_pixels']:,}")
        print(f"    擴展遮罩像素:   {result['region_stats']['dilated_pixels']:,}")
        print(f"    真皮延伸像素:   {result['region_stats']['extension_pixels']:,}")
        print(f"    延伸比例:       {result['region_stats']['extension_percentage']:.2f}%")
        print(f"\n  背景修正:")
        print(f"    原始 ROI Mean:  {result['roi_stats']['roi_mean']:.2f}")
        print(f"    修正後 Mean:    {result['roi_stats']['corrected_mean']:.2f}")
        print(f"\n  Otsu 閾值:")
        print(f"    閾值:           {result['otsu_stats'].get('otsu_threshold', 'N/A')}")
        print(f"    標記像素:       {result['otsu_stats']['otsu_pixels']:,}")
        print(f"    標記比例:       {result['otsu_stats']['label_percentage']:.2f}%")

    except Exception as e:
        logger.error(f"處理失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
