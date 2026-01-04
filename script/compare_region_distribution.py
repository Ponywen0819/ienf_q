"""
比對表皮區域與真皮區域的像素分布

修改配置區塊即可調整比對參數
輸出疊加直方圖以視覺化兩個區域的像素強度分布差異
"""
import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from src.preprocessing import SkinAnalysisPipeline


# ========================================
# 影像路徑設定
# ========================================
IMAGE_ID = 'S163-2_a'
LABEL_IMAGE_PATH = f'data/Label/{IMAGE_ID}.tif'
EPIDERMIS_MASK_PATH = f'data/Mask/{IMAGE_ID}.tif'
ORIGINAL_IMAGE_PATH = f'data/Original/{IMAGE_ID}.tif'


# ========================================
# Pipeline 配置
# ========================================
PIPELINE_CONFIG = {
    'morphology': {
        'closing_kernel': 3,
        'opening_kernel': 3
    },
    'mask': {
        'dilate_offset': 100  # 真皮區域向下延伸的像素數
    },
    'background': {
        'method': 'rolling_ball',
        'radius': 2,
        'light_background': False
    },
    'threshold': {
        'method': 'binary'
    },
    'normalization': {
        'enabled': True      # 是否啟用區域正規化
    }
}


# ========================================
# 直方圖配置
# ========================================
HISTOGRAM_BINS = 256          # 直方圖 bin 數量
HISTOGRAM_ALPHA = 0.6         # 直方圖透明度
EPIDERMIS_COLOR = 'green'     # 表皮區域顏色
DERMIS_COLOR = 'orange'       # 真皮區域顏色
FIGURE_SIZE = (12, 8)         # 圖表尺寸
DPI = 150                     # 輸出解析度


# ========================================
# 輸出配置
# ========================================
OUTPUT_DIR = 'output/region_distribution'


def compute_statistics(pixels: np.ndarray) -> dict:
    """
    計算像素統計資訊

    Args:
        pixels: 像素陣列

    Returns:
        統計資訊字典
    """
    return {
        'count': len(pixels),
        'mean': np.mean(pixels),
        'std': np.std(pixels),
        'median': np.median(pixels),
        'min': np.min(pixels),
        'max': np.max(pixels),
        'q25': np.percentile(pixels, 25),
        'q75': np.percentile(pixels, 75)
    }


def create_overlayed_histogram(
    epidermis_pixels: np.ndarray,
    dermis_pixels: np.ndarray,
    epidermis_stats: dict,
    dermis_stats: dict,
    output_path: str,
    title: str = "Epidermis vs Dermis Pixel Intensity Distribution"
) -> None:
    """
    建立疊加直方圖

    Args:
        epidermis_pixels: 表皮區域像素
        dermis_pixels: 真皮區域像素
        epidermis_stats: 表皮區域統計
        dermis_stats: 真皮區域統計
        output_path: 輸出路徑
        title: 圖表標題
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # 繪製疊加直方圖
    ax.hist(
        epidermis_pixels,
        bins=HISTOGRAM_BINS,
        range=(0, 256),
        alpha=HISTOGRAM_ALPHA,
        color=EPIDERMIS_COLOR,
        label=f'Epidermis (n={epidermis_stats["count"]:,})',
        edgecolor='black',
        linewidth=0.3
    )

    ax.hist(
        dermis_pixels,
        bins=HISTOGRAM_BINS,
        range=(0, 256),
        alpha=HISTOGRAM_ALPHA,
        color=DERMIS_COLOR,
        label=f'Dermis (n={dermis_stats["count"]:,})',
        edgecolor='black',
        linewidth=0.3
    )

    # 添加平均值標記線
    ax.axvline(
        epidermis_stats['mean'],
        color='darkgreen',
        linestyle='--',
        linewidth=2,
        label=f'Epidermis Mean: {epidermis_stats["mean"]:.1f}'
    )
    ax.axvline(
        dermis_stats['mean'],
        color='darkorange',
        linestyle='--',
        linewidth=2,
        label=f'Dermis Mean: {dermis_stats["mean"]:.1f}'
    )

    # 設定標籤和標題
    ax.set_xlabel('Pixel Intensity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_ylim(0, 24000)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 255)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加統計資訊文字框
    stats_text = (
        f"Epidermis Statistics:\n"
        f"  Mean: {epidermis_stats['mean']:.2f}\n"
        f"  Std:  {epidermis_stats['std']:.2f}\n"
        f"  Median: {epidermis_stats['median']:.2f}\n"
        f"  Range: [{epidermis_stats['min']:.0f}, {epidermis_stats['max']:.0f}]\n"
        f"\n"
        f"Dermis Statistics:\n"
        f"  Mean: {dermis_stats['mean']:.2f}\n"
        f"  Std:  {dermis_stats['std']:.2f}\n"
        f"  Median: {dermis_stats['median']:.2f}\n"
        f"  Range: [{dermis_stats['min']:.0f}, {dermis_stats['max']:.0f}]"
    )

    # 在圖表左上角添加統計資訊
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def analyze_label_brightness(label_mask: np.ndarray, image: np.ndarray, output_dir: str) -> None:
    """
    統計原始標記遮罩範圍內的像素亮度分布並繪製直方圖
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取標記區域的像素
    label_pixels = image[label_mask > 0]
    
    if len(label_pixels) == 0:
        print("警告: 標記區域內無像素")
        return

    # 計算統計資訊
    stats = compute_statistics(label_pixels)
    
    print("\n" + "=" * 60)
    print("原始標記區域 (Label) 亮度統計")
    print("=" * 60)
    print(f"  像素數量:  {stats['count']:,}")
    print(f"  平均亮度:  {stats['mean']:.2f}")
    print(f"  標準差:    {stats['std']:.2f}")
    print(f"  中位數:    {stats['median']:.2f}")
    print(f"  最小值:    {stats['min']:.0f}")
    print(f"  最大值:    {stats['max']:.0f}")
    
    # 繪製直方圖
    plt.figure(figsize=FIGURE_SIZE)
    plt.hist(
        label_pixels,
        bins=HISTOGRAM_BINS,
        range=(0, 256),
        alpha=0.7,
        color='red',
        label=f'Label Region (n={stats["count"]:,})',
        edgecolor='black',
        linewidth=0.3
    )
    
    plt.axvline(
        stats['mean'],
        color='darkred',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {stats["mean"]:.1f}'
    )
    
    plt.xlabel('Pixel Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Pixel Intensity Distribution within Label Mask', fontsize=14, fontweight='bold')
    plt.xlim(0, 255)
    plt.ylim(0, 1200)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加統計文字
    stats_text = (
        f"Label Statistics:\n"
        f"  Mean: {stats['mean']:.2f}\n"
        f"  Std:  {stats['std']:.2f}\n"
        f"  Median: {stats['median']:.2f}\n"
        f"  Range: [{stats['min']:.0f}, {stats['max']:.0f}]"
    )
    
    plt.text(
        0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    output_path = os.path.join(output_dir, 'label_brightness_distribution.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n儲存標記區域亮度直方圖至: {output_path}")


if __name__ == "__main__":
    # ========================================
    # 載入影像
    # ========================================
    print("載入影像...")
    label_image = cv2.imread(LABEL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    epidermis_mask = cv2.imread(EPIDERMIS_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

    if label_image is None:
        raise FileNotFoundError(f"無法載入 label 影像: {LABEL_IMAGE_PATH}")
    if epidermis_mask is None:
        raise FileNotFoundError(f"無法載入 epidermis mask: {EPIDERMIS_MASK_PATH}")
    if original_image is None:
        raise FileNotFoundError(f"無法載入原始影像: {ORIGINAL_IMAGE_PATH}")

    # 提取綠色通道
    original_green = original_image[:, :, 1]

    print(f"  影像尺寸: {original_green.shape}")
    print(f"  Label 影像: {LABEL_IMAGE_PATH}")
    print(f"  Mask 影像: {EPIDERMIS_MASK_PATH}")
    print(f"  原始影像: {ORIGINAL_IMAGE_PATH}")

    # ========================================
    # 執行 Pipeline (debug 模式)
    # ========================================
    print("\n執行 Pipeline (debug 模式)...")
    pipeline = SkinAnalysisPipeline(PIPELINE_CONFIG)
    final_label, roi_image, debug_output = pipeline.run(
        label_image,
        epidermis_mask,
        original_green,
        debug=True
    )

    # print(f"  背景校正 radius: {PIPELINE_CONFIG['background']['radius']}")
    # print(f"  Dilate offset: {PIPELINE_CONFIG['mask']['dilate_offset']}")

    # ========================================
    # 提取區域像素
    # ========================================
    print("\n提取區域像素...")

    # 表皮區域：Pipeline 處理後（含背景校正與正規化）的影像在表皮 mask 內的像素
    epidermis_pixels = roi_image[epidermis_mask > 0]

    # 真皮區域：Pipeline 處理後（含背景校正與正規化）的影像在真皮 ROI mask 內的像素
    dermis_pixels = roi_image[debug_output.dermis_roi_mask > 0]

    print(f"  表皮區域像素數: {len(epidermis_pixels):,}")
    print(f"  真皮區域像素數: {len(dermis_pixels):,}")

    # ========================================
    # 計算統計資訊
    # ========================================
    print("\n計算統計資訊...")
    epidermis_stats = compute_statistics(epidermis_pixels)
    dermis_stats = compute_statistics(dermis_pixels)

    # ========================================
    # 輸出統計結果
    # ========================================
    print("\n" + "=" * 60)
    print("區域像素分布統計")
    print("=" * 60)

    print("\n表皮區域 (Epidermis):")
    print(f"  像素數量:  {epidermis_stats['count']:,}")
    print(f"  平均值:    {epidermis_stats['mean']:.2f}")
    print(f"  標準差:    {epidermis_stats['std']:.2f}")
    print(f"  中位數:    {epidermis_stats['median']:.2f}")
    print(f"  最小值:    {epidermis_stats['min']:.0f}")
    print(f"  最大值:    {epidermis_stats['max']:.0f}")
    print(f"  Q25:       {epidermis_stats['q25']:.2f}")
    print(f"  Q75:       {epidermis_stats['q75']:.2f}")

    print("\n真皮區域 (Dermis):")
    print(f"  像素數量:  {dermis_stats['count']:,}")
    print(f"  平均值:    {dermis_stats['mean']:.2f}")
    print(f"  標準差:    {dermis_stats['std']:.2f}")
    print(f"  中位數:    {dermis_stats['median']:.2f}")
    print(f"  最小值:    {dermis_stats['min']:.0f}")
    print(f"  最大值:    {dermis_stats['max']:.0f}")
    print(f"  Q25:       {dermis_stats['q25']:.2f}")
    print(f"  Q75:       {dermis_stats['q75']:.2f}")

    # 計算差異
    mean_diff = epidermis_stats['mean'] - dermis_stats['mean']
    print(f"\n平均值差異 (表皮 - 真皮): {mean_diff:.2f}")

    # ========================================
    # 建立輸出目錄並儲存圖表
    # ========================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 建立標題（包含配置資訊）
    title = (
        f"Epidermis vs Dermis Pixel Intensity Distribution\n"
        # f"(BG Sigma={PIPELINE_CONFIG['background']['sigma']}, "
        # f"Dilate Offset={PIPELINE_CONFIG['mask']['dilate_offset']})"
    )

    output_path = os.path.join(OUTPUT_DIR, 'region_distribution_histogram.png')
    print(f"\n儲存直方圖至: {output_path}")

    create_overlayed_histogram(
        epidermis_pixels,
        dermis_pixels,
        epidermis_stats,
        dermis_stats,
        output_path,
        title=title
    )

    # ========================================
    # 分析標記區域亮度
    # ========================================
    analyze_label_brightness(label_image, roi_image, OUTPUT_DIR)

    print("\n完成！")
