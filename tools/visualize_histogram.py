"""
影像直方圖視覺化工具

分析和視覺化影像的強度分布直方圖

使用方式:
python visualize_histogram.py --input image.tif --output histogram.png
python visualize_histogram.py --input image.tif --output-dir output/ --detailed
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


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


def calculate_histogram(
    image: np.ndarray,
    bins: int = 256,
    range_min: int = 0,
    range_max: int = 255,
    exclude_zero: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算影像直方圖

    Args:
        image: 輸入灰階影像
        bins: 直方圖 bin 數量
        range_min: 強度範圍最小值
        range_max: 強度範圍最大值
        exclude_zero: 是否排除強度值為 0 的像素

    Returns:
        (直方圖計數, bin 邊界)
    """
    image_data = image.flatten()

    # 排除強度值為 0 的像素
    if exclude_zero:
        image_data = image_data[image_data > 0]

    hist, bin_edges = np.histogram(
        image_data,
        bins=bins,
        range=(range_min, range_max)
    )

    return hist, bin_edges


def calculate_statistics(image: np.ndarray, exclude_zero: bool = False) -> dict:
    """
    計算影像統計資訊

    Args:
        image: 輸入灰階影像
        exclude_zero: 是否排除強度值為 0 的像素

    Returns:
        統計資訊字典
    """
    # 排除強度值為 0 的像素
    if exclude_zero:
        image_data = image[image > 0]
        if len(image_data) == 0:
            # 如果所有像素都是 0，返回預設值
            return {
                'mean': 0,
                'std': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'q25': 0,
                'q75': 0,
                'non_zero_pixels': 0,
                'total_pixels': image.size,
                'non_zero_percentage': 0
            }
    else:
        image_data = image

    stats = {
        'mean': np.mean(image_data),
        'std': np.std(image_data),
        'median': np.median(image_data),
        'min': np.min(image_data),
        'max': np.max(image_data),
        'q25': np.percentile(image_data, 25),
        'q75': np.percentile(image_data, 75),
        'non_zero_pixels': np.count_nonzero(image),
        'total_pixels': image.size,
        'non_zero_percentage': (np.count_nonzero(image) / image.size) * 100
    }

    return stats


def visualize_histogram_basic(
    image: np.ndarray,
    output_path: str,
    bins: int = 256,
    title: str = None,
    color: str = 'steelblue',
    dpi: int = 150,
    exclude_zero: bool = False
):
    """
    基本直方圖視覺化

    Args:
        image: 輸入灰階影像
        output_path: 輸出路徑
        bins: 直方圖 bin 數量
        title: 圖表標題
        color: 直方圖顏色
        dpi: 輸出解析度
        exclude_zero: 是否排除強度值為 0 的像素
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    # 計算直方圖
    hist, bin_edges = calculate_histogram(image, bins=bins, exclude_zero=exclude_zero)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 繪製直方圖
    ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0],
           color=color, alpha=0.7, edgecolor='none')

    # 計算統計
    stats = calculate_statistics(image, exclude_zero=exclude_zero)

    # 添加統計線
    ax.axvline(stats['mean'], color='red', linestyle='--',
              linewidth=2, label=f"Mean: {stats['mean']:.2f}")
    ax.axvline(stats['median'], color='green', linestyle='--',
              linewidth=2, label=f"Median: {stats['median']:.2f}")

    # 設定標籤
    ax.set_xlabel('Intensity', fontsize=12, weight='bold')
    ax.set_ylabel('Frequency (pixels)', fontsize=12, weight='bold')

    if title:
        ax.set_title(title, fontsize=14, weight='bold')
    else:
        ax.set_title('Image Intensity Histogram', fontsize=14, weight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存基本直方圖: {output_path}")


def visualize_histogram_detailed(
    image: np.ndarray,
    output_path: str,
    bins: int = 256,
    title: str = None,
    dpi: int = 150,
    exclude_zero: bool = False
):
    """
    詳細直方圖視覺化（含累積分布和統計資訊）

    Args:
        image: 輸入灰階影像
        output_path: 輸出路徑
        bins: 直方圖 bin 數量
        title: 圖表標題
        dpi: 輸出解析度
        exclude_zero: 是否排除強度值為 0 的像素
    """
    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 計算統計
    stats = calculate_statistics(image, exclude_zero=exclude_zero)
    hist, bin_edges = calculate_histogram(image, bins=bins, exclude_zero=exclude_zero)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 1. 直方圖（左上）
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0],
           color='steelblue', alpha=0.7, edgecolor='none')
    ax1.axvline(stats['mean'], color='red', linestyle='--',
               linewidth=2, label=f"Mean: {stats['mean']:.2f}")
    ax1.axvline(stats['median'], color='green', linestyle='--',
               linewidth=2, label=f"Median: {stats['median']:.2f}")
    ax1.set_xlabel('Intensity', fontsize=11, weight='bold')
    ax1.set_ylabel('Frequency (pixels)', fontsize=11, weight='bold')
    ax1.set_title('Intensity Histogram', fontsize=12, weight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 2. 累積分布函數（中左）
    ax2 = fig.add_subplot(gs[1, 0])
    cumsum = np.cumsum(hist)
    cumsum_normalized = cumsum / cumsum[-1] * 100
    ax2.plot(bin_centers, cumsum_normalized, color='purple', linewidth=2)
    ax2.fill_between(bin_centers, cumsum_normalized, alpha=0.3, color='purple')
    ax2.set_xlabel('Intensity', fontsize=11, weight='bold')
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=11, weight='bold')
    ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 100])

    # 3. 對數尺度直方圖（中右）
    ax3 = fig.add_subplot(gs[1, 1])
    hist_nonzero = np.where(hist > 0, hist, 1)  # 避免 log(0)
    ax3.bar(bin_centers, hist_nonzero, width=bin_edges[1]-bin_edges[0],
           color='orange', alpha=0.7, edgecolor='none')
    ax3.set_yscale('log')
    ax3.set_xlabel('Intensity', fontsize=11, weight='bold')
    ax3.set_ylabel('Frequency (log scale)', fontsize=11, weight='bold')
    ax3.set_title('Histogram (Log Scale)', fontsize=12, weight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # 4. 統計資訊表格（下左）
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    stats_text = [
        ['Statistic', 'Value'],
        ['Mean', f"{stats['mean']:.2f}"],
        ['Std Dev', f"{stats['std']:.2f}"],
        ['Median', f"{stats['median']:.2f}"],
        ['Min', f"{stats['min']:.0f}"],
        ['Max', f"{stats['max']:.0f}"],
        ['Q25', f"{stats['q25']:.2f}"],
        ['Q75', f"{stats['q75']:.2f}"],
        ['Non-zero pixels', f"{stats['non_zero_pixels']:,}"],
        ['Non-zero %', f"{stats['non_zero_percentage']:.2f}%"],
    ]

    table = ax4.table(cellText=stats_text, cellLoc='left',
                     colWidths=[0.5, 0.5], loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 設定表頭樣式
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 設定交替行顏色
    for i in range(1, len(stats_text)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax4.set_title('Statistical Summary', fontsize=12, weight='bold', pad=10)

    # 5. 箱線圖（下右）
    ax5 = fig.add_subplot(gs[2, 1])
    bp = ax5.boxplot([image.flatten()], vert=True, patch_artist=True,
                     labels=['Intensity'])
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax5.set_ylabel('Intensity', fontsize=11, weight='bold')
    ax5.set_title('Box Plot', fontsize=12, weight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 整體標題
    if title:
        fig.suptitle(title, fontsize=16, weight='bold', y=0.98)
    else:
        fig.suptitle('Detailed Histogram Analysis', fontsize=16, weight='bold', y=0.98)

    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存詳細直方圖: {output_path}")


def visualize_histogram_with_image(
    image: np.ndarray,
    output_path: str,
    bins: int = 256,
    title: str = None,
    dpi: int = 150,
    exclude_zero: bool = False
):
    """
    直方圖與影像並排視覺化

    Args:
        image: 輸入灰階影像
        output_path: 輸出路徑
        bins: 直方圖 bin 數量
        title: 圖表標題
        dpi: 輸出解析度
        exclude_zero: 是否排除強度值為 0 的像素
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)

    # 左側：影像
    im = ax1.imshow(image, cmap='gray')
    ax1.set_title('Image', fontsize=12, weight='bold')
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, label='Intensity', shrink=0.8)

    # 右側：直方圖
    hist, bin_edges = calculate_histogram(image, bins=bins, exclude_zero=exclude_zero)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    stats = calculate_statistics(image, exclude_zero=exclude_zero)

    ax2.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0],
           color='steelblue', alpha=0.7, edgecolor='none')
    ax2.axvline(stats['mean'], color='red', linestyle='--',
               linewidth=2, label=f"Mean: {stats['mean']:.2f}")
    ax2.axvline(stats['median'], color='green', linestyle='--',
               linewidth=2, label=f"Median: {stats['median']:.2f}")

    ax2.set_xlabel('Intensity', fontsize=11, weight='bold')
    ax2.set_ylabel('Frequency (pixels)', fontsize=11, weight='bold')
    ax2.set_title('Intensity Histogram', fontsize=12, weight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    if title:
        fig.suptitle(title, fontsize=14, weight='bold')
    else:
        fig.suptitle('Image and Histogram', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存影像與直方圖: {output_path}")


def compare_histograms(
    images: List[np.ndarray],
    labels: List[str],
    output_path: str,
    bins: int = 256,
    title: str = None,
    dpi: int = 150,
    exclude_zero: bool = False
):
    """
    比較多個影像的直方圖

    Args:
        images: 影像列表
        labels: 標籤列表
        output_path: 輸出路徑
        bins: 直方圖 bin 數量
        title: 圖表標題
        dpi: 輸出解析度
        exclude_zero: 是否排除強度值為 0 的像素
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    colors = plt.cm.tab10(np.linspace(0, 1, len(images)))

    for idx, (image, label) in enumerate(zip(images, labels)):
        hist, bin_edges = calculate_histogram(image, bins=bins, exclude_zero=exclude_zero)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 標準化直方圖（轉為機率密度）
        hist_normalized = hist / hist.sum()

        ax.plot(bin_centers, hist_normalized, color=colors[idx],
               linewidth=2, label=label, alpha=0.8)
        ax.fill_between(bin_centers, hist_normalized, alpha=0.2, color=colors[idx])

    ax.set_xlabel('Intensity', fontsize=12, weight='bold')
    ax.set_ylabel('Normalized Frequency', fontsize=12, weight='bold')

    if title:
        ax.set_title(title, fontsize=14, weight='bold')
    else:
        ax.set_title('Histogram Comparison', fontsize=14, weight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存直方圖比較: {output_path}")


def save_statistics_report(stats: dict, output_path: str):
    """
    保存統計報告

    Args:
        stats: 統計資訊字典
        output_path: 輸出路徑
    """
    report = []
    report.append("=" * 60)
    report.append("影像強度統計報告")
    report.append("=" * 60)
    report.append("")

    report.append("【基本統計】")
    report.append(f"  平均值 (Mean):     {stats['mean']:.2f}")
    report.append(f"  標準差 (Std Dev):  {stats['std']:.2f}")
    report.append(f"  中位數 (Median):   {stats['median']:.2f}")
    report.append("")

    report.append("【範圍】")
    report.append(f"  最小值 (Min):      {stats['min']:.0f}")
    report.append(f"  最大值 (Max):      {stats['max']:.0f}")
    report.append(f"  範圍 (Range):      {stats['max'] - stats['min']:.0f}")
    report.append("")

    report.append("【四分位數】")
    report.append(f"  第一四分位數 (Q25): {stats['q25']:.2f}")
    report.append(f"  第二四分位數 (Q50): {stats['median']:.2f}")
    report.append(f"  第三四分位數 (Q75): {stats['q75']:.2f}")
    report.append(f"  四分位距 (IQR):     {stats['q75'] - stats['q25']:.2f}")
    report.append("")

    report.append("【像素統計】")
    report.append(f"  總像素數:          {stats['total_pixels']:,}")
    report.append(f"  非零像素數:        {stats['non_zero_pixels']:,}")
    report.append(f"  非零像素百分比:    {stats['non_zero_percentage']:.2f}%")
    report.append("")

    report.append("=" * 60)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"  ✓ 已保存統計報告: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='視覺化影像強度直方圖',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本直方圖
  python visualize_histogram.py \\
      --input image.tif \\
      --output histogram.png

  # 詳細直方圖分析
  python visualize_histogram.py \\
      --input image.tif \\
      --output-dir output/histogram/ \\
      --detailed

  # 影像與直方圖並排
  python visualize_histogram.py \\
      --input image.tif \\
      --output histogram_with_image.png \\
      --with-image

  # 完整分析（所有視覺化 + 統計報告）
  python visualize_histogram.py \\
      --input image.tif \\
      --output-dir output/histogram/ \\
      --detailed \\
      --with-image \\
      --statistics

  # 自訂 bins 數量
  python visualize_histogram.py \\
      --input image.tif \\
      --output histogram.png \\
      --bins 128

  # 排除黑色像素 (強度值為 0)
  python visualize_histogram.py \\
      --input image.tif \\
      --output histogram.png \\
      --exclude-zero
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

    # 可選參數
    parser.add_argument(
        '--bins', '-b',
        type=int,
        default=256,
        help='直方圖 bin 數量（預設: 256）'
    )

    parser.add_argument(
        '--green-channel', '-g',
        action='store_true',
        help='只使用綠色通道'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='生成詳細直方圖分析（需要 --output-dir）'
    )

    parser.add_argument(
        '--with-image',
        action='store_true',
        help='生成影像與直方圖並排視覺化'
    )

    parser.add_argument(
        '--statistics', '-s',
        action='store_true',
        help='生成統計報告'
    )

    parser.add_argument(
        '--color',
        default='steelblue',
        help='直方圖顏色（預設: steelblue）'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='輸出解析度（預設: 150）'
    )

    parser.add_argument(
        '--title',
        default=None,
        help='圖表標題（可選）'
    )

    parser.add_argument(
        '--exclude-zero',
        action='store_true',
        help='排除強度值為 0 的像素（黑色像素）'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 錯誤: 輸入檔案不存在: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("影像直方圖視覺化")
    print("=" * 60)

    try:
        # 步驟 1: 載入影像
        print(f"\n[1/3] 載入影像...")
        print(f"  輸入: {args.input}")
        image = load_image(args.input)

        # 提取綠色通道（如果需要）
        if args.green_channel or len(image.shape) == 3:
            print("  提取綠色通道...")
            image = extract_green_channel(image)

        print(f"  影像尺寸: {image.shape}")
        print(f"  強度範圍: {image.min()} - {image.max()}")

        # 步驟 2: 計算統計
        print(f"\n[2/3] 計算統計資訊...")
        if args.exclude_zero:
            print("  (排除強度值為 0 的像素)")
        stats = calculate_statistics(image, exclude_zero=args.exclude_zero)
        print(f"  平均值: {stats['mean']:.2f}")
        print(f"  標準差: {stats['std']:.2f}")
        print(f"  中位數: {stats['median']:.2f}")

        # 步驟 3: 生成視覺化
        print(f"\n[3/3] 生成視覺化...")

        if args.output_dir:
            # 輸出到目錄模式
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = input_path.stem

            # 基本直方圖
            basic_path = output_dir / f"{base_name}_histogram.png"
            visualize_histogram_basic(
                image=image,
                output_path=str(basic_path),
                bins=args.bins,
                title=args.title,
                color=args.color,
                dpi=args.dpi,
                exclude_zero=args.exclude_zero
            )

            # 詳細直方圖
            if args.detailed:
                detailed_path = output_dir / f"{base_name}_histogram_detailed.png"
                visualize_histogram_detailed(
                    image=image,
                    output_path=str(detailed_path),
                    bins=args.bins,
                    title=args.title,
                    dpi=args.dpi,
                    exclude_zero=args.exclude_zero
                )

            # 影像與直方圖
            if args.with_image:
                with_image_path = output_dir / f"{base_name}_histogram_with_image.png"
                visualize_histogram_with_image(
                    image=image,
                    output_path=str(with_image_path),
                    bins=args.bins,
                    title=args.title,
                    dpi=args.dpi,
                    exclude_zero=args.exclude_zero
                )

            # 統計報告
            if args.statistics:
                stats_path = output_dir / f"{base_name}_statistics.txt"
                save_statistics_report(stats, str(stats_path))

        else:
            # 單一輸出檔案模式
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            visualize_histogram_basic(
                image=image,
                output_path=args.output,
                bins=args.bins,
                title=args.title,
                color=args.color,
                dpi=args.dpi,
                exclude_zero=args.exclude_zero
            )

        print("\n" + "=" * 60)
        print("✓ 視覺化完成！")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 視覺化失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
