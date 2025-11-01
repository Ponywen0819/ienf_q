#!/usr/bin/env python3
"""
影像標準化腳本 (Image Normalization Script)

實作百分位數標準化演算法，用於統一神經纖維影像的動態範圍並去除極端值。

使用方式:
    python normalization.py -i enhanced.png -o normalized.png -l 1.0 -u 99.0

作者: Generated with Claude Code
日期: 2025-10-22
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageNormalizer:
    """百分位數影像標準化器"""

    def __init__(
        self,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        output_min: int = 0,
        output_max: int = 255,
        verbose: bool = False
    ):
        """
        初始化影像標準化器

        Args:
            lower_percentile: 下界百分位數 (0-100)
            upper_percentile: 上界百分位數 (0-100)
            output_min: 輸出範圍最小值
            output_max: 輸出範圍最大值
            verbose: 是否輸出詳細資訊
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.output_min = output_min
        self.output_max = output_max
        self.verbose = verbose

        # 參數驗證
        if not (0 <= lower_percentile < upper_percentile <= 100):
            raise ValueError(
                f"百分位數範圍錯誤: lower_percentile ({lower_percentile}) "
                f"必須小於 upper_percentile ({upper_percentile})，且都在 [0, 100] 範圍內"
            )

        if output_min >= output_max:
            raise ValueError(
                f"輸出範圍錯誤: output_min ({output_min}) 必須小於 output_max ({output_max})"
            )

    def load_image(self, image_path: str) -> np.ndarray:
        """
        載入灰階影像

        Args:
            image_path: 影像檔案路徑

        Returns:
            灰階影像陣列

        Raises:
            FileNotFoundError: 檔案不存在
            ValueError: 無法讀取影像或影像不是單通道
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"影像檔案不存在: {image_path}")

        # 強制以灰階模式讀取
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        # 驗證是否為單通道
        if len(image.shape) != 2:
            raise ValueError(
                f"輸入影像必須是單通道灰階影像，"
                f"但收到的是 {len(image.shape)} 通道影像"
            )

        if self.verbose:
            print(f"✓ 成功載入灰階影像: {image_path}")
            print(f"  影像尺寸: {image.shape[1]}x{image.shape[0]} (寬x高)")
            print(f"  像素值範圍: [{image.min()}, {image.max()}]")
            print(f"  平均強度: {image.mean():.2f}")
            print(f"  標準差: {image.std():.2f}")

        return image

    def normalize(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        執行百分位數標準化

        處理流程:
        1. 計算指定的下界和上界百分位數
        2. 裁剪超出範圍的極端值
        3. 線性映射到目標輸出範圍 [output_min, output_max]

        效果:
        - 移除異常亮點和暗點（如染色瑕疵、雜訊）
        - 統一影像動態範圍
        - 增強主要訊號的對比度

        Args:
            image: 輸入灰階影像

        Returns:
            (標準化後影像, 統計資訊字典)
        """
        if self.verbose:
            print(f"\n執行百分位數標準化...")
            print(f"  下界百分位數: {self.lower_percentile}%")
            print(f"  上界百分位數: {self.upper_percentile}%")
            print(f"  輸出範圍: [{self.output_min}, {self.output_max}]")

        # 1. 計算百分位數
        p_low = np.percentile(image, self.lower_percentile)
        p_high = np.percentile(image, self.upper_percentile)

        if self.verbose:
            print(f"\n  計算得到的百分位數值:")
            print(f"    {self.lower_percentile}% 百分位數: {p_low:.2f}")
            print(f"    {self.upper_percentile}% 百分位數: {p_high:.2f}")
            print(f"    動態範圍: {p_high - p_low:.2f}")

        # 2. 裁剪極端值
        clipped = np.clip(image, p_low, p_high)

        # 計算被裁剪的像素數量
        n_clipped_low = np.sum(image < p_low)
        n_clipped_high = np.sum(image > p_high)
        total_pixels = image.size

        if self.verbose:
            print(f"\n  裁剪統計:")
            print(f"    低於下界的像素: {n_clipped_low} ({n_clipped_low/total_pixels*100:.2f}%)")
            print(f"    高於上界的像素: {n_clipped_high} ({n_clipped_high/total_pixels*100:.2f}%)")
            print(f"    總裁剪像素: {n_clipped_low + n_clipped_high} "
                  f"({(n_clipped_low + n_clipped_high)/total_pixels*100:.2f}%)")

        # 3. 線性映射到輸出範圍
        if p_high > p_low:
            # 標準線性映射公式
            normalized = (clipped - p_low) / (p_high - p_low)
            normalized = normalized * (self.output_max - self.output_min) + self.output_min
            normalized = normalized.astype(np.uint8)
        else:
            # 邊界情況：如果上下界相同（影像幾乎均勻）
            if self.verbose:
                print(f"  警告: 上下界百分位數相同，影像可能接近均勻灰度")
            normalized = np.full_like(image, (self.output_max + self.output_min) // 2, dtype=np.uint8)

        if self.verbose:
            print(f"\n✓ 標準化完成")
            print(f"  標準化後像素值範圍: [{normalized.min()}, {normalized.max()}]")
            print(f"  標準化後平均強度: {normalized.mean():.2f}")
            print(f"  標準化後標準差: {normalized.std():.2f}")

        # 收集統計資訊
        stats = {
            'p_low': p_low,
            'p_high': p_high,
            'n_clipped_low': n_clipped_low,
            'n_clipped_high': n_clipped_high,
            'total_pixels': total_pixels,
            'original_range': (image.min(), image.max()),
            'normalized_range': (normalized.min(), normalized.max()),
            'original_mean': image.mean(),
            'normalized_mean': normalized.mean(),
            'original_std': image.std(),
            'normalized_std': normalized.std()
        }

        return normalized, stats

    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        儲存影像

        Args:
            image: 要儲存的影像陣列
            output_path: 輸出檔案路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise IOError(f"無法儲存影像: {output_path}")

        if self.verbose:
            print(f"✓ 影像已儲存: {output_path}")

    def visualize_comparison(
        self,
        original: np.ndarray,
        normalized: np.ndarray,
        stats: dict,
        save_path: Optional[str] = None
    ) -> None:
        """
        視覺化原始影像和標準化結果的對比

        Args:
            original: 原始影像
            normalized: 標準化後影像
            stats: 統計資訊字典
            save_path: 儲存路徑 (None 則顯示而不儲存)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Original image
        axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Normalized image
        axes[0, 1].imshow(normalized, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('Percentile Normalized', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Original image histogram
        axes[1, 0].hist(original.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(stats['p_low'], color='red', linestyle='--', linewidth=2,
                          label=f'{self.lower_percentile}% percentile: {stats["p_low"]:.1f}')
        axes[1, 0].axvline(stats['p_high'], color='red', linestyle='--', linewidth=2,
                          label=f'{self.upper_percentile}% percentile: {stats["p_high"]:.1f}')
        axes[1, 0].set_title('Original Histogram', fontsize=12)
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)

        # Normalized image histogram
        axes[1, 1].hist(normalized.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(normalized.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {normalized.mean():.1f}')
        axes[1, 1].set_title('Normalized Histogram', fontsize=12)
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

        # Main title
        clipped_percent = (stats['n_clipped_low'] + stats['n_clipped_high']) / stats['total_pixels'] * 100
        plt.suptitle(
            f'Percentile Normalization ({self.lower_percentile}%-{self.upper_percentile}%, '
            f'clipped {clipped_percent:.2f}% pixels)',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"✓ 視覺化圖表已儲存: {save_path}")
        else:
            plt.show()

        plt.close()

        if self.verbose and not save_path:
            print("✓ 視覺化完成")

    def save_histogram(
        self,
        original: np.ndarray,
        normalized: np.ndarray,
        stats: dict,
        output_path: str
    ) -> None:
        """
        單獨儲存直方圖對比圖

        Args:
            original: 原始影像
            normalized: 標準化後影像
            stats: 統計資訊字典
            output_path: 輸出檔案路徑
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Original image histogram
        axes[0].hist(original.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, edgecolor='black')
        axes[0].axvline(stats['p_low'], color='red', linestyle='--', linewidth=2,
                       label=f'{self.lower_percentile}% percentile: {stats["p_low"]:.1f}')
        axes[0].axvline(stats['p_high'], color='red', linestyle='--', linewidth=2,
                       label=f'{self.upper_percentile}% percentile: {stats["p_high"]:.1f}')
        axes[0].axvline(original.mean(), color='orange', linestyle=':', linewidth=2,
                       label=f'Mean: {original.mean():.1f}')
        axes[0].set_title('Original Histogram', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Pixel Value', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Normalized image histogram
        axes[1].hist(normalized.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7, edgecolor='black')
        axes[1].axvline(normalized.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {normalized.mean():.1f}')
        axes[1].axvline(normalized.std(), color='orange', linestyle=':', linewidth=2,
                       label=f'Std Dev: {normalized.std():.1f}')
        axes[1].set_title('Normalized Histogram', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Pixel Value', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # Main title
        clipped_percent = (stats['n_clipped_low'] + stats['n_clipped_high']) / stats['total_pixels'] * 100
        plt.suptitle(
            f'Histogram Comparison ({self.lower_percentile}%-{self.upper_percentile}%, clipped {clipped_percent:.2f}% pixels)',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ 直方圖已儲存: {output_path}")

    def process(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        visualize: bool = False,
        save_histogram: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        完整的影像標準化流程

        Args:
            input_path: 輸入影像路徑 (單通道灰階)
            output_path: 輸出影像路徑 (None 則自動生成)
            visualize: 是否顯示視覺化對比
            save_histogram: 是否儲存直方圖對比圖

        Returns:
            (標準化後影像, 統計資訊字典)
        """
        if self.verbose:
            print("=" * 60)
            print("百分位數標準化")
            print("=" * 60)

        # 1. 載入影像
        image = self.load_image(input_path)

        # 2. 執行標準化
        normalized, stats = self.normalize(image)

        # 3. 儲存結果
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(
                input_path_obj.parent /
                f"{input_path_obj.stem}_normalized{input_path_obj.suffix}"
            )

        self.save_image(normalized, output_path)

        # 4. 儲存直方圖 (可選)
        if save_histogram:
            hist_path = str(Path(output_path).parent /
                          f"{Path(output_path).stem}_histogram.png")
            self.save_histogram(image, normalized, stats, hist_path)

        # 5. 視覺化 (可選)
        if visualize:
            self.visualize_comparison(image, normalized, stats)

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 影像標準化完成!")
            print("=" * 60)

        return normalized, stats


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='神經纖維影像標準化工具 (百分位數標準化)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用 (1%-99% 百分位數標準化)
  python %(prog)s -i enhanced.png -o normalized.png

  # 自訂百分位數範圍
  python %(prog)s -i enhanced.png -l 0.5 -u 99.5

  # 顯示視覺化對比
  python %(prog)s -i enhanced.png -v

  # 完整參數
  python %(prog)s -i data/green_enhanced.png -o data/green_normalized.png -l 1.0 -u 99.0 -v --save-histogram

參數說明:
  百分位數範圍:
    - lower-percentile (下界): 通常設為 0.5-2.0，用於去除過暗區域
    - upper-percentile (上界): 通常設為 98.0-99.5，用於去除過亮區域
    - 較寬的範圍 (如 0.1%-99.9%) 保留更多細節但可能包含雜訊
    - 較窄的範圍 (如 2%-98%) 更激進地去除極端值

  輸出範圍:
    - 預設映射到 [0, 255] (標準 8-bit 灰階)
    - 可根據後續處理需求調整

處理流程:
  1. 計算指定百分位數 → 確定有效動態範圍
  2. 裁剪極端值 → 去除異常像素
  3. 線性映射 → 統一到目標範圍

注意事項:
  - 輸入影像必須是單通道灰階影像
  - 建議在 CLAHE 增強後使用
  - 標準化後的影像可直接用於種子點提取
        """
    )

    # 必填參數
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='輸入影像路徑 (必須是單通道灰階影像)'
    )

    # 選填參數
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='輸出影像路徑 (預設: {input}_normalized.png)'
    )

    parser.add_argument(
        '-l', '--lower-percentile',
        type=float,
        default=1.0,
        metavar='PERCENT',
        help='下界百分位數 (預設: 1.0, 範圍: 0-100)'
    )

    parser.add_argument(
        '-u', '--upper-percentile',
        type=float,
        default=99.0,
        metavar='PERCENT',
        help='上界百分位數 (預設: 99.0, 範圍: 0-100)'
    )

    parser.add_argument(
        '--output-min',
        type=int,
        default=0,
        help='輸出範圍最小值 (預設: 0)'
    )

    parser.add_argument(
        '--output-max',
        type=int,
        default=255,
        help='輸出範圍最大值 (預設: 255)'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='顯示原始影像、標準化結果和直方圖的對比圖'
    )

    parser.add_argument(
        '--save-histogram',
        action='store_true',
        help='儲存直方圖對比圖為獨立檔案'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='輸出詳細處理資訊'
    )

    return parser.parse_args()


def main():
    """主程式進入點"""
    args = parse_arguments()

    # 參數驗證
    if args.lower_percentile >= args.upper_percentile:
        print(
            f"錯誤: lower_percentile ({args.lower_percentile}) "
            f"必須小於 upper_percentile ({args.upper_percentile})",
            file=sys.stderr
        )
        return 1

    if not (0 <= args.lower_percentile <= 100 and 0 <= args.upper_percentile <= 100):
        print(f"錯誤: 百分位數必須在 [0, 100] 範圍內", file=sys.stderr)
        return 1

    try:
        # 建立標準化器
        normalizer = ImageNormalizer(
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            output_min=args.output_min,
            output_max=args.output_max,
            verbose=args.verbose
        )

        # 執行標準化
        normalizer.process(
            input_path=args.input,
            output_path=args.output,
            visualize=args.visualize,
            save_histogram=args.save_histogram
        )

        return 0

    except FileNotFoundError as e:
        print(f"錯誤: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"錯誤: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"未預期的錯誤: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
