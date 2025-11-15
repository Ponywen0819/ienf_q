#!/usr/bin/env python3
"""
對比度增強腳本 (Contrast Enhancement Script)

實作 CLAHE (對比度受限自適應直方圖均衡化) 演算法，用於增強神經纖維影像的局部對比度。

使用方式:
    python contrast_enhancement.py -i green_channel.png -o enhanced.png -c 3.0 -t 8

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


class ContrastEnhancer:
    """CLAHE 對比度增強器"""

    def __init__(
        self,
        clip_limit: float = 3.0,
        tile_size: int = 8,
        verbose: bool = False
    ):
        """
        初始化對比度增強器

        Args:
            clip_limit: 對比度限制參數 (2.0-4.0)，限制局部對比度增強的程度
            tile_size: 網格大小 (8 或 16)，影像被分割成 tile_size x tile_size 的網格
            verbose: 是否輸出詳細資訊
        """
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.verbose = verbose

        # 建立 CLAHE 物件
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
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

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        執行 CLAHE 對比度增強

        CLAHE 原理:
        1. 將影像分割成小網格 (tiles)
        2. 對每個網格執行直方圖均衡化
        3. 限制對比度增強程度 (clip_limit) 以避免雜訊放大
        4. 使用雙線性插值平滑網格邊界

        效果:
        - 平衡不同區域的亮度
        - 增強局部對比度
        - 提升神經纖維與背景的區分度

        Args:
            image: 輸入灰階影像

        Returns:
            增強後的影像
        """
        if self.verbose:
            print(f"\n執行 CLAHE 對比度增強...")
            print(f"  對比度限制 (clip_limit): {self.clip_limit}")
            print(f"  網格大小 (tile_size): {self.tile_size}x{self.tile_size}")
            print(f"  網格數量: {image.shape[0]//self.tile_size} x {image.shape[1]//self.tile_size}")

        enhanced = self.clahe.apply(image)

        if self.verbose:
            print(f"✓ 對比度增強完成")
            print(f"  增強後像素值範圍: [{enhanced.min()}, {enhanced.max()}]")
            print(f"  增強後平均強度: {enhanced.mean():.2f}")
            print(f"  增強後標準差: {enhanced.std():.2f}")

            # 計算對比度改善
            original_contrast = image.std()
            enhanced_contrast = enhanced.std()
            improvement = ((enhanced_contrast - original_contrast) / original_contrast) * 100
            print(f"  對比度改善: {improvement:+.2f}%")

        return enhanced

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
        enhanced: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        視覺化原始影像和增強結果的對比

        Args:
            original: 原始影像
            enhanced: 增強後影像
            save_path: 儲存路徑 (None 則顯示而不儲存)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original image
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Enhanced image
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title('CLAHE Enhanced', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Original image histogram
        axes[1, 0].hist(original.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        axes[1, 0].set_title('Original Histogram', fontsize=12)
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Enhanced image histogram
        axes[1, 1].hist(enhanced.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
        axes[1, 1].set_title('Enhanced Histogram', fontsize=12)
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        # Main title
        plt.suptitle(
            f'CLAHE Contrast Enhancement (clip_limit={self.clip_limit}, tile_size={self.tile_size})',
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
        enhanced: np.ndarray,
        output_path: str
    ) -> None:
        """
        單獨儲存直方圖對比圖

        Args:
            original: 原始影像
            enhanced: 增強後影像
            output_path: 輸出檔案路徑
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Original image histogram
        axes[0].hist(original.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_title('Original Histogram', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)

        # Statistics
        axes[0].axvline(original.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {original.mean():.1f}')
        axes[0].legend()

        # Enhanced image histogram
        axes[1].hist(enhanced.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7, edgecolor='black')
        axes[1].set_title('CLAHE Enhanced Histogram', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)

        # Statistics
        axes[1].axvline(enhanced.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {enhanced.mean():.1f}')
        axes[1].legend()

        plt.suptitle(
            f'Histogram Comparison (clip_limit={self.clip_limit}, tile_size={self.tile_size})',
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
    ) -> np.ndarray:
        """
        完整的對比度增強流程

        Args:
            input_path: 輸入影像路徑 (單通道灰階)
            output_path: 輸出影像路徑 (None 則自動生成)
            visualize: 是否顯示視覺化對比
            save_histogram: 是否儲存直方圖對比圖

        Returns:
            增強後的影像陣列
        """
        if self.verbose:
            print("=" * 60)
            print("CLAHE 對比度增強")
            print("=" * 60)

        # 1. 載入影像
        image = self.load_image(input_path)

        # 2. 執行對比度增強
        enhanced = self.enhance(image)

        # 3. 儲存結果
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(
                input_path_obj.parent /
                f"{input_path_obj.stem}_enhanced{input_path_obj.suffix}"
            )

        self.save_image(enhanced, output_path)

        # 4. 儲存直方圖 (可選)
        if save_histogram:
            hist_path = str(Path(output_path).parent /
                          f"{Path(output_path).stem}_histogram.png")
            self.save_histogram(image, enhanced, hist_path)

        # 5. 視覺化 (可選)
        if visualize:
            self.visualize_comparison(image, enhanced)

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 對比度增強完成!")
            print("=" * 60)

        return enhanced


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='神經纖維影像對比度增強工具 (CLAHE 演算法)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用
  python %(prog)s -i green_channel.png -o enhanced.png

  # 調整 CLAHE 參數
  python %(prog)s -i green_channel.png -c 2.5 -t 16

  # 顯示視覺化對比
  python %(prog)s -i green_channel.png -v

  # 完整參數
  python %(prog)s -i data/green_corrected.png -o data/enhanced.png -c 3.0 -t 8 -v --save-histogram

參數說明:
  對比度限制 (--clip-limit):
    - 建議範圍: 2.0-4.0
    - 較大的值會產生更強的對比度增強
    - 但過大可能會放大雜訊

  網格大小 (--tile-size):
    - 8x8: 更細緻的局部對比度調整
    - 16x16: 更平滑的整體效果
    - 較小的網格可以更好地處理局部亮度變化

注意事項:
  - 輸入影像必須是單通道灰階影像（例如綠色通道）
  - 建議在背景校正後使用此腳本
  - CLAHE 會增強局部對比度，有助於後續的分割和特徵提取
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
        help='輸出影像路徑 (預設: {input}_enhanced.png)'
    )

    parser.add_argument(
        '-c', '--clip-limit',
        type=float,
        default=3.0,
        metavar='LIMIT',
        help='對比度限制參數 (預設: 3.0, 建議範圍: 2.0-4.0)'
    )

    parser.add_argument(
        '-t', '--tile-size',
        type=int,
        choices=[8, 16],
        default=8,
        help='網格大小 (預設: 8, 可選: 8 或 16)'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='顯示原始影像、增強結果和直方圖的對比圖'
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
    if args.clip_limit < 1.0:
        print(f"警告: clip_limit ({args.clip_limit}) 過小，增強效果可能不明顯", file=sys.stderr)
    elif args.clip_limit > 10.0:
        print(f"警告: clip_limit ({args.clip_limit}) 過大，可能會過度放大雜訊", file=sys.stderr)

    try:
        # 建立對比度增強器
        enhancer = ContrastEnhancer(
            clip_limit=args.clip_limit,
            tile_size=args.tile_size,
            verbose=args.verbose
        )

        # 執行對比度增強
        enhancer.process(
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
