#!/usr/bin/env python3
"""
背景不均勻矯正腳本 (Background Correction Script)

實作 Rolling Ball 背景扣除演算法，用於消除神經纖維影像中的背景染色不均問題。

使用方式:
    python background_correction.py -i input.tif -o output.png -r 50

作者: Generated with Claude Code
日期: 2025-10-22
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage import restoration, exposure
import matplotlib.pyplot as plt


class BackgroundCorrector:
    """背景不均勻矯正器"""

    CHANNELS = {
        'red': 2,    # OpenCV uses BGR format
        'green': 1,
        'blue': 0,
        'all': None
    }

    def __init__(self, ball_radius: int = 50, verbose: bool = False):
        """
        初始化背景矯正器

        Args:
            ball_radius: Rolling Ball 演算法的球體半徑 (像素)
            verbose: 是否輸出詳細資訊
        """
        self.ball_radius = ball_radius
        self.verbose = verbose

    def load_image(self, image_path: str) -> np.ndarray:
        """
        載入影像

        Args:
            image_path: 影像檔案路徑

        Returns:
            BGR 格式的影像陣列

        Raises:
            FileNotFoundError: 檔案不存在
            ValueError: 無法讀取影像
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"影像檔案不存在: {image_path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        if self.verbose:
            print(f"✓ 成功載入影像: {image_path}")
            print(f"  影像尺寸: {image.shape[1]}x{image.shape[0]} (寬x高)")
            print(f"  色彩空間: {'RGB' if len(image.shape) == 3 else 'Grayscale'}")

        return image

    def extract_channel(self, image: np.ndarray, channel: str = 'green') -> np.ndarray:
        """
        提取指定色彩通道

        Args:
            image: BGR 格式影像
            channel: 通道名稱 ('red', 'green', 'blue', 'all')

        Returns:
            單通道影像陣列或原始影像
        """
        if channel == 'all':
            if self.verbose:
                print(f"✓ 使用完整 RGB 影像")
            return image

        channel_idx = self.CHANNELS[channel]
        extracted = image[:, :, channel_idx]

        if self.verbose:
            print(f"✓ 提取 {channel} 通道")
            print(f"  通道強度範圍: [{extracted.min()}, {extracted.max()}]")
            print(f"  平均強度: {extracted.mean():.2f}")

        return extracted

    def rolling_ball_correction(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        執行 Rolling Ball 背景扣除

        原理:
        1. 模擬球體在影像強度表面下方滾動
        2. 球接觸的表面視為背景
        3. 從原影像扣除背景，得到校正後影像

        Args:
            image: 輸入影像 (灰階或彩色)

        Returns:
            (背景影像, 校正後影像) 的元組
        """
        if self.verbose:
            print(f"\n執行 Rolling Ball 背景扣除...")
            print(f"  球體半徑: {self.ball_radius} 像素")

        # 判斷是否為彩色影像
        is_color = len(image.shape) == 3

        if is_color:
            # 對每個通道分別處理
            background = np.zeros_like(image)
            corrected = np.zeros_like(image)

            for i in range(3):
                channel = image[:, :, i]
                bg = restoration.rolling_ball(channel, radius=self.ball_radius)
                background[:, :, i] = bg
                corrected[:, :, i] = cv2.subtract(channel, bg)

            if self.verbose:
                print(f"  處理模式: 彩色影像 (3通道)")
        else:
            # 單通道處理
            background = restoration.rolling_ball(image, radius=self.ball_radius)
            corrected = cv2.subtract(image, background)

            if self.verbose:
                print(f"  處理模式: 灰階影像")

        if self.verbose:
            print(f"✓ 背景扣除完成")
            print(f"  校正後強度範圍: [{corrected.min()}, {corrected.max()}]")

        return background, corrected

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
        background: np.ndarray,
        corrected: np.ndarray,
        channel: str = 'green'
    ) -> None:
        """
        視覺化原始影像、背景和校正結果的對比

        Args:
            original: 原始影像
            background: 估計的背景
            corrected: 校正後影像
            channel: 通道名稱 (用於標題)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 統一使用灰階色彩映射
        cmap = 'gray'

        # Original image
        axes[0].imshow(original, cmap=cmap)
        axes[0].set_title(f'Original Image ({channel} channel)', fontsize=12)
        axes[0].axis('off')

        # Estimated background
        axes[1].imshow(background, cmap=cmap)
        axes[1].set_title('Estimated Background (Rolling Ball)', fontsize=12)
        axes[1].axis('off')

        # Corrected result
        axes[2].imshow(corrected, cmap=cmap)
        axes[2].set_title('Background Corrected', fontsize=12)
        axes[2].axis('off')

        plt.suptitle(
            f'Background Correction (ball radius: {self.ball_radius}px)',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

        if self.verbose:
            print("✓ 視覺化完成")

    def process(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        channel: str = 'green',
        visualize: bool = False,
        save_intermediate: bool = False
    ) -> np.ndarray:
        """
        完整的背景校正流程

        Args:
            input_path: 輸入影像路徑
            output_path: 輸出影像路徑 (None 則自動生成)
            channel: 要處理的通道
            visualize: 是否顯示視覺化對比
            save_intermediate: 是否儲存中間結果 (背景影像)

        Returns:
            校正後的影像陣列
        """
        if self.verbose:
            print("=" * 60)
            print("背景不均勻矯正 - Rolling Ball 演算法")
            print("=" * 60)

        # 1. 載入影像
        image = self.load_image(input_path)

        # 2. 提取通道
        channel_image = self.extract_channel(image, channel)

        # 3. 執行背景校正
        background, corrected = self.rolling_ball_correction(channel_image)

        # 4. 儲存結果
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(
                input_path_obj.parent /
                f"{input_path_obj.stem}_corrected{input_path_obj.suffix}"
            )

        self.save_image(corrected, output_path)

        # 5. 儲存中間結果 (可選)
        if save_intermediate:
            bg_path = str(Path(output_path).parent /
                         f"{Path(output_path).stem}_background{Path(output_path).suffix}")
            self.save_image(background, bg_path)
            if self.verbose:
                print(f"✓ 背景影像已儲存: {bg_path}")

        # 6. 視覺化 (可選)
        if visualize:
            self.visualize_comparison(channel_image, background, corrected, channel)

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 背景校正完成!")
            print("=" * 60)

        return corrected


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='神經纖維影像背景不均勻矯正工具 (Rolling Ball 演算法)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用 (處理綠色通道)
  python %(prog)s -i input.tif -o output.png

  # 指定球體半徑
  python %(prog)s -i input.tif -r 60

  # 顯示視覺化對比
  python %(prog)s -i input.tif -v

  # 完整參數
  python %(prog)s -i data/raw/sample.tif -o data/processed/sample_corrected.png -c green -r 55 -v --save-intermediate

參數說明:
  球體半徑 (--ball-radius):
    - 建議範圍: 40-60 像素
    - 應大於神經纖維最大寬度
    - 較大的半徑可處理更大範圍的背景變化
        """
    )

    # 必填參數
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='輸入影像路徑 (支援 .tif, .png, .jpg 等格式)'
    )

    # 選填參數
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='輸出影像路徑 (預設: {input}_corrected.png)'
    )

    parser.add_argument(
        '-c', '--channel',
        type=str,
        choices=['red', 'green', 'blue', 'all'],
        default='green',
        help='要處理的色彩通道 (預設: green)'
    )

    parser.add_argument(
        '-r', '--ball-radius',
        type=int,
        default=50,
        metavar='RADIUS',
        help='Rolling Ball 球體半徑，單位：像素 (預設: 50, 建議範圍: 40-60)'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='顯示原始影像、背景和校正結果的對比圖'
    )

    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='儲存中間結果 (背景影像)'
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
    if args.ball_radius < 10:
        print(f"警告: 球體半徑 ({args.ball_radius}) 過小，可能無法有效去除背景", file=sys.stderr)
    elif args.ball_radius > 200:
        print(f"警告: 球體半徑 ({args.ball_radius}) 過大，可能會去除神經纖維訊號", file=sys.stderr)

    try:
        # 建立背景矯正器
        corrector = BackgroundCorrector(
            ball_radius=args.ball_radius,
            verbose=args.verbose
        )

        # 執行背景校正
        corrector.process(
            input_path=args.input,
            output_path=args.output,
            channel=args.channel,
            visualize=args.visualize,
            save_intermediate=args.save_intermediate
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
