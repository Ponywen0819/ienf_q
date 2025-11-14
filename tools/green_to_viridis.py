"""
綠色通道轉 Viridis 視覺化工具

將輸入影像的綠色通道提取並使用 Viridis 色彩映射進行視覺化

使用方式:
python green_to_viridis.py --input image.tif --output output.png
python green_to_viridis.py --input image.tif --output output.png --colorbar --title "My Image"
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def extract_green_channel(image_path: str) -> np.ndarray:
    """
    提取影像的綠色通道

    Args:
        image_path: 影像路徑

    Returns:
        綠色通道灰階影像
    """
    # 讀取影像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"無法讀取影像: {image_path}")

    # 如果是灰階影像，直接返回
    if len(image.shape) == 2:
        print("  輸入為灰階影像，直接使用")
        return image

    # 如果是彩色影像，提取綠色通道
    if len(image.shape) == 3:
        # OpenCV 讀取順序是 BGR
        if image.shape[2] >= 3:
            green_channel = image[:, :, 1]  # G 通道
            print(f"  已提取綠色通道，尺寸: {green_channel.shape}")
            return green_channel
        else:
            raise ValueError(f"影像通道數不足: {image.shape[2]}")

    raise ValueError(f"不支援的影像格式: {image.shape}")


def apply_viridis_colormap(
    green_channel: np.ndarray,
    output_path: str,
    title: str = None,
    show_colorbar: bool = True,
    alpha: float = 1.0,
    dpi: int = 150,
    figsize: tuple = None,
    vmin: int = None,
    vmax: int = None
):
    """
    將綠色通道應用 Viridis 色彩映射並保存

    Args:
        green_channel: 綠色通道灰階影像
        output_path: 輸出路徑
        title: 圖表標題（可選）
        show_colorbar: 是否顯示 colorbar
        alpha: 透明度 (0-1)
        dpi: 輸出解析度
        figsize: 圖表尺寸 (width, height)，None 則自動計算
        vmin: 最小顯示值（None 則使用影像最小值）
        vmax: 最大顯示值（None 則使用影像最大值）
    """
    # 計算合適的圖表尺寸
    if figsize is None:
        height, width = green_channel.shape
        # 以 100 像素/英寸為基準計算
        figsize = (width / 100, height / 100)
        # 限制最大尺寸
        max_size = 20
        if figsize[0] > max_size or figsize[1] > max_size:
            scale = max_size / max(figsize)
            figsize = (figsize[0] * scale, figsize[1] * scale)

    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 應用 Viridis 色彩映射
    im = ax.imshow(green_channel, cmap='viridis', alpha=alpha,
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    # 添加 colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, label='Green Channel Intensity',
                           pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

    # 設定標題
    if title:
        ax.set_title(title, fontsize=14, weight='bold', pad=10)

    # 設定軸標籤
    ax.set_xlabel('X coordinate (pixels)', fontsize=10)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=10)

    # 顯示網格（可選）
    # ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 調整佈局
    plt.tight_layout()

    # 保存圖表
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存 Viridis 視覺化: {output_path}")
    print(f"    - 尺寸: {green_channel.shape}")
    print(f"    - 強度範圍: {green_channel.min()} - {green_channel.max()}")


def apply_viridis_colormap_direct(
    green_channel: np.ndarray,
    output_path: str,
    vmin: int = None,
    vmax: int = None
):
    """
    直接將 Viridis 色彩映射應用到影像並保存（無邊框、軸等）

    Args:
        green_channel: 綠色通道灰階影像
        output_path: 輸出路徑
        vmin: 最小顯示值（None 則使用影像最小值）
        vmax: 最大顯示值（None 則使用影像最小值）
    """
    # 標準化到 0-1
    if vmin is None:
        vmin = green_channel.min()
    if vmax is None:
        vmax = green_channel.max()

    normalized = (green_channel - vmin) / (vmax - vmin)
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
    print(f"    - 尺寸: {green_channel.shape}")
    print(f"    - 強度範圍: {green_channel.min()} - {green_channel.max()}")


def main():
    parser = argparse.ArgumentParser(
        description='將影像的綠色通道轉換為 Viridis 色彩映射',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用
  python green_to_viridis.py --input image.tif --output output.png

  # 添加標題和 colorbar
  python green_to_viridis.py --input image.tif --output output.png \\
      --title "Nerve Fiber Image" --colorbar

  # 調整透明度和解析度
  python green_to_viridis.py --input image.tif --output output.png \\
      --alpha 0.8 --dpi 300

  # 無邊框模式（直接輸出影像）
  python green_to_viridis.py --input image.tif --output output.png --direct

  # 指定顯示範圍
  python green_to_viridis.py --input image.tif --output output.png \\
      --vmin 0 --vmax 255
        """
    )

    # 必要參數
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='輸入影像路徑'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='輸出影像路徑'
    )

    # 可選參數
    parser.add_argument(
        '--title', '-t',
        default=None,
        help='圖表標題（可選）'
    )

    parser.add_argument(
        '--colorbar',
        action='store_true',
        help='顯示 colorbar（預設: False）'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='透明度 0-1（預設: 1.0）'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='輸出解析度（預設: 150）'
    )

    parser.add_argument(
        '--width',
        type=float,
        default=None,
        help='圖表寬度（英寸，可選）'
    )

    parser.add_argument(
        '--height',
        type=float,
        default=None,
        help='圖表高度（英寸，可選）'
    )

    parser.add_argument(
        '--vmin',
        type=int,
        default=None,
        help='最小顯示值（可選，預設使用影像最小值）'
    )

    parser.add_argument(
        '--vmax',
        type=int,
        default=None,
        help='最大顯示值（可選，預設使用影像最大值）'
    )

    parser.add_argument(
        '--direct',
        action='store_true',
        help='直接模式：輸出無邊框的 Viridis 影像（忽略 colorbar, title 等選項）'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 錯誤: 輸入檔案不存在: {args.input}")
        sys.exit(1)

    # 創建輸出目錄
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("綠色通道轉 Viridis 視覺化")
    print("=" * 60)

    try:
        # 步驟 1: 提取綠色通道
        print(f"\n[1/2] 讀取影像並提取綠色通道...")
        print(f"  輸入: {args.input}")
        green_channel = extract_green_channel(args.input)

        # 步驟 2: 應用 Viridis 並保存
        print(f"\n[2/2] 應用 Viridis 色彩映射...")

        if args.direct:
            # 直接模式
            apply_viridis_colormap_direct(
                green_channel=green_channel,
                output_path=args.output,
                vmin=args.vmin,
                vmax=args.vmax
            )
        else:
            # 完整模式
            figsize = None
            if args.width and args.height:
                figsize = (args.width, args.height)

            apply_viridis_colormap(
                green_channel=green_channel,
                output_path=args.output,
                title=args.title,
                show_colorbar=args.colorbar,
                alpha=args.alpha,
                dpi=args.dpi,
                figsize=figsize,
                vmin=args.vmin,
                vmax=args.vmax
            )

        print("\n" + "=" * 60)
        print("✓ 轉換完成！")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 轉換失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
