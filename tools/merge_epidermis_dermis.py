"""
表皮與真皮影像組合工具

將分開的表皮內（Epidermis）和表皮外（Dermis）影像組合回完整影像

使用方式:
python merge_epidermis_dermis.py \\
    --epidermis split/image_epidermis.tif \\
    --dermis split/image_dermis.tif \\
    --output merged.tif

python merge_epidermis_dermis.py \\
    --epidermis split/image_epidermis.tif \\
    --dermis split/image_dermis.tif \\
    --mask split/image_mask_binary.tif \\
    --output merged.tif
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


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


def create_mask_from_images(
    epidermis: np.ndarray,
    dermis: np.ndarray,
    background_value: int = 0
) -> np.ndarray:
    """
    從表皮和真皮影像反推 mask

    Args:
        epidermis: 表皮內影像
        dermis: 表皮外影像
        background_value: 背景值

    Returns:
        推測的 mask (255 為表皮內，0 為表皮外)
    """
    # 表皮內的非背景區域
    epidermis_mask = (epidermis != background_value).astype(np.uint8) * 255

    # 真皮內的非背景區域
    dermis_mask = (dermis != background_value).astype(np.uint8) * 255

    # 檢查是否有重疊
    overlap = np.logical_and(epidermis_mask > 0, dermis_mask > 0)
    if np.any(overlap):
        print(f"  ⚠ 警告: 檢測到 {np.sum(overlap)} 個重疊像素")

    return epidermis_mask


def merge_images(
    epidermis: np.ndarray,
    dermis: np.ndarray,
    mask: Optional[np.ndarray] = None,
    background_value: int = 0,
    blend_border: bool = False,
    border_width: int = 2
) -> np.ndarray:
    """
    組合表皮和真皮影像

    Args:
        epidermis: 表皮內影像（背景為 background_value）
        dermis: 表皮外影像（背景為 background_value）
        mask: 可選的 mask（255 為表皮內，0 為表皮外）
        background_value: 背景值
        blend_border: 是否在邊界處進行混合
        border_width: 邊界混合寬度（僅當 blend_border=True 時有效）

    Returns:
        合併後的影像
    """
    # 驗證尺寸
    if epidermis.shape != dermis.shape:
        raise ValueError(f"表皮影像尺寸 {epidermis.shape} 與真皮影像尺寸 {dermis.shape} 不匹配")

    # 創建輸出影像
    merged = np.zeros_like(epidermis)

    if mask is not None:
        # 使用提供的 mask
        if mask.shape[:2] != epidermis.shape[:2]:
            raise ValueError(f"Mask 尺寸 {mask.shape[:2]} 與影像尺寸 {epidermis.shape[:2]} 不匹配")

        mask_binary = (mask > 0).astype(bool)
    else:
        # 從影像反推 mask
        print("  未提供 mask，從影像反推...")
        mask_generated = create_mask_from_images(epidermis, dermis, background_value)
        mask_binary = (mask_generated > 0).astype(bool)

    if not blend_border:
        # 簡單組合：表皮內用表皮影像，表皮外用真皮影像
        merged[mask_binary] = epidermis[mask_binary]
        merged[~mask_binary] = dermis[~mask_binary]

        # 統計
        num_epidermis = np.sum(mask_binary)
        num_dermis = np.sum(~mask_binary)
        print(f"  表皮內像素: {num_epidermis} ({num_epidermis/epidermis.size*100:.2f}%)")
        print(f"  表皮外像素: {num_dermis} ({num_dermis/epidermis.size*100:.2f}%)")

    else:
        # 邊界混合模式
        print(f"  使用邊界混合，寬度: {border_width} 像素")

        # 找出邊界
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = mask_binary.astype(np.uint8) * 255
        dilated = cv2.dilate(mask_uint8, kernel, iterations=border_width)
        eroded = cv2.erode(mask_uint8, kernel, iterations=border_width)
        border = dilated - eroded
        border_mask = border > 0

        # 先填充內部和外部
        merged[mask_binary & ~border_mask] = epidermis[mask_binary & ~border_mask]
        merged[~mask_binary & ~border_mask] = dermis[~mask_binary & ~border_mask]

        # 邊界混合
        if np.any(border_mask):
            alpha = 0.5  # 簡單的 50/50 混合
            merged[border_mask] = (
                epidermis[border_mask].astype(np.float32) * alpha +
                dermis[border_mask].astype(np.float32) * (1 - alpha)
            ).astype(epidermis.dtype)

            print(f"  邊界混合像素: {np.sum(border_mask)} ({np.sum(border_mask)/epidermis.size*100:.2f}%)")

    return merged


def visualize_merge_result(
    epidermis: np.ndarray,
    dermis: np.ndarray,
    merged: np.ndarray,
    mask: Optional[np.ndarray],
    output_path: str,
    title: str = None,
    dpi: int = 150
):
    """
    視覺化組合結果

    Args:
        epidermis: 表皮內影像
        dermis: 表皮外影像
        merged: 合併後影像
        mask: Mask (可選)
        output_path: 輸出路徑
        title: 圖表標題
        dpi: 輸出解析度
    """
    if mask is not None:
        # 2x2 佈局：epidermis, dermis, mask, merged
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=dpi)

        axes[0, 0].imshow(epidermis, cmap='gray')
        axes[0, 0].set_title('Epidermis (Inside)', fontsize=12, weight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(dermis, cmap='gray')
        axes[0, 1].set_title('Dermis (Outside)', fontsize=12, weight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Mask', fontsize=12, weight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(merged, cmap='gray')
        axes[1, 1].set_title('Merged Result', fontsize=12, weight='bold')
        axes[1, 1].axis('off')

    else:
        # 1x3 佈局：epidermis, dermis, merged
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)

        axes[0].imshow(epidermis, cmap='gray')
        axes[0].set_title('Epidermis (Inside)', fontsize=12, weight='bold')
        axes[0].axis('off')

        axes[1].imshow(dermis, cmap='gray')
        axes[1].set_title('Dermis (Outside)', fontsize=12, weight='bold')
        axes[1].axis('off')

        axes[2].imshow(merged, cmap='gray')
        axes[2].set_title('Merged Result', fontsize=12, weight='bold')
        axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
    else:
        fig.suptitle('Epidermis and Dermis Merging', fontsize=14, weight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存視覺化結果: {output_path}")


def visualize_merge_viridis(
    epidermis: np.ndarray,
    dermis: np.ndarray,
    merged: np.ndarray,
    mask: Optional[np.ndarray],
    output_path: str,
    title: str = None,
    dpi: int = 150
):
    """
    使用 Viridis 色彩映射視覺化組合結果

    Args:
        epidermis: 表皮內影像
        dermis: 表皮外影像
        merged: 合併後影像
        mask: Mask (可選)
        output_path: 輸出路徑
        title: 圖表標題
        dpi: 輸出解析度
    """
    if mask is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=dpi)

        im0 = axes[0, 0].imshow(epidermis, cmap='viridis')
        axes[0, 0].set_title('Epidermis (Inside)', fontsize=12, weight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], label='Intensity', shrink=0.8)

        im1 = axes[0, 1].imshow(dermis, cmap='viridis')
        axes[0, 1].set_title('Dermis (Outside)', fontsize=12, weight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Intensity', shrink=0.8)

        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Mask', fontsize=12, weight='bold')
        axes[1, 0].axis('off')

        im3 = axes[1, 1].imshow(merged, cmap='viridis')
        axes[1, 1].set_title('Merged Result', fontsize=12, weight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], label='Intensity', shrink=0.8)

    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=dpi)

        im0 = axes[0].imshow(epidermis, cmap='viridis')
        axes[0].set_title('Epidermis (Inside)', fontsize=12, weight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], label='Intensity', shrink=0.8)

        im1 = axes[1].imshow(dermis, cmap='viridis')
        axes[1].set_title('Dermis (Outside)', fontsize=12, weight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], label='Intensity', shrink=0.8)

        im2 = axes[2].imshow(merged, cmap='viridis')
        axes[2].set_title('Merged Result', fontsize=12, weight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], label='Intensity', shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
    else:
        fig.suptitle('Epidermis and Dermis Merging (Viridis)', fontsize=14, weight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  ✓ 已保存 Viridis 視覺化結果: {output_path}")


def calculate_statistics(epidermis: np.ndarray, dermis: np.ndarray, merged: np.ndarray) -> dict:
    """
    計算組合統計資訊

    Args:
        epidermis: 表皮內影像
        dermis: 表皮外影像
        merged: 合併後影像

    Returns:
        統計資訊字典
    """
    stats = {
        'epidermis_mean': np.mean(epidermis[epidermis > 0]) if np.any(epidermis > 0) else 0,
        'epidermis_std': np.std(epidermis[epidermis > 0]) if np.any(epidermis > 0) else 0,
        'dermis_mean': np.mean(dermis[dermis > 0]) if np.any(dermis > 0) else 0,
        'dermis_std': np.std(dermis[dermis > 0]) if np.any(dermis > 0) else 0,
        'merged_mean': np.mean(merged),
        'merged_std': np.std(merged),
        'merged_min': np.min(merged),
        'merged_max': np.max(merged),
    }
    return stats


def print_statistics(stats: dict):
    """
    印出統計資訊

    Args:
        stats: 統計資訊字典
    """
    print("\n" + "=" * 60)
    print("組合統計資訊")
    print("=" * 60)
    print(f"\n【表皮內（非背景區域）】")
    print(f"  平均值: {stats['epidermis_mean']:.2f}")
    print(f"  標準差: {stats['epidermis_std']:.2f}")
    print(f"\n【真皮（非背景區域）】")
    print(f"  平均值: {stats['dermis_mean']:.2f}")
    print(f"  標準差: {stats['dermis_std']:.2f}")
    print(f"\n【組合結果】")
    print(f"  平均值: {stats['merged_mean']:.2f}")
    print(f"  標準差: {stats['merged_std']:.2f}")
    print(f"  最小值: {stats['merged_min']:.0f}")
    print(f"  最大值: {stats['merged_max']:.0f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='組合表皮與真皮影像',
        formatter_class=argparse.RawDescriptionHelpFormatter,
       
    )

    # 必要參數
    parser.add_argument(
        '--epidermis', '-e',
        required=True,
        help='表皮內影像路徑'
    )

    parser.add_argument(
        '--dermis', '-d',
        required=True,
        help='表皮外（真皮）影像路徑'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='輸出影像路徑'
    )

    # 可選參數
    parser.add_argument(
        '--mask', '-m',
        default=None,
        help='表皮 mask 路徑（可選，若無則從影像反推）'
    )

    parser.add_argument(
        '--background', '-b',
        type=int,
        default=0,
        help='背景值（預設: 0）'
    )

    parser.add_argument(
        '--blend-border',
        action='store_true',
        help='在表皮與真皮邊界處進行混合'
    )

    parser.add_argument(
        '--border-width',
        type=int,
        default=2,
        help='邊界混合寬度（像素，預設: 2）'
    )

    parser.add_argument(
        '--green-channel', '-g',
        action='store_true',
        help='只使用綠色通道'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='生成視覺化比較圖'
    )

    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='不顯示統計資訊'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='視覺化輸出解析度（預設: 150）'
    )

    parser.add_argument(
        '--title',
        default=None,
        help='圖表標題（可選）'
    )

    args = parser.parse_args()

    # 驗證輸入檔案
    for file_path, name in [(args.epidermis, '表皮'), (args.dermis, '真皮')]:
        if not Path(file_path).exists():
            print(f"✗ 錯誤: {name}影像不存在: {file_path}")
            sys.exit(1)

    if args.mask and not Path(args.mask).exists():
        print(f"✗ 錯誤: Mask 檔案不存在: {args.mask}")
        sys.exit(1)

    print("=" * 60)
    print("表皮與真皮影像組合")
    print("=" * 60)

    try:
        # 步驟 1: 載入影像
        print(f"\n[1/4] 載入影像...")
        print(f"  表皮: {args.epidermis}")
        epidermis = load_image(args.epidermis)

        print(f"  真皮: {args.dermis}")
        dermis = load_image(args.dermis)

        mask = None
        if args.mask:
            print(f"  Mask: {args.mask}")
            mask = load_image(args.mask)

        # 提取綠色通道（如果需要）
        if args.green_channel or len(epidermis.shape) == 3:
            print("  提取綠色通道...")
            epidermis = extract_green_channel(epidermis)
            dermis = extract_green_channel(dermis)
            if mask is not None and len(mask.shape) == 3:
                mask = extract_green_channel(mask)

        # 步驟 2: 驗證尺寸
        print(f"\n[2/4] 驗證尺寸...")
        print(f"  表皮尺寸: {epidermis.shape}")
        print(f"  真皮尺寸: {dermis.shape}")
        if mask is not None:
            print(f"  Mask 尺寸: {mask.shape}")

        # 步驟 3: 執行組合
        print(f"\n[3/4] 執行影像組合...")
        print(f"  背景值: {args.background}")

        merged = merge_images(
            epidermis=epidermis,
            dermis=dermis,
            mask=mask,
            background_value=args.background,
            blend_border=args.blend_border,
            border_width=args.border_width
        )

        # 步驟 4: 保存結果
        print(f"\n[4/4] 保存結果...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), merged)
        print(f"  ✓ 已保存組合影像: {output_path}")

        # 視覺化
        if args.visualize:
            print(f"\n  生成視覺化...")
            base_name = output_path.stem

            # 灰階視覺化
            vis_gray_path = output_path.parent / f"{base_name}_merge_comparison_gray.png"
            visualize_merge_result(
                epidermis=epidermis,
                dermis=dermis,
                merged=merged,
                mask=mask,
                output_path=str(vis_gray_path),
                title=args.title,
                dpi=args.dpi
            )

            # Viridis 視覺化
            vis_viridis_path = output_path.parent / f"{base_name}_merge_comparison_viridis.png"
            visualize_merge_viridis(
                epidermis=epidermis,
                dermis=dermis,
                merged=merged,
                mask=mask,
                output_path=str(vis_viridis_path),
                title=args.title,
                dpi=args.dpi
            )

        # 統計資訊
        if not args.no_stats:
            stats = calculate_statistics(epidermis, dermis, merged)
            print_statistics(stats)

        print("\n✓ 組合完成！")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 組合失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
