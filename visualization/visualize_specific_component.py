"""
Specific Component Connection Visualization
特定元件連接視覺化

這個腳本專門用於視覺化特定元件與其相關的所有連接(包括成功和被拒絕的連接)

使用方式:
    # 從原始影像生成特定元件視覺化
    python visualization/visualize_specific_component.py \\
        --annotation output/preprocessing/final_label.png \\
        --green-channel output/preprocessing/roi_image.png \\
        --component-id 0 \\
        --output output/component_0_connections.png

    # 視覺化多個元件
    python visualization/visualize_specific_component.py \\
        --annotation output/preprocessing/final_label.png \\
        --green-channel output/preprocessing/roi_image.png \\
        --component-ids 0 1 2 3 \\
        --output-dir output/specific_components

功能:
    - 顯示目標元件(用黃色圓圈高亮)
    - 顯示與目標元件連接的所有其他元件
    - 顯示所有成功的連接(綠色實線)
    - 顯示所有被拒絕的連接(彩色虛線,不同顏色代表不同的拒絕原因)
    - 顯示種子點和元件標籤
    - 提供詳細的統計信息
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import cv2
import argparse
import logging
from typing import Optional, List

# Import from the main visualization module
from visualize_component_pairing import (
    visualize_specific_component_connections,
    visualize_component_pairing_from_images
)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='視覺化特定元件與其相關的所有連接',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 視覺化單個元件
  python visualization/visualize_specific_component.py \\
      --annotation output/preprocessing/final_label.png \\
      --green-channel output/preprocessing/roi_image.png \\
      --component-id 0 \\
      --output output/component_0.png

  # 視覺化多個元件
  python visualization/visualize_specific_component.py \\
      --annotation output/preprocessing/final_label.png \\
      --green-channel output/preprocessing/roi_image.png \\
      --component-ids 0 1 2 3 \\
      --output-dir output/specific_components
        """
    )

    # Required arguments
    parser.add_argument(
        '--annotation',
        type=str,
        required=True,
        help='標註影像路徑 (二值 mask)'
    )
    parser.add_argument(
        '--green-channel',
        type=str,
        required=True,
        help='綠色通道影像路徑'
    )

    # Component selection
    component_group = parser.add_mutually_exclusive_group(required=True)
    component_group.add_argument(
        '--component-id',
        type=int,
        help='要視覺化的元件 ID (單個元件)'
    )
    component_group.add_argument(
        '--component-ids',
        type=int,
        nargs='+',
        help='要視覺化的元件 ID 列表 (多個元件)'
    )

    # Output
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--output',
        type=str,
        help='輸出影像路徑 (用於單個元件)'
    )
    output_group.add_argument(
        '--output-dir',
        type=str,
        help='輸出目錄 (用於多個元件)'
    )

    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置檔路徑 (可選,預設使用 config/default.yaml)'
    )
    parser.add_argument(
        '--no-components',
        action='store_true',
        help='不顯示元件疊加層'
    )
    parser.add_argument(
        '--no-seeds',
        action='store_true',
        help='不顯示種子點'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='不顯示元件標籤'
    )
    parser.add_argument(
        '--component-alpha',
        type=float,
        default=0.3,
        help='元件疊加層透明度 (0.0-1.0,預設: 0.3)'
    )
    parser.add_argument(
        '--line-thickness',
        type=int,
        default=2,
        help='連接線粗細 (預設: 2)'
    )
    parser.add_argument(
        '--seed-radius',
        type=int,
        default=5,
        help='種子點半徑 (預設: 5)'
    )
    parser.add_argument(
        '--crop-size',
        type=int,
        default=200,
        help='裁切大小 (預設: 200x200 像素)'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=800,
        help='輸出影像大小 (預設: 800x800 像素)'
    )

    args = parser.parse_args()

    # 解析元件 ID 列表
    if args.component_id is not None:
        component_ids = [args.component_id]
        output_dir = None
        output_path = args.output
    else:
        component_ids = args.component_ids
        output_dir = args.output_dir
        output_path = None

    # 創建輸出目錄
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # 步驟 1: 運行完整 pipeline 獲取配對結果
    logger.info("\n" + "="*70)
    logger.info("Step 1: Running Neural Reconstruction Pipeline")
    logger.info("="*70)

    temp_output_dir = "output/temp_component_pairing"
    results = visualize_component_pairing_from_images(
        annotation_path=args.annotation,
        green_channel_path=args.green_channel,
        output_dir=temp_output_dir,
        config_path=args.config,
        show_successful=True,
        show_rejected=True,
        show_components=False,
        show_seeds=False,
        show_labels=False,
        show_legend=False,
        component_alpha=0.3,
        line_thickness=1,
        seed_radius=1
    )

    # 步驟 2: 為每個指定的元件生成視覺化
    logger.info("\n" + "="*70)
    logger.info("Step 2: Creating Specific Component Visualizations")
    logger.info("="*70)

    for comp_id in component_ids:
        try:
            # 決定輸出路徑
            if output_path:
                current_output_path = output_path
            else:
                current_output_path = str(output_dir / f"component_{comp_id}_connections.png")

            logger.info(f"\nVisualizing component {comp_id}...")

            # 生成視覺化
            component_info = visualize_specific_component_connections(
                pairing_results=results['pairing_results'],
                components_data=results['components_data'],
                green_channel=results['green_channel'],
                target_component_id=comp_id,
                output_path=current_output_path,
                show_components=not args.no_components,
                show_seeds=not args.no_seeds,
                show_labels=not args.no_labels,
                component_alpha=args.component_alpha,
                line_thickness=args.line_thickness,
                seed_radius=args.seed_radius,
                crop_size=args.crop_size,
                output_size=args.output_size
            )

            # 顯示摘要
            logger.info(f"\nComponent {comp_id} Summary:")
            logger.info(f"  Output: {current_output_path}")
            logger.info(f"  Connected to components: {component_info['connected_components']}")
            logger.info(f"  Successful connections: {component_info['stats']['num_successful']}")

            total_rejected = component_info['stats']['num_pairs_analyzed'] - component_info['stats']['num_successful']
            logger.info(f"  Rejected connections: {total_rejected}")

            if total_rejected > 0:
                logger.info(f"    - Distance too far: {component_info['stats']['rejected_distance_too_far']}")
                logger.info(f"    - Cost exceeds threshold: {component_info['stats']['rejected_cost_exceeds_threshold']}")
                logger.info(f"    - No valid path: {component_info['stats']['rejected_no_valid_path']}")
                logger.info(f"    - No seeds: {component_info['stats']['rejected_no_seeds']}")

        except Exception as e:
            logger.error(f"Error visualizing component {comp_id}: {e}", exc_info=True)

    logger.info("\n" + "="*70)
    logger.info("Visualization Complete")
    logger.info("="*70)


if __name__ == '__main__':
    main()
