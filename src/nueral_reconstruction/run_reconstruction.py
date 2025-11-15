"""
MST 神經纖維重建 - CLI 執行腳本

使用方式:
python run_reconstruction.py \
    --graph output/network/network.graphml \
    --seeds output/seeds/seeds.json \
    --image test/green_channel.png \
    --output output/reconstruction \
    --config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

from src.nueral_reconstruction.reconstruction_runner import ReconstructionRunner, ReconstructionConfig

# Import configuration loader
from src.config_loader import load_config, ReconstructionConfig as PydanticReconstructionConfig


def main():
    parser = argparse.ArgumentParser(
        description='MST 神經纖維重建',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 使用 YAML 配置文件（推薦）
  python run_reconstruction.py \\
      --graph output/network/network.graphml \\
      --seeds output/seeds/seeds.json \\
      --image test/green_channel.png \\
      --output output/reconstruction \\
      --config config/default.yaml

  # 使用預設參數
  python run_reconstruction.py \\
      --graph output/network/network.graphml \\
      --seeds output/seeds/seeds.json \\
      --image test/green_channel.png \\
      --output output/reconstruction

注意:
  - 圖檔案應來自階段三的輸出
  - 種子檔案應來自階段二的輸出
  - 影像應為綠色通道影像
        """
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML 配置文件路徑（可選，CLI 參數會覆蓋配置文件）'
    )

    # 必要參數
    parser.add_argument(
        '--graph',
        required=True,
        help='網路圖檔案路徑 (network.graphml)'
    )

    parser.add_argument(
        '--seeds',
        required=True,
        help='種子檔案路徑 (seeds.json)'
    )

    parser.add_argument(
        '--image',
        required=True,
        help='綠色通道影像路徑'
    )

    # 可選參數
    parser.add_argument(
        '--mask',
        default=None,
        help='表皮標注 mask 路徑（可選，用於繪製表皮-真皮界線）'
    )

    parser.add_argument(
        '--output',
        default='output/reconstruction',
        help='輸出目錄（預設: output/reconstruction）'
    )

    parser.add_argument(
        '--max-cost',
        type=float,
        default=None,
        help='最大邊成本閾值'
    )

    parser.add_argument(
        '--min-branch-angle',
        type=float,
        default=None,
        help='銳角分支閾值（度數）'
    )

    parser.add_argument(
        '--min-quality',
        type=float,
        default=None,
        help='最小路徑質量閾值'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='靜音模式，減少輸出'
    )

    args = parser.parse_args()

    try:
        # Load configuration from YAML (if provided) or use defaults
        if args.config:
            full_config = load_config(args.config)
            pydantic_config = full_config.reconstruction
            if not args.quiet:
                print(f"✓ 載入配置文件: {args.config}")
        else:
            # Use default configuration
            pydantic_config = PydanticReconstructionConfig()

        # Apply CLI overrides (CLI takes precedence over YAML)
        if args.max_cost is not None:
            pydantic_config.max_edge_cost = args.max_cost
        if args.min_branch_angle is not None:
            pydantic_config.min_branch_angle = args.min_branch_angle
        if args.min_quality is not None:
            pydantic_config.min_quality_threshold = args.min_quality
        if args.quiet:
            pydantic_config.verbose = False

        # 驗證輸入檔案
        if not Path(args.graph).exists():
            print(f"錯誤: 圖檔案不存在: {args.graph}")
            sys.exit(1)

        if not Path(args.seeds).exists():
            print(f"錯誤: 種子檔案不存在: {args.seeds}")
            sys.exit(1)

        if not Path(args.image).exists():
            print(f"錯誤: 影像檔案不存在: {args.image}")
            sys.exit(1)

        # Convert pydantic config to dataclass config for ReconstructionRunner
        config = ReconstructionConfig(
            max_edge_cost=pydantic_config.max_edge_cost,
            min_branch_angle=pydantic_config.min_branch_angle,
            min_quality_threshold=pydantic_config.min_quality_threshold,
            verbose=pydantic_config.verbose
        )

        # 創建並執行重建流程
        runner = ReconstructionRunner(config)

        results = runner.run(
            graph_path=args.graph,
            seeds_path=args.seeds,
            green_channel_path=args.image,
            output_dir=args.output,
            mask_path=args.mask
        )

        print(f"\n✓ 重建完成！輸出目錄: {args.output}")
        print(f"  - MST 森林: {results['stats']['total_nodes']} 節點, "
              f"{results['stats']['total_edges']} 邊")
        print(f"  - 連通分量: {results['stats']['num_components']} 個")

        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 重建失敗: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
