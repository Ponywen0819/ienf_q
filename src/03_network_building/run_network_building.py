#!/usr/bin/env python3
"""
網路建構 CLI 工具

使用範例:
    python run_network_building.py \
        --seeds output/seeds/seeds.json \
        --image test/green_channel.png \
        --output output/network \
        --config config/default.yaml \
        --verbose
"""

import argparse
import sys
from pathlib import Path

# Import configuration loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config, NetworkBuildingConfig

def main():
    parser = argparse.ArgumentParser(
        description='網路建構 - IENF 神經纖維重建',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 使用 YAML 配置文件（推薦）
  %(prog)s -s output/seeds/seeds.json -i test/green_channel.png -o output/network --config config/default.yaml

  # 使用預設參數
  %(prog)s -s output/seeds/seeds.json -i test/green_channel.png -o output/network

  # CLI 覆蓋 YAML 參數
  %(prog)s -s seeds.json -i image.png -o network/ --config config/default.yaml --k-neighbors 15
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
    required = parser.add_argument_group('必要參數')
    required.add_argument(
        '-s', '--seeds',
        required=True,
        metavar='PATH',
        help='種子 JSON 檔案路徑 (seeds.json)'
    )
    required.add_argument(
        '-i', '--image',
        required=True,
        metavar='PATH',
        help='綠色通道影像路徑'
    )
    required.add_argument(
        '-o', '--output',
        required=True,
        metavar='DIR',
        help='輸出目錄'
    )

    # 可選參數
    optional = parser.add_argument_group('可選參數')
    optional.add_argument(
        '--max-edge-cost',
        type=float,
        default=None,
        metavar='FLOAT',
        help='最大邊成本閾值'
    )
    optional.add_argument(
        '--k-neighbors',
        type=int,
        default=None,
        metavar='INT',
        help='密度估算的鄰居數'
    )
    optional.add_argument(
        '--alpha',
        type=float,
        default=None,
        metavar='FLOAT',
        help='幾何成本權重'
    )
    optional.add_argument(
        '--beta',
        type=float,
        default=None,
        metavar='FLOAT',
        help='影像成本權重'
    )
    optional.add_argument(
        '--gamma',
        type=float,
        default=None,
        metavar='FLOAT',
        help='曲率成本權重'
    )
    optional.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='顯示詳細資訊'
    )
    
    args = parser.parse_args()

    try:
        # Load configuration from YAML (if provided) or use defaults
        if args.config:
            full_config = load_config(args.config)
            config = full_config.network_building
            if args.verbose:
                print(f"✓ 載入配置文件: {args.config}")
        else:
            # Use default configuration
            config = NetworkBuildingConfig()

        # Apply CLI overrides (CLI takes precedence over YAML)
        if args.max_edge_cost is not None:
            config.network.max_edge_cost = args.max_edge_cost
        if args.k_neighbors is not None:
            config.network.k_neighbors = args.k_neighbors
        if args.alpha is not None:
            config.cost_weights.alpha = args.alpha
        if args.beta is not None:
            config.cost_weights.beta = args.beta
        if args.gamma is not None:
            config.cost_weights.gamma = args.gamma
        if args.verbose:
            config.network.verbose = True

        # 驗證輸入檔案
        seeds_path = Path(args.seeds)
        image_path = Path(args.image)

        if not seeds_path.exists():
            print(f"❌ 錯誤: 種子檔案不存在: {seeds_path}", file=sys.stderr)
            return 1

        if not image_path.exists():
            print(f"❌ 錯誤: 影像檔案不存在: {image_path}", file=sys.stderr)
            return 1

        # 顯示配置
        print("\n" + "=" * 60)
        print("網路建構配置")
        print("=" * 60)
        print(f"種子檔案: {seeds_path}")
        print(f"影像檔案: {image_path}")
        print(f"輸出目錄: {args.output}")
        print(f"最大邊成本: {config.network.max_edge_cost}")
        print(f"k-近鄰數: {config.network.k_neighbors}")
        print(f"成本權重: α={config.cost_weights.alpha}, β={config.cost_weights.beta}, γ={config.cost_weights.gamma}")
        print(f"詳細模式: {'是' if config.network.verbose else '否'}")

        # 執行網路建構
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        from network_builder import NetworkBuilder, NetworkConfig

        # Convert pydantic config to dataclass config for NetworkBuilder
        builder_config = NetworkConfig(
            k_neighbors=config.network.k_neighbors,
            max_edge_cost=config.network.max_edge_cost,
            verbose=config.network.verbose
        )

        builder = NetworkBuilder(builder_config)

        # Pass cost weights to the builder
        builder.cost_calculator.alpha = config.cost_weights.alpha
        builder.cost_calculator.beta = config.cost_weights.beta
        builder.cost_calculator.gamma = config.cost_weights.gamma

        # Pass density and pathfinding config
        builder.density_estimator.dense_threshold = config.density.dense_threshold
        builder.density_estimator.moderate_threshold = config.density.moderate_threshold
        builder.density_estimator.dense_radius = config.density.dense_radius
        builder.density_estimator.moderate_radius = config.density.moderate_radius
        builder.density_estimator.sparse_radius = config.density.sparse_radius

        builder.cost_calculator.pathfinder.max_distance_multiplier = config.pathfinding.max_distance_multiplier
        builder.cost_calculator.pathfinder.distance_from_start_cutoff = config.pathfinding.distance_from_start_cutoff

        G = builder.build_network(
            seeds_json=str(seeds_path),
            green_channel_image=str(image_path),
            output_dir=args.output
        )

        print(f"\n✅ 執行成功! 結果已保存到: {args.output}")
        return 0

    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
