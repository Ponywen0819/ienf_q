#!/usr/bin/env python3
"""
網路建構 CLI 工具

使用範例:
    python run_network_building.py \
        --seeds output/seeds/seeds.json \
        --image test/green_channel.png \
        --output output/network \
        --verbose
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='網路建構 - IENF 神經纖維重建',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  %(prog)s -s output/seeds/seeds.json -i test/green_channel.png -o output/network
  %(prog)s --seeds seeds.json --image image.png --output network/ --verbose
        """
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
        default=150.0,
        metavar='FLOAT',
        help='最大邊成本閾值 (預設: 150.0)'
    )
    optional.add_argument(
        '--k-neighbors',
        type=int,
        default=10,
        metavar='INT',
        help='密度估算的鄰居數 (預設: 10)'
    )
    optional.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='顯示詳細資訊'
    )
    
    args = parser.parse_args()
    
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
    print(f"最大邊成本: {args.max_edge_cost}")
    print(f"k-近鄰數: {args.k_neighbors}")
    print(f"詳細模式: {'是' if args.verbose else '否'}")
    
    # 執行網路建構
    try:
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        from network_builder import NetworkBuilder, NetworkConfig
        
        config = NetworkConfig(
            k_neighbors=args.k_neighbors,
            max_edge_cost=args.max_edge_cost,
            verbose=args.verbose
        )
        
        builder = NetworkBuilder(config)
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
