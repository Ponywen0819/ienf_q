#!/usr/bin/env python3
"""
單一種子配對視覺化工具

此工具用於視覺化單一神經纖維種子與其鄰居的配對過程，
幫助理解和調試網路建構階段的演算法參數。

使用範例:
    python tools/visualize_seed_pairing.py \
        --seed-id 150 \
        --seeds output/intermediates/seeds/seeds.json \
        --image output/intermediates/green_channel.png \
        --config config/default.yaml \
        --output output/visualizations/seed_150_pairing.png \
        --zoom-radius 250
"""

import argparse
import sys
import importlib.util
from pathlib import Path

# 將專案根目錄添加到 Python 路徑中，以便能夠導入 src 中的模組
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.config_loader import IENFConfig

# --- 從 run_pipeline.py 複製的動態模組導入輔助函數 ---
def _import_module_from_path(module_name: str, file_path: Path):
    """動態導入模組（支援數字開頭的目錄名）"""
    module_dir = str(file_path.parent)
    if module_dir in sys.path:
        sys.path.remove(module_dir)
    sys.path.insert(0, module_dir)
    
    # 清理可能的緩存
    modules_to_clear = [
        key for key in sys.modules 
        if key in ['visualization', 'mst_builder', 'seed_loader', 'cost_calculator',
                   'density_estimator', 'pathfinding', 'seed_pairing', 'graph_builder']
    ]
    for key in modules_to_clear:
        if key in sys.modules:
            del sys.modules[key]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"無法從 {file_path} 找到模組 {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# --- 動態導入 NetworkBuilder ---
try:
    network_builder_module = _import_module_from_path(
        "network_builder",
        project_root / "src" / "03_network_building" / "network_builder.py"
    )
    NetworkBuilder = network_builder_module.NetworkBuilder
    NetworkConfig = network_builder_module.NetworkConfig
except ImportError as e:
    print(f"✗ 致命錯誤：無法動態導入 NetworkBuilder 模組。", file=sys.stderr)
    print(f"  {e}", file=sys.stderr)
    sys.exit(1)


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description='單一種子配對視覺化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    required = parser.add_argument_group('必要參數')
    required.add_argument('--seed-id', required=True, type=int, help='要分析的目標種子 ID')
    required.add_argument('--seeds', required=True, type=str, help='種子 JSON 檔案路徑 (例如: intermediates/seeds/seeds.json)')
    required.add_argument('--image', required=True, type=str, help='綠色通道影像路徑 (例如: intermediates/green_channel.png)')
    required.add_argument('--output', required=True, type=str, help='視覺化結果的儲存路徑 (例如: seed_pairing.png)')
    
    optional = parser.add_argument_group('可選參數')
    optional.add_argument('--config', type=str, default='config/default.yaml', help='IENF 配置文件路徑 (預設: config/default.yaml)')
    optional.add_argument('--zoom-radius', type=int, default=200, help='視覺化結果的縮放半徑,單位為像素 (預設: 200)')

    args = parser.parse_args()

    # --- 載入配置 ---
    try:
        ienf_config: IENFConfig = load_config(args.config)
        # 手動從 ienf_config 映射到 NetworkConfig，這是正確的做法
        network_config = NetworkConfig(
            k_neighbors=ienf_config.network_building.network.k_neighbors,
            max_edge_cost=ienf_config.network_building.network.max_edge_cost,
            verbose=False, # 在工具中通常不需要詳細日誌
            # Cost weights
            alpha=ienf_config.network_building.cost_weights.alpha,
            beta=ienf_config.network_building.cost_weights.beta,
            gamma=ienf_config.network_building.cost_weights.gamma,
            # Density parameters
            dense_threshold=ienf_config.network_building.density.dense_threshold,
            moderate_threshold=ienf_config.network_building.density.moderate_threshold,
            dense_radius=ienf_config.network_building.density.dense_radius,
            moderate_radius=ienf_config.network_building.density.moderate_radius,
            sparse_radius=ienf_config.network_building.density.sparse_radius,
            # Pathfinding parameters
            max_distance_multiplier=ienf_config.network_building.pathfinding.max_distance_multiplier,
            distance_from_start_cutoff=ienf_config.network_building.pathfinding.distance_from_start_cutoff
        )
        print(f"✓ 成功載入配置文件: {args.config}")
    except Exception as e:
        print(f"✗ 載入配置失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # --- 執行視覺化 ---
    try:
        # 初始化 NetworkBuilder
        builder = NetworkBuilder(config=network_config)

        # 呼叫視覺化方法
        builder.visualize_single_seed_pairing(
            target_seed_id=args.seed_id,
            seeds_json=args.seeds,
            green_channel_image=args.image,
            output_path=args.output,
            zoom_radius=args.zoom_radius
        )
        
        return 0

    except FileNotFoundError as e:
        print(f"\n✗ 錯誤: 找不到必要的輸入檔案。")
        print(f"  - {e}")
        print(f"  請確保您提供的 --seeds 和 --image 路徑正確。")
        print(f"  這些檔案通常在執行 `run_pipeline.py --save-intermediates` 後，生成於 `output/intermediates/` 目錄下。")
        return 1
    except ValueError as e:
        print(f"\n✗ 錯誤: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 執行視覺化時發生未知錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
