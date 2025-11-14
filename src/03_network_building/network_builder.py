"""
網路建構器 - 主流程協調
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import networkx as nx
import json
import numpy as np

from seed_loader import SeedLoader
from density_estimator import DensityEstimator
from cost_calculator import CostCalculator
from seed_pairing import SeedPairer
from graph_builder import GraphBuilder
from netowk_visualization import NetworkVisualizer
from pathfinding import ImagePathfinder


@dataclass
class NetworkConfig:
    """網路建構配置"""
    k_neighbors: int = 10
    max_edge_cost: float = 150.0
    verbose: bool = False
    # Cost weights
    alpha: float = 0.05
    beta: float = 0.9
    gamma: float = 0.05
    # Density parameters
    dense_threshold: int = 30
    moderate_threshold: int = 70
    dense_radius: int = 30
    moderate_radius: int = 50
    sparse_radius: int = 80
    # Pathfinding parameters
    max_distance_multiplier: int = 200
    distance_from_start_cutoff: int = 40


class NetworkBuilder:
    """
    網路建構器 - 協調所有模組的主流程
    
    流程:
    1. 載入種子與影像
    2. 計算局部密度
    3. 種子配對
    4. 計算邊成本
    5. 建構圖結構
    6. 儲存與視覺化
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config

        # 初始化所有模組 (CostCalculator 將在 build_network 中動態建立)
        self.seed_loader = SeedLoader(verbose=config.verbose)
        self.density_estimator = DensityEstimator(k=config.k_neighbors)
        self.seed_pairer = SeedPairer(verbose=config.verbose)
        self.graph_builder = GraphBuilder(max_edge_cost=config.max_edge_cost)
        self.visualizer = NetworkVisualizer()

        # 設定密度參數
        self.density_estimator.dense_threshold = config.dense_threshold
        self.density_estimator.moderate_threshold = config.moderate_threshold
        self.density_estimator.dense_radius = config.dense_radius
        self.density_estimator.moderate_radius = config.moderate_radius
        self.density_estimator.sparse_radius = config.sparse_radius
    
    def build_network(
        self,
        seeds_json: str,
        green_channel_image: str,
        output_dir: str
    ) -> Optional[nx.Graph]:
        """
        執行完整網路建構流程
        
        Args:
            seeds_json: 種子 JSON 檔案路徑
            green_channel_image: 綠色通道影像路徑
            output_dir: 輸出目錄
        
        Returns:
            G: NetworkX 圖物件 (目前為 None,待模組實作)
        """
        print("=" * 60)
        print("網路建構流程")
        print("=" * 60)
        
        # ========== 階段 1: 載入資料 ==========
        print("\n[1/6] 載入種子與影像...")
        try:
            seeds = self.seed_loader.load_seeds(seeds_json)
            green_channel = self.seed_loader.load_green_channel(green_channel_image)
            kdtree = self.seed_loader.build_spatial_index()
            print(f"  ✓ 成功載入 {len(seeds)} 個種子")
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
            raise

        # --- 動態建立依賴影像的模組 ---
        pathfinder = ImagePathfinder(green_channel, verbose=self.config.verbose)
        pathfinder.max_distance_multiplier = self.config.max_distance_multiplier
        pathfinder.distance_from_start_cutoff = self.config.distance_from_start_cutoff

        cost_calculator = CostCalculator(
            pathfinder,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            verbose=self.config.verbose
        )
        
        # ========== 階段 2: 計算局部密度 ==========
        print("\n[2/6] 計算局部密度...")
        density_info = {}
        try:
            for seed in seeds:
                density = self.density_estimator.calculate_local_density(
                    seed, kdtree, k=self.config.k_neighbors
                )
                radius = self.density_estimator.determine_adaptive_radius(density)
                density_info[seed.id] = {
                    'local_density': density,
                    'pairing_radius': radius
                }
            print(f"  ✓ 完成密度分析")
        except NotImplementedError:
            print(f"  ⚠️  模組尚未實作,跳過")
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
        
        # ========== 階段 3: 種子配對 ==========
        print("\n[3/6] 種子配對...")
        seed_pairs = []
        try:
            seed_pairs = self.seed_pairer.pair_seeds(seeds, density_info, kdtree)
            print(f"  ✓ 生成 {len(seed_pairs)} 對候選連接")
        except NotImplementedError:
            print(f"  ⚠️  模組尚未實作,跳過")
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
        
        # ========== 階段 4: 計算邊成本 ==========
        print("\n[4/6] 計算邊成本...")
        edges_with_costs = []
        try:
            # 移除 NotImplementedError, 因為我們現在要實作它
            for i, (seed_i, seed_j, edge_type) in enumerate(seed_pairs):
                if self.config.verbose and i % 100 == 0:
                    print(f"  進度: {i}/{len(seed_pairs)}")
                
                costs = cost_calculator.calculate_total_cost(seed_i, seed_j)
                
                edges_with_costs.append({
                    'source_id': seed_i.id,
                    'target_id': seed_j.id,
                    'edge_type': edge_type,
                    **costs
                })
            
            print(f"  ✓ 成功計算 {len(edges_with_costs)} 條邊的成本")
        except NotImplementedError:
            print(f"  ⚠️  模組尚未實作,跳過")
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
        
        # ========== 階段 5: 建構圖 ==========
        print("\n[5/6] 建構圖結構...")
        G = None
        try:
            G = self.graph_builder.build_graph(seeds, edges_with_costs)
            stats = self.graph_builder.get_statistics(G)
            print(f"  ✓ 節點數: {stats.get('num_nodes', 'N/A')}")
            print(f"  ✓ 邊數: {stats.get('num_edges', 'N/A')}")
        except NotImplementedError:
            print(f"  ⚠️  模組尚未實作,跳過")
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
        
        # ========== 階段 6: 儲存與視覺化 ==========
        print("\n[6/6] 儲存結果...")
        try:
            # 將 green_channel 和 density_info 傳遞給 _save_outputs
            self._save_outputs(G, seeds, green_channel, edges_with_costs, density_info, output_dir)
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
        
        print("\n" + "=" * 60)
        if G is not None:
            print("✓ 網路建構完成!")
        else:
            print("✓ 主流程執行完成!")
            print("⚠️  注意: 部分模組尚未實作,目前僅驗證流程")
        print("=" * 60)
        
        return G
    
    def _save_outputs(self, G, seeds, green_channel, edges_with_costs, density_info, output_dir):
        """儲存所有輸出檔案"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存基本狀態
        status_file = output_dir / 'status.txt'
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(f"網路建構執行狀態\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"種子數: {len(seeds)}\n")
            f.write(f"邊數: {len(edges_with_costs)}\n")
            if G is not None:
                stats = self.graph_builder.get_statistics(G)
                f.write(f"圖節點數: {stats['num_nodes']}\n")
                f.write(f"圖邊數: {stats['num_edges']}\n")
                f.write(f"連通元件數: {stats['num_components']}\n")
            else:
                f.write("圖狀態: 未建構\n")
        print(f"  ✓ 狀態檔: {status_file}")
        
        # 如果圖已建構,保存並視覺化
        if G is not None:
            # 保存 GraphML
            try:
                graphml_file = output_dir / 'network.graphml'
                nx.write_graphml(G, str(graphml_file))
                print(f"  ✓ GraphML: {graphml_file}")
            except Exception as e:
                print(f"  ✗ 無法保存 GraphML: {e}")

        # 保存完整的邊資料（包含路徑）為 JSON（只保存有效邊）
        try:
            # 過濾掉找不到路徑的邊（total_cost = Infinity）
            valid_edges = [
                edge for edge in edges_with_costs
                if np.isfinite(edge.get('total_cost', float('inf')))
            ]

            edges_json_file = output_dir / 'edges_with_paths.json'
            with open(edges_json_file, 'w', encoding='utf-8') as f:
                json.dump(valid_edges, f, indent=2, default=str)

            print(f"  ✓ 邊資料 JSON: {edges_json_file}")
            print(f"    - 有效邊數: {len(valid_edges)}/{len(edges_with_costs)}")
        except Exception as e:
            print(f"  ✗ 無法保存邊資料 JSON: {e}")

        if G is not None:
            
            # 網路視覺化
            try:
                viz_file = output_dir / 'network.png'
                self.visualizer.visualize_network(G, seeds, green_channel, str(viz_file))
                print(f"  ✓ 網路視覺化: {viz_file}")
            except Exception as e:
                print(f"  ✗ 網路視覺化錯誤: {e}")

            # 放大圖視覺化
            try:
                zoom_viz_file = output_dir / 'network_zoomed.png'
                self.visualizer.visualize_zoomed_view(G, seeds, green_channel, str(zoom_viz_file))
                print(f"  ✓ 放大圖視覺化: {zoom_viz_file}")
            except Exception as e:
                print(f"  ✗ 放大圖視覺化錯誤: {e}")

            # 成本分布視覺化
            try:
                cost_dist_file = output_dir / 'cost_distribution.png'
                self.visualizer.visualize_cost_distribution(G, str(cost_dist_file))
                print(f"  ✓ 成本分布圖: {cost_dist_file}")
            except Exception as e:
                print(f"  ✗ 成本分布圖錯誤: {e}")

            # 路徑樣本視覺化（放大顯示）
            try:
                path_viz_file = output_dir / 'sample_paths.png'
                self.visualizer.visualize_sample_paths(G, seeds, green_channel, str(path_viz_file), num_samples=15, zoom_radius=200)
            except Exception as e:
                print(f"  ✗ 路徑視覺化錯誤: {e}")

        # 密度熱力圖視覺化（不需要圖物件也可以生成）
        try:
            density_heatmap_file = output_dir / 'density_heatmap.png'
            self.visualizer.visualize_density_heatmap(seeds, density_info, green_channel, str(density_heatmap_file))
        except Exception as e:
            print(f"  ✗ 密度熱力圖錯誤: {e}")

        # 成本比值分析圖（幫助決定 pathfinding 倍數參數）
        try:
            cost_ratio_file = output_dir / 'cost_ratio_analysis.png'
            self.visualizer.visualize_cost_ratio_analysis(edges_with_costs, str(cost_ratio_file))
        except Exception as e:
            print(f"  ✗ 成本比值分析錯誤: {e}")


if __name__ == '__main__':
    # 測試程式碼
    config = NetworkConfig(
        k_neighbors=10,
        max_edge_cost=150.0,
        verbose=True
    )
    
    builder = NetworkBuilder(config)
    G = builder.build_network(
        seeds_json='output/seeds/seeds.json',
        green_channel_image='test/green_channel.png',
        output_dir='output/network'
    )
