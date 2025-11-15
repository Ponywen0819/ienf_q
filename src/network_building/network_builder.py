"""
網路建構器 - 主流程協調
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import networkx as nx
import json
import numpy as np
import cv2

from src.network_building.seed_loader import SeedLoader
from src.network_building.density_estimator import DensityEstimator
from src.network_building.cost_calculator import CostCalculator
from src.network_building.seed_pairing import SeedPairer
from src.network_building.graph_builder import GraphBuilder
from src.network_building.netowk_visualization import NetworkVisualizer
from src.shared.pathfinding import ImagePathfinder


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

    def visualize_single_seed_pairing(
        self,
        target_seed_id: int,
        seeds_json: str,
        green_channel_image: str,
        output_path: str,
        zoom_radius: int = 200
    ):
        """
        視覺化單一種子與其鄰居的配對嘗試過程（使用自適應半徑，完全符合實際網路建構邏輯）。

        Args:
            target_seed_id: 要分析的目標種子 ID。
            seeds_json: 種子 JSON 檔案路徑。
            green_channel_image: 綠色通道影像路徑。
            output_path: 視覺化結果的儲存路徑。
            zoom_radius: 視覺化結果的縮放半徑 (預設: 200px)。
        """
        print("=" * 60)
        print(f"單一種子配對視覺化 (ID: {target_seed_id}, 縮放半徑: {zoom_radius}px)")
        print("=" * 60)

        # --- 1. 載入資料 ---
        print("\n[1/5] 載入資料...")
        try:
            seeds = self.seed_loader.load_seeds(seeds_json)
            seeds_map = {s.id: s for s in seeds}
            green_channel = self.seed_loader.load_green_channel(green_channel_image)
            kdtree = self.seed_loader.build_spatial_index()

            if target_seed_id not in seeds_map:
                raise ValueError(f"找不到 ID 為 {target_seed_id} 的種子。")

            source_seed = seeds_map[target_seed_id]
            print(f"  ✓ 成功載入 {len(seeds)} 個種子並找到目標種子。")
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
            raise

        # --- 2. 計算局部密度和自適應半徑 ---
        print("\n[2/5] 計算局部密度和自適應半徑...")
        local_density = self.density_estimator.calculate_local_density(
            source_seed, kdtree, k=self.config.k_neighbors
        )
        adaptive_radius = self.density_estimator.determine_adaptive_radius(local_density)

        print(f"  ✓ 目標種子 ID {target_seed_id}:")
        print(f"    - 局部密度: {local_density:.2f}")
        print(f"    - 自適應配對半徑: {adaptive_radius} px")

        # --- 3. 尋找鄰居（使用自適應半徑 + 額外參考鄰居）---
        print(f"\n[3/5] 查詢鄰居（半徑內 + 參考鄰居）...")

        # 查詢自適應半徑內的鄰居
        indices_in_radius = kdtree.query_radius([[source_seed.x, source_seed.y]], r=adaptive_radius)[0]
        neighbors_in_radius = [seeds[i] for i in indices_in_radius if seeds[i].id != source_seed.id]

        # 額外查詢一些半徑外的參考鄰居（用於顯示對比）
        distances, indices_knn = kdtree.query([[source_seed.x, source_seed.y]], k=min(20, len(seeds)))
        all_potential_neighbors = [seeds[i] for i in indices_knn[0] if seeds[i].id != source_seed.id]

        # 分類鄰居：半徑內 vs 半徑外
        neighbors_in_radius_ids = {s.id for s in neighbors_in_radius}
        neighbors_outside_radius = [
            s for s in all_potential_neighbors
            if s.id not in neighbors_in_radius_ids
        ][:5]  # 最多取 5 個半徑外鄰居作為參考

        print(f"  ✓ 半徑內鄰居: {len(neighbors_in_radius)} 個")
        print(f"  ✓ 半徑外參考鄰居: {len(neighbors_outside_radius)} 個")

        # --- 4. 準備繪圖和計算模組 ---
        print("\n[4/5] 分析所有鄰居的配對結果...")
        pathfinder = ImagePathfinder(green_channel, verbose=self.config.verbose)
        cost_calculator = CostCalculator(
            pathfinder,
            alpha=self.config.alpha, beta=self.config.beta, gamma=self.config.gamma
        )

        # 定義裁切區域 (ROI) 並縮放
        h, w = green_channel.shape
        x, y = source_seed.x, source_seed.y
        y_min, y_max = max(0, y - zoom_radius), min(h, y + zoom_radius)
        x_min, x_max = max(0, x - zoom_radius), min(w, x + zoom_radius)

        roi = green_channel[y_min:y_max, x_min:x_max]
        scale_factor = 2.5
        roi_height, roi_width = roi.shape
        scaled_width = int(roi_width * scale_factor)
        scaled_height = int(roi_height * scale_factor)
        roi_scaled = cv2.resize(roi, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        offset = np.array([x_min, y_min])

        # 分析半徑內鄰居
        viz_results = []
        stats = {'success': 0, 'high_cost': 0, 'cutoff_distance': 0, 'cutoff_cost': 0, 'no_path': 0, 'outside_radius': 0}

        for neighbor_seed in neighbors_in_radius:
            distance = np.linalg.norm([neighbor_seed.x - source_seed.x, neighbor_seed.y - source_seed.y])
            start_pos = (source_seed.y, source_seed.x)
            end_pos = (neighbor_seed.y, neighbor_seed.x)

            path_result = pathfinder.find_path(
                start_pos, end_pos,
                max_distance_from_start=self.config.distance_from_start_cutoff
            )

            status = path_result['status']
            result_data = {
                'neighbor': neighbor_seed,
                'in_radius': True,
                'euclidean_distance': distance,
                **path_result
            }

            if status == 'success':
                costs = cost_calculator.calculate_total_cost(source_seed, neighbor_seed)
                result_data.update(costs)
                if costs['total_cost'] > self.config.max_edge_cost:
                    result_data['status'] = 'high_cost'
                    stats['high_cost'] += 1
                else:
                    stats['success'] += 1
            elif status == 'cutoff':
                # 細分 cutoff 原因
                cutoff_reason = path_result.get('reason', 'unknown')
                if cutoff_reason == 'distance_from_start':
                    result_data['status'] = 'cutoff_distance'
                    stats['cutoff_distance'] += 1
                else:  # 'max_g_cost'
                    result_data['status'] = 'cutoff_cost'
                    stats['cutoff_cost'] += 1
            else:  # no_path
                stats['no_path'] += 1

            viz_results.append(result_data)

        # 分析半徑外鄰居（僅用於顯示對比）
        for neighbor_seed in neighbors_outside_radius:
            distance = np.linalg.norm([neighbor_seed.x - source_seed.x, neighbor_seed.y - source_seed.y])
            result_data = {
                'neighbor': neighbor_seed,
                'in_radius': False,
                'euclidean_distance': distance,
                'status': 'outside_radius',
                'path': None
            }
            viz_results.append(result_data)
            stats['outside_radius'] += 1

        print(f"  ✓ 完成 {len(viz_results)} 個鄰居的分析")
        print(f"    - ✅ 成功: {stats['success']}")
        print(f"    - ⚠️  成本過高: {stats['high_cost']}")
        print(f"    - 🔴 A* 距離 cutoff: {stats['cutoff_distance']}")
        print(f"    - 🟣 A* 成本 cutoff: {stats['cutoff_cost']}")
        print(f"    - ⚫ 無路徑: {stats['no_path']}")
        print(f"    - ⚪ 半徑外: {stats['outside_radius']}")

        # --- 5. 繪製視覺化結果 ---
        print("\n[5/5] 繪製視覺化結果...")

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec

        # 計算子圖佈局
        num_neighbors = len(viz_results)
        num_cols = min(3, num_neighbors)
        num_rows = max(1, (num_neighbors + num_cols - 1) // num_cols)

        # 創建圖形
        fig = plt.figure(figsize=(22, max(8, num_rows * 4)))
        gs = GridSpec(max(num_rows, 2), 5, figure=fig, hspace=0.4, wspace=0.3, width_ratios=[1.2, 1, 1, 1, 0.3])

        # === Main plot: Overview with adaptive radius circle ===
        ax_main = fig.add_subplot(gs[:, 0])
        ax_main.imshow(roi_scaled, cmap='gray')
        ax_main.set_title(
            f'Seed {target_seed_id} - Pairing Overview\n'
            f'Local Density: {local_density:.1f} | Pairing Radius: {adaptive_radius}px',
            fontsize=13, fontweight='bold'
        )
        ax_main.axis('off')

        def transform_coord_plot(x, y):
            """轉換座標用於 matplotlib"""
            return ((x - offset[0]) * scale_factor, (y - offset[1]) * scale_factor)

        # 繪製自適應半徑圓圈
        source_pos = transform_coord_plot(source_seed.x, source_seed.y)
        circle = plt.Circle(
            source_pos, adaptive_radius * scale_factor,
            color='cyan', fill=False, linewidth=2.5, linestyle='--', alpha=0.7, zorder=2
        )
        ax_main.add_patch(circle)

        # 繪製所有路徑和鄰居
        for idx, res in enumerate(viz_results, 1):
            neighbor = res['neighbor']
            path = res.get('path')
            status = res['status']

            # 根據狀態選擇顏色和樣式
            if status == 'success':
                color, alpha, linestyle = 'green', 0.7, '-'
            elif status == 'high_cost':
                color, alpha, linestyle = 'orange', 0.6, '-'
            elif status == 'cutoff_distance':
                color, alpha, linestyle = 'red', 0.5, '--'
            elif status == 'cutoff_cost':
                color, alpha, linestyle = 'purple', 0.5, '--'
            elif status == 'no_path':
                color, alpha, linestyle = 'gray', 0.4, ':'
            else:  # outside_radius
                color, alpha, linestyle = 'lightgray', 0.3, ':'

            # 繪製路徑或直線
            if path:
                pts = [transform_coord_plot(pt[1], pt[0]) for pt in path]
                xs, ys = zip(*pts)
                ax_main.plot(xs, ys, color=color, alpha=alpha, linewidth=1.5, linestyle=linestyle)
            else:
                pt1 = transform_coord_plot(source_seed.x, source_seed.y)
                pt2 = transform_coord_plot(neighbor.x, neighbor.y)
                ax_main.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                           color=color, alpha=alpha, linestyle=linestyle, linewidth=1)

            # 標記鄰居編號
            neighbor_pos = transform_coord_plot(neighbor.x, neighbor.y)
            marker_size = 100 if res['in_radius'] else 60
            ax_main.scatter([neighbor_pos[0]], [neighbor_pos[1]],
                          c=color, s=marker_size, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
            if res['in_radius']:
                ax_main.text(neighbor_pos[0], neighbor_pos[1], str(idx),
                           ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)

        # 標記源種子
        ax_main.scatter([source_pos[0]], [source_pos[1]],
                      c='blue', s=250, marker='*', edgecolors='white', linewidths=2.5, zorder=10)
        ax_main.text(source_pos[0], source_pos[1] - 25, 'SOURCE',
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor='blue', alpha=0.8))

        # === 子圖：為每個鄰居創建獨立的詳細視圖 ===
        for idx, res in enumerate(viz_results):
            row = idx // num_cols
            col = (idx % num_cols) + 1

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(roi_scaled, cmap='gray')
            ax.axis('off')

            neighbor = res['neighbor']
            path = res.get('path')
            status = res['status']

            # Set color and status text
            if status == 'success':
                color = 'green'
                status_text = f"SUCCESS\nCost: {res.get('total_cost', 0):.1f}"
            elif status == 'high_cost':
                color = 'orange'
                status_text = f"HIGH COST\n{res.get('total_cost', 0):.1f} > {self.config.max_edge_cost}"
            elif status == 'cutoff_distance':
                color = 'red'
                status_text = f"CUTOFF (Distance)\nMax: {self.config.distance_from_start_cutoff}px"
            elif status == 'cutoff_cost':
                color = 'purple'
                status_text = f"CUTOFF (Cost)\nAccum. too high"
            elif status == 'no_path':
                color = 'gray'
                status_text = "NO PATH\nObstacle/Unreachable"
            else:  # outside_radius
                color = 'lightgray'
                status_text = f"OUTSIDE RADIUS\nDist: {res['euclidean_distance']:.1f}px"

            # 繪製路徑
            if path:
                pts = [transform_coord_plot(pt[1], pt[0]) for pt in path]
                xs, ys = zip(*pts)
                linestyle = '-' if status in ['success', 'high_cost'] else '--'
                ax.plot(xs, ys, color=color, linewidth=3, alpha=0.8, linestyle=linestyle, zorder=3)
            else:
                pt1 = transform_coord_plot(source_seed.x, source_seed.y)
                pt2 = transform_coord_plot(neighbor.x, neighbor.y)
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                       color=color, linestyle=':', linewidth=2, alpha=0.5, zorder=3)

            # 標記種子點
            source_pos = transform_coord_plot(source_seed.x, source_seed.y)
            neighbor_pos = transform_coord_plot(neighbor.x, neighbor.y)
            ax.scatter([source_pos[0]], [source_pos[1]],
                      c='blue', s=150, marker='*', edgecolors='white', linewidths=1.5, zorder=5)
            ax.scatter([neighbor_pos[0]], [neighbor_pos[1]],
                      c=color, s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)

            # 標題
            ax.set_title(f"Neighbor #{idx+1}\n{status_text}",
                        fontsize=9, fontweight='bold', color=color, pad=10)

        # === Statistics summary panel (right side) ===
        ax_stats = fig.add_subplot(gs[:, 4])
        ax_stats.axis('off')

        stats_text = (
            f"STATISTICS\n"
            f"{'='*20}\n\n"
            f"Total Neighbors:\n  {len(viz_results)}\n\n"
            f"In Radius ({adaptive_radius}px):\n  {len(neighbors_in_radius)}\n\n"
            f"Success:\n  {stats['success']}\n\n"
            f"High Cost:\n  {stats['high_cost']}\n\n"
            f"Cutoff (Dist):\n  {stats['cutoff_distance']}\n\n"
            f"Cutoff (Cost):\n  {stats['cutoff_cost']}\n\n"
            f"No Path:\n  {stats['no_path']}\n\n"
            f"Outside:\n  {stats['outside_radius']}\n"
        )

        ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8))

        # Add overall legend
        legend_elements = [
            mpatches.Patch(color='green', label='Success'),
            mpatches.Patch(color='orange', label='High Cost'),
            mpatches.Patch(color='red', label='Cutoff (Distance)'),
            mpatches.Patch(color='purple', label='Cutoff (Cost)'),
            mpatches.Patch(color='gray', label='No Path'),
            mpatches.Patch(color='lightgray', label='Outside Radius'),
        ]
        fig.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9, ncol=2)

        # 儲存圖形
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  ✓ 視覺化結果已儲存至: {output_path}")
        print(f"  ✓ 生成 1 個總覽圖 + {num_neighbors} 個詳細子圖 + 統計面板")
        print("=" * 60)


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
