'''
網路視覺化器
'''

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from typing import List, Dict, Any
from scipy.interpolate import griddata

# 類型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .seed_loader import Seed

class NetworkVisualizer:
    """
    網路視覺化工具
    - 繪製網路圖並疊加在背景影像上
    - 根據節點和邊的屬性使用不同樣式
    """
    
    def __init__(self):
        # 樣式設定
        self.seed_styles = {
            'endpoint': {'marker': 'o', 'color': '#e63946', 's': 30, 'zorder': 10}, # 亮紅色
            'branchpoint': {'marker': 's', 'color': '#457b9d', 's': 40, 'zorder': 10}, # 藍色
            'centroid': {'marker': 'D', 'color': '#fca311', 's': 30, 'zorder': 10}, # 橘黃色
            'curvature': {'marker': '*', 'color': '#f1faee', 's': 80, 'zorder': 11, 'edgecolors': 'black'}, # 白色帶黑邊
            'regular': {'marker': '.', 'color': '#a8dadc', 's': 10, 'zorder': 9}, # 淡藍色
            'default': {'marker': 'x', 'color': 'gray', 's': 20, 'zorder': 9}
        }

    def _draw_network_on_ax(self, ax, G: nx.Graph, seeds: List['Seed'], green_channel: np.ndarray,
                            enhanced_edges: bool = False):
        """在給定的 Matplotlib Axes 上繪製網路

        Args:
            enhanced_edges: 是否使用增強的邊樣式（用於放大圖）
        """
        # 1. 繪製背景影像
        ax.imshow(green_channel, cmap='gray')

        # 建立 seed_id -> seed 物件的映射,方便查找
        seed_map = {seed.id: seed for seed in seeds}

        # 2. 繪製邊
        norm_factor = self.get_norm_factor(G)
        for u, v, data in G.edges(data=True):
            seed_u = seed_map.get(u)
            seed_v = seed_map.get(v)

            if not seed_u or not seed_v:
                continue

            # 使用更高對比的顏色
            if enhanced_edges:
                color = '#00d9ff' if data.get('edge_type') == 'intra_component' else '#ff00ff'  # 亮青色/洋紅色
                weight = data.get('weight', 150)
                linewidth = max(2.0, 5.0 - (weight / norm_factor))  # 更粗的線
                alpha = 0.9  # 更不透明
            else:
                color = '#457b9d' if data.get('edge_type') == 'intra_component' else '#e63946'
                weight = data.get('weight', 150)
                linewidth = max(1.5, 4.0 - (weight / norm_factor))  # 增加最小線寬
                alpha = 0.8  # 提高透明度

            ax.plot([seed_u.x, seed_v.x], [seed_u.y, seed_v.y],
                   color=color, linewidth=linewidth, alpha=alpha, zorder=5)

        # 3. 繪製節點 (按類型分組繪製以產生圖例)
        seeds_by_type = self.group_seeds_by_type(seeds)
        for seed_type, type_seeds in seeds_by_type.items():
            style = self.seed_styles.get(seed_type, self.seed_styles['default'])
            
            xs = [s.x for s in type_seeds]
            ys = [s.y for s in type_seeds]
            
            scatter_kwargs = style.copy()
            if 'edgecolors' not in scatter_kwargs and seed_type != 'regular':
                scatter_kwargs['edgecolors'] = 'black'
                scatter_kwargs['linewidths'] = 0.5

            ax.scatter(xs, ys, label=seed_type, **scatter_kwargs)

    def visualize_network(
        self,
        G: nx.Graph,
        seeds: List['Seed'],
        green_channel: np.ndarray,
        output_path: str
    ):
        """繪製完整的網路圖"""
        if G is None:
            print("⚠️  圖物件為空,跳過視覺化")
            return

        fig, ax = plt.subplots(figsize=(24, 18), dpi=150)
        self._draw_network_on_ax(ax, G, seeds, green_channel)

        # 美化與儲存
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), frameon=True, facecolor='lightgray', fontsize='small')
        ax.set_title('Neural Network Reconstruction (Full View)', fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def visualize_zoomed_view(
        self,
        G: nx.Graph,
        seeds: List['Seed'],
        green_channel: np.ndarray,
        output_path: str,
        zoom_radius: int = 200
    ):
        """繪製以隨機節點為中心的局部放大圖"""
        if G is None or G.number_of_nodes() == 0:
            print("⚠️  圖物件為空或沒有節點,跳過放大圖視覺化")
            return

        fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
        self._draw_network_on_ax(ax, G, seeds, green_channel, enhanced_edges=True)

        # 隨機選擇一個節點作為中心
        seed_map = {seed.id: seed for seed in seeds}
        center_node_id = random.choice(list(G.nodes()))
        center_seed = seed_map.get(center_node_id)

        if center_seed:
            # 設定放大範圍
            x_center, y_center = center_seed.x, center_seed.y
            ax.set_xlim(x_center - zoom_radius, x_center + zoom_radius)
            ax.set_ylim(y_center + zoom_radius, y_center - zoom_radius) # y軸反轉
            
            # 在中心點做個標記
            ax.scatter(x_center, y_center, s=300, facecolors='none', edgecolors='yellow', linewidths=2, zorder=12)
            title = f'Zoomed View (around node {center_node_id})'
        else:
            title = 'Zoomed View (Random Area)'

        # 美化與儲存
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title(title, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def get_norm_factor(self, G: nx.Graph) -> float:
        """取得用於標準化線寬的成本因子"""
        if G.number_of_edges() == 0:
            return 1.0
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        return np.mean(weights) if weights else 1.0

    def group_seeds_by_type(self, seeds: List['Seed']) -> Dict[str, List['Seed']]:
        """將種子按類型分組"""
        grouped = {}
        for seed in seeds:
            if seed.seed_type not in grouped:
                grouped[seed.seed_type] = []
            grouped[seed.seed_type].append(seed)
        return grouped

    def visualize_cost_distribution(self, G: nx.Graph, output_path: str):
        """繪製成本分布直方圖"""
        if G is None or G.number_of_edges() == 0:
            print("⚠️  圖物件為空或沒有邊,跳過成本分布圖繪製")
            return

        costs = [data['weight'] for _, _, data in G.edges(data=True)]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(costs, bins=50, color='#457b9d', edgecolor='black')
        ax.set_title('Edge Cost Distribution', fontsize=16)
        ax.set_xlabel('Total Cost')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.75)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    def visualize_density_heatmap(
        self,
        seeds: List['Seed'],
        density_info: Dict[int, Dict[str, float]],
        green_channel: np.ndarray,
        output_path: str
    ):
        """
        繪製密度熱力圖（平滑插值版本）

        Args:
            seeds: 種子列表
            density_info: 密度資訊字典 {seed_id: {'local_density': float, 'pairing_radius': float}}
            green_channel: 背景影像
            output_path: 輸出路徑
        """
        if not seeds or not density_info:
            print("⚠️  沒有種子或密度資訊，跳過密度熱力圖繪製")
            return

        # 提取種子位置和密度值
        points = []
        densities = []
        for seed in seeds:
            if seed.id in density_info:
                points.append([seed.x, seed.y])
                densities.append(density_info[seed.id]['local_density'])

        if len(points) == 0:
            print("⚠️  沒有有效的密度數據，跳過密度熱力圖繪製")
            return

        points = np.array(points)
        densities = np.array(densities)

        # 建立插值網格
        height, width = green_channel.shape
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, width-1, width//4),  # 降低解析度以加快速度
            np.linspace(0, height-1, height//4)
        )

        # 使用插值生成平滑的密度分布
        grid_density = griddata(points, densities, (grid_x, grid_y), method='cubic', fill_value=np.nan)

        # 繪圖
        fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

        # 1. 繪製背景影像
        ax.imshow(green_channel, cmap='gray', alpha=0.7)

        # 2. 疊加密度熱力圖
        heatmap = ax.contourf(grid_x, grid_y, grid_density, levels=20, cmap='hot_r', alpha=0.6)

        # 3. 添加 colorbar
        cbar = plt.colorbar(heatmap, ax=ax, label='Local Density (pixels)')

        # 4. 在 colorbar 上標示密度等級邊界
        cbar.ax.axhline(y=30, color='cyan', linewidth=2, linestyle='--', label='Dense/Medium (30px)')
        cbar.ax.axhline(y=70, color='yellow', linewidth=2, linestyle='--', label='Medium/Sparse (70px)')

        # 5. 添加文字標註
        cbar.ax.text(1.5, 15, 'Dense', fontsize=10, color='cyan', weight='bold')
        cbar.ax.text(1.5, 50, 'Medium', fontsize=10, color='yellow', weight='bold')
        cbar.ax.text(1.5, 85, 'Sparse', fontsize=10, color='orange', weight='bold')

        # 美化
        ax.set_title('Neural Network Density Heatmap', fontsize=18, weight='bold')
        ax.set_xlabel('X coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y coordinate (pixels)', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        print(f"  ✓ 密度熱力圖: {output_path}")
        print(f"    - 密度範圍: {densities.min():.1f} ~ {densities.max():.1f} px")
        print(f"    - 平均密度: {densities.mean():.1f} px")

    def visualize_cost_ratio_analysis(
        self,
        edges_with_costs: List[Dict[str, Any]],
        output_path: str
    ):
        """
        繪製路徑成本與直線距離的比值分析圖

        Args:
            edges_with_costs: 邊成本資料列表
            output_path: 輸出路徑
        """
        if not edges_with_costs:
            print("⚠️  沒有邊成本資料，跳過成本比值分析")
            return

        # 提取資料並計算比值
        geometric_costs = []
        cost_ratios = []
        edge_types = []

        for edge in edges_with_costs:
            geometric = edge.get('geometric_cost', 0)
            path_cost = edge.get('path_cost', 0)

            # 過濾掉無效數據
            if geometric > 0 and np.isfinite(geometric) and np.isfinite(path_cost):
                ratio = path_cost / geometric
                if np.isfinite(ratio):  # 確保比值有效
                    geometric_costs.append(geometric)
                    cost_ratios.append(ratio)
                    edge_types.append(edge.get('edge_type', 'unknown'))

        if len(cost_ratios) == 0:
            print("⚠️  沒有有效的成本比值數據")
            return

        geometric_costs = np.array(geometric_costs)
        cost_ratios = np.array(cost_ratios)

        # 計算統計資訊
        percentiles = [50, 75, 90, 95, 99]
        percentile_values = np.percentile(cost_ratios, percentiles)

        # 創建三個子圖
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # === 子圖 1: 散點圖 ===
        ax1 = axes[0]

        # 根據 edge_type 分組繪製
        intra_mask = np.array([et == 'intra_component' for et in edge_types])
        inter_mask = ~intra_mask

        if intra_mask.any():
            ax1.scatter(geometric_costs[intra_mask], cost_ratios[intra_mask],
                       alpha=0.5, s=20, c='#457b9d', label='Intra-component', edgecolors='none')
        if inter_mask.any():
            ax1.scatter(geometric_costs[inter_mask], cost_ratios[inter_mask],
                       alpha=0.5, s=20, c='#e63946', label='Inter-component', edgecolors='none')

        # 添加參考線
        reference_lines = [1.0, 2.0, 3.0, 5.0, 10.0]
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        for ref, color in zip(reference_lines, colors):
            ax1.axhline(y=ref, color=color, linestyle='--', linewidth=1, alpha=0.7, label=f'{ref}x')

        ax1.set_xlabel('Geometric Distance (pixels)', fontsize=11)
        ax1.set_ylabel('Cost Ratio (path_cost / geometric_cost)', fontsize=11)
        ax1.set_title('Cost Ratio vs Geometric Distance', fontsize=13, weight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, min(max(cost_ratios), 20))  # 限制 Y 軸範圍避免極端值

        # === 子圖 2: 比值分布直方圖 ===
        ax2 = axes[1]

        # 過濾極端值以改善視覺效果
        filtered_ratios = cost_ratios[cost_ratios <= 15]

        ax2.hist(filtered_ratios, bins=50, color='#457b9d', edgecolor='black', alpha=0.7)

        # 標示當前使用的倍數 (5.0)
        ax2.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Current (5.0x)')

        # 標示統計值
        median_val = np.median(cost_ratios)
        ax2.axvline(x=median_val, color='green', linestyle='-.', linewidth=1.5, label=f'Median ({median_val:.2f}x)')

        ax2.set_xlabel('Cost Ratio', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Cost Ratio Distribution', fontsize=13, weight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        # === 子圖 3: 累積分布圖 (CDF) ===
        ax3 = axes[2]

        sorted_ratios = np.sort(cost_ratios)
        cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios) * 100

        ax3.plot(sorted_ratios, cumulative, color='#457b9d', linewidth=2)

        # 標示關鍵倍數的覆蓋率
        key_multipliers = [1.5, 2.0, 3.0, 5.0, 10.0]
        for mult in key_multipliers:
            coverage = (cost_ratios <= mult).sum() / len(cost_ratios) * 100
            ax3.axvline(x=mult, color='red' if mult == 5.0 else 'gray',
                       linestyle='--', alpha=0.7, linewidth=1.5 if mult == 5.0 else 1)
            ax3.text(mult, coverage + 2, f'{mult}x\n{coverage:.1f}%',
                    fontsize=8, ha='center', weight='bold' if mult == 5.0 else 'normal')

        ax3.set_xlabel('Cost Ratio', fontsize=11)
        ax3.set_ylabel('Cumulative Coverage (%)', fontsize=11)
        ax3.set_title('Cumulative Distribution (CDF)', fontsize=13, weight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 12)
        ax3.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 輸出統計摘要
        print(f"  ✓ 成本比值分析: {output_path}")
        print(f"    - 總邊數: {len(cost_ratios)}")
        print(f"    - 比值範圍: {cost_ratios.min():.2f} ~ {cost_ratios.max():.2f}")
        print(f"    - 平均值: {cost_ratios.mean():.2f}x")
        print(f"    - 中位數: {median_val:.2f}x")
        print(f"    - 百分位數:")
        for p, v in zip(percentiles, percentile_values):
            coverage = (cost_ratios <= v).sum() / len(cost_ratios) * 100
            print(f"      {p}%: {v:.2f}x (覆蓋 {coverage:.1f}% 的邊)")

        # 給出建議
        p95_val = percentile_values[-2]  # 95百分位
        print(f"    - 建議: 若要保留 95% 的邊，可設定倍數為 {p95_val:.1f}x")

    def visualize_sample_paths(
        self,
        G: nx.Graph,
        seeds: List['Seed'],
        green_channel: np.ndarray,
        output_path: str,
        num_samples: int = 15,
        zoom_radius: int = 200
    ):
        """
        視覺化隨機選取的邊的實際路徑（放大顯示）

        Args:
            G: NetworkX 圖物件
            seeds: 種子列表
            green_channel: 背景影像
            output_path: 輸出路徑
            num_samples: 隨機選取的邊數量
            zoom_radius: 放大半徑（像素）
        """
        if G is None or G.number_of_edges() == 0:
            print("⚠️  圖物件為空或沒有邊，跳過路徑視覺化")
            return

        # 隨機選取邊
        all_edges = list(G.edges(data=True))
        num_samples = min(num_samples, len(all_edges))
        sampled_edges = random.sample(all_edges, num_samples)

        # 建立 seed_id -> seed 物件的映射
        seed_map = {seed.id: seed for seed in seeds}

        # 創建圖表
        fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

        # 1. 繪製背景影像
        ax.imshow(green_channel, cmap='gray', alpha=0.8)

        # 2. 準備顏色映射
        colors = plt.cm.tab20(np.linspace(0, 1, num_samples))

        # 3. 收集所有路徑點以計算放大中心
        all_path_points = []

        # 4. 繪製每條邊的路徑
        for idx, (source_id, target_id, data) in enumerate(sampled_edges):
            # 獲取路徑（需要反序列化）
            path_str = data.get('path', 'None')
            if path_str == 'None' or path_str is None:
                continue

            # 反序列化路徑字串
            try:
                import ast
                path = ast.literal_eval(path_str)
            except:
                continue

            if not path or len(path) < 2:
                continue

            # 提取 x, y 座標
            ys = [pos[0] for pos in path]
            xs = [pos[1] for pos in path]

            # 收集所有點用於計算中心
            all_path_points.extend(path)

            # 繪製路徑（增加線寬以適應放大）
            ax.plot(xs, ys, color=colors[idx], linewidth=3, alpha=0.8, label=f'Edge {idx+1}')

            # 標示起點（綠色）和終點（紅色）（增加標記大小）
            ax.scatter(xs[0], ys[0], color='lime', s=120, marker='o', edgecolors='black', linewidths=1.5, zorder=10)
            ax.scatter(xs[-1], ys[-1], color='red', s=120, marker='o', edgecolors='black', linewidths=1.5, zorder=10)

            # 標註統計資訊（在起點附近）
            stats_text = f"L={len(path)}, T={data.get('tortuosity', 0):.2f}"
            ax.text(xs[0], ys[0]-15, stats_text, fontsize=8, color=colors[idx],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=colors[idx]))

        # 5. 計算放大中心並設定視圖範圍
        if len(all_path_points) > 0:
            center_x = np.median([p[1] for p in all_path_points])
            center_y = np.median([p[0] for p in all_path_points])

            # 設定放大範圍
            ax.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
            ax.set_ylim(center_y + zoom_radius, center_y - zoom_radius)  # y軸反轉

            title_suffix = f' (Zoomed at [{center_x:.0f}, {center_y:.0f}], radius={zoom_radius}px)'
        else:
            center_x, center_y = 0, 0
            title_suffix = ''

        # 6. 添加圖例和標題
        ax.set_title('Sample Edge Paths Visualization' + title_suffix, fontsize=14, weight='bold')
        ax.set_xlabel('X coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y coordinate (pixels)', fontsize=12)

        # 添加起點/終點圖例
        ax.scatter([], [], color='lime', s=120, marker='o', edgecolors='black', label='Start', linewidths=1.5)
        ax.scatter([], [], color='red', s=120, marker='o', edgecolors='black', label='End', linewidths=1.5)

        ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0), fontsize=9, frameon=True, facecolor='white')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"  ✓ 路徑樣本視覺化: {output_path}")
        print(f"    - 繪製 {len(all_path_points)//num_samples if all_path_points else 0} 條邊的路徑")
        if len(all_path_points) > 0:
            print(f"    - 放大中心: ({center_x:.0f}, {center_y:.0f}), 半徑: {zoom_radius}px")
