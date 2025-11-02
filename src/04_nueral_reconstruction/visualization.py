"""
MST 森林視覺化

提供多種視覺化方法展示 MST 重建結果
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional


class MSTVisualizer:
    """
    MST 森林視覺化器

    提供四種主要視覺化：
    1. MST 森林全景（完整圖和放大圖）
    2. 連通分量分解（多子圖）
    3. 路徑質量熱力圖
    4. 驗證報告視覺化（階段二）
    """

    def __init__(self, green_channel: np.ndarray):
        """
        初始化視覺化器

        Args:
            green_channel: 背景影像（灰階）
        """
        self.green_channel = green_channel
        self.seed_map = {}  # seed_id -> seed dict

    def set_seeds(self, seeds: List[dict]):
        """
        設定種子映射

        Args:
            seeds: 種子列表，每個種子包含 id, x, y
        """
        # 同時支持 int 和 str 作為 key（因為 GraphML 可能轉換類型）
        for seed in seeds:
            seed_id = seed['id']
            self.seed_map[seed_id] = seed
            self.seed_map[str(seed_id)] = seed  # 也用字串作為 key

    def visualize_mst_forest(
        self,
        forest: nx.Graph,
        output_path: str,
        zoom: bool = False,
        zoom_radius: int = 200
    ):
        """
        繪製 MST 森林全景或放大圖

        Args:
            forest: MST 森林
            output_path: 輸出路徑
            zoom: 是否繪製放大圖
            zoom_radius: 放大半徑（像素）
        """
        if forest is None or forest.number_of_nodes() == 0:
            print("⚠️  森林為空，跳過視覺化")
            return

        print(f"\n繪製 MST 森林{'放大圖' if zoom else '全景'}...")

        # 創建圖表
        fig, ax = plt.subplots(figsize=(20, 15), dpi=150)

        # 1. 繪製背景
        ax.imshow(self.green_channel, cmap='gray', alpha=0.7)

        # 2. 獲取連通分量
        components = list(nx.connected_components(forest))
        print(f"  連通分量數: {len(components)}")

        # 3. 為每個分量分配顏色
        colors = self._get_component_colors(len(components))

        # 4. 繪製每個分量
        all_points = []  # 用於計算放大中心
        for comp_id, component_nodes in enumerate(components):
            color = colors[comp_id]

            # 提取子圖
            subgraph = forest.subgraph(component_nodes)

            # 繪製邊
            for u, v, data in subgraph.edges(data=True):
                self._draw_edge(ax, u, v, data, color)

                # 收集點用於放大
                if zoom:
                    seed_u = self.seed_map.get(u) or self.seed_map.get(str(u))
                    seed_v = self.seed_map.get(v) or self.seed_map.get(str(v))
                    if seed_u and seed_v:
                        all_points.extend([(seed_u['x'], seed_u['y']),
                                          (seed_v['x'], seed_v['y'])])

            # 繪製節點
            for node_id in component_nodes:
                seed = self.seed_map.get(node_id) or self.seed_map.get(str(node_id))
                if seed:
                    degree = subgraph.degree(node_id)
                    self._draw_node(ax, seed, degree, color)

        # 5. 如果是放大圖，設定視圖範圍
        if zoom and all_points:
            center_x = np.median([p[0] for p in all_points])
            center_y = np.median([p[1] for p in all_points])
            ax.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
            ax.set_ylim(center_y + zoom_radius, center_y - zoom_radius)
            title_suffix = f' (Zoomed at [{center_x:.0f}, {center_y:.0f}])'
        else:
            title_suffix = ''

        # 6. 添加標題和圖例
        ax.set_title(f'MST Forest Reconstruction{title_suffix}',
                    fontsize=18, weight='bold')
        ax.set_xlabel('X coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y coordinate (pixels)', fontsize=12)

        # 創建圖例
        legend_elements = [
            mpatches.Patch(color=colors[i], label=f'Component {i+1}')
            for i in range(min(10, len(components)))  # 最多顯示 10 個
        ]
        if len(components) > 10:
            legend_elements.append(
                mpatches.Patch(color='gray', label=f'... ({len(components)-10} more)')
            )

        ax.legend(handles=legend_elements, loc='upper right',
                 bbox_to_anchor=(1.08, 1.0), fontsize=9, frameon=True)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"  ✓ 已保存: {output_path}")

    def visualize_component_breakdown(
        self,
        forest: nx.Graph,
        output_path: str,
        max_components: int = 12
    ):
        """
        繪製連通分量分解圖（多子圖）

        Args:
            forest: MST 森林
            output_path: 輸出路徑
            max_components: 最多顯示的分量數
        """
        if forest is None or forest.number_of_nodes() == 0:
            print("⚠️  森林為空，跳過分量分解視覺化")
            return

        print(f"\n繪製連通分量分解...")

        # 獲取連通分量並按大小排序
        components = list(nx.connected_components(forest))
        components.sort(key=lambda c: len(c), reverse=True)

        num_components = min(len(components), max_components)
        print(f"  顯示前 {num_components} 個分量")

        # 計算子圖佈局
        n_cols = 4
        n_rows = (num_components + n_cols - 1) // n_cols

        # 創建圖表
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), dpi=150)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # 為每個分量分配顏色
        colors = self._get_component_colors(num_components)

        # 繪製每個分量
        for idx in range(num_components):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            component_nodes = components[idx]
            subgraph = forest.subgraph(component_nodes)
            color = colors[idx]

            # 計算分量的邊界框
            xs, ys = [], []
            for node_id in component_nodes:
                seed = self.seed_map.get(node_id) or self.seed_map.get(str(node_id))
                if seed:
                    xs.append(seed['x'])
                    ys.append(seed['y'])

            if not xs:
                ax.axis('off')
                continue

            # 計算放大範圍
            x_min, x_max = min(xs) - 50, max(xs) + 50
            y_min, y_max = min(ys) - 50, max(ys) + 50

            # 繪製背景（裁剪到分量範圍）
            x_min = max(0, int(x_min))
            x_max = min(self.green_channel.shape[1], int(x_max))
            y_min = max(0, int(y_min))
            y_max = min(self.green_channel.shape[0], int(y_max))

            cropped = self.green_channel[y_min:y_max, x_min:x_max]
            ax.imshow(cropped, cmap='gray', alpha=0.7,
                     extent=[x_min, x_max, y_max, y_min])

            # 繪製邊
            for u, v, data in subgraph.edges(data=True):
                self._draw_edge(ax, u, v, data, color)

            # 繪製節點
            for node_id in component_nodes:
                seed = self.seed_map.get(node_id) or self.seed_map.get(str(node_id))
                if seed:
                    degree = subgraph.degree(node_id)
                    self._draw_node(ax, seed, degree, color)

            # 統計資訊
            weights = [d['weight'] for _, _, d in subgraph.edges(data=True)]
            avg_weight = np.mean(weights) if weights else 0

            # 設定標題
            ax.set_title(
                f'Component {idx+1}\n'
                f'{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges\n'
                f'avg weight: {avg_weight:.1f}',
                fontsize=10, weight='bold'
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
            ax.set_xticks([])
            ax.set_yticks([])

        # 隱藏多餘的子圖
        for idx in range(num_components, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle('MST Forest - Component Breakdown', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"  ✓ 已保存: {output_path}")

    def visualize_quality_heatmap(
        self,
        forest: nx.Graph,
        output_path: str
    ):
        """
        繪製路徑質量熱力圖

        Args:
            forest: MST 森林
            output_path: 輸出路徑
        """
        if forest is None or forest.number_of_nodes() == 0:
            print("⚠️  森林為空，跳過質量熱力圖")
            return

        print(f"\n繪製路徑質量熱力圖...")

        # 創建圖表
        fig, ax = plt.subplots(figsize=(20, 15), dpi=150)

        # 繪製背景
        ax.imshow(self.green_channel, cmap='gray', alpha=0.7)

        # 計算每條邊的質量（路徑平均強度）
        edge_qualities = []
        for u, v, data in forest.edges(data=True):
            quality = self._calculate_edge_quality(data)
            edge_qualities.append(quality)

        if not edge_qualities:
            print("  ⚠️  無法計算邊質量")
            return

        # 標準化質量到 0-1
        min_q = min(edge_qualities)
        max_q = max(edge_qualities)
        norm_qualities = [(q - min_q) / (max_q - min_q) if max_q > min_q else 0.5
                         for q in edge_qualities]

        # 繪製邊（用質量著色）
        cmap = plt.cm.RdYlGn  # 紅黃綠
        for idx, (u, v, data) in enumerate(forest.edges(data=True)):
            norm_quality = norm_qualities[idx]
            color = cmap(norm_quality)

            self._draw_edge(ax, u, v, data, color, linewidth=2)

        # 繪製節點（統一顏色）
        for node_id in forest.nodes():
            seed = self.seed_map.get(node_id) or self.seed_map.get(str(node_id))
            if seed:
                degree = forest.degree(node_id)
                self._draw_node(ax, seed, degree, 'white', edgecolor='black')

        # 添加 colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_q, vmax=max_q))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Path Quality (avg intensity)')

        # 標題
        ax.set_title('MST Forest - Path Quality Heatmap', fontsize=18, weight='bold')
        ax.set_xlabel('X coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y coordinate (pixels)', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"  ✓ 已保存: {output_path}")
        print(f"    - 質量範圍: {min_q:.1f} ~ {max_q:.1f}")

    def _draw_edge(self, ax, u, v, data, color, linewidth=None):
        """繪製邊（沿著路徑或直線）"""
        seed_u = self.seed_map.get(u) or self.seed_map.get(str(u))
        seed_v = self.seed_map.get(v) or self.seed_map.get(str(v))

        if not seed_u or not seed_v:
            return

        # 根據權重決定線寬
        if linewidth is None:
            weight = data.get('weight', 50)
            linewidth = max(0.5, 3.0 - weight / 100)

        # 嘗試沿著路徑繪製
        path_str = data.get('path', 'None')
        if path_str and path_str != 'None':
            try:
                import ast
                path = ast.literal_eval(path_str)
                if path and len(path) >= 2:
                    ys = [pos[0] for pos in path]
                    xs = [pos[1] for pos in path]
                    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.8, zorder=5)
                    return
            except:
                pass

        # 否則繪製直線
        ax.plot([seed_u['x'], seed_v['x']], [seed_u['y'], seed_v['y']],
               color=color, linewidth=linewidth, alpha=0.8, zorder=5)

    def _draw_node(self, ax, seed, degree, color, edgecolor='black'):
        """繪製節點"""
        x, y = seed['x'], seed['y']

        # 分支點用較大的標記
        if degree >= 3:
            marker = 's'  # 方塊
            size = 10
            linewidth = 2
        else:
            marker = 'o'  # 圓點
            size = 5
            linewidth = 1

        ax.scatter(x, y, c=[color], marker=marker, s=size,
                  edgecolors=edgecolor, linewidths=linewidth, zorder=10)

    def _calculate_edge_quality(self, edge_data: dict) -> float:
        """計算邊的質量（路徑平均強度）"""
        path_str = edge_data.get('path', 'None')

        if not path_str or path_str == 'None':
            return 128  # 預設中等質量

        try:
            import ast
            path = ast.literal_eval(path_str)

            if not path:
                return 128

            # 提取路徑上的強度
            intensities = []
            for y, x in path:
                if 0 <= y < self.green_channel.shape[0] and 0 <= x < self.green_channel.shape[1]:
                    intensities.append(self.green_channel[y, x])

            return np.mean(intensities) if intensities else 128

        except:
            return 128

    def _get_component_colors(self, num_components: int) -> list:
        """生成分量顏色列表"""
        if num_components <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.hsv

        colors = [cmap(i / max(num_components, 1)) for i in range(num_components)]
        return colors


if __name__ == '__main__':
    print("MST 視覺化器測試")
    # 需要實際資料才能測試
