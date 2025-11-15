"""
MST 森林構建器

從候選連接圖構建約束 MST 森林，允許多個獨立連通分量
"""

import networkx as nx
from typing import Dict, Any, List


class MSTBuilder:
    """
    MST 森林構建器

    根據 README 第 312-330 行的策略：
    1. 設定最大可接受成本
    2. 拒絕成本超過上限的邊
    3. 允許產生森林（多個連通分量）
    """

    def __init__(self, max_edge_cost: float = 150):
        """
        初始化 MST 構建器

        Args:
            max_edge_cost: 最大可接受邊成本，超過此成本的邊將被拒絕
        """
        self.max_edge_cost = max_edge_cost

    def build_constrained_mst_forest(self, G: nx.Graph) -> nx.Graph:
        """
        構建約束 MST 森林

        策略（README 321-329 行）：
        1. 過濾高成本邊（cost >= max_edge_cost）
        2. 對每個連通分量獨立構建 MST
        3. 合併所有分量的 MST 形成森林

        Args:
            G: 輸入的完整連接圖（來自階段三）

        Returns:
            forest: MST 森林，可能包含多個連通分量
        """
        if G is None or G.number_of_nodes() == 0:
            print("⚠️  輸入圖為空")
            return nx.Graph()

        print(f"\n構建 MST 森林...")
        print(f"  輸入圖: {G.number_of_nodes()} 節點, {G.number_of_edges()} 邊")

        # 1. 過濾高成本邊
        filtered_graph = self._filter_high_cost_edges(G)
        print(f"  過濾後: {filtered_graph.number_of_nodes()} 節點, {filtered_graph.number_of_edges()} 邊")

        # 2. 找出所有連通分量
        components = list(nx.connected_components(filtered_graph))
        print(f"  連通分量數: {len(components)}")

        # 3. 對每個分量構建 MST
        forest = nx.Graph()

        for i, component_nodes in enumerate(components):
            # 提取子圖
            subgraph = filtered_graph.subgraph(component_nodes).copy()

            # 如果只有一個節點，直接添加
            if subgraph.number_of_nodes() == 1:
                forest.add_nodes_from(subgraph.nodes(data=True))
                continue

            # 構建 MST
            mst = nx.minimum_spanning_tree(subgraph, weight='weight')

            # 添加到森林中
            forest.add_nodes_from(mst.nodes(data=True))
            forest.add_edges_from(mst.edges(data=True))

            if i < 5 or len(components) <= 10:  # 只顯示前 5 個或所有分量（如果 <= 10）
                print(f"    分量 {i+1}: {mst.number_of_nodes()} 節點, {mst.number_of_edges()} 邊")

        if len(components) > 10:
            print(f"    ... ({len(components) - 5} 個分量未顯示)")

        print(f"  ✓ MST 森林: {forest.number_of_nodes()} 節點, {forest.number_of_edges()} 邊")

        return forest

    def _filter_high_cost_edges(self, G: nx.Graph) -> nx.Graph:
        """
        過濾掉高成本邊

        Args:
            G: 輸入圖

        Returns:
            filtered_graph: 只包含成本 < max_edge_cost 的邊的圖
        """
        filtered_graph = nx.Graph()

        # 保留所有節點
        filtered_graph.add_nodes_from(G.nodes(data=True))

        # 只保留低成本邊
        for u, v, data in G.edges(data=True):
            cost = data.get('weight', float('inf'))
            if cost < self.max_edge_cost:
                filtered_graph.add_edge(u, v, **data)

        return filtered_graph

    def get_forest_statistics(self, forest: nx.Graph) -> Dict[str, Any]:
        """
        獲取森林統計資訊

        Args:
            forest: MST 森林

        Returns:
            stats: 包含統計資訊的字典
        """
        if forest is None or forest.number_of_nodes() == 0:
            return {
                'num_components': 0,
                'total_nodes': 0,
                'total_edges': 0,
                'components_info': []
            }

        # 找出所有連通分量
        components = list(nx.connected_components(forest))

        # 收集每個分量的資訊
        components_info = []
        for i, component_nodes in enumerate(components):
            subgraph = forest.subgraph(component_nodes)

            # 計算平均權重
            if subgraph.number_of_edges() > 0:
                weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
                avg_weight = sum(weights) / len(weights)
                min_weight = min(weights)
                max_weight = max(weights)
            else:
                avg_weight = 0
                min_weight = 0
                max_weight = 0

            # 計算度數統計
            degrees = [d for _, d in subgraph.degree()]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            max_degree = max(degrees) if degrees else 0

            components_info.append({
                'component_id': i,
                'num_nodes': subgraph.number_of_nodes(),
                'num_edges': subgraph.number_of_edges(),
                'avg_weight': avg_weight,
                'min_weight': min_weight,
                'max_weight': max_weight,
                'avg_degree': avg_degree,
                'max_degree': max_degree
            })

        # 按節點數排序（大到小）
        components_info.sort(key=lambda x: x['num_nodes'], reverse=True)

        return {
            'num_components': len(components),
            'total_nodes': forest.number_of_nodes(),
            'total_edges': forest.number_of_edges(),
            'components_info': components_info
        }

    def print_statistics(self, stats: Dict[str, Any], verbose: bool = False):
        """
        打印森林統計資訊

        Args:
            stats: 統計資訊字典
            verbose: 是否顯示每個分量的詳細資訊
        """
        print("\n" + "=" * 60)
        print("MST 森林統計")
        print("=" * 60)
        print(f"連通分量數: {stats['num_components']}")
        print(f"總節點數: {stats['total_nodes']}")
        print(f"總邊數: {stats['total_edges']}")

        if verbose and stats['components_info']:
            print("\n各分量詳情：")
            print(f"{'ID':<5} {'節點':<8} {'邊':<8} {'平均權重':<12} {'最大度數':<10}")
            print("-" * 60)

            for info in stats['components_info'][:10]:  # 只顯示前 10 個
                print(f"{info['component_id']:<5} "
                      f"{info['num_nodes']:<8} "
                      f"{info['num_edges']:<8} "
                      f"{info['avg_weight']:<12.2f} "
                      f"{info['max_degree']:<10}")

            if len(stats['components_info']) > 10:
                print(f"... ({len(stats['components_info']) - 10} 個分量未顯示)")

        print("=" * 60)


if __name__ == '__main__':
    # 測試程式碼
    print("MST 構建器測試")

    # 創建測試圖
    G = nx.Graph()
    G.add_edge(1, 2, weight=10)
    G.add_edge(2, 3, weight=20)
    G.add_edge(3, 4, weight=30)
    G.add_edge(1, 4, weight=100)  # 高成本邊
    G.add_edge(5, 6, weight=15)   # 另一個分量

    # 測試 MST 構建
    builder = MSTBuilder(max_edge_cost=50)
    forest = builder.build_constrained_mst_forest(G)
    stats = builder.get_forest_statistics(forest)
    builder.print_statistics(stats, verbose=True)
