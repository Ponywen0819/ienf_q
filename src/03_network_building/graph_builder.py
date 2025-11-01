'''
圖建構器
'''

import networkx as nx
from typing import List, Dict, Any

# 類型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .seed_loader import Seed

class GraphBuilder:
    """
    NetworkX 圖建構與邊過濾
    
    - 從種子和邊列表建立圖
    - 過濾掉成本高於閾值的邊
    """
    
    def __init__(self, max_edge_cost: float = 150.0):
        """
        初始化圖建構器

        Args:
            max_edge_cost: 允許的最大邊成本,超過此成本的邊將被過濾
        """
        self.max_edge_cost = max_edge_cost
    
    def build_graph(
        self,
        seeds: List['Seed'],
        edges_with_costs: List[Dict[str, Any]]
    ) -> nx.Graph:
        """
        建構 NetworkX 圖

        步驟:
        1. 創建空圖 G = nx.Graph()
        2. 添加節點 (所有種子)
        3. 添加邊 (只保留 cost < max_edge_cost)
        4. 返回圖
        """
        G = nx.Graph()
        
        # 1. 添加節點
        for seed in seeds:
            G.add_node(
                seed.id,
                x=seed.x,
                y=seed.y,
                component_id=seed.component_id,
                seed_type=seed.seed_type
            )
        
        # 2. 添加邊 (並根據成本過濾)
        for edge in edges_with_costs:
            total_cost = edge.get('total_cost', float('inf'))
            edge_is_reachable = total_cost != float('inf')
            if edge_is_reachable and (total_cost < self.max_edge_cost or self.max_edge_cost <= 0):
                G.add_edge(
                    edge['source_id'],
                    edge['target_id'],
                    weight=total_cost,
                    edge_type=edge.get('edge_type', 'unknown'),
                    geometric_cost=edge.get('geometric_cost'),
                    image_cost=edge.get('image_cost'),
                    curvature_cost=edge.get('curvature_cost'),
                    tortuosity=edge.get('tortuosity'),
                    path_cost=edge.get('path_cost'),
                    path=str(edge.get('path'))  # 序列化為字串供 GraphML 保存
                )
        
        return G
    
    def get_statistics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        獲取圖的統計資訊
        
        Args:
            G: NetworkX 圖物件
            
        Returns:
            一個包含統計數據的字典
        """
        if G.number_of_nodes() == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'num_components': 0,
                'avg_degree': 0
            }
            
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_components': nx.number_connected_components(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
        }