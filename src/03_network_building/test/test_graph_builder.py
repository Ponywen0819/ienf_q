'''
獨立測試腳本 for GraphBuilder
'''
import sys
from dataclasses import dataclass
import networkx as nx
import os

# 確保可以從當前目錄匯入模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_builder import GraphBuilder

# 為了獨立測試,我們在腳本內定義一個模擬的 Seed 類別
@dataclass
class MockSeed:
    id: int
    x: int
    y: int
    component_id: int
    seed_type: str

def create_mock_data():
    """建立測試用的種子和邊列表"""
    
    # 1. 建立 4 個模擬種子
    mock_seeds = [
        MockSeed(id=10, x=1, y=1, component_id=1, seed_type='endpoint'),
        MockSeed(id=20, x=10, y=10, component_id=1, seed_type='branchpoint'),
        MockSeed(id=30, x=20, y=20, component_id=2, seed_type='endpoint'),
        MockSeed(id=40, x=30, y=30, component_id=2, seed_type='centroid'),
    ]
    
    # 2. 建立 3 條邊,其中一條成本過高
    mock_edges = [
        # 這條邊應該被加入 (成本 50 < 150)
        {
            'source_id': 10, 'target_id': 20, 'edge_type': 'intra_component',
            'total_cost': 50.0, 'geometric_cost': 12.7, 'image_cost': 30.0,
            'curvature_cost': 7.3, 'tortuosity': 1.1
        },
        # 這條邊應該被過濾掉 (成本 200 > 150)
        {
            'source_id': 20, 'target_id': 30, 'edge_type': 'inter_component',
            'total_cost': 200.0, 'geometric_cost': 12.7, 'image_cost': 180.0,
            'curvature_cost': 7.3, 'tortuosity': 1.8
        },
        # 這條邊應該被加入 (成本 140 < 150)
        {
            'source_id': 30, 'target_id': 40, 'edge_type': 'intra_component',
            'total_cost': 140.0, 'geometric_cost': 12.7, 'image_cost': 120.0,
            'curvature_cost': 7.3, 'tortuosity': 1.2
        }
    ]
    
    return mock_seeds, mock_edges

def run_tests():
    """執行所有測試案例"""
    print("="*50)
    print("🚀 開始測試 GraphBuilder")
    print("="*50)

    # 1. 準備資料和物件
    seeds, edges = create_mock_data()
    # 使用預設的 max_edge_cost=150.0
    graph_builder = GraphBuilder(max_edge_cost=150.0)
    print("✓ 測試環境建立成功 (模擬資料、GraphBuilder)")
    print(f"  - 測試種子數: {len(seeds)}")
    print(f"  - 測試邊數: {len(edges)}")
    print(f"  - 邊成本過濾閾值: {graph_builder.max_edge_cost}")
    print("-"*50)

    # --- 執行測試案例 ---

    # 案例 1: 測試 build_graph
    print("▶️  案例 1: 測試 build_graph 方法...")
    G = graph_builder.build_graph(seeds, edges)
    
    # 驗證節點
    assert G.number_of_nodes() == 4, "節點數應為 4"
    print("  ✅ 節點數驗證通過 (4)")
    assert G.nodes[10]['seed_type'] == 'endpoint', "節點屬性應正確設定"
    print("  ✅ 節點屬性驗證通過")
    
    # 驗證邊
    assert G.number_of_edges() == 2, "邊數應為 2 (1條成本>150的邊應被過濾)"
    print("  ✅ 邊過濾功能驗證通過 (3 條邊過濾後剩 2 條)")
    
    assert G.has_edge(10, 20), "邊 (10, 20) 應存在"
    assert not G.has_edge(20, 30), "邊 (20, 30) 不應存在 (成本過高)"
    assert G.has_edge(30, 40), "邊 (30, 40) 應存在"
    print("  ✅ 邊存在性驗證通過")
    
    assert G.edges[10, 20]['weight'] == 50.0, "邊屬性 (權重) 應正確設定"
    print("  ✅ 邊屬性驗證通過")
    print("-"*50)

    # 案例 2: 測試 get_statistics
    print("▶️  案例 2: 測試 get_statistics 方法...")
    stats = graph_builder.get_statistics(G)
    
    print(f"  - 統計結果: {stats}")
    assert stats['num_nodes'] == 4, "統計: 節點數應為 4"
    assert stats['num_edges'] == 2, "統計: 邊數應為 2"
    # 節點10-20, 30-40, 所以是兩個獨立的元件
    assert stats['num_components'] == 2, "統計: 連通元件數應為 2"
    print("  ✅ 統計數據驗證通過")
    print("-"*50)

    print("🎉 所有測試案例執行完畢!")
    print("="*50)

if __name__ == "__main__":
    run_tests()
