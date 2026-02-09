"""
測試 HausdorffCalculator 類的集成測試腳本

驗證從圖中提取節點和邊路徑點，並計算平均 Hausdorff 距離的完整流程。
"""

import numpy as np
import networkx as nx
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from tools.evaluate_dataset import HausdorffCalculator


def create_test_graph_with_paths():
    """創建一個包含節點和邊路徑的測試圖"""
    graph = nx.Graph()

    # 添加節點
    graph.add_node((0, 0))
    graph.add_node((10, 0))
    graph.add_node((10, 10))

    # 添加帶有路徑的邊
    # 邊 1: (0,0) -> (10,0)，路徑包含中間點
    path1 = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
    graph.add_edge((0, 0), (10, 0), path=path1)

    # 邊 2: (10,0) -> (10,10)，路徑包含中間點
    path2 = [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)]
    graph.add_edge((10, 0), (10, 10), path=path2)

    return graph


def create_test_graph_with_path_coordinates():
    """創建一個使用 'path-coordinates' 屬性的測試圖（模擬 GT 圖）"""
    graph = nx.Graph()

    # 添加節點
    graph.add_node((0, 0))
    graph.add_node((5, 5))

    # 添加帶有路徑的邊（使用 'path-coordinates' 屬性）
    path = [(1, 1), (2, 2), (3, 3), (4, 4)]
    graph.add_edge((0, 0), (5, 5), **{'path-coordinates': path})

    return graph


def test_extract_points_from_graph_with_path():
    """測試從圖中提取點集（使用 'path' 屬性）"""
    print("\n測試 1: 提取點集（'path' 屬性）")
    print("-" * 60)

    calculator = HausdorffCalculator()
    graph = create_test_graph_with_paths()

    points = calculator._extract_all_points(graph)

    # 預期點數：3 個節點 + 9 + 9 = 21 個點
    # 但去重後可能會少一些（如果節點與路徑端點重複）
    print(f"提取的點數: {len(points)}")
    print(f"節點數: {graph.number_of_nodes()}")
    print(f"邊數: {graph.number_of_edges()}")

    assert len(points) >= graph.number_of_nodes(), "點數應至少等於節點數"
    assert points.shape[1] == 2, "每個點應該是 2D 座標"
    print("✓ 測試通過：成功提取點集")


def test_extract_points_from_graph_with_path_coordinates():
    """測試從圖中提取點集（使用 'path-coordinates' 屬性）"""
    print("\n測試 2: 提取點集（'path-coordinates' 屬性）")
    print("-" * 60)

    calculator = HausdorffCalculator()
    graph = create_test_graph_with_path_coordinates()

    points = calculator._extract_all_points(graph)

    # 預期點數：2 個節點 + 4 個路徑點 = 6 個點
    print(f"提取的點數: {len(points)}")
    print(f"節點數: {graph.number_of_nodes()}")

    assert len(points) >= graph.number_of_nodes(), "點數應至少等於節點數"
    assert points.shape[1] == 2, "每個點應該是 2D 座標"
    print("✓ 測試通過：成功提取點集（path-coordinates）")


def test_extract_points_from_graph_without_paths():
    """測試從只有節點的圖中提取點集"""
    print("\n測試 3: 提取點集（無邊路徑）")
    print("-" * 60)

    calculator = HausdorffCalculator()
    graph = nx.Graph()

    # 只添加節點，不添加邊
    graph.add_node((0, 0))
    graph.add_node((5, 5))
    graph.add_node((10, 10))

    points = calculator._extract_all_points(graph)

    print(f"提取的點數: {len(points)}")
    print(f"節點數: {graph.number_of_nodes()}")

    assert len(points) == graph.number_of_nodes(), "點數應等於節點數"
    assert points.shape[1] == 2, "每個點應該是 2D 座標"
    print("✓ 測試通過：成功提取點集（無邊路徑）")


def test_compute_distance_between_graphs():
    """測試計算兩個圖之間的平均 Hausdorff 距離"""
    print("\n測試 4: 計算兩個圖之間的距離")
    print("-" * 60)

    calculator = HausdorffCalculator()

    # 創建兩個相似的圖
    graph1 = create_test_graph_with_paths()
    graph2 = create_test_graph_with_path_coordinates()

    distance = calculator.compute(graph1, graph2)

    print(f"計算得到的平均 Hausdorff 距離: {distance:.4f}")

    assert distance is not None, "距離不應為 None"
    assert distance >= 0, "距離應為非負數"
    print("✓ 測試通過：成功計算圖之間的距離")


def test_compute_distance_identical_graphs():
    """測試計算相同圖之間的距離（應為 0）"""
    print("\n測試 5: 計算相同圖之間的距離")
    print("-" * 60)

    calculator = HausdorffCalculator()
    graph = create_test_graph_with_paths()

    distance = calculator.compute(graph, graph)

    print(f"計算得到的平均 Hausdorff 距離: {distance:.4f}")

    assert distance is not None, "距離不應為 None"
    assert abs(distance) < 1e-10, f"相同圖的距離應為 0，實際為 {distance}"
    print("✓ 測試通過：相同圖的距離為 0")


def test_compute_distance_with_none_graph():
    """測試當圖為 None 時的處理"""
    print("\n測試 6: 處理 None 圖")
    print("-" * 60)

    calculator = HausdorffCalculator()
    graph = create_test_graph_with_paths()

    distance1 = calculator.compute(None, graph)
    distance2 = calculator.compute(graph, None)

    print(f"None vs 圖的距離: {distance1}")
    print(f"圖 vs None 的距離: {distance2}")

    assert distance1 is None, "None 圖應返回 None"
    assert distance2 is None, "None 圖應返回 None"
    print("✓ 測試通過：正確處理 None 圖")


def test_compute_distance_with_empty_graph():
    """測試當圖為空時的處理"""
    print("\n測試 7: 處理空圖")
    print("-" * 60)

    calculator = HausdorffCalculator()
    empty_graph = nx.Graph()
    graph = create_test_graph_with_paths()

    distance = calculator.compute(empty_graph, graph)

    print(f"空圖 vs 圖的距離: {distance}")

    assert distance is None, "空圖應返回 None"
    print("✓ 測試通過：正確處理空圖")


def main():
    """運行所有集成測試"""
    print("=" * 60)
    print("開始測試 HausdorffCalculator 類...")
    print("=" * 60)

    try:
        test_extract_points_from_graph_with_path()
        test_extract_points_from_graph_with_path_coordinates()
        test_extract_points_from_graph_without_paths()
        test_compute_distance_between_graphs()
        test_compute_distance_identical_graphs()
        test_compute_distance_with_none_graph()
        test_compute_distance_with_empty_graph()

        print("\n" + "=" * 60)
        print("✓ 所有集成測試通過！")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ 測試失敗：{e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 錯誤：{e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
