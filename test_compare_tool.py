"""
測試拓樸比對工具

創建測試拓樸並使用新的比對工具進行測試。
"""

import sys
from pathlib import Path
import networkx as nx

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from tools.compare_topologies import TopologyLoader, TopologyComparator


def create_test_topologies():
    """創建測試拓樸並儲存"""
    output_dir = Path("output/test_topologies")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 創建拓樸 1：簡單的線段
    graph1 = nx.Graph()
    graph1.add_node((0, 0))
    graph1.add_node((10, 0))
    graph1.add_edge((0, 0), (10, 0), path=[(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])

    # 創建拓樸 2：類似但稍有偏移的線段
    graph2 = nx.Graph()
    graph2.add_node((0, 1))  # y 偏移 1
    graph2.add_node((10, 1))
    graph2.add_edge((0, 1), (10, 1), **{'path-coordinates': [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]})

    # 儲存為不同格式
    loader = TopologyLoader()

    print("=" * 80)
    print("創建並儲存測試拓樸...")
    print("=" * 80)

    # Pickle 格式
    pkl_path1 = output_dir / "topology1.pkl"
    pkl_path2 = output_dir / "topology2.pkl"
    loader.save(graph1, pkl_path1, 'pickle')
    loader.save(graph2, pkl_path2, 'pickle')
    print(f"✓ 已儲存 Pickle: {pkl_path1}, {pkl_path2}")

    # JSON 格式
    json_path1 = output_dir / "topology1.json"
    json_path2 = output_dir / "topology2.json"
    loader.save(graph1, json_path1, 'json')
    loader.save(graph2, json_path2, 'json')
    print(f"✓ 已儲存 JSON: {json_path1}, {json_path2}")

    # GraphML 格式
    graphml_path1 = output_dir / "topology1.graphml"
    graphml_path2 = output_dir / "topology2.graphml"
    loader.save(graph1, graphml_path1, 'graphml')
    loader.save(graph2, graphml_path2, 'graphml')
    print(f"✓ 已儲存 GraphML: {graphml_path1}, {graphml_path2}")

    return output_dir


def test_loading_and_comparison():
    """測試載入和比對功能"""
    print("\n" + "=" * 80)
    print("測試拓樸載入和比對...")
    print("=" * 80)

    output_dir = Path("output/test_topologies")

    loader = TopologyLoader()
    comparator = TopologyComparator()

    # 測試 Pickle 格式
    print("\n1. 測試 Pickle 格式")
    graph1 = loader.load(output_dir / "topology1.pkl")
    graph2 = loader.load(output_dir / "topology2.pkl")

    if graph1 and graph2:
        result = comparator.compare(graph1, graph2, "Topology1", "Topology2")
        print(f"   ✓ Pickle 載入成功")
        print(f"   - 拓樸 1: {result['num_nodes1']} 節點, {result['num_edges1']} 邊, {result['num_points1']} 點")
        print(f"   - 拓樸 2: {result['num_nodes2']} 節點, {result['num_edges2']} 邊, {result['num_points2']} 點")
        print(f"   - Hausdorff 距離: {result['hausdorff_distance']:.4f}")
    else:
        print("   ✗ Pickle 載入失敗")

    # 測試 JSON 格式
    print("\n2. 測試 JSON 格式")
    graph1 = loader.load(output_dir / "topology1.json")
    graph2 = loader.load(output_dir / "topology2.json")

    if graph1 and graph2:
        result = comparator.compare(graph1, graph2, "Topology1", "Topology2")
        print(f"   ✓ JSON 載入成功")
        print(f"   - Hausdorff 距離: {result['hausdorff_distance']:.4f}")
    else:
        print("   ✗ JSON 載入失敗")

    # 測試 GraphML 格式
    print("\n3. 測試 GraphML 格式")
    graph1 = loader.load(output_dir / "topology1.graphml")
    graph2 = loader.load(output_dir / "topology2.graphml")

    if graph1 and graph2:
        result = comparator.compare(graph1, graph2, "Topology1", "Topology2")
        if result['status'] == 'success':
            print(f"   ✓ GraphML 載入成功")
            print(f"   - Hausdorff 距離: {result['hausdorff_distance']:.4f}")
        else:
            print(f"   ✗ GraphML 比對失敗: {result['error']}")
    else:
        print("   ✗ GraphML 載入失敗")

    print("\n" + "=" * 80)
    print("測試完成！")
    print("=" * 80)


def main():
    # 創建測試拓樸
    output_dir = create_test_topologies()

    # 測試載入和比對
    test_loading_and_comparison()

    print(f"\n測試檔案位於: {output_dir}")
    print("\n現在可以使用命令列工具測試:")
    print(f"  python tools/compare_topologies.py \\")
    print(f"    --topology1 {output_dir}/topology1.pkl \\")
    print(f"    --topology2 {output_dir}/topology2.pkl")

    return 0


if __name__ == "__main__":
    sys.exit(main())
