#!/usr/bin/env python3
"""
使用提取的數據集拓樸示例

展示如何載入和使用 extract_dataset_topologies.py 生成的 GT 拓樸文件。
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.compare_topologies import TopologyLoader, TopologyComparator


def example_1_load_single_topology():
    """示例 1: 載入單個拓樸文件"""
    print("=" * 80)
    print("示例 1: 載入單個拓樸文件")
    print("=" * 80)

    loader = TopologyLoader()
    topology_path = Path("output/topologies/S1585-2_a_gt.pkl")

    if not topology_path.exists():
        print(f"錯誤: 文件不存在 {topology_path}")
        print("請先運行: uv run python tools/extract_dataset_topologies.py")
        return

    # 載入拓樸
    graph = loader.load(topology_path)

    if graph is None:
        print("載入失敗")
        return

    print(f"✓ 成功載入: {topology_path.name}")
    print(f"  節點數: {graph.number_of_nodes()}")
    print(f"  邊數: {graph.number_of_edges()}")

    # 檢查邊的路徑屬性
    edges_with_path = 0
    total_path_points = 0

    for u, v, data in graph.edges(data=True):
        path = data.get("path-coordinates") or data.get("path")
        if path is not None:
            edges_with_path += 1
            total_path_points += len(path)

    print(f"  包含路徑的邊: {edges_with_path}/{graph.number_of_edges()}")
    print(f"  總路徑點數: {total_path_points}")
    print()


def example_2_compare_two_topologies():
    """示例 2: 比對兩個拓樸"""
    print("=" * 80)
    print("示例 2: 比對兩個拓樸")
    print("=" * 80)

    loader = TopologyLoader()
    comparator = TopologyComparator()

    # 載入兩個拓樸
    topo1_path = Path("output/topologies/S1585-2_a_gt.pkl")
    topo2_path = Path("output/topologies/S1585-2_b_gt.pkl")

    if not (topo1_path.exists() and topo2_path.exists()):
        print("錯誤: 拓樸文件不存在")
        print("請先運行: uv run python tools/extract_dataset_topologies.py")
        return

    graph1 = loader.load(topo1_path)
    graph2 = loader.load(topo2_path)

    if graph1 is None or graph2 is None:
        print("載入失敗")
        return

    # 比對
    result = comparator.compare(
        graph1, graph2, topo1_path.stem, topo2_path.stem
    )

    if result["status"] == "success":
        print(f"✓ 比對成功")
        print(f"  拓樸 1: {result['label1']}")
        print(f"    - 節點: {result['num_nodes1']}, 邊: {result['num_edges1']}, 總點: {result['num_points1']}")
        print(f"  拓樸 2: {result['label2']}")
        print(f"    - 節點: {result['num_nodes2']}, 邊: {result['num_edges2']}, 總點: {result['num_points2']}")
        print(f"  平均 Hausdorff 距離: {result['hausdorff_distance']:.4f} 像素")
    else:
        print(f"✗ 比對失敗: {result.get('error', 'Unknown error')}")

    print()


def example_3_list_all_topologies():
    """示例 3: 列出所有可用的拓樸"""
    print("=" * 80)
    print("示例 3: 列出所有可用的拓樸")
    print("=" * 80)

    topology_dir = Path("output/topologies")

    if not topology_dir.exists():
        print(f"錯誤: 目錄不存在 {topology_dir}")
        print("請先運行: uv run python tools/extract_dataset_topologies.py")
        return

    pkl_files = sorted(topology_dir.glob("*_gt.pkl"))

    if not pkl_files:
        print(f"在 {topology_dir} 中未找到拓樸文件")
        return

    print(f"找到 {len(pkl_files)} 個 GT 拓樸文件:\n")

    loader = TopologyLoader()

    for i, pkl_path in enumerate(pkl_files, 1):
        graph = loader.load(pkl_path)
        if graph is not None:
            sample_id = pkl_path.stem.replace("_gt", "")
            print(
                f"{i:2d}. {sample_id:15s} - "
                f"{graph.number_of_nodes():4d} 節點, "
                f"{graph.number_of_edges():4d} 邊"
            )

    print()


def example_4_batch_statistics():
    """示例 4: 批量統計所有拓樸"""
    print("=" * 80)
    print("示例 4: 批量統計所有拓樸")
    print("=" * 80)

    topology_dir = Path("output/topologies")

    if not topology_dir.exists():
        print(f"錯誤: 目錄不存在 {topology_dir}")
        return

    pkl_files = sorted(topology_dir.glob("*_gt.pkl"))

    if not pkl_files:
        print("未找到拓樸文件")
        return

    loader = TopologyLoader()

    total_nodes = 0
    total_edges = 0
    total_path_points = 0

    for pkl_path in pkl_files:
        graph = loader.load(pkl_path)
        if graph is not None:
            total_nodes += graph.number_of_nodes()
            total_edges += graph.number_of_edges()

            # 統計路徑點
            for u, v, data in graph.edges(data=True):
                path = data.get("path-coordinates") or data.get("path")
                if path is not None:
                    total_path_points += len(path)

    count = len(pkl_files)

    print(f"統計 {count} 個拓樸文件:\n")
    print(f"總節點數: {total_nodes:,}")
    print(f"總邊數: {total_edges:,}")
    print(f"總路徑點數: {total_path_points:,}")
    print(f"\n平均每個樣本:")
    print(f"  節點: {total_nodes / count:.1f}")
    print(f"  邊: {total_edges / count:.1f}")
    print(f"  路徑點: {total_path_points / count:.1f}")
    print()


def main():
    """運行所有示例"""
    print("\n" + "=" * 80)
    print("使用提取的數據集拓樸示例")
    print("=" * 80)
    print()

    # 運行所有示例
    example_1_load_single_topology()
    example_2_compare_two_topologies()
    example_3_list_all_topologies()
    example_4_batch_statistics()

    print("=" * 80)
    print("所有示例完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
