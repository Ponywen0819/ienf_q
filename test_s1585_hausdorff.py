"""
直接測試 S1585-2_a 樣本的 Hausdorff 距離計算

繞過 Pipeline，直接從圖像創建圖並計算距離。
"""

import numpy as np
import networkx as nx
from PIL import Image
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from tools.evaluate_dataset import HausdorffCalculator, TopologyExtractor


def main():
    print("=" * 80)
    print("測試 S1585-2_a 樣本的平均 Hausdorff 距離計算")
    print("=" * 80)

    # 載入圖像
    data_dir = Path("/home/pony/projects/ienf_q/data/S1585-2_a")

    print("\n1. 載入圖像...")
    annotation = np.array(Image.open(data_dir / "annotation.png"))
    label = np.array(Image.open(data_dir / "label.png"))

    print(f"   - Annotation 形狀: {annotation.shape}")
    print(f"   - Label (GT) 形狀: {label.shape}")

    # 創建拓樸萃取器
    print("\n2. 創建拓樸萃取器...")
    extractor = TopologyExtractor()

    # 從 GT 萃取拓樸
    print("\n3. 從 GT 標註萃取拓樸...")
    graph_gt = extractor.extract_from_gt(label)

    if graph_gt is None:
        print("   ✗ GT 拓樸萃取失敗")
        return 1

    print(f"   ✓ GT 圖:")
    print(f"     - 節點數: {graph_gt.number_of_nodes()}")
    print(f"     - 邊數: {graph_gt.number_of_edges()}")

    # 從 annotation 萃取拓樸（作為"預測"）
    print("\n4. 從 Annotation 萃取拓樸...")
    graph_pred = extractor.extract_from_gt(annotation)

    if graph_pred is None:
        print("   ✗ Annotation 拓樸萃取失敗")
        return 1

    print(f"   ✓ Annotation 圖:")
    print(f"     - 節點數: {graph_pred.number_of_nodes()}")
    print(f"     - 邊數: {graph_pred.number_of_edges()}")

    # 計算 Hausdorff 距離
    print("\n5. 計算平均 Hausdorff 距離...")
    calculator = HausdorffCalculator()

    # 提取點集
    points_pred = calculator._extract_all_points(graph_pred)
    points_gt = calculator._extract_all_points(graph_gt)

    print(f"   ✓ 提取點集:")
    print(f"     - Annotation 總點數: {len(points_pred)}")
    print(f"     - GT 總點數: {len(points_gt)}")

    # 計算距離
    distance = calculator.compute(graph_pred, graph_gt)

    if distance is None:
        print("   ✗ 距離計算失敗")
        return 1

    print(f"\n6. 結果:")
    print(f"   ✓ 平均 Hausdorff 距離: {distance:.4f} 像素")

    # 統計邊的路徑屬性
    print(f"\n7. 邊路徑統計:")

    # 檢查 annotation 圖的邊
    edges_with_path_pred = sum(1 for _, _, data in graph_pred.edges(data=True)
                                if 'path' in data or 'path-coordinates' in data)
    print(f"   - Annotation 圖: {edges_with_path_pred}/{graph_pred.number_of_edges()} 條邊包含路徑")

    # 檢查 GT 圖的邊
    edges_with_path_gt = sum(1 for _, _, data in graph_gt.edges(data=True)
                            if 'path' in data or 'path-coordinates' in data)
    print(f"   - GT 圖: {edges_with_path_gt}/{graph_gt.number_of_edges()} 條邊包含路徑")

    # 顯示一些邊的樣本
    if graph_gt.number_of_edges() > 0:
        print(f"\n8. GT 圖邊屬性樣本（前 3 條）:")
        for i, (u, v, data) in enumerate(list(graph_gt.edges(data=True))[:3]):
            path = data.get('path')
            if path is None:
                path = data.get('path-coordinates')
            if path is not None:
                print(f"   - 邊 {i+1}: {len(path)} 個路徑點")
            else:
                print(f"   - 邊 {i+1}: 無路徑點")

    print("\n" + "=" * 80)
    print("測試完成！")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
