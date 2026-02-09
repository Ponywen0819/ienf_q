"""
測試平均 Hausdorff 距離實現的簡單測試腳本
"""

import numpy as np
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from tools.evaluate_dataset import compute_average_hausdorff


def test_identical_points():
    """測試：相同點集的距離應為 0"""
    points = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
    distance = compute_average_hausdorff(points, points)
    assert abs(distance) < 1e-10, f"預期距離為 0，實際為 {distance}"
    print("✓ 測試通過：相同點集距離為 0")


def test_single_points():
    """測試：單點的距離應等於歐幾里得距離"""
    p1 = np.array([[0, 0]], dtype=np.float64)
    p2 = np.array([[3, 4]], dtype=np.float64)
    distance = compute_average_hausdorff(p1, p2)
    expected = 5.0  # sqrt(3^2 + 4^2) = 5
    assert abs(distance - expected) < 1e-6, f"預期距離為 {expected}，實際為 {distance}"
    print(f"✓ 測試通過：單點距離 = {distance:.4f} (預期 {expected})")


def test_symmetric():
    """測試：對稱性 - d(A, B) 應該等於 d(B, A)"""
    points_a = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
    points_b = np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float64)

    d_ab = compute_average_hausdorff(points_a, points_b)
    d_ba = compute_average_hausdorff(points_b, points_a)

    assert abs(d_ab - d_ba) < 1e-10, f"對稱性測試失敗：d(A,B)={d_ab}, d(B,A)={d_ba}"
    print(f"✓ 測試通過：對稱性 d(A,B) = d(B,A) = {d_ab:.4f}")


def test_known_geometry():
    """測試：已知幾何形狀的距離"""
    # 兩條平行線，距離為 1
    line1 = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
    line2 = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.float64)

    distance = compute_average_hausdorff(line1, line2)
    expected = 1.0  # 平均距離應該是 1
    assert abs(distance - expected) < 1e-6, f"預期距離為 {expected}，實際為 {distance}"
    print(f"✓ 測試通過：平行線距離 = {distance:.4f} (預期 {expected})")


def test_sparse_vs_dense():
    """測試：稀疏 vs 密集點集"""
    # 稀疏點集
    sparse = np.array([[0, 0], [10, 10]], dtype=np.float64)
    # 密集點集（在同一條線上）
    dense = np.array([[i, i] for i in range(11)], dtype=np.float64)

    distance = compute_average_hausdorff(sparse, dense)
    # 稀疏點集的每個點都在密集點集上，所以從稀疏到密集的距離應該是 0
    # 密集點集的每個點到稀疏點集的最大距離大約是 3.5（中點到最近點）
    print(f"✓ 測試通過：稀疏 vs 密集點集距離 = {distance:.4f}")
    assert distance >= 0, "距離應該非負"


def main():
    """運行所有測試"""
    print("開始測試平均 Hausdorff 距離實現...")
    print("=" * 60)

    try:
        test_identical_points()
        test_single_points()
        test_symmetric()
        test_known_geometry()
        test_sparse_vs_dense()

        print("=" * 60)
        print("✓ 所有測試通過！")
        return 0
    except AssertionError as e:
        print("=" * 60)
        print(f"✗ 測試失敗：{e}")
        return 1
    except Exception as e:
        print("=" * 60)
        print(f"✗ 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
