"""
測試 SeedPairer 的實作
"""

import sys
from pathlib import Path

# 添加當前目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from seed_loader import SeedLoader
from density_estimator import DensityEstimator
from seed_pairing import SeedPairer


def test_seed_pairer():
    """測試種子配對功能"""

    print("=" * 60)
    print("測試 SeedPairer 實作")
    print("=" * 60)

    # 1. 載入種子和影像
    print("\n[1/4] 載入測試資料...")
    loader = SeedLoader(verbose=True)
    seeds = loader.load_seeds('output/seeds/seeds.json')
    kdtree = loader.build_spatial_index()
    print(f"✓ 載入完成: {len(seeds)} 個種子")

    # 2. 計算密度
    print("\n[2/4] 計算局部密度...")
    estimator = DensityEstimator(k=10)
    density_info = {}

    for seed in seeds:
        density = estimator.calculate_local_density(seed, kdtree, k=10)
        radius = estimator.determine_adaptive_radius(density)
        density_info[seed.id] = {
            'local_density': density,
            'pairing_radius': radius
        }

    print(f"✓ 密度計算完成")

    # 統計半徑分布
    radius_dist = {}
    for info in density_info.values():
        r = info['pairing_radius']
        radius_dist[r] = radius_dist.get(r, 0) + 1

    print(f"  半徑分布: {radius_dist}")

    # 3. 執行配對
    print("\n[3/4] 執行種子配對...")
    pairer = SeedPairer(verbose=True)
    pairs = pairer.pair_seeds(seeds, density_info, kdtree)

    # 4. 驗證結果
    print("\n[4/4] 驗證配對結果...")

    # 驗證 1: 配對數量合理性
    print(f"\n驗證 1: 配對數量")
    print(f"  總配對數: {len(pairs)}")
    assert len(pairs) > 0, "配對數應該大於0"
    print(f"  ✓ 配對數量正常")

    # 驗證 2: 無自我配對
    print(f"\n驗證 2: 無自我配對")
    self_pairs = [p for p in pairs if p[0].id == p[1].id]
    assert len(self_pairs) == 0, f"發現 {len(self_pairs)} 個自我配對"
    print(f"  ✓ 無自我配對")

    # 驗證 3: 無重複配對
    print(f"\n驗證 3: 無重複配對")
    pair_set = set()
    duplicates = []
    for s1, s2, _ in pairs:
        key = tuple(sorted([s1.id, s2.id]))
        if key in pair_set:
            duplicates.append(key)
        pair_set.add(key)

    assert len(duplicates) == 0, f"發現 {len(duplicates)} 個重複配對"
    print(f"  ✓ 無重複配對")

    # 驗證 4: edge_type 正確性
    print(f"\n驗證 4: edge_type 正確性")
    wrong_types = []
    for s1, s2, edge_type in pairs:
        expected = "intra_component" if s1.component_id == s2.component_id else "inter_component"
        if edge_type != expected:
            wrong_types.append((s1.id, s2.id, edge_type, expected))

    assert len(wrong_types) == 0, f"發現 {len(wrong_types)} 個錯誤的 edge_type"
    print(f"  ✓ edge_type 全部正確")

    # 統計元件內/元件間比例
    intra = sum(1 for _, _, t in pairs if t == "intra_component")
    inter = sum(1 for _, _, t in pairs if t == "inter_component")

    print(f"\n配對分類統計:")
    print(f"  元件內配對: {intra} ({intra/len(pairs)*100:.1f}%)")
    print(f"  元件間配對: {inter} ({inter/len(pairs)*100:.1f}%)")

    # 驗證 5: 配對距離檢查 (抽樣)
    print(f"\n驗證 5: 配對距離檢查 (抽樣10對)")
    import numpy as np
    sample_pairs = pairs[:10] if len(pairs) >= 10 else pairs

    for s1, s2, edge_type in sample_pairs:
        distance = np.sqrt((s1.x - s2.x)**2 + (s1.y - s2.y)**2)
        radius = density_info[s1.id]['pairing_radius']
        print(f"  種子 {s1.id} <-> {s2.id}: 距離={distance:.1f}px, 半徑={radius}px, 類型={edge_type}")
        assert distance <= radius, f"距離 {distance} 超過半徑 {radius}"

    print(f"  ✓ 距離檢查通過")

    print("\n" + "=" * 60)
    print("✓ 所有測試通過!")
    print("=" * 60)

    return pairs


if __name__ == '__main__':
    try:
        pairs = test_seed_pairer()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
