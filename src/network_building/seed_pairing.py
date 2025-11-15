"""
種子配對器
使用自適應半徑進行種子配對
"""

class SeedPairer:
    """
    種子配對策略

    功能:
    - 使用自適應半徑查找鄰近種子
    - 區分元件內/元件間配對
    - 避免重複配對和自我配對
    """

    def __init__(self, verbose: bool = False):
        """
        初始化配對器

        Args:
            verbose: 是否顯示詳細配對統計
        """
        self.verbose = verbose
    
    def pair_seeds(self, seeds, density_info, kdtree):
        """
        執行種子配對 (元件內/元件間)

        策略:
        1. 對每個種子,獲取其自適應半徑
        2. 使用 KDTree 查詢半徑內的所有鄰居
        3. 判斷元件內配對(component_id相同)或元件間配對
        4. 避免重複配對(只配對索引大於當前種子的鄰居)

        Args:
            seeds: 種子列表
            density_info: 密度資訊字典 {seed_id: {'local_density': float, 'pairing_radius': float}}
            kdtree: KDTree 空間索引物件

        Returns:
            pairs: [(seed_i, seed_j, edge_type), ...]
                   edge_type 為 "intra_component" 或 "inter_component"
        """
        pairs = []
        intra_count = 0  # 元件內配對計數
        inter_count = 0  # 元件間配對計數

        # 遍歷每個種子
        for i, seed in enumerate(seeds):
            # 獲取該種子的自適應半徑
            radius = density_info[seed.id]['pairing_radius']

            # 使用 KDTree 查詢半徑內的所有鄰居
            query_point = [[seed.x, seed.y]]
            indices = kdtree.query_radius(query_point, r=radius)[0]

            # 處理每個鄰居
            for idx in indices:
                # 關鍵: 避免重複配對和自我配對
                # idx <= i 確保:
                # 1. idx == i: 不與自己配對
                # 2. idx < i: 避免重複 (如果 (A,B) 已記錄,就不記錄 (B,A))
                if idx <= i:
                    continue

                neighbor = seeds[idx]

                # 判斷元件內/元件間
                if seed.component_id == neighbor.component_id:
                    edge_type = "intra_component"
                    intra_count += 1
                else:
                    edge_type = "inter_component"
                    inter_count += 1

                # 添加配對
                pairs.append((seed, neighbor, edge_type))

        # 詳細輸出模式
        if self.verbose:
            print(f"  配對統計:")
            print(f"    元件內配對: {intra_count}")
            print(f"    元件間配對: {inter_count}")
            print(f"    總配對數: {len(pairs)}")
            if len(pairs) > 0:
                intra_ratio = intra_count / len(pairs) * 100
                print(f"    元件內比例: {intra_ratio:.1f}%")

        return pairs
