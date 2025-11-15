"""
種子載入器
載入種子資料、影像,並建立空間索引
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.neighbors import KDTree


@dataclass
class Seed:
    """種子點數據結構"""
    id: int
    x: int
    y: int
    component_id: int
    seed_type: str
    curvature_degrees: float = None
    path_id: int = None

    @property
    def position(self) -> Tuple[int, int]:
        """返回 (y, x) 座標用於索引"""
        return (self.y, self.x)

    @property
    def position_xy(self) -> Tuple[int, int]:
        """返回 (x, y) 座標用於顯示"""
        return (self.x, self.y)


class SeedLoader:
    """
    種子載入器

    功能:
    1. 從 seeds.json 載入種子資料
    2. 載入綠色通道影像
    3. 建立 KD-tree 空間索引
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.seeds: List[Seed] = []
        self.kdtree: KDTree = None

    def load_seeds(self, seeds_json_path: str) -> List[Seed]:
        """
        載入種子資料

        Args:
            seeds_json_path: seeds.json 檔案路徑

        Returns:
            seeds: Seed 物件列表
        """
        seeds_json_path = Path(seeds_json_path)

        if not seeds_json_path.exists():
            raise FileNotFoundError(f"種子檔案不存在: {seeds_json_path}")

        with open(seeds_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        seeds = []
        for seed_data in data['seeds']:
            seed = Seed(
                id=seed_data['seed_id'],
                x=seed_data['position']['x'],
                y=seed_data['position']['y'],
                component_id=seed_data['component_id'],
                seed_type=seed_data['type'],
                curvature_degrees=seed_data.get('curvature_degrees'),
                path_id=seed_data.get('path_id')
            )
            seeds.append(seed)

        self.seeds = seeds

        if self.verbose:
            print(f"✓ 載入 {len(seeds)} 個種子")
            print(f"  來源: {seeds_json_path}")

            # 統計種子類型
            seed_types = {}
            for seed in seeds:
                seed_types[seed.seed_type] = seed_types.get(seed.seed_type, 0) + 1
            print(f"  種子類型分布: {seed_types}")

        return seeds

    def load_green_channel(self, image_path: str) -> np.ndarray:
        """
        載入綠色通道影像

        Args:
            image_path: 影像檔案路徑 (可以是 RGB 或灰階)

        Returns:
            green_channel: 綠色通道影像 (uint8)
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"影像檔案不存在: {image_path}")

        # 載入影像
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        # 提取綠色通道
        if len(img.shape) == 3:
            # RGB 影像
            green_channel = img[:, :, 1]  # OpenCV 是 BGR 順序
        elif len(img.shape) == 2:
            # 已經是灰階影像
            green_channel = img
        else:
            raise ValueError(f"不支援的影像格式: shape={img.shape}")

        if self.verbose:
            print(f"✓ 載入綠色通道影像")
            print(f"  來源: {image_path}")
            print(f"  尺寸: {green_channel.shape}")
            print(f"  數值範圍: [{green_channel.min()}, {green_channel.max()}]")

        return green_channel

    def build_spatial_index(self, seeds: List[Seed] = None) -> KDTree:
        """
        建立 KD-tree 空間索引

        Args:
            seeds: 種子列表 (若為 None 則使用 self.seeds)

        Returns:
            kdtree: sklearn KDTree 物件
        """
        if seeds is None:
            seeds = self.seeds

        if len(seeds) == 0:
            raise ValueError("種子列表為空,無法建立空間索引")

        # 提取座標 (x, y)
        positions = np.array([[seed.x, seed.y] for seed in seeds])

        # 建立 KD-tree
        kdtree = KDTree(positions, leaf_size=30, metric='euclidean')

        self.kdtree = kdtree

        if self.verbose:
            print(f"✓ 建立 KD-tree 空間索引")
            print(f"  節點數: {len(seeds)}")
            print(f"  葉子大小: 30")

        return kdtree

    def query_neighbors(
        self,
        seed: Seed,
        k: int = None,
        radius: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        查詢種子的鄰居

        Args:
            seed: 查詢的種子
            k: 返回最近的 k 個鄰居 (與 radius 互斥)
            radius: 返回半徑內的所有鄰居 (與 k 互斥)

        Returns:
            distances: 距離陣列
            indices: 索引陣列
        """
        if self.kdtree is None:
            raise ValueError("尚未建立空間索引,請先調用 build_spatial_index()")

        if k is None and radius is None:
            raise ValueError("必須指定 k 或 radius 其中之一")

        if k is not None and radius is not None:
            raise ValueError("k 和 radius 不能同時指定")

        query_point = np.array([[seed.x, seed.y]])

        if k is not None:
            # k-近鄰查詢
            distances, indices = self.kdtree.query(query_point, k=k)
            return distances[0], indices[0]
        else:
            # 半徑查詢
            indices = self.kdtree.query_radius(query_point, r=radius)[0]
            distances = np.array([
                np.sqrt((seed.x - self.seeds[i].x)**2 +
                       (seed.y - self.seeds[i].y)**2)
                for i in indices
            ])
            return distances, indices

    def get_seed_by_index(self, index: int) -> Seed:
        """根據索引獲取種子"""
        return self.seeds[index]

    def get_statistics(self) -> Dict:
        """獲取種子統計資訊"""
        if len(self.seeds) == 0:
            return {}

        # 按元件統計
        components = {}
        for seed in self.seeds:
            if seed.component_id not in components:
                components[seed.component_id] = []
            components[seed.component_id].append(seed)

        # 按類型統計
        seed_types = {}
        for seed in self.seeds:
            seed_types[seed.seed_type] = seed_types.get(seed.seed_type, 0) + 1

        return {
            'total_seeds': len(self.seeds),
            'total_components': len(components),
            'seeds_per_component': {
                'mean': np.mean([len(s) for s in components.values()]),
                'min': min([len(s) for s in components.values()]),
                'max': max([len(s) for s in components.values()])
            },
            'seed_types': seed_types
        }


if __name__ == '__main__':
    # 測試程式碼
    loader = SeedLoader(verbose=True)

    # 載入種子
    seeds = loader.load_seeds('output/seeds/seeds.json')

    # 建立空間索引
    kdtree = loader.build_spatial_index()

    # 測試查詢
    test_seed = seeds[0]
    distances, indices = loader.query_neighbors(test_seed, k=10)
    print(f"\n測試種子 {test_seed.id} 的 10 個最近鄰:")
    print(f"  距離: {distances}")
    print(f"  索引: {indices}")

    # 統計資訊
    stats = loader.get_statistics()
    print(f"\n種子統計:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
