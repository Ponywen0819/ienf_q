#!/usr/bin/env python3
"""
元件配對與連接判斷模組 (Component Pairing and Connection Decision)

此模組提供兩個連通元件之間的配對分析與連接判斷功能。

主要流程：
1. 計算兩個元件之間的最近點距離
2. 距離預篩選：距離太遠直接跳過，避免不必要的計算
3. 找到距離最近的種子對
4. 使用 A* 計算連接路徑和成本
5. 根據成本閾值判斷兩元件是否應該連接

使用範例:
    from src.nueral_reconstruction.component_pairing import ComponentPairAnalyzer

    analyzer = ComponentPairAnalyzer(
        green_channel=green_channel_image,
        max_distance_threshold=100.0,
        max_cost_threshold=150.0
    )

    # 分析兩個元件的連接可能性
    result = analyzer.analyze_component_pair(
        component_a_seeds,
        component_b_seeds
    )

    if result['should_connect']:
        print(f"元件應連接，成本: {result['cost']}, 路徑長度: {len(result['path'])}")

作者: Generated with Claude Code
日期: 2025-11-17
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial.distance import cdist

from .pathfinding import AStarPathfinder

# 設定 logger
logger = logging.getLogger(__name__)


class ComponentPairAnalyzer:
    """
    元件配對分析器

    分析兩個連通元件之間的連接可能性。
    使用 A* 路徑搜尋計算種子點之間的連接路徑和成本。
    """

    def __init__(
        self,
        green_channel: np.ndarray,
        max_distance_threshold: float = 100.0,
        max_cost_threshold: float = 150.0
    ):
        """
        初始化元件配對分析器

        Args:
            green_channel: 綠色通道影像 (uint8, 0-255)
            max_distance_threshold: 最大距離閾值（像素），超過此距離不進行配對
            max_cost_threshold: 最大成本閾值，超過此值則判定為不連接
        """
        self.pathfinder = AStarPathfinder(green_channel)
        self.max_distance_threshold = max_distance_threshold
        self.max_cost_threshold = max_cost_threshold

        logger.info("=" * 70)
        logger.info("元件配對分析器初始化完成")
        logger.info("=" * 70)
        logger.info(f"最大距離閾值: {max_distance_threshold} 像素")
        logger.info(f"最大成本閾值: {max_cost_threshold}")
        logger.info(f"距離計算方式: nearest (最近點)")
        logger.info("=" * 70)

    def calculate_component_distance(
        self,
        seeds_a: List[Dict],
        seeds_b: List[Dict]
    ) -> Dict[str, Any]:
        """
        計算兩個元件之間的最近點距離

        Args:
            seeds_a: 元件 A 的種子列表，每個種子為 {'position': (y, x), ...}
            seeds_b: 元件 B 的種子列表

        Returns:
            距離資訊字典：
                {
                    'distance': float,  # 最近點距離
                    'nearest_pair': ((y1, x1), (y2, x2))  # 最近點對
                }
        """
        # 提取種子座標
        coords_a = np.array([seed['position'] for seed in seeds_a])
        coords_b = np.array([seed['position'] for seed in seeds_b])

        # 計算所有點對之間的距離矩陣
        dist_matrix = cdist(coords_a, coords_b, metric='euclidean')

        # 找到最小距離及其索引
        min_idx = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
        min_distance = dist_matrix[min_idx]

        nearest_a = tuple(coords_a[min_idx[0]])
        nearest_b = tuple(coords_b[min_idx[1]])

        return {
            'distance': min_distance,
            'nearest_pair': (nearest_a, nearest_b)
        }

    def analyze_component_pair(
        self,
        seeds_a: List[Dict],
        seeds_b: List[Dict],
        component_id_a: Optional[int] = None,
        component_id_b: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        分析兩個元件之間的配對關係並判斷是否應該連接

        完整流程：
        1. 計算元件最近點距離
        2. 距離預篩選（太遠直接跳過）
        3. 找到距離最近的種子對
        4. 呼叫成本計算器計算該種子對的成本
        5. 根據閾值判斷是否連接

        Args:
            seeds_a: 元件 A 的種子列表
            seeds_b: 元件 B 的種子列表
            component_id_a: 元件 A 的 ID（可選，用於日誌）
            component_id_b: 元件 B 的 ID（可選，用於日誌）

        Returns:
            分析結果字典：
                {
                'should_connect': bool,  # 是否應該連接
                'cost': float,           # 連接成本
                'distance': float,       # 元件間距離
                'seed_pair': (seed_a, seed_b) or None,  # 種子對
                'path': List[(y, x)] or None,  # 連接路徑
                'skipped_reason': str or None  # 如果跳過，原因說明
            }
        """
        comp_label_a = f"元件 {component_id_a}" if component_id_a is not None else "元件 A"
        comp_label_b = f"元件 {component_id_b}" if component_id_b is not None else "元件 B"

        logger.info(f"\n分析 {comp_label_a} 與 {comp_label_b} 的配對關係...")

        results = {
                'should_connect': False,
                'cost': float('inf'),
                'distance': float('inf'),
                'seed_pair': None,
                'path': None,
                'skipped_reason': 'no_seeds'
            }

        # ========== 步驟 0: 檢查種子列表是否為空 ==========
        if not seeds_a or not seeds_b:
            logger.info(f"  ✗ 其中一個元件沒有種子點，跳過配對")
            return results

        # ========== 步驟 1: 計算元件距離 ==========
        distance_info = self.calculate_component_distance(seeds_a, seeds_b)
        distance = distance_info['distance']
        results['distance'] = distance

        logger.info(f"  最近點距離: {distance:.2f} 像素")

        # ========== 步驟 2: 距離預篩選 ==========
        if distance > self.max_distance_threshold:
            logger.info(f"  ✗ 距離超過閾值 {self.max_distance_threshold}，跳過配對")

            results['skipped_reason'] = 'distance_too_far'
            return results

        # ========== 步驟 3: 找到距離最近的種子對 ==========
        # 提取種子座標
        coords_a = np.array([seed['position'] for seed in seeds_a])
        coords_b = np.array([seed['position'] for seed in seeds_b])

        # 計算所有點對之間的距離矩陣
        dist_matrix = cdist(coords_a, coords_b, metric='euclidean')

        # 找到最小距離及其索引
        min_idx = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
        
        nearest_seed_a = seeds_a[min_idx[0]]
        nearest_seed_b = seeds_b[min_idx[1]]
        nearest_distance = dist_matrix[min_idx]

        logger.info(f"  找到最近種子對:")
        logger.info(f"    元件 A 種子索引: {min_idx[0]} / {len(seeds_a)}")
        logger.info(f"    元件 B 種子索引: {min_idx[1]} / {len(seeds_b)}")
        logger.info(f"    種子間距離: {nearest_distance:.2f} 像素")

        # ========== 步驟 4: 使用 A* 計算路徑和成本 ==========
        start = nearest_seed_a['position']
        end = nearest_seed_b['position']
        results['seed_pair'] = (nearest_seed_a, nearest_seed_b)
        

        path = self.pathfinder.find_path(start, end)
        results['path'] = path
        
        if path is None:
            logger.info(f"  ✗ 無法找到有效的連接路徑")
            results['skipped_reason'] = 'no_valid_path'
            return results

        # 計算路徑成本
        cost = self.pathfinder.calculate_path_cost(path)
        results['cost'] = cost


        max_cost = self.pathfinder._cost_function(255) * distance
        # ========== 步驟 5: 根據成本閾值判斷是否連接 ==========
        should_connect = cost <= (max_cost * 0.98)

        

        logger.info(f"\n  配對分析結果:")
        logger.info(f"    路徑長度: {len(path)} 像素")
        logger.info(f"    連接成本: {cost:.2f}")
        logger.info(f"    成本閾值: {self.max_cost_threshold}")

        if should_connect:
            logger.info(f"  ✓ 判定: 應該連接")
        else:
            logger.info(f"  ✗ 判定: 不應連接（成本超過閾值）")

        results['skipped_reason'] = None if should_connect else 'cost_exceeds_threshold'
        results['should_connect'] = should_connect
        return results

    def batch_analyze_components(
        self,
        components_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        批次分析多個元件之間的配對關係

        Args:
            components_data: 元件資料列表，每個元件包含：
                {
                    'component_id': int,
                    'seeds': List[Dict]  # 種子列表
                }

        Returns:
            批次分析結果：
                {
                    'num_components': int,
                    'num_pairs_analyzed': int,
                    'num_connections': int,
                    'connections': List[Dict],  # 所有應該連接的元件對
                    'all_pair_results': List[Dict]  # 所有配對分析結果（包含不連接的）
                }
        """
        num_components = len(components_data)

        logger.info("\n" + "=" * 70)
        logger.info("批次元件配對分析")
        logger.info("=" * 70)
        logger.info(f"元件總數: {num_components}")
        logger.info(f"配對總數: {num_components * (num_components - 1) // 2}")

        connections = []
        all_pair_results = []
        num_pairs_analyzed = 0

        # 對所有元件對進行配對分析
        for i in range(num_components):
            for j in range(i + 1, num_components):
                comp_a = components_data[i]
                comp_b = components_data[j]

                result = self.analyze_component_pair(
                    seeds_a=comp_a['seeds'],
                    seeds_b=comp_b['seeds'],
                    component_id_a=comp_a['component_id'],
                    component_id_b=comp_b['component_id']
                )

                num_pairs_analyzed += 1

                # 記錄所有配對結果
                pair_result = {
                    'component_a_id': comp_a['component_id'],
                    'component_b_id': comp_b['component_id'],
                    **result
                }
                all_pair_results.append(pair_result)

                # 如果應該連接，加入連接列表
                if result['should_connect']:
                    connections.append({
                        'component_a_id': comp_a['component_id'],
                        'component_b_id': comp_b['component_id'],
                        'cost': result['cost'],
                        'distance': result['distance'],
                        'seed_pair': result['seed_pair'],
                        'path': result['path']
                    })

        logger.info("\n" + "=" * 70)
        logger.info("批次分析完成")
        logger.info("=" * 70)
        logger.info(f"分析配對數: {num_pairs_analyzed}")
        logger.info(f"應連接數: {len(connections)}")
        if num_pairs_analyzed > 0:
            logger.info(f"連接率: {len(connections) / num_pairs_analyzed * 100:.1f}%")
        logger.info("=" * 70)

        return {
            'num_components': num_components,
            'num_pairs_analyzed': num_pairs_analyzed,
            'num_connections': len(connections),
            'connections': connections,
            'all_pair_results': all_pair_results
        }

