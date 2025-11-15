#!/usr/bin/env python3
"""
曲率感知的種子提取腳本 (Curvature-Aware Seed Extraction)

從骨架化結果中提取種子點，基於曲率感知的自適應分段策略。
在神經彎折處放置種子點，確保後續 MST 重建不會將彎折拉直。

使用方式:
    python seed_extraction.py -i output/skeletons -o output/seeds -v

作者: Generated with Claude Code
日期: 2025-10-22
"""

import argparse
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Set, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from skimage.color import label2rgb

# Import configuration loader
from src.config_loader import load_config, SeedExtractionConfig


class PathDecomposer:
    """骨架路徑分解器 - 使用圖論方法將骨架分解為獨立路徑"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def decompose_skeleton(
        self,
        skeleton: np.ndarray,
        endpoints: List[Tuple[int, int]],
        branchpoints: List[Tuple[int, int]],
        component_id: int
    ) -> List[Dict]:
        """
        將骨架分解為獨立路徑列表

        Args:
            skeleton: 單個元件的骨架 mask
            endpoints: 端點列表 [(y, x), ...]
            branchpoints: 分支點列表 [(y, x), ...]
            component_id: 元件 ID

        Returns:
            paths: 路徑列表
        """
        # 轉換為集合便於查找
        endpoints_set = {tuple(ep) for ep in endpoints}
        branchpoints_set = {tuple(bp) for bp in branchpoints}
        keypoints_set = endpoints_set | branchpoints_set

        if len(keypoints_set) == 0:
            # 沒有關鍵點（可能是環形或單點）
            if self.verbose:
                print(f"    警告: 元件 {component_id} 沒有端點或分支點")
            return []

        paths = []
        visited_edges = set()

        # 從每個關鍵點出發追蹤路徑
        for start_point in keypoints_set:
            neighbors = self._get_skeleton_neighbors(skeleton, start_point)

            for neighbor in neighbors:
                edge_id = self._make_edge_id(start_point, neighbor)

                if edge_id in visited_edges:
                    continue

                # 追蹤路徑直到下一個關鍵點
                path_coords = self._trace_path_to_keypoint(
                    skeleton, neighbor, start_point, keypoints_set
                )

                # 完整路徑 = [start] + path
                full_path = [start_point] + path_coords

                # 標記所有經過的邊為已訪問
                for i in range(len(full_path) - 1):
                    edge_id = self._make_edge_id(full_path[i], full_path[i+1])
                    visited_edges.add(edge_id)

                # 計算路徑長度
                path_length = self._calculate_path_length(full_path)

                # 判斷起點和終點類型
                start_type = 'endpoint' if start_point in endpoints_set else 'branchpoint'
                end_type = 'endpoint' if full_path[-1] in endpoints_set else 'branchpoint'

                paths.append({
                    'path_id': len(paths),
                    'component_id': component_id,
                    'start': {'pos': start_point, 'type': start_type},
                    'end': {'pos': full_path[-1], 'type': end_type},
                    'coordinates': full_path,
                    'length': path_length
                })

        return paths

    def _get_skeleton_neighbors(
        self,
        skeleton: np.ndarray,
        point: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """獲取骨架點的 8-鄰域骨架鄰居"""
        y, x = point
        neighbors = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = y + dy, x + dx

                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx] > 0:
                        neighbors.append((ny, nx))

        return neighbors

    def _make_edge_id(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """創建唯一的邊 ID（排序後的點對）"""
        return tuple(sorted([point1, point2]))

    def _trace_path_to_keypoint(
        self,
        skeleton: np.ndarray,
        start: Tuple[int, int],
        came_from: Tuple[int, int],
        keypoints: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        從 start 追蹤路徑直到遇到關鍵點

        Args:
            skeleton: 骨架 mask
            start: 起始點
            came_from: 來源點（避免回頭）
            keypoints: 關鍵點集合

        Returns:
            path: 路徑座標列表
        """
        path = []
        current = start
        visited = {came_from}

        while True:
            path.append(current)
            visited.add(current)

            # 如果當前點是關鍵點（且不是起點），停止
            if current in keypoints and current != start:
                break

            # 獲取未訪問的鄰居
            neighbors = self._get_skeleton_neighbors(skeleton, current)
            unvisited = [n for n in neighbors if n not in visited]

            # 沒有未訪問的鄰居，停止
            if not unvisited:
                break

            # 繼續追蹤（對於非分支點，應該只有一個未訪問鄰居）
            current = unvisited[0]

        return path

    def _calculate_path_length(self, path_coords: List[Tuple[int, int]]) -> float:
        """計算路徑長度（考慮對角線距離）"""
        if len(path_coords) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path_coords)):
            prev = np.array(path_coords[i-1])
            curr = np.array(path_coords[i])
            total_length += np.linalg.norm(curr - prev)

        return total_length


class CurvatureCalculator:
    """曲率計算器 - 使用 5 點窗口和不對稱邊界處理"""

    def __init__(self, window_size: int = 5, verbose: bool = False):
        """
        Args:
            window_size: 窗口大小（骨架點數），必須是奇數
            verbose: 是否輸出詳細資訊
        """
        if window_size % 2 == 0:
            raise ValueError(f"window_size 必須是奇數，但收到 {window_size}")

        self.window_size = window_size
        self.half_window = window_size // 2
        self.verbose = verbose

    def calculate_curvature_along_path(
        self,
        path_coords: List[Tuple[int, int]],
        branchpoint_indices: Optional[List[int]] = None,
        skip_branchpoint_range: int = 5
    ) -> List[Optional[float]]:
        """
        計算路徑上每個點的曲率

        Args:
            path_coords: 路徑座標列表 [(y, x), ...]
            branchpoint_indices: 分支點在路徑中的索引
            skip_branchpoint_range: 分支點附近跳過的範圍（骨架點數）

        Returns:
            curvatures: 每個點的曲率（度數），None 表示跳過
        """
        n = len(path_coords)
        curvatures = []

        # 處理分支點附近的跳過區域
        skip_indices = set()
        if branchpoint_indices:
            for bp_idx in branchpoint_indices:
                for offset in range(-skip_branchpoint_range, skip_branchpoint_range + 1):
                    idx = bp_idx + offset
                    if 0 <= idx < n:
                        skip_indices.add(idx)

        for i in range(n):
            # 跳過分支點附近
            if i in skip_indices:
                curvatures.append(None)
                continue

            # 確定實際窗口範圍（不對稱邊界處理）
            start_idx = max(0, i - self.half_window)
            end_idx = min(n, i + self.half_window + 1)

            # 窗口至少需要 3 個點才能計算曲率
            if end_idx - start_idx < 3:
                curvatures.append(None)
                continue

            window_points = path_coords[start_idx:end_idx]

            # 計算當前點在窗口中的相對位置
            mid_idx = i - start_idx

            # 分為前半和後半（都包含當前點）
            front_half = window_points[:mid_idx+1]
            back_half = window_points[mid_idx:]

            # 計算方向向量（端點連線）
            if len(front_half) >= 2:
                v1 = np.array(front_half[-1]) - np.array(front_half[0])
            else:
                curvatures.append(None)
                continue

            if len(back_half) >= 2:
                v2 = np.array(back_half[-1]) - np.array(back_half[0])
            else:
                curvatures.append(None)
                continue

            # 處理零向量
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 < 1e-6 or norm_v2 < 1e-6:
                # 零向量，標記為直線（曲率 = 0）
                curvatures.append(0.0)
                continue

            # 計算夾角（arccos）
            dot_product = np.dot(v1, v2) / (norm_v1 * norm_v2)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # 數值穩定性

            curvature_radians = np.arccos(dot_product)
            curvature_degrees = np.degrees(curvature_radians)

            curvatures.append(curvature_degrees)

        return curvatures


class SeedExtractor:
    """種子提取器 - 基於曲率感知的自適應分段"""

    def __init__(
        self,
        base_segment_length: float = 10.0,
        max_segment_length: float = 20.0,
        curvature_threshold: float = 30.0,
        min_path_points: int = 10,
        verbose: bool = False
    ):
        """
        Args:
            base_segment_length: 基礎分段長度（像素）
            max_segment_length: 防呆最大長度（像素）
            curvature_threshold: 曲率閾值（度數）
            min_path_points: 路徑最小點數，低於此值只放端點/分支點
            verbose: 是否輸出詳細資訊
        """
        self.base_segment_length = base_segment_length
        self.max_segment_length = max_segment_length
        self.curvature_threshold = curvature_threshold
        self.min_path_points = min_path_points
        self.verbose = verbose

    def extract_seeds_from_path(
        self,
        path_info: Dict,
        curvatures: List[Optional[float]]
    ) -> List[Dict]:
        """
        從單條路徑提取種子點

        Args:
            path_info: 路徑資訊字典
            curvatures: 每個點的曲率列表

        Returns:
            seeds: 種子列表
        """
        path_coords = path_info['coordinates']
        path_length = path_info['length']

        seeds = []

        # 特殊情況：極短路徑（< min_path_points 個點）
        if len(path_coords) < self.min_path_points:
            # 只在端點/分支點放種子
            seeds.append({
                'position': path_coords[0],
                'type': path_info['start']['type'],
                'curvature_degrees': None,
                'path_id': path_info['path_id']
            })

            # 避免重複（起點和終點相同）
            if path_coords[-1] != path_coords[0]:
                seeds.append({
                    'position': path_coords[-1],
                    'type': path_info['end']['type'],
                    'curvature_degrees': None,
                    'path_id': path_info['path_id']
                })

            return seeds

        # 正常處理：曲率感知分段

        # 1. 強制添加起點（端點或分支點）
        seeds.append({
            'position': path_coords[0],
            'type': path_info['start']['type'],
            'curvature_degrees': None,
            'path_id': path_info['path_id']
        })

        # 2. 沿路徑行進，累積長度並監測曲率
        current_segment_start_idx = 0
        accumulated_length = 0.0

        for i in range(1, len(path_coords)):
            # 計算到前一個點的距離
            prev_point = np.array(path_coords[i-1])
            curr_point = np.array(path_coords[i])
            step_distance = np.linalg.norm(curr_point - prev_point)
            accumulated_length += step_distance

            curvature = curvatures[i]

            # 判斷是否創建種子
            should_create_seed = False
            seed_type = None
            seed_curvature = None

            # 優先級 1: 顯著彎折（最高優先級）
            if curvature is not None and curvature > self.curvature_threshold:
                should_create_seed = True
                seed_type = 'curvature'
                seed_curvature = curvature

            # 優先級 2: 防呆機制（路徑過長）
            elif accumulated_length >= self.max_segment_length:
                should_create_seed = True
                seed_type = 'regular'
                seed_curvature = None

            # 優先級 3: 基礎分段長度
            elif accumulated_length >= self.base_segment_length:
                should_create_seed = True
                seed_type = 'regular'
                seed_curvature = None

            if should_create_seed:
                # 計算該段的中心點（路徑中點）
                segment_coords = path_coords[current_segment_start_idx:i+1]
                seed_position = self._find_path_midpoint(segment_coords)

                seeds.append({
                    'position': seed_position,
                    'type': seed_type,
                    'curvature_degrees': seed_curvature,
                    'path_id': path_info['path_id']
                })

                # 重置累積長度和起點
                current_segment_start_idx = i
                accumulated_length = 0.0

        # 3. 強制添加終點（端點或分支點）
        if path_coords[-1] != path_coords[0]:
            seeds.append({
                'position': path_coords[-1],
                'type': path_info['end']['type'],
                'curvature_degrees': None,
                'path_id': path_info['path_id']
            })

        return seeds

    def _find_path_midpoint(
        self,
        path_coords: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        找到路徑的中點（基於累積長度）

        Args:
            path_coords: 路徑座標列表

        Returns:
            midpoint: 中點座標
        """
        if len(path_coords) == 1:
            return path_coords[0]

        # 計算累積長度
        cumulative_lengths = [0.0]
        for i in range(1, len(path_coords)):
            prev = np.array(path_coords[i-1])
            curr = np.array(path_coords[i])
            dist = np.linalg.norm(curr - prev)
            cumulative_lengths.append(cumulative_lengths[-1] + dist)

        total_length = cumulative_lengths[-1]

        if total_length < 1e-6:
            # 退化情況：所有點重合
            return path_coords[0]

        target_length = total_length / 2.0

        # 找到累積長度最接近 target_length 的點
        for i in range(len(cumulative_lengths)):
            if cumulative_lengths[i] >= target_length:
                return path_coords[i]

        # 理論上不應到達這裡，返回最後一個點
        return path_coords[-1]


class SeedDeduplicator:
    """種子去重器 - 端點/分支點優先，位置去重"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def deduplicate_seeds(self, seeds: List[Dict]) -> List[Dict]:
        """
        去重策略：
        1. 端點和分支點按位置去重
        2. 其他種子（curvature, regular）不去重

        Args:
            seeds: 所有種子列表

        Returns:
            deduplicated_seeds: 去重後的種子列表
        """
        # 分類種子
        priority_seeds = []  # endpoint, branchpoint, centroid
        other_seeds = []     # curvature, regular

        for seed in seeds:
            if seed['type'] in ['endpoint', 'branchpoint', 'centroid']:
                priority_seeds.append(seed)
            else:
                other_seeds.append(seed)

        # 端點/分支點/質心位置去重
        unique_priority = []
        seen_positions = set()

        for seed in priority_seeds:
            pos_tuple = tuple(seed['position'])
            if pos_tuple not in seen_positions:
                seen_positions.add(pos_tuple)
                unique_priority.append(seed)

        if self.verbose:
            removed = len(priority_seeds) - len(unique_priority)
            if removed > 0:
                print(f"    去重: 移除 {removed} 個重複的端點/分支點/質心")

        # 合併：優先種子 + 其他種子（不去重）
        return unique_priority + other_seeds


class SeedExtractionPipeline:
    """種子提取主流程"""

    def __init__(
        self,
        window_size: int = 5,
        base_segment_length: float = 10.0,
        max_segment_length: float = 20.0,
        curvature_threshold: float = 30.0,
        skip_branchpoint_range: int = 5,
        min_path_points: int = 10,
        verbose: bool = False
    ):
        """初始化種子提取流程"""
        self.window_size = window_size
        self.base_segment_length = base_segment_length
        self.max_segment_length = max_segment_length
        self.curvature_threshold = curvature_threshold
        self.skip_branchpoint_range = skip_branchpoint_range
        self.min_path_points = min_path_points
        self.verbose = verbose

        # 初始化各模塊
        self.path_decomposer = PathDecomposer(verbose=verbose)
        self.curvature_calculator = CurvatureCalculator(
            window_size=window_size,
            verbose=verbose
        )
        self.seed_extractor = SeedExtractor(
            base_segment_length=base_segment_length,
            max_segment_length=max_segment_length,
            curvature_threshold=curvature_threshold,
            min_path_points=min_path_points,
            verbose=verbose
        )
        self.deduplicator = SeedDeduplicator(verbose=verbose)

    def load_skeletons_data(
        self,
        skeletons_dir: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        載入骨架資料

        Args:
            skeletons_dir: 骨架資料夾路徑

        Returns:
            (labeled_skeletons, skeletons_json)
        """
        skeletons_dir = Path(skeletons_dir)

        if not skeletons_dir.exists():
            raise FileNotFoundError(f"骨架資料夾不存在: {skeletons_dir}")

        # 載入標籤骨架影像
        labeled_skeletons_path = skeletons_dir / 'labeled_skeletons.png'
        if not labeled_skeletons_path.exists():
            raise FileNotFoundError(f"找不到標籤骨架影像: {labeled_skeletons_path}")

        labeled_skeletons = cv2.imread(str(labeled_skeletons_path), cv2.IMREAD_UNCHANGED)
        if labeled_skeletons is None:
            raise ValueError(f"無法讀取標籤骨架影像: {labeled_skeletons_path}")

        # 載入 JSON 元數據
        json_path = skeletons_dir / 'skeletons.json'
        if not json_path.exists():
            raise FileNotFoundError(f"找不到骨架元數據: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            skeletons_data = json.load(f)

        if self.verbose:
            print(f"✓ 載入骨架資料")
            print(f"  標籤骨架影像: {labeled_skeletons_path}")
            print(f"  影像尺寸: {labeled_skeletons.shape[1]}x{labeled_skeletons.shape[0]}")
            print(f"  骨架數量: {len(skeletons_data['skeletons'])}")

        return labeled_skeletons, skeletons_data

    def process_component(
        self,
        labeled_skeletons: np.ndarray,
        skeleton_info: Dict
    ) -> List[Dict]:
        """
        處理單個元件的骨架，提取種子點

        Args:
            labeled_skeletons: 標籤骨架影像
            skeleton_info: 骨架資訊字典

        Returns:
            seeds: 種子列表
        """
        component_id = skeleton_info['component_id']

        if self.verbose:
            print(f"\n處理元件 {component_id}...")
            print(f"  骨架長度: {skeleton_info['skeleton_length']:.2f} 像素")
            print(f"  端點數: {skeleton_info['num_endpoints']}")
            print(f"  分支點數: {skeleton_info['num_branchpoints']}")

        # 檢查是否有有效骨架，無骨架則使用質心作為種子
        if skeleton_info['skeleton_pixels'] == 0:
            if self.verbose:
                print(f"  警告: 元件無骨架，使用質心作為種子")

            centroid = skeleton_info.get('centroid')
            if centroid and 'x' in centroid and 'y' in centroid:
                centroid_seed = {
                    'position': (int(centroid['y']), int(centroid['x'])),
                    'type': 'centroid',
                    'curvature_degrees': None,
                    'path_id': None,
                    'component_id': component_id
                }
                if self.verbose:
                    print(f"  提取 1 個質心種子: ({centroid['x']:.1f}, {centroid['y']:.1f})")
                return [centroid_seed]
            else:
                if self.verbose:
                    print(f"  錯誤: 無質心資訊可用")
                return []

        # 提取該元件的骨架 mask
        skeleton_mask = (labeled_skeletons == component_id).astype(np.uint8) * 255

        # 轉換端點和分支點為 tuple
        endpoints = [tuple([ep['y'], ep['x']]) for ep in skeleton_info['endpoints']]
        branchpoints = [tuple([bp['y'], bp['x']]) for bp in skeleton_info['branchpoints']]

        # 1. 分解骨架為路徑
        paths = self.path_decomposer.decompose_skeleton(
            skeleton_mask, endpoints, branchpoints, component_id
        )

        if self.verbose:
            print(f"  分解為 {len(paths)} 條路徑")

        # 檢查是否有路徑，無路徑則使用質心作為種子
        if len(paths) == 0:
            if self.verbose:
                print(f"  警告: 元件無有效路徑，使用質心作為種子")

            centroid = skeleton_info.get('centroid')
            if centroid and 'x' in centroid and 'y' in centroid:
                centroid_seed = {
                    'position': (int(centroid['y']), int(centroid['x'])),
                    'type': 'centroid',
                    'curvature_degrees': None,
                    'path_id': None,
                    'component_id': component_id
                }
                if self.verbose:
                    print(f"  提取 1 個質心種子: ({centroid['x']:.1f}, {centroid['y']:.1f})")
                return [centroid_seed]
            else:
                if self.verbose:
                    print(f"  錯誤: 無質心資訊可用")
                return []

        all_seeds = []

        # 2. 對每條路徑提取種子
        for path in paths:
            if self.verbose:
                print(f"    路徑 {path['path_id']}: "
                      f"{path['start']['type']} → {path['end']['type']}, "
                      f"長度 {path['length']:.1f}px, "
                      f"{len(path['coordinates'])} 個點")

            # 查找分支點在路徑中的索引
            branchpoint_indices = []
            branchpoint_set = set(branchpoints)
            for i, coord in enumerate(path['coordinates']):
                if tuple(coord) in branchpoint_set:
                    branchpoint_indices.append(i)

            # 計算曲率
            curvatures = self.curvature_calculator.calculate_curvature_along_path(
                path['coordinates'],
                branchpoint_indices,
                self.skip_branchpoint_range
            )

            # 提取種子
            path_seeds = self.seed_extractor.extract_seeds_from_path(path, curvatures)

            if self.verbose:
                print(f"      提取 {len(path_seeds)} 個種子")

            all_seeds.extend(path_seeds)

        # 3. 去重（端點/分支點位置去重）
        deduplicated_seeds = self.deduplicator.deduplicate_seeds(all_seeds)

        if self.verbose:
            print(f"  元件 {component_id} 總計: {len(deduplicated_seeds)} 個種子")
            seed_types = {}
            for seed in deduplicated_seeds:
                seed_types[seed['type']] = seed_types.get(seed['type'], 0) + 1
            print(f"    類型分布: {seed_types}")

        return deduplicated_seeds

    def save_seeds_json(
        self,
        all_seeds: List[Dict],
        input_dir: str,
        output_path: str
    ) -> None:
        """儲存種子資料為 JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 為每個種子分配唯一 ID
        for i, seed in enumerate(all_seeds, start=1):
            seed['seed_id'] = i
            # 轉換 position 為 dict
            seed['position'] = {'x': int(seed['position'][1]), 'y': int(seed['position'][0])}

        # 統計資訊
        seeds_by_type = {}
        for seed in all_seeds:
            seed_type = seed['type']
            seeds_by_type[seed_type] = seeds_by_type.get(seed_type, 0) + 1

        # 計算每個元件的種子數
        seeds_per_component = {}
        for seed in all_seeds:
            comp_id = seed.get('component_id')
            if comp_id:
                seeds_per_component[comp_id] = seeds_per_component.get(comp_id, 0) + 1

        avg_seeds_per_component = (
            sum(seeds_per_component.values()) / len(seeds_per_component)
            if seeds_per_component else 0.0
        )

        metadata = {
            'metadata': {
                'total_components': len(seeds_per_component),
                'total_seeds': len(all_seeds),
                'timestamp': datetime.now().isoformat(),
                'source_skeletons': str(Path(input_dir) / 'skeletons.json'),
                'parameters': {
                    'window_size': self.window_size,
                    'base_segment_length': self.base_segment_length,
                    'max_segment_length': self.max_segment_length,
                    'curvature_threshold': self.curvature_threshold,
                    'skip_branchpoint_range': self.skip_branchpoint_range,
                    'min_path_points': self.min_path_points
                }
            },
            'seeds': all_seeds,
            'statistics': {
                'seeds_by_type': seeds_by_type,
                'avg_seeds_per_component': float(avg_seeds_per_component)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"✓ 種子資料已儲存: {output_path}")

    def visualize_seeds_on_skeleton(
        self,
        labeled_skeletons: np.ndarray,
        all_seeds: List[Dict],
        output_path: str
    ) -> None:
        """視覺化 1: 種子點標記在骨架上"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成彩色骨架
        colored_skeleton = label2rgb(
            labeled_skeletons,
            bg_label=0,
            bg_color=(0, 0, 0)
        )

        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(colored_skeleton)

        # Define visual styles for seed types
        seed_styles = {
            'endpoint': {'marker': 'o', 'color': 'red', 'size': 80, 'label': 'Endpoints'},
            'branchpoint': {'marker': 's', 'color': 'blue', 'size': 80, 'label': 'Branchpoints'},
            'curvature': {'marker': '*', 'color': 'yellow', 'size': 120, 'label': 'Curvature Points'},
            'regular': {'marker': '^', 'color': 'lime', 'size': 60, 'label': 'Regular Points'},
            'centroid': {'marker': 'D', 'color': 'magenta', 'size': 80, 'label': 'Centroids'}
        }

        # Plot seed points
        for seed_type, style in seed_styles.items():
            seeds_of_type = [s for s in all_seeds if s['type'] == seed_type]
            if seeds_of_type:
                xs = [s['position']['x'] for s in seeds_of_type]
                ys = [s['position']['y'] for s in seeds_of_type]
                ax.scatter(xs, ys,
                          marker=style['marker'],
                          c=style['color'],
                          s=style['size'],
                          label=style['label'],
                          edgecolors='black',
                          linewidths=1,
                          alpha=0.9,
                          zorder=10)

        ax.axis('off')
        ax.set_title(
            f'Seed Points Visualization ({len(all_seeds)} seeds)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ 種子點視覺化已儲存: {output_path}")

    def visualize_seeds_overlay(
        self,
        labeled_components: np.ndarray,
        all_seeds: List[Dict],
        output_path: str,
        components_dir: str
    ) -> None:
        """視覺化 2: 種子點疊加在原始元件上"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 載入原始標籤元件
        components_path = Path(components_dir) / 'labeled_components.png'
        if components_path.exists():
            labeled_components = cv2.imread(str(components_path), cv2.IMREAD_UNCHANGED)

        # 生成半透明元件背景
        components_colored = label2rgb(
            labeled_components,
            bg_label=0,
            bg_color=(0, 0, 0),
            alpha=0.3
        )

        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(components_colored)

        # Plot seed points (same as above)
        seed_styles = {
            'endpoint': {'marker': 'o', 'color': 'red', 'size': 80, 'label': 'Endpoints'},
            'branchpoint': {'marker': 's', 'color': 'blue', 'size': 80, 'label': 'Branchpoints'},
            'curvature': {'marker': '*', 'color': 'yellow', 'size': 120, 'label': 'Curvature Points'},
            'regular': {'marker': '^', 'color': 'lime', 'size': 60, 'label': 'Regular Points'},
            'centroid': {'marker': 'D', 'color': 'magenta', 'size': 80, 'label': 'Centroids'}
        }

        for seed_type, style in seed_styles.items():
            seeds_of_type = [s for s in all_seeds if s['type'] == seed_type]
            if seeds_of_type:
                xs = [s['position']['x'] for s in seeds_of_type]
                ys = [s['position']['y'] for s in seeds_of_type]
                ax.scatter(xs, ys,
                          marker=style['marker'],
                          c=style['color'],
                          s=style['size'],
                          label=style['label'],
                          edgecolors='black',
                          linewidths=1,
                          alpha=0.9,
                          zorder=10)

        ax.axis('off')
        ax.set_title(
            f'Seed Points Overlay Visualization',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ 疊加視覺化已儲存: {output_path}")

    def visualize_curvature_heatmap(
        self,
        labeled_skeletons: np.ndarray,
        skeletons_data: Dict,
        all_seeds: List[Dict],
        output_path: str
    ) -> None:
        """視覺化 3: 曲率熱力圖"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 創建曲率影像（初始化為 NaN）
        curvature_map = np.full(labeled_skeletons.shape, np.nan, dtype=np.float32)

        # 對每個元件重新計算曲率並填充
        for skeleton_info in skeletons_data['skeletons']:
            component_id = skeleton_info['component_id']
            skeleton_mask = (labeled_skeletons == component_id).astype(np.uint8) * 255

            endpoints = [tuple([ep['y'], ep['x']]) for ep in skeleton_info['endpoints']]
            branchpoints = [tuple([bp['y'], bp['x']]) for bp in skeleton_info['branchpoints']]

            paths = self.path_decomposer.decompose_skeleton(
                skeleton_mask, endpoints, branchpoints, component_id
            )

            for path in paths:
                branchpoint_indices = []
                branchpoint_set = set(branchpoints)
                for i, coord in enumerate(path['coordinates']):
                    if tuple(coord) in branchpoint_set:
                        branchpoint_indices.append(i)

                curvatures = self.curvature_calculator.calculate_curvature_along_path(
                    path['coordinates'],
                    branchpoint_indices,
                    self.skip_branchpoint_range
                )

                # 填充曲率圖
                for coord, curv in zip(path['coordinates'], curvatures):
                    if curv is not None:
                        curvature_map[coord[0], coord[1]] = curv

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 14))

        # Use mask to handle NaN values
        curvature_masked = np.ma.masked_invalid(curvature_map)

        im = ax.imshow(
            curvature_masked,
            cmap='coolwarm',
            vmin=0,
            vmax=180,
            interpolation='nearest'
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Curvature (degrees)', fontsize=12)

        # Plot curvature threshold line
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2,
                   label=f'Curvature threshold: {self.curvature_threshold}°', alpha=0)

        # Mark curvature type seeds
        curvature_seeds = [s for s in all_seeds if s['type'] == 'curvature']
        if curvature_seeds:
            xs = [s['position']['x'] for s in curvature_seeds]
            ys = [s['position']['y'] for s in curvature_seeds]
            ax.scatter(xs, ys,
                      marker='*',
                      c='yellow',
                      s=120,
                      label='Curvature Seeds',
                      edgecolors='black',
                      linewidths=1,
                      zorder=10)

        ax.axis('off')
        ax.set_title(
            f'Curvature Heatmap',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ 曲率熱力圖已儲存: {output_path}")

    def process(
        self,
        input_dir: str,
        output_dir: str,
        visualize_seeds: bool = False,
        visualize_overlay: bool = False,
        visualize_curvature: bool = False
    ) -> Tuple[List[Dict], Dict]:
        """
        完整的種子提取流程

        Args:
            input_dir: 骨架資料夾路徑
            output_dir: 輸出目錄
            visualize_seeds: 是否生成種子標記視覺化
            visualize_overlay: 是否生成疊加視覺化
            visualize_curvature: 是否生成曲率熱力圖

        Returns:
            (all_seeds, skeletons_data)
        """
        if self.verbose:
            print("=" * 60)
            print("曲率感知的種子提取")
            print("=" * 60)
            print(f"參數:")
            print(f"  窗口大小: {self.window_size} 個骨架點")
            print(f"  基礎分段長度: {self.base_segment_length} 像素")
            print(f"  防呆最大長度: {self.max_segment_length} 像素")
            print(f"  曲率閾值: {self.curvature_threshold} 度")
            print(f"  分支點跳過範圍: ±{self.skip_branchpoint_range} 個骨架點")

        # 1. 載入骨架資料
        labeled_skeletons, skeletons_data = self.load_skeletons_data(input_dir)

        # 2. 處理每個元件
        all_seeds = []

        for skeleton_info in skeletons_data['skeletons']:
            component_seeds = self.process_component(labeled_skeletons, skeleton_info)
            # 添加 component_id
            for seed in component_seeds:
                if 'component_id' not in seed:
                    seed['component_id'] = skeleton_info['component_id']
            all_seeds.extend(component_seeds)

        if self.verbose:
            print(f"\n總計提取 {len(all_seeds)} 個種子")

        # 3. 建立輸出目錄
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4. 儲存 JSON
        json_path = output_dir / 'seeds.json'
        self.save_seeds_json(all_seeds, input_dir, str(json_path))

        # 5. 生成視覺化
        if visualize_seeds:
            viz_path = output_dir / 'visualization_seeds.png'
            self.visualize_seeds_on_skeleton(labeled_skeletons, all_seeds, str(viz_path))

        if visualize_overlay:
            viz_path = output_dir / 'visualization_overlay.png'
            # 嘗試從 components 目錄載入
            components_dir = Path(input_dir).parent / 'components'
            self.visualize_seeds_overlay(
                labeled_skeletons, all_seeds, str(viz_path), str(components_dir)
            )

        if visualize_curvature:
            viz_path = output_dir / 'visualization_curvature.png'
            self.visualize_curvature_heatmap(
                labeled_skeletons, skeletons_data, all_seeds, str(viz_path)
            )

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 種子提取完成!")
            print("=" * 60)
            print(f"\n輸出檔案:")
            print(f"  - 種子資料: {json_path}")
            if visualize_seeds:
                print(f"  - 種子視覺化: {output_dir / 'visualization_seeds.png'}")
            if visualize_overlay:
                print(f"  - 疊加視覺化: {output_dir / 'visualization_overlay.png'}")
            if visualize_curvature:
                print(f"  - 曲率熱力圖: {output_dir / 'visualization_curvature.png'}")

        return all_seeds, skeletons_data


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='曲率感知的種子提取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 使用 YAML 配置文件（推薦）
  python %(prog)s -i output/skeletons -o output/seeds --config config/default.yaml

  # 基本使用（預設參數）
  python %(prog)s -i output/skeletons -o output/seeds

  # 完整參數 + 所有視覺化
  python %(prog)s -i output/skeletons -o output/seeds -v --verbose

  # 調整參數（CLI 覆蓋 YAML）
  python %(prog)s -i output/skeletons --config config/default.yaml --curvature-threshold 25.0

  # 僅生成曲率熱力圖
  python %(prog)s -i output/skeletons --viz-curvature

演算法說明:
  1. 骨架路徑分解（端點↔端點/分支點）
  2. 5點窗口計算曲率（不對稱邊界）
  3. 三種觸發條件提取種子：
     - 曲率 > 30度（優先級最高）
     - 長度 >= 20px（防呆）
     - 長度 >= 10px（基礎分段）
  4. 種子位置 = 段落路徑中點
  5. 端點/分支點去重

種子類型:
  - endpoint: 端點（紅色圓點）
  - branchpoint: 分支點（藍色方塊）
  - curvature: 彎折點（黃色星形）
  - regular: 規律點（綠色三角形）

輸出說明:
  output_dir/
  ├── seeds.json                   # 種子資料
  ├── visualization_seeds.png      # 種子標記在骨架上
  ├── visualization_overlay.png    # 疊加在原始影像
  └── visualization_curvature.png  # 曲率熱力圖
        """
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML 配置文件路徑（可選，CLI 參數會覆蓋配置文件）'
    )

    # 必填參數
    parser.add_argument(
        '-i', '--input-skeletons',
        type=str,
        required=True,
        help='骨架資料夾路徑（包含 labeled_skeletons.png 和 skeletons.json）'
    )

    # 選填參數
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='輸出目錄（預設: ./output/seeds）'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=None,
        help='曲率計算窗口大小（骨架點數，必須是奇數）'
    )

    parser.add_argument(
        '--base-length',
        type=float,
        default=None,
        metavar='PIXELS',
        help='基礎分段長度（像素）'
    )

    parser.add_argument(
        '--max-length',
        type=float,
        default=None,
        metavar='PIXELS',
        help='防呆最大長度（像素）'
    )

    parser.add_argument(
        '--curvature-threshold',
        type=float,
        default=None,
        metavar='DEGREES',
        help='曲率閾值（度數）'
    )

    parser.add_argument(
        '--skip-branchpoint',
        type=int,
        default=None,
        help='分支點附近跳過範圍（骨架點數）'
    )

    # 視覺化選項
    viz_group = parser.add_argument_group('視覺化選項')
    viz_group.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='生成所有視覺化圖'
    )
    viz_group.add_argument(
        '--viz-seeds',
        action='store_true',
        help='僅生成種子標記視覺化'
    )
    viz_group.add_argument(
        '--viz-overlay',
        action='store_true',
        help='僅生成疊加視覺化'
    )
    viz_group.add_argument(
        '--viz-curvature',
        action='store_true',
        help='僅生成曲率熱力圖'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='輸出詳細處理資訊'
    )

    return parser.parse_args()


def main():
    """主程式進入點"""
    args = parse_arguments()

    try:
        # Load configuration from YAML (if provided) or use defaults
        if args.config:
            full_config = load_config(args.config)
            config = full_config.seed_extraction
            if args.verbose:
                print(f"✓ 載入配置文件: {args.config}")
        else:
            # Use default configuration
            config = SeedExtractionConfig()

        # Apply CLI overrides (CLI takes precedence over YAML)
        overrides = {}
        if args.window_size is not None:
            overrides['window_size'] = args.window_size
        if args.base_length is not None:
            overrides['base_segment_length'] = args.base_length
        if args.max_length is not None:
            overrides['max_segment_length'] = args.max_length
        if args.curvature_threshold is not None:
            overrides['curvature_threshold'] = args.curvature_threshold
        if args.skip_branchpoint is not None:
            overrides['skip_branchpoint_range'] = args.skip_branchpoint

        # Apply overrides to config
        for key, value in overrides.items():
            setattr(config, key, value)

        # Determine output directory
        output_dir = args.output_dir if args.output_dir else './output/seeds'

        # Validate parameters
        if config.window_size % 2 == 0:
            print(f"錯誤: window_size 必須是奇數，但收到 {config.window_size}", file=sys.stderr)
            return 1

        if config.base_segment_length <= 0 or config.max_segment_length <= 0:
            print(f"錯誤: base_segment_length 和 max_segment_length 必須大於 0", file=sys.stderr)
            return 1

        if config.base_segment_length > config.max_segment_length:
            print(f"警告: base_segment_length ({config.base_segment_length}) > max_segment_length ({config.max_segment_length})")

        # 決定視覺化選項
        visualize_seeds = args.visualize or args.viz_seeds
        visualize_overlay = args.visualize or args.viz_overlay
        visualize_curvature = args.visualize or args.viz_curvature

        # 建立種子提取流程
        pipeline = SeedExtractionPipeline(
            window_size=config.window_size,
            base_segment_length=config.base_segment_length,
            max_segment_length=config.max_segment_length,
            curvature_threshold=config.curvature_threshold,
            skip_branchpoint_range=config.skip_branchpoint_range,
            min_path_points=config.min_path_points,
            verbose=args.verbose
        )

        # 執行種子提取
        pipeline.process(
            input_dir=args.input_skeletons,
            output_dir=output_dir,
            visualize_seeds=visualize_seeds,
            visualize_overlay=visualize_overlay,
            visualize_curvature=visualize_curvature
        )

        return 0

    except FileNotFoundError as e:
        print(f"錯誤: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"錯誤: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"未預期的錯誤: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
