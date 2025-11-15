#!/usr/bin/env python3
"""
骨架化與結構分析腳本 (Skeletonization and Structure Analysis)

對連通元件執行形態學骨架化，並分析骨架結構（端點、分支點、長度），
供後續曲率感知的種子提取使用。

使用方式:
    python skeletonization.py -i output/components -o output/skeletons -v

作者: Generated with Claude Code
日期: 2025-10-22
"""

import argparse
import json
import sys
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from skimage import morphology, measure
from skimage.color import label2rgb


class SkeletonAnalyzer:
    """骨架化與結構分析器"""

    AVAILABLE_METHODS = ['zhang-suen', 'medial-axis', 'guo-hall']

    def __init__(
        self,
        method: str = 'zhang-suen',
        spur_threshold: int = 3,
        verbose: bool = False
    ):
        """
        初始化骨架分析器

        Args:
            method: 骨架化方法 ('zhang-suen', 'medial-axis', 'guo-hall')
            spur_threshold: 短刺過濾閾值（像素），0 表示不過濾
            verbose: 是否輸出詳細資訊
        """
        self.method = method
        self.spur_threshold = spur_threshold
        self.verbose = verbose

        # 驗證並檢查骨架化方法
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"未知的骨架化方法: {method}，可用方法: {self.AVAILABLE_METHODS}")

        # 檢查 guo-hall 方法的依賴
        if method == 'guo-hall':
            if not self._check_opencv_contrib():
                if self.verbose:
                    print("警告: 無法使用 guo-hall 方法，降級為 zhang-suen")
                self.method = 'zhang-suen'

    def _check_opencv_contrib(self) -> bool:
        """檢查 opencv-contrib-python 是否可用"""
        try:
            import cv2.ximgproc
            # 測試是否可以調用 thinning 函數
            cv2.ximgproc.thinning
            return True
        except (ImportError, AttributeError):
            return False

    def load_components(
        self,
        components_dir: str
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        載入連通元件資料

        Args:
            components_dir: 連通元件資料夾路徑

        Returns:
            (標籤影像, 元件列表)

        Raises:
            FileNotFoundError: 找不到必要檔案
            ValueError: JSON 格式錯誤
        """
        components_dir = Path(components_dir)

        # 檢查資料夾是否存在
        if not components_dir.exists():
            raise FileNotFoundError(f"資料夾不存在: {components_dir}")

        # 載入標籤影像
        labeled_image_path = components_dir / 'labeled_components.png'
        if not labeled_image_path.exists():
            raise FileNotFoundError(f"找不到標籤影像: {labeled_image_path}")

        labeled_image = cv2.imread(str(labeled_image_path), cv2.IMREAD_UNCHANGED)
        if labeled_image is None:
            raise ValueError(f"無法讀取標籤影像: {labeled_image_path}")

        # 載入 JSON 元數據
        json_path = components_dir / 'components.json'
        if not json_path.exists():
            raise FileNotFoundError(f"找不到元數據: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        components = metadata.get('components', [])

        if self.verbose:
            print(f"✓ 載入連通元件資料")
            print(f"  標籤影像: {labeled_image_path}")
            print(f"  影像尺寸: {labeled_image.shape[1]}x{labeled_image.shape[0]}")
            print(f"  元件數量: {len(components)}")
            print(f"  最大元件 ID: {labeled_image.max()}")

        return labeled_image, components

    def skeletonize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        對單個 mask 執行骨架化

        Args:
            mask: 二值 mask (0 或 255)

        Returns:
            骨架影像 (0 或 255)
        """
        # 確保 mask 是二值的
        binary = (mask > 0).astype(np.uint8)

        if self.method == 'zhang-suen':
            # scikit-image 的 Zhang-Suen 演算法
            skeleton = morphology.skeletonize(binary)
            skeleton = (skeleton * 255).astype(np.uint8)

        elif self.method == 'medial-axis':
            # scikit-image 的中軸變換
            skeleton = morphology.medial_axis(binary)
            skeleton = (skeleton * 255).astype(np.uint8)

        elif self.method == 'guo-hall':
            # OpenCV 的 Guo-Hall 演算法
            import cv2.ximgproc
            skeleton = cv2.ximgproc.thinning(
                binary * 255,
                thinningType=cv2.ximgproc.THINNING_GUOHALL
            )

        return skeleton

    def detect_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """
        檢測骨架端點（鄰居數 = 1）

        Args:
            skeleton: 骨架影像 (0 或 255)

        Returns:
            端點座標陣列 [[y1, x1], [y2, x2], ...]
        """
        # 二值化
        binary = (skeleton > 0).astype(np.uint8)

        # 8-鄰域卷積核
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)

        # 計算每個骨架點的鄰居數
        neighbor_count = cv2.filter2D(binary, -1, kernel)

        # 端點: 骨架點 且 鄰居數 = 1
        endpoints_mask = (binary > 0) & (neighbor_count == 1)
        endpoints = np.argwhere(endpoints_mask)

        return endpoints

    def detect_branchpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """
        檢測骨架分支點（鄰居數 >= 3）

        Args:
            skeleton: 骨架影像 (0 或 255)

        Returns:
            分支點座標陣列 [[y1, x1], [y2, x2], ...]
        """
        # 二值化
        binary = (skeleton > 0).astype(np.uint8)

        # 8-鄰域卷積核
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)

        # 計算每個骨架點的鄰居數
        neighbor_count = cv2.filter2D(binary, -1, kernel)

        # 分支點: 骨架點 且 鄰居數 >= 3
        branchpoints_mask = (binary > 0) & (neighbor_count >= 3)
        branchpoints = np.argwhere(branchpoints_mask)

        return branchpoints

    def trace_from_endpoint(
        self,
        skeleton: np.ndarray,
        start_point: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        從端點追蹤骨架路徑，直到遇到分支點或另一端點

        Args:
            skeleton: 骨架影像
            start_point: 起始端點 (y, x)

        Returns:
            路徑點列表 [(y1, x1), (y2, x2), ...]
        """
        binary = (skeleton > 0).astype(np.uint8)
        path = [tuple(start_point)]
        visited = {tuple(start_point)}

        current = start_point

        while True:
            # 尋找 8-鄰域中的下一個骨架點
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = current[0] + dy, current[1] + dx

                    # 檢查邊界
                    if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]:
                        if binary[ny, nx] > 0 and (ny, nx) not in visited:
                            neighbors.append((ny, nx))

            # 沒有未訪問的鄰居，結束
            if not neighbors:
                break

            # 選擇第一個鄰居繼續（對於分支，只選一條路徑）
            next_point = neighbors[0]
            path.append(next_point)
            visited.add(next_point)
            current = next_point

            # 如果遇到分支點（>2 個鄰居），停止
            kernel = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], dtype=np.uint8)
            neighbor_count = cv2.filter2D(binary, -1, kernel)[current[0], current[1]]
            if neighbor_count >= 3:
                break

        return path

    def remove_spurs(self, skeleton: np.ndarray, threshold: int) -> np.ndarray:
        """
        移除短刺（長度 < threshold 的末端分支）

        Args:
            skeleton: 骨架影像
            threshold: 短刺長度閾值

        Returns:
            過濾後的骨架影像
        """
        if threshold <= 0:
            return skeleton

        cleaned = skeleton.copy()
        iteration = 0
        max_iterations = 100  # 防止無限循環

        while iteration < max_iterations:
            iteration += 1
            endpoints = self.detect_endpoints(cleaned)

            if len(endpoints) == 0:
                break

            changed = False

            for endpoint in endpoints:
                # 從端點追蹤路徑
                path = self.trace_from_endpoint(cleaned, tuple(endpoint))

                # 如果路徑長度 < threshold，移除
                if len(path) < threshold:
                    for point in path:
                        cleaned[point[0], point[1]] = 0
                    changed = True

            if not changed:
                break

        if self.verbose and iteration > 1:
            print(f"    短刺過濾: {iteration} 次迭代")

        return cleaned

    def calculate_skeleton_length(self, skeleton: np.ndarray) -> float:
        """
        計算骨架長度（考慮對角線距離）

        Args:
            skeleton: 骨架影像

        Returns:
            骨架長度（浮點數）
        """
        binary = (skeleton > 0).astype(np.uint8)
        skeleton_points = np.argwhere(binary > 0)

        if len(skeleton_points) == 0:
            return 0.0

        # 使用邊計數法：遍歷所有骨架點，計算與鄰居的距離總和
        total_length = 0.0
        visited_edges = set()

        for point in skeleton_points:
            y, x = point[0], point[1]

            # 檢查 8-鄰域
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ny, nx = y + dy, x + dx

                # 檢查邊界
                if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]:
                    if binary[ny, nx] > 0:
                        # 創建唯一的邊標識（排序以避免重複）
                        edge = tuple(sorted([(y, x), (ny, nx)]))

                        if edge not in visited_edges:
                            visited_edges.add(edge)

                            # 對角線距離 sqrt(2), 水平/垂直距離 1
                            dist = np.sqrt(2) if (dy != 0 and dx != 0) else 1.0
                            total_length += dist

        return total_length

    def analyze_skeleton(
        self,
        skeleton: np.ndarray,
        component_info: Dict
    ) -> Dict:
        """
        分析骨架結構

        Args:
            skeleton: 骨架影像
            component_info: 原始元件資訊

        Returns:
            骨架分析結果字典
        """
        # 檢測端點和分支點
        endpoints = self.detect_endpoints(skeleton)
        branchpoints = self.detect_branchpoints(skeleton)

        # 計算骨架長度
        skeleton_length = self.calculate_skeleton_length(skeleton)

        # 計算骨架像素數
        skeleton_pixels = np.sum(skeleton > 0)

        # 構建結果
        result = {
            'component_id': component_info['id'],
            'skeleton_length': float(skeleton_length),
            'skeleton_pixels': int(skeleton_pixels),
            'num_endpoints': len(endpoints),
            'num_branchpoints': len(branchpoints),
            'endpoints': [{'x': int(pt[1]), 'y': int(pt[0])} for pt in endpoints],
            'branchpoints': [{'x': int(pt[1]), 'y': int(pt[0])} for pt in branchpoints],
            'bbox': component_info.get('bbox', {}),
            'centroid': component_info.get('centroid', {})
        }

        return result

    def save_labeled_skeletons(
        self,
        labeled_skeletons: np.ndarray,
        output_path: str
    ) -> None:
        """
        儲存 16-bit 標籤骨架影像

        Args:
            labeled_skeletons: 標籤骨架影像
            output_path: 輸出路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 確保為 16-bit 格式
        labeled_16bit = labeled_skeletons.astype(np.uint16)

        success = cv2.imwrite(str(output_path), labeled_16bit)
        if not success:
            raise IOError(f"無法儲存標籤骨架影像: {output_path}")

        if self.verbose:
            print(f"✓ 標籤骨架影像已儲存: {output_path}")

    def save_metadata(
        self,
        skeletons_info: List[Dict],
        input_dir: str,
        output_path: str
    ) -> None:
        """
        儲存骨架元數據為 JSON

        Args:
            skeletons_info: 骨架資訊列表
            input_dir: 輸入資料夾路徑
            output_path: 輸出 JSON 路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 計算統計資訊
        total_skeleton_length = sum(s['skeleton_length'] for s in skeletons_info)
        total_endpoints = sum(s['num_endpoints'] for s in skeletons_info)
        total_branchpoints = sum(s['num_branchpoints'] for s in skeletons_info)

        metadata = {
            'metadata': {
                'total_components': len(skeletons_info),
                'timestamp': datetime.now().isoformat(),
                'source_components': str(Path(input_dir) / 'components.json'),
                'source_labeled_image': str(Path(input_dir) / 'labeled_components.png'),
                'skeletonization_method': self.method,
                'spur_threshold': self.spur_threshold
            },
            'skeletons': skeletons_info,
            'statistics': {
                'total_skeleton_length': float(total_skeleton_length),
                'avg_skeleton_length': float(total_skeleton_length / len(skeletons_info)) if skeletons_info else 0.0,
                'total_endpoints': int(total_endpoints),
                'total_branchpoints': int(total_branchpoints)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"✓ 骨架元數據已儲存: {output_path}")

    def visualize_skeleton_only(
        self,
        labeled_skeletons: np.ndarray,
        output_path: str
    ) -> None:
        """
        視覺化 1: 純骨架線條（彩色）

        Args:
            labeled_skeletons: 標籤骨架影像
            output_path: 輸出路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use label2rgb to generate pseudo-color
        colored = label2rgb(labeled_skeletons, bg_label=0, bg_color=(0, 0, 0))

        # Plot
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(colored)
        ax.axis('off')
        ax.set_title(
            f'Skeleton Visualization ({labeled_skeletons.max()} components)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ 骨架視覺化已儲存: {output_path}")

    def visualize_keypoints(
        self,
        labeled_skeletons: np.ndarray,
        skeletons_info: List[Dict],
        output_path: str
    ) -> None:
        """
        視覺化 2: 骨架 + 關鍵點（端點、分支點）

        Args:
            labeled_skeletons: 標籤骨架影像
            skeletons_info: 骨架資訊列表
            output_path: 輸出路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate colored skeleton
        colored = label2rgb(labeled_skeletons, bg_label=0, bg_color=(0, 0, 0))

        # Plot
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(colored)

        # Plot endpoints and branchpoints
        for skeleton_info in skeletons_info:
            # Endpoints (red circles)
            for ep in skeleton_info['endpoints']:
                circle = Circle(
                    (ep['x'], ep['y']),
                    radius=3,
                    color='red',
                    fill=True,
                    alpha=0.8
                )
                ax.add_patch(circle)

            # Branchpoints (blue squares)
            for bp in skeleton_info['branchpoints']:
                rect = Rectangle(
                    (bp['x'] - 2, bp['y'] - 2),
                    width=4,
                    height=4,
                    color='blue',
                    fill=True,
                    alpha=0.8
                )
                ax.add_patch(rect)

        ax.axis('off')
        ax.set_title(
            f'Skeleton + Keypoints Visualization',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Endpoints'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                   markersize=8, label='Branchpoints')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ 關鍵點視覺化已儲存: {output_path}")

    def visualize_overlay(
        self,
        labeled_components: np.ndarray,
        labeled_skeletons: np.ndarray,
        skeletons_info: List[Dict],
        output_path: str
    ) -> None:
        """
        視覺化 3: 骨架疊加在原始元件上

        Args:
            labeled_components: 原始標籤元件影像
            labeled_skeletons: 標籤骨架影像
            skeletons_info: 骨架資訊列表
            output_path: 輸出路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate semi-transparent component background
        components_colored = label2rgb(
            labeled_components,
            bg_label=0,
            bg_color=(0, 0, 0),
            alpha=0.3
        )

        # Generate skeleton
        skeletons_colored = label2rgb(
            labeled_skeletons,
            bg_label=0,
            bg_color=(0, 0, 0)
        )

        # Overlay: use skeleton color where skeleton exists, otherwise use component color
        overlay = components_colored.copy()
        skeleton_mask = labeled_skeletons > 0
        overlay[skeleton_mask] = skeletons_colored[skeleton_mask]

        # Plot
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(overlay)

        # Plot keypoints
        for skeleton_info in skeletons_info:
            # Endpoints (red circles)
            for ep in skeleton_info['endpoints']:
                circle = Circle(
                    (ep['x'], ep['y']),
                    radius=3,
                    color='red',
                    fill=True,
                    alpha=0.9
                )
                ax.add_patch(circle)

            # Branchpoints (blue squares)
            for bp in skeleton_info['branchpoints']:
                rect = Rectangle(
                    (bp['x'] - 2, bp['y'] - 2),
                    width=4,
                    height=4,
                    color='blue',
                    fill=True,
                    alpha=0.9
                )
                ax.add_patch(rect)

        ax.axis('off')
        ax.set_title(
            f'Skeleton Overlay Visualization',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Endpoints'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                   markersize=8, label='Branchpoints'),
            Line2D([0], [0], color='gray', linewidth=3, alpha=0.3,
                   label='Original Components')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ 疊加視覺化已儲存: {output_path}")

    def process(
        self,
        input_dir: str,
        output_dir: str,
        visualize_skeleton: bool = False,
        visualize_keypoints: bool = False,
        visualize_overlay: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        完整的骨架化處理流程

        Args:
            input_dir: 連通元件資料夾路徑
            output_dir: 輸出目錄
            visualize_skeleton: 是否生成純骨架視覺化
            visualize_keypoints: 是否生成關鍵點視覺化
            visualize_overlay: 是否生成疊加視覺化

        Returns:
            (標籤骨架影像, 骨架資訊列表)
        """
        if self.verbose:
            print("=" * 60)
            print("骨架化與結構分析")
            print("=" * 60)
            print(f"  骨架化方法: {self.method}")
            print(f"  短刺過濾閾值: {self.spur_threshold} 像素")

        # 1. 載入連通元件
        labeled_components, components = self.load_components(input_dir)

        # 2. 初始化標籤骨架影像
        labeled_skeletons = np.zeros_like(labeled_components, dtype=np.uint16)
        skeletons_info = []

        if self.verbose:
            print(f"\n開始處理 {len(components)} 個元件...")

        # 3. 對每個元件執行骨架化
        for i, component in enumerate(components, 1):
            comp_id = component['id']

            if self.verbose:
                print(f"\n處理元件 {i}/{len(components)} (ID={comp_id})...")

            # 提取該元件的 mask
            mask = (labeled_components == comp_id).astype(np.uint8) * 255

            # 執行骨架化
            skeleton = self.skeletonize_mask(mask)

            if self.verbose:
                skeleton_pixels_before = np.sum(skeleton > 0)
                print(f"  骨架化完成，骨架像素數: {skeleton_pixels_before}")

            # 過濾短刺
            if self.spur_threshold > 0:
                skeleton = self.remove_spurs(skeleton, self.spur_threshold)
                if self.verbose:
                    skeleton_pixels_after = np.sum(skeleton > 0)
                    removed = skeleton_pixels_before - skeleton_pixels_after
                    print(f"  短刺過濾完成，移除 {removed} 像素")

            # 分析骨架結構
            skeleton_info = self.analyze_skeleton(skeleton, component)
            skeletons_info.append(skeleton_info)

            if self.verbose:
                print(f"  骨架長度: {skeleton_info['skeleton_length']:.2f} 像素")
                print(f"  端點數: {skeleton_info['num_endpoints']}")
                print(f"  分支點數: {skeleton_info['num_branchpoints']}")

            # 將骨架加入標籤影像
            labeled_skeletons[skeleton > 0] = comp_id

        # 4. 建立輸出目錄
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 5. 儲存標籤骨架影像
        labeled_skeletons_path = output_dir / 'labeled_skeletons.png'
        self.save_labeled_skeletons(labeled_skeletons, str(labeled_skeletons_path))

        # 6. 儲存元數據
        metadata_path = output_dir / 'skeletons.json'
        self.save_metadata(skeletons_info, input_dir, str(metadata_path))

        # 7. 生成視覺化
        if self.verbose:
            print(f"\n生成視覺化...")

        if visualize_skeleton:
            viz_path = output_dir / 'visualization_skeleton.png'
            self.visualize_skeleton_only(labeled_skeletons, str(viz_path))

        if visualize_keypoints:
            viz_path = output_dir / 'visualization_keypoints.png'
            self.visualize_keypoints(labeled_skeletons, skeletons_info, str(viz_path))

        if visualize_overlay:
            viz_path = output_dir / 'visualization_overlay.png'
            self.visualize_overlay(
                labeled_components,
                labeled_skeletons,
                skeletons_info,
                str(viz_path)
            )

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 骨架化處理完成!")
            print("=" * 60)
            print(f"\n輸出檔案:")
            print(f"  - 標籤骨架影像: {labeled_skeletons_path}")
            print(f"  - 骨架元數據: {metadata_path}")
            if visualize_skeleton:
                print(f"  - 骨架視覺化: {output_dir / 'visualization_skeleton.png'}")
            if visualize_keypoints:
                print(f"  - 關鍵點視覺化: {output_dir / 'visualization_keypoints.png'}")
            if visualize_overlay:
                print(f"  - 疊加視覺化: {output_dir / 'visualization_overlay.png'}")

        return labeled_skeletons, skeletons_info


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='骨架化與結構分析工具 - 對連通元件執行骨架化並分析結構',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用（預設 zhang-suen 方法）
  python %(prog)s -i output/components -o output/skeletons

  # 使用 medial-axis 方法 + 生成所有視覺化
  python %(prog)s -i output/components -o output/skeletons --method medial-axis -v --verbose

  # 不過濾短刺 + 僅生成關鍵點視覺化
  python %(prog)s -i output/components --spur-threshold 0 --viz-keypoints

  # 調整短刺閾值為 5 像素
  python %(prog)s -i output/components --spur-threshold 5 -v

  # 測試不同骨架化方法
  python %(prog)s -i output/components --method guo-hall -v

骨架化方法說明:
  zhang-suen (預設):
    - 使用 scikit-image 的 Zhang-Suen 演算法
    - 標準骨架化方法，結果穩定
    - 單像素寬度骨架

  medial-axis:
    - 使用 scikit-image 的中軸變換
    - 基於距離變換，結果平滑
    - 適合較粗的神經纖維

  guo-hall:
    - 使用 OpenCV 的 Guo-Hall 演算法
    - 需要安裝 opencv-contrib-python
    - 效能較好，適合大影像
    - 若未安裝會自動降級為 zhang-suen

關鍵點檢測:
  - 端點: 鄰居數 = 1 的骨架點
  - 分支點: 鄰居數 >= 3 的骨架點
  - 使用 8-鄰域檢測

短刺過濾:
  - 移除長度 < threshold 的末端分支
  - 迭代過濾直到無短刺
  - 設為 0 則不過濾

輸出說明:
  output_dir/
  ├── labeled_skeletons.png          # 16-bit 標籤骨架影像
  ├── skeletons.json                 # 骨架結構資訊
  ├── visualization_skeleton.png     # 純骨架（--viz-skeleton 或 -v）
  ├── visualization_keypoints.png    # 骨架+關鍵點（--viz-keypoints 或 -v）
  └── visualization_overlay.png      # 疊加視覺化（--viz-overlay 或 -v）

後續使用:
  種子提取腳本可以讀取標籤骨架影像和 JSON：

  import cv2
  import json

  labeled_skeletons = cv2.imread('labeled_skeletons.png', cv2.IMREAD_UNCHANGED)
  with open('skeletons.json') as f:
      data = json.load(f)

  for skeleton_info in data['skeletons']:
      comp_id = skeleton_info['component_id']
      skeleton_mask = (labeled_skeletons == comp_id).astype(np.uint8)
      # 在骨架上進行路徑追蹤和曲率計算...
        """
    )

    # 必填參數
    parser.add_argument(
        '-i', '--input-components',
        type=str,
        required=True,
        help='連通元件資料夾路徑（包含 labeled_components.png 和 components.json）'
    )

    # 選填參數
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./output/skeletons',
        help='輸出目錄（預設: ./output/skeletons）'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['zhang-suen', 'medial-axis', 'guo-hall'],
        default='zhang-suen',
        help='骨架化方法（預設: zhang-suen）'
    )

    parser.add_argument(
        '--spur-threshold',
        type=int,
        default=3,
        metavar='PIXELS',
        help='短刺過濾閾值（像素），設為 0 則不過濾（預設: 3）'
    )

    # 視覺化選項
    viz_group = parser.add_argument_group('視覺化選項')
    viz_group.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='生成所有視覺化圖（skeleton + keypoints + overlay）'
    )
    viz_group.add_argument(
        '--viz-skeleton',
        action='store_true',
        help='僅生成純骨架視覺化'
    )
    viz_group.add_argument(
        '--viz-keypoints',
        action='store_true',
        help='僅生成骨架+關鍵點視覺化'
    )
    viz_group.add_argument(
        '--viz-overlay',
        action='store_true',
        help='僅生成疊加視覺化'
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

    # 參數驗證
    if args.spur_threshold < 0:
        print(f"錯誤: spur_threshold 不能為負數", file=sys.stderr)
        return 1

    # 決定視覺化選項
    visualize_skeleton = args.visualize or args.viz_skeleton
    visualize_keypoints = args.visualize or args.viz_keypoints
    visualize_overlay = args.visualize or args.viz_overlay

    try:
        # 建立骨架分析器
        analyzer = SkeletonAnalyzer(
            method=args.method,
            spur_threshold=args.spur_threshold,
            verbose=args.verbose
        )

        # 執行骨架化
        analyzer.process(
            input_dir=args.input_components,
            output_dir=args.output_dir,
            visualize_skeleton=visualize_skeleton,
            visualize_keypoints=visualize_keypoints,
            visualize_overlay=visualize_overlay
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
