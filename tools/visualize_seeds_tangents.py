#!/usr/bin/env python3
"""
種子與切線視覺化腳本

使用 ComponentAnalyzer 分析影像中的每個組件，
提取種子點和切線方向，
將其繪製在輸入影像的綠色通道上。
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
import logging
from typing import Tuple

from skimage.measure import label, regionprops
from skimage import io
import networkx as nx

# 匯入本專案的模組
from neural_reconstruction.core.construction.component_analyzer import ComponentAnalyzer
from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SeedTangentVisualizer:
    """種子與切線視覺化器"""

    def __init__(
        self,
        segment_length=10.0,
        min_edge_length=10.0,
        tangent_length=20,
        seed_radius=3,
        pipeline_config=None,
        seed_color="yellow",
        tangent_color="green",
    ):
        """
        Args:
            segment_length: 種子間隔長度
            min_edge_length: 最小邊長度
            tangent_length: 切線長度 (像素)
            seed_radius: 種子半徑 (像素)
            pipeline_config: 前處理管道設定
            seed_color: 種子點顏色 ('yellow', 'green', 'red', 'cyan', 'magenta', 'white')
            tangent_color: 切線顏色 ('yellow', 'green', 'red', 'cyan', 'magenta', 'white')
        """
        self.analyzer = ComponentAnalyzer(
            segment_length=segment_length, min_edge_length=min_edge_length
        )
        self.tangent_length = tangent_length
        self.seed_radius = seed_radius

        # 定義顏色映射
        self.color_map = {
            "yellow": (255, 255, 0),  # 黃色
            "green": (0, 255, 0),  # 綠色
            "red": (255, 0, 0),  # 紅色
            "cyan": (0, 255, 255),  # 青色
            "magenta": (255, 0, 255),  # 洋紅色
            "white": (255, 255, 255),  # 白色
        }
        self.seed_color = self.color_map.get(seed_color, self.color_map["yellow"])
        self.tangent_color = self.color_map.get(tangent_color, self.color_map["green"])

        # 初始化前處理管道
        if pipeline_config is None:
            pipeline_config = {
                "morphology": {"closing_kernel": 3, "opening_kernel": 3},
                "mask": {"dilate_offset": 50},
                "background": {
                    "method": "morphology",
                    "radius": 12,
                    "light_background": True,
                },
                "threshold": {"method": "binary"},
            }

        self.pipeline = SkinAnalysisPipeline(pipeline_config)

    def get_edge_tangent(
        self, graph: nx.MultiGraph, u, v, data: dict
    ) -> Tuple[float, float]:
        """
        計算邊的切線方向 (平均方向)

        Args:
            graph: NetworkX graph
            u, v: 邊的端點
            data: 邊的資料

        Returns:
            (angle, magnitude): 角度和大小
        """
        path_coords = data.get("path", [])

        if len(path_coords) < 2:
            # 如果路徑太短，使用端點方向
            y1, x1 = u
            y2, x2 = v
            dy = y2 - y1
            dx = x2 - x1
        else:
            # 使用路徑的平均方向
            path_coords = np.array([u] + [tuple(p) for p in path_coords] + [v])
            # 計算所有段的方向向量
            dys = np.diff(path_coords[:, 0])
            dxs = np.diff(path_coords[:, 1])

            # 取平均方向
            dy = np.mean(dys)
            dx = np.mean(dxs)

        # 計算角度
        angle = np.arctan2(dy, dx)
        magnitude = np.sqrt(dx**2 + dy**2)

        return angle, magnitude

    def draw_seed_and_tangent(
        self,
        output_image: np.ndarray,
        graph: nx.MultiGraph,
        bbox: Tuple[int, int, int, int],
        seed_color=(255, 255, 0),
        tangent_color=(0, 255, 0),
    ):
        """
        在輸出影像上繪製種子點和切線

        Args:
            output_image: 輸出影像 (將在此繪製)，RGB 圖像
            graph: 拓樸圖
            bbox: 組件的邊界框 (minr, minc, maxr, maxc)
            seed_color: 種子點顏色 (RGB 元組)
            tangent_color: 切線顏色 (RGB 元組)
        """
        minr, minc, maxr, maxc = bbox
        height, width = output_image.shape[:2]

        # 繪製邊（切線）
        for u, v, data in graph.edges(data=True):
            y1, x1 = u
            y2, x2 = v

            # 轉換為全局座標
            gy1, gx1 = minr + y1, minc + x1
            gy2, gx2 = minr + y2, minc + x2

            # 檢查邊界
            if not (
                0 <= gy1 < height
                and 0 <= gx1 < width
                and 0 <= gy2 < height
                and 0 <= gx2 < width
            ):
                continue

            # 計算切線方向
            angle, magnitude = self.get_edge_tangent(graph, u, v, data)

            # 如果magnitude太小，跳過
            if magnitude < 0.1:
                continue

            # 從種子點出發繪製切線
            tx = int(np.cos(angle) * self.tangent_length)
            ty = int(np.sin(angle) * self.tangent_length)

            # 繪製切線 (使用 Bresenham 線演算法的簡單實現)
            self._draw_line(
                output_image,
                int(gy1),
                int(gx1),
                int(gy1 + ty),
                int(gx1 + tx),
                tangent_color,
            )

            # 也繪製反向切線
            self._draw_line(
                output_image,
                int(gy1),
                int(gx1),
                int(gy1 - ty),
                int(gx1 - tx),
                tangent_color,
            )

        # 繪製種子點（節點）
        for node in graph.nodes():
            y, x = node

            # 轉換為全局座標
            gy, gx = minr + y, minc + x

            # 檢查邊界
            if not (0 <= gy < height and 0 <= gx < width):
                continue

            # 繪製圓形種子點
            self._draw_circle(
                output_image, int(gy), int(gx), self.seed_radius, seed_color
            )

    def _draw_line(
        self, image: np.ndarray, y1: int, x1: int, y2: int, x2: int, color: Tuple
    ):
        """
        使用 Bresenham 演算法繪製線段
        """
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        sy = 1 if y2 > y1 else -1
        sx = 1 if x2 > x1 else -1
        err = dx - dy

        y, x = y1, x1
        h, w = image.shape[:2]

        while True:
            if 0 <= y < h and 0 <= x < w:
                image[y, x] = color

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _draw_circle(
        self, image: np.ndarray, cy: int, cx: int, radius: int, color: Tuple
    ):
        """
        使用 Bresenham 圓演算法繪製圓形
        """
        h, w = image.shape[:2]

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < h and 0 <= x < w:
                        image[y, x] = color

    def process_image(
        self, image_path, mask_path=None, label_path=None, output_path=None
    ):
        """
        處理影像：套用前處理、分析組件並繪製種子與切線

        Args:
            image_path: 輸入影像路徑
            mask_path: 表皮 mask 路徑 (可選)
            label_path: Label 影像路徑 (可選)
            output_path: 輸出影像路徑

        Returns:
            output_image: 輸出影像
        """
        # 載入影像
        try:
            img = io.imread(image_path)
            logger.info(f"載入影像: {image_path} (形狀: {img.shape})")
        except Exception as e:
            logger.error(f"無法載入影像 {image_path}: {e}")
            return None

        # 確保是灰度圖
        if len(img.shape) == 3:
            # 如果是 RGB，轉為灰度
            img_gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8)
        else:
            img_gray = img

        # 如果提供了 mask 和 label，使用前處理管道
        if mask_path is not None and label_path is not None:
            try:
                mask = io.imread(mask_path)
                label_img = io.imread(label_path)

                # 確保是二值圖
                if mask.max() > 1:
                    mask = (mask > 0).astype(np.uint8) * 255
                if label_img.max() > 1:
                    label_img = (label_img > 0).astype(np.uint8) * 255

                logger.info(f"載入 mask: {mask_path}")
                logger.info(f"載入 label: {label_path}")

                # 套用前處理
                processed_label, roi_image = self.pipeline.run(
                    label_img, mask, img_gray
                )

                logger.info("已套用前處理")
                binary = processed_label

            except Exception as e:
                logger.warning(f"無法使用前處理: {e}，改用原始影像")
                binary = (img_gray > 0).astype(np.uint8) * 255
        else:
            # 沒有 mask 和 label，使用簡單二值化
            binary = (img_gray > 0).astype(np.uint8) * 255

        # 標記連通組件
        labeled_array = label(binary)
        regions = regionprops(labeled_array)

        logger.info(f"找到 {len(regions)} 個組件")

        # 建立輸出影像（RGB）- 將灰階複製到三個通道
        output_image = np.zeros((*img_gray.shape, 3), dtype=np.uint8)
        output_image[:, :, 0] = img_gray  # R 通道
        output_image[:, :, 1] = img_gray  # G 通道
        output_image[:, :, 2] = img_gray  # B 通道

        # 處理每個組件
        for region in regions:
            try:
                # 分析組件
                result = self.analyzer.analyze(region)

                logger.info(
                    f"組件 {result.component_id}: "
                    f"節點數={result.topology.number_of_nodes()}, "
                    f"邊數={result.topology.number_of_edges()}"
                )

                # 繪製種子和切線
                self.draw_seed_and_tangent(
                    output_image,
                    result.topology,
                    result.bbox,
                    seed_color=self.seed_color,
                    tangent_color=self.tangent_color,
                )

            except Exception as e:
                logger.warning(f"處理組件 {region.label} 時出錯: {e}")
                continue

        # 儲存輸出影像
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_seeds_tangents.png"
        else:
            output_path = Path(output_path)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_img = Image.fromarray(output_image)
            output_img.save(output_path)
            logger.info(f"已儲存輸出: {output_path}")
        except Exception as e:
            logger.error(f"無法儲存輸出 {output_path}: {e}")
            return output_image

        return output_image


def main():
    """命令列介面"""
    parser = argparse.ArgumentParser(description="繪製種子點和切線")

    parser.add_argument("-i", "--input", type=str, required=True, help="輸入影像路徑")

    parser.add_argument(
        "-m", "--mask", type=str, default=None, help="表皮 mask 路徑 (可選)"
    )

    parser.add_argument(
        "-l", "--label", type=str, default=None, help="Label 影像路徑 (可選)"
    )

    parser.add_argument("-o", "--output", type=str, default=None, help="輸出影像路徑")

    parser.add_argument(
        "-s", "--segment-length", type=float, default=10.0, help="種子間隔長度"
    )

    parser.add_argument(
        "-e", "--min-edge-length", type=float, default=0.0, help="最小邊長度"
    )

    parser.add_argument(
        "-t", "--tangent-length", type=int, default=10, help="切線長度 (像素)"
    )

    parser.add_argument(
        "-r", "--seed-radius", type=int, default=1, help="種子半徑 (像素)"
    )

    parser.add_argument(
        "-sc",
        "--seed-color",
        type=str,
        default="yellow",
        choices=["yellow", "green", "red", "cyan", "magenta", "white"],
        help="種子點顏色 (預設: yellow)",
    )

    parser.add_argument(
        "-tc",
        "--tangent-color",
        type=str,
        default="green",
        choices=["yellow", "green", "red", "cyan", "magenta", "white"],
        help="切線顏色 (預設: green)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"輸入檔案不存在: {input_path}")
        return

    visualizer = SeedTangentVisualizer(
        segment_length=args.segment_length,
        min_edge_length=args.min_edge_length,
        tangent_length=args.tangent_length,
        seed_radius=args.seed_radius,
        seed_color=args.seed_color,
        tangent_color=args.tangent_color,
    )

    visualizer.process_image(input_path, args.mask, args.label, args.output)


if __name__ == "__main__":
    main()
