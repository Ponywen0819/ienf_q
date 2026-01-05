"""
MST Neural Reconstruction Visualization Module
MST 神經重構視覺化模組

This module visualizes the final MST (Minimum Spanning Tree) reconstruction results
from the neural reconstruction pipeline.

Main Features:
1. Visualizes only the final MST edges (filtered connections)
2. Colors connections by tree membership in the MST forest
3. Shows seed points at connection endpoints
4. Displays MST statistics overlay
5. Component labels at centroids

Usage:
    python visualization/visualize_mst_reconstruction.py \\
        --annotation closing_3.png \\
        --green-channel split/S163-2_a_epidermis_correct_12.png

    # With custom configuration
    python visualization/visualize_mst_reconstruction.py \\
        --annotation closing_3.png \\
        --green-channel split/S163-2_a_epidermis_correct_12.png \\
        --config config/high_quality.yaml \\
        --output output/mst_visualization/mst_reconstruction.png

Outputs:
- mst_reconstruction.png: MST visualization on green channel
- mst_statistics.txt: Statistical summary
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import cv2
import numpy as np
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional
import colorsys
import networkx as nx

# Import pipeline components
from nueral_reconstruction.pipeline import NeuralReconstructionPipeline
from nueral_reconstruction.config_loader import load_config
from preprocessing import SkinAnalysisPipeline


# ========================================
# 影像路徑設定
# ========================================
IMAGE_ID = 'S163-2_a'
LABEL_IMAGE_PATH = f'data/Label/{IMAGE_ID}.tif'
EPIDERMIS_MASK_PATH = f'data/Mask/{IMAGE_ID}.tif'
ORIGINAL_IMAGE_PATH = f'data/Original/{IMAGE_ID}.tif'
# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions - Color Generation
# ============================================================================

def generate_tree_colors(num_trees: int) -> List[Tuple[int, int, int]]:
    """
    為 MST 森林中的每棵樹生成唯一顏色

    使用 HSV 色彩空間循環色相,生成視覺上易區分的顏色

    Args:
        num_trees: 樹的數量

    Returns:
        colors: BGR 格式的顏色列表
    """
    colors = []
    for i in range(num_trees):
        # HSV: H 在 0-1 範圍, S=1, V=1
        hue = i / max(num_trees, 1)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # 轉換為 BGR (OpenCV 格式) 並縮放到 0-255
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def draw_components_overlay(
    base_image: np.ndarray,
    components: List[Dict],
    colors: List[Tuple[int, int, int]],
    alpha: float = 0.3
) -> np.ndarray:
    """
    在底圖上繪製組件疊加層

    Args:
        base_image: 底圖 (BGR 格式)
        components: 組件列表,每個包含 'id', 'mask'
        colors: 組件顏色列表
        alpha: 疊加層透明度

    Returns:
        overlaid_image: 疊加後的影像
    """
    overlay = base_image.copy()

    for comp in components:
        comp_id = comp['id']
        mask = comp.get('mask')

        if mask is None:
            logger.warning(f"Component {comp_id} has no mask, skipping overlay")
            continue

        color = colors[comp_id % len(colors)]

        # 創建彩色疊加層
        colored_mask = np.zeros_like(base_image)
        colored_mask[mask > 0] = color

        # Alpha 混合
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

    return overlay


def get_tree_membership(mst_forest: nx.Graph) -> Dict[int, int]:
    """
    識別 MST 森林中每個節點所屬的樹

    Args:
        mst_forest: NetworkX MST 森林圖

    Returns:
        node_to_tree: 節點ID -> 樹ID 的映射
    """
    node_to_tree = {}
    trees = list(nx.connected_components(mst_forest))

    for tree_id, tree_nodes in enumerate(trees):
        for node in tree_nodes:
            node_to_tree[node] = tree_id

    return node_to_tree


# ============================================================================
# Core Drawing Functions
# ============================================================================

def draw_mst_path(
    image: np.ndarray,
    path: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = 2
) -> None:
    """
    在圖像上繪製 MST 路徑

    Args:
        image: 要繪製的圖像 (會被修改)
        path: 路徑座標列表 [(y, x), ...]
        color: BGR 顏色
        thickness: 線條粗細
    """
    if len(path) < 2:
        return

    # 轉換 (y, x) 為 (x, y) for OpenCV
    path_points = [(int(pt[1]), int(pt[0])) for pt in path]

    # 繪製路徑線段
    for i in range(len(path_points) - 1):
        cv2.line(
            image,
            path_points[i],
            path_points[i + 1],
            color,
            thickness,
            cv2.LINE_AA  # 抗鋸齒
        )


def draw_seed_markers(
    image: np.ndarray,
    seed_a: Dict,
    seed_b: Dict,
    radius: int = 5
) -> None:
    """
    在圖像上繪製種子點標記

    Args:
        image: 要繪製的圖像 (會被修改)
        seed_a: 起點種子 {'position': (y, x), ...}
        seed_b: 終點種子 {'position': (y, x), ...}
        radius: 圓點半徑
    """
    # 起點: 綠色
    pos_a = seed_a['position']
    cv2.circle(
        image,
        (int(pos_a[1]), int(pos_a[0])),  # (x, y)
        radius,
        (0, 255, 0),  # Green
        -1  # 填充
    )

    # 終點: 紅色
    pos_b = seed_b['position']
    cv2.circle(
        image,
        (int(pos_b[1]), int(pos_b[0])),  # (x, y)
        radius,
        (0, 0, 255),  # Red
        -1  # 填充
    )


def draw_component_labels(
    image: np.ndarray,
    components: List[Dict],
    font_scale: float = 0.5,
    thickness: int = 1
) -> None:
    """
    在組件質心位置繪製標籤

    Args:
        image: 要繪製的圖像 (會被修改)
        components: 組件列表
        font_scale: 字體大小
        thickness: 字體粗細
    """
    for comp in components:
        comp_id = comp['id']

        # 優先使用 seeds 計算質心（與參考實現一致，seeds 已經是全局座標）
        if 'seeds' in comp and comp['seeds']:
            # 從 seed 位置計算質心
            positions = np.array([s['position'] for s in comp['seeds']])
            centroid = positions.mean(axis=0).astype(int)
            # centroid 是 (row, col) = (y, x)，轉換為 OpenCV 的 (x, y)
            position = (int(centroid[1]), int(centroid[0]))
        elif 'centroid' in comp:
            centroid = comp['centroid']
            # centroid 是 (row, col) = (y, x)，轉換為 OpenCV 的 (x, y)
            position = (int(centroid[1]), int(centroid[0]))
        elif 'region' in comp and hasattr(comp['region'], 'centroid'):
            centroid = comp['region'].centroid
            # region.centroid 是 (row, col) = (y, x)，轉換為 OpenCV 的 (x, y)
            position = (int(centroid[1]), int(centroid[0]))
        else:
            continue

        # 繪製文字
        label = f"{comp_id}"
        cv2.putText(
            image,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White
            thickness,
            cv2.LINE_AA
        )


def add_mst_statistics_overlay(
    image: np.ndarray,
    mst_stage: Dict,
    node_to_tree: Dict[int, int]
) -> None:
    """
    在圖像上添加 MST 統計信息疊加層

    Args:
        image: 要繪製的圖像 (會被修改)
        mst_stage: MST 階段結果數據
        node_to_tree: 節點到樹的映射
    """
    h, w = image.shape[:2]

    # 創建半透明背景
    overlay = image.copy()
    box_height = 180
    box_width = 350
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + box_width, 10 + box_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # 準備統計文字
    num_trees = len(set(node_to_tree.values()))
    stats_text = [
        f"MST Reconstruction Statistics",
        f"=" * 40,
        f"MST Edges: {mst_stage['num_mst_edges']}",
        f"MST Nodes: {mst_stage['num_mst_nodes']}",
        f"Trees in Forest: {num_trees}",
        f"Components: {mst_stage['num_connected_components']}",
        f"",
        f"Connections Kept: {mst_stage['connections_kept']}",
        f"Connections Removed: {mst_stage['connections_removed']}",
    ]

    # 繪製文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20

    for i, text in enumerate(stats_text):
        y = 30 + i * line_height
        cv2.putText(
            image,
            text,
            (20, y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )


# ============================================================================
# Main Visualization Function
# ============================================================================

def visualize_mst_reconstruction(
    pipeline_results: Dict,
    green_channel_path: str,
    output_path: str,
    show_components: bool = True,
    component_alpha: float = 0.3,
    show_seeds: bool = True,
    show_component_labels: bool = True,
    line_thickness: int = 2,
    seed_radius: int = 5
) -> None:
    """
    視覺化 MST 重構結果

    Args:
        pipeline_results: Pipeline 完整結果
        green_channel_path: 綠通道圖像路徑
        output_path: 輸出圖像路徑
        show_components: 是否顯示組件疊加層
        component_alpha: 組件疊加層透明度 (0.0-1.0)
        show_seeds: 是否顯示種子點
        show_component_labels: 是否顯示組件標籤
        line_thickness: 線條粗細
        seed_radius: 種子點半徑
    """
    logger.info("開始生成 MST 重構視覺化...")

    # 1. 提取 MST 數據
    mst_stage = pipeline_results['stages']['mst_reconstruction']
    mst_edges = mst_stage['mst_with_paths']['edges']
    mst_forest = mst_stage['mst_forest']

    logger.info(f"MST 邊數: {len(mst_edges)}")
    logger.info(f"MST 節點數: {mst_stage['num_mst_nodes']}")

    # 2. 載入綠通道圖像並應用 viridis colormap
    logger.info(f"載入綠通道圖像: {green_channel_path}")
    green_channel = cv2.imread(green_channel_path, cv2.IMREAD_GRAYSCALE)

    if green_channel is None:
        raise ValueError(f"無法載入綠通道圖像: {green_channel_path}")

    # 正規化並應用 viridis colormap
    normalized = cv2.normalize(green_channel, None, 0, 255, cv2.NORM_MINMAX)
    output_image = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)

    logger.info(f"圖像尺寸: {output_image.shape}")

    # 3. 繪製組件疊加層（在 MST 邊之前）
    if show_components:
        logger.info("繪製組件疊加層...")
        regions = pipeline_results['stages']['connected_components']['regions']
        image_shape = output_image.shape[:2]

        # 為每個組件創建 mask
        components_data = []
        for region in regions:
            mask = np.zeros(image_shape, dtype=np.uint8)
            coords = region.coords
            mask[coords[:, 0], coords[:, 1]] = 255

            components_data.append({
                'id': region.label,
                'mask': mask
            })

        # 生成組件顏色
        component_colors = generate_tree_colors(len(components_data))

        # 繪製組件疊加層
        output_image = draw_components_overlay(
            output_image,
            components_data,
            component_colors,
            component_alpha
        )

        logger.info(f"已繪製 {len(components_data)} 個組件")

    # 4. 識別 MST 森林中的樹並分配顏色
    node_to_tree = get_tree_membership(mst_forest)
    num_trees = len(set(node_to_tree.values()))
    tree_colors = generate_tree_colors(num_trees)

    logger.info(f"MST 森林包含 {num_trees} 棵樹")

    # 5. 繪製所有 MST 邊
    logger.info("繪製 MST 邊...")
    for edge in mst_edges:
        comp_a_id = edge['component_a_id']
        comp_b_id = edge['component_b_id']
        path = edge['path']

        # 根據起點組件所屬的樹選擇顏色
        tree_id = node_to_tree.get(comp_a_id, 0)
        color = tree_colors[tree_id % len(tree_colors)]

        # 繪製路徑
        draw_mst_path(output_image, path, color, line_thickness)

        # 繪製種子點
        if show_seeds:
            seed_a, seed_b = edge['seed_pair']
            draw_seed_markers(output_image, seed_a, seed_b, seed_radius)

    # 6. 繪製組件標籤
    if show_component_labels:
        logger.info("繪製組件標籤...")
        regions = pipeline_results['stages']['connected_components']['regions']
        all_seeds = pipeline_results['stages']['topology_and_seeds']['seeds']

        # 將 regions 轉換為適合 draw_component_labels 的格式，包含 seeds
        components_for_labels = []
        for region in regions:
            comp_id = region.label
            # 找到屬於這個組件的所有 seeds
            component_seeds = [
                s for s in all_seeds
                if s.get('component_id') == comp_id
            ]
            components_for_labels.append({
                'id': comp_id,
                'region': region,
                'seeds': component_seeds
            })

        draw_component_labels(output_image, components_for_labels)

    # 7. 添加統計信息疊加層
    logger.info("添加統計信息...")
    add_mst_statistics_overlay(output_image, mst_stage, node_to_tree)

    # 8. 保存圖像
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(output_path, output_image)
    logger.info(f"視覺化已保存到: {output_path}")

    # 8. 保存統計信息到文本文件
    stats_path = output_path.replace('.png', '_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("MST Reconstruction Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"MST Edges: {mst_stage['num_mst_edges']}\n")
        f.write(f"MST Nodes: {mst_stage['num_mst_nodes']}\n")
        f.write(f"Trees in Forest: {num_trees}\n")
        f.write(f"Connected Components: {mst_stage['num_connected_components']}\n")
        f.write(f"\nConnections Kept: {mst_stage['connections_kept']}\n")
        f.write(f"Connections Removed: {mst_stage['connections_removed']}\n")

        # 樹的詳細信息
        f.write(f"\nTree Details:\n")
        f.write("-" * 50 + "\n")
        trees = list(nx.connected_components(mst_forest))
        for i, tree_nodes in enumerate(trees):
            f.write(f"Tree {i}: {len(tree_nodes)} components\n")
            f.write(f"  Components: {sorted(tree_nodes)}\n")

    logger.info(f"統計信息已保存到: {stats_path}")


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize MST neural reconstruction results'
    )

    # Required arguments
    parser.add_argument(
        '--annotation',
        type=str,
        required=True,
        help='Path to annotation image (e.g., closing_3.png)'
    )
    parser.add_argument(
        '--green-channel',
        type=str,
        required=True,
        help='Path to green channel image'
    )

    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file (default: config/default.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/mst_visualization/mst_reconstruction.png',
        help='Output image path (default: output/mst_visualization/mst_reconstruction.png)'
    )
    parser.add_argument(
        '--no-seeds',
        action='store_true',
        help='Do not show seed markers'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Do not show component labels'
    )
    parser.add_argument(
        '--no-components',
        action='store_true',
        help='Do not show component overlays'
    )
    parser.add_argument(
        '--component-alpha',
        type=float,
        default=0.3,
        help='Component overlay transparency (0.0-1.0, default: 0.3)'
    )
    parser.add_argument(
        '--line-thickness',
        type=int,
        default=2,
        help='Line thickness for paths (default: 2)'
    )
    parser.add_argument(
        '--seed-radius',
        type=int,
        default=5,
        help='Radius for seed markers (default: 5)'
    )

    args = parser.parse_args()

    try:
        # 1. 載入配置
        logger.info(f"載入配置: {args.config}")
        config = load_config(args.config)

        # 2. 運行 pipeline
        logger.info("運行神經重構 pipeline...")
        pipeline = NeuralReconstructionPipeline(config=config)

        # 計算輸出目錄
        output_dir = str(Path(args.output).parent)

        results = pipeline.run(
            input_image_path=args.annotation,
            green_channel_path=args.green_channel,
            output_dir=output_dir
        )

        # 檢查 MST 階段是否成功
        if results['stages']['mst_reconstruction']['status'] != 'completed':
            logger.error("MST 重構階段未完成")
            return 1

        # 3. 生成視覺化
        visualize_mst_reconstruction(
            pipeline_results=results,
            green_channel_path=args.green_channel,
            output_path=args.output,
            show_components=not args.no_components,
            component_alpha=args.component_alpha,
            show_seeds=not args.no_seeds,
            show_component_labels=not args.no_labels,
            line_thickness=args.line_thickness,
            seed_radius=args.seed_radius
        )

        logger.info("MST 視覺化完成!")
        return 0

    except Exception as e:
        logger.error(f"錯誤: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    # Example usage - direct execution
    annotation_path = "/Users/ponywen/projects/ienf_q/output/preprocessing/final_label.png"
    green_channel_path = "/Users/ponywen/projects/ienf_q/output/preprocessing/roi_image.png"
    config_path = "config/default.yaml"
    output_path = "output/mst_visualization/mst_reconstruction.png"

    try:
        # 1. 載入配置
        logger.info(f"載入配置: {config_path}")
        config = load_config(config_path)

        # 2. 運行 pipeline
        logger.info("運行神經重構 pipeline...")
        pipeline = NeuralReconstructionPipeline(config=config)


        PIPELINE_CONFIG = {
            'morphology': {
                'closing_kernel': 3,
                'opening_kernel': 3
            },
            'mask': {
                'dilate_offset': 100  # 真皮區域向下延伸的像素數
            },
            'background': {
                'method': 'rolling_ball',
                'radius': 2,
                'light_background': False
            },
            'threshold': {
                'method': 'binary'
            },
            'normalization': {
                'enabled': True      # 是否啟用區域正規化
            }
        }

        label_image = cv2.imread(LABEL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        epidermis_mask = cv2.imread(EPIDERMIS_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

        original_green = original_image[:, :, 1]

        preprocessing = SkinAnalysisPipeline(PIPELINE_CONFIG)
        final_label, roi_image = preprocessing.run(
            label_image,
            epidermis_mask,
            original_green,
            debug=False
        )
        results = pipeline.run(
            input_image=final_label,
            green_image=roi_image,
            output_dir="output/mst_visualization"
        )

        # 檢查 MST 階段是否成功
        if results['stages']['mst_reconstruction']['status'] != 'completed':
            logger.error("MST 重構階段未完成")
            sys.exit(1)

        # 3. 生成視覺化
        visualize_mst_reconstruction(
            pipeline_results=results,
            green_channel_path=green_channel_path,
            output_path=output_path,
            show_components=True,
            component_alpha=0.3,
            show_seeds=False,
            show_component_labels=False,
            line_thickness=1,
            seed_radius=1
        )

        logger.info("MST 視覺化完成!")

    except Exception as e:
        logger.error(f"錯誤: {e}", exc_info=True)
        sys.exit(1)
