"""
Component Pairing Visualization Module
組件配對視覺化模組

This module visualizes the component pairing analysis results from the neural reconstruction pipeline.

Main Features:
1. Self-contained pipeline execution - only needs annotation and green channel images
2. Automatic component pairing analysis
3. Visualization of successful connections (green solid lines)
4. Visualization of rejected connections with reasons (colored dashed lines)
5. Statistical charts and analysis
6. Component color-coding for easy identification

Usage:
    # Basic usage - only annotation and green channel required
    python visualization/visualize_component_pairing.py \\
        --annotation closing_3.png \\
        --green-channel split/S163-2_a_epidermis_correct_12.png

    # With custom configuration
    python visualization/visualize_component_pairing.py \\
        --annotation closing_3.png \\
        --green-channel split/S163-2_a_epidermis_correct_12.png \\
        --config config/high_quality.yaml

Outputs:
- pairing_overview.png: Complete visualization with all connections
- successful_connections.png: Only successful connections
- rejected_connections.png: Only rejected connections
- pairing_statistics.png: Statistical analysis charts
- pairing_results.json: Detailed pairing data

Legacy Mode:
    For backwards compatibility, you can still provide pre-computed JSON files:
    python visualization/visualize_component_pairing.py \\
        --pairing-results output/pairing_results.json \\
        --components-data output/components_data.json \\
        --green-channel split/S163-2_a_epidermis_correct_12.png
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional
import colorsys

# Import pipeline components
from nueral_reconstruction.pipeline import NeuralReconstructionPipeline
from nueral_reconstruction.config_loader import load_config

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Color Definitions (BGR format for OpenCV)
# ============================================================================

COLORS = {
    'successful': (0, 255, 0),        # Green - 成功連接
    'distance_too_far': (0, 0, 255),  # Red - 距離太遠
    'cost_exceeds_threshold': (0, 165, 255),  # Orange - 成本過高
    'no_valid_path': (0, 255, 255),   # Yellow - 無有效路徑
    'no_seeds': (128, 128, 128),      # Gray - 無種子點
}

REJECTION_REASON_NAMES = {
    'distance_too_far': '距離太遠 (Distance Too Far)',
    'cost_exceeds_threshold': '成本過高 (Cost Exceeds Threshold)',
    'no_valid_path': '無有效路徑 (No Valid Path)',
    'no_seeds': '無種子點 (No Seeds)',
}


# ============================================================================
# Helper Functions - Color Generation
# ============================================================================

def generate_component_colors(num_components: int) -> List[Tuple[int, int, int]]:
    """
    生成不同組件的唯一顏色

    使用 HSV 色彩空間循環色相,生成視覺上易區分的顏色

    Args:
        num_components: 組件數量

    Returns:
        colors: BGR 格式的顏色列表
    """
    colors = []
    for i in range(num_components):
        # HSV: H 在 0-1 範圍, S=1, V=1
        hue = i / max(num_components, 1)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # 轉換為 BGR (OpenCV 格式) 並縮放到 0-255
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


# ============================================================================
# Core Drawing Functions
# ============================================================================

def draw_components_overlay(
    base_image: np.ndarray,
    components: List[Dict],
    colors: List[Tuple[int, int, int]],
    alpha: float = 0.5
) -> np.ndarray:
    """
    在底圖上繪製組件疊加層

    Args:
        base_image: 底圖 (BGR 格式)
        components: 組件列表,每個包含 'id', 'mask', 'seeds' 等
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


def draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10
) -> None:
    """
    繪製虛線

    Args:
        img: 影像
        pt1: 起點 (x, y)
        pt2: 終點 (x, y)
        color: 顏色 (BGR)
        thickness: 線條粗細
        dash_length: 虛線段長度(像素)
    """
    dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    if dist < 1:
        return

    num_dashes = int(dist / dash_length)
    if num_dashes == 0:
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return

    for i in range(0, num_dashes, 2):  # 每隔一段繪製
        start_ratio = i / num_dashes
        end_ratio = min((i + 1) / num_dashes, 1.0)

        start = (
            int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
            int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
        )
        end = (
            int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
            int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
        )

        cv2.line(img, start, end, color, thickness, cv2.LINE_AA)


def draw_connection_path(
    img: np.ndarray,
    path: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dashed: bool = False
) -> None:
    """
    繪製連接路徑

    Args:
        img: 影像
        path: 路徑座標列表 [(y, x), ...]
        color: 顏色 (BGR)
        thickness: 線條粗細
        dashed: 是否繪製虛線
    """
    if len(path) < 2:
        return

    for i in range(len(path) - 1):
        pt1 = (path[i][1], path[i][0])  # (x, y)
        pt2 = (path[i+1][1], path[i+1][0])

        if dashed:
            draw_dashed_line(img, pt1, pt2, color, thickness)
        else:
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_connections(
    img: np.ndarray,
    connections: List[Dict],
    connection_type: str,
    line_thickness: int = 2
) -> None:
    """
    繪製連接線

    Args:
        img: 影像
        connections: 連接列表
        connection_type: 'successful' 或 rejection reason
        line_thickness: 線條粗細
    """
    color = COLORS.get(connection_type, (255, 255, 255))
    is_dashed = (connection_type != 'successful')

    for conn in connections:
        path = conn.get('path')
        if path is None or len(path) < 2:
            continue

        draw_connection_path(img, path, color, line_thickness, dashed=is_dashed)


def draw_seeds(
    img: np.ndarray,
    seed_pairs: List[Tuple[Dict, Dict]],
    seed_radius: int = 5
) -> None:
    """
    繪製種子點

    Args:
        img: 影像
        seed_pairs: 種子對列表 [(seed_a, seed_b), ...]
        seed_radius: 種子點半徑
    """
    for seed_a, seed_b in seed_pairs:
        if seed_a is None or seed_b is None:
            continue

        pos_a = seed_a.get('position')
        pos_b = seed_b.get('position')

        if pos_a:
            # 起點: 綠色圓圈
            cv2.circle(img, (pos_a[1], pos_a[0]), seed_radius, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (pos_a[1], pos_a[0]), seed_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

        if pos_b:
            # 終點: 紅色圓圈
            cv2.circle(img, (pos_b[1], pos_b[0]), seed_radius, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (pos_b[1], pos_b[0]), seed_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)


def draw_component_labels(
    img: np.ndarray,
    components: List[Dict],
    colors: List[Tuple[int, int, int]]
) -> None:
    """
    在組件中心繪製 ID 標籤

    Args:
        img: 影像
        components: 組件列表
        colors: 組件顏色列表
    """
    for comp in components:
        comp_id = comp['id']
        seeds = comp.get('seeds', [])

        if not seeds:
            continue

        # 計算組件種子的質心作為標籤位置
        positions = np.array([s['position'] for s in seeds])
        centroid = positions.mean(axis=0).astype(int)

        label = f"C{comp_id}"
        position = (centroid[1], centroid[0])  # (x, y)

        # 繪製文字(白色背景 + 組件顏色文字)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # 獲取文字大小
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # 繪製背景框
        bg_pt1 = (position[0] - 3, position[1] - text_h - 3)
        bg_pt2 = (position[0] + text_w + 3, position[1] + baseline + 3)
        cv2.rectangle(img, bg_pt1, bg_pt2, (255, 255, 255), -1)

        # 繪製文字
        color = colors[comp_id % len(colors)]
        cv2.putText(img, label, position, font, font_scale, color, font_thickness, cv2.LINE_AA)


def add_legend_and_stats(
    img: np.ndarray,
    stats: Dict,
    position: str = 'top_right'
) -> np.ndarray:
    """
    添加圖例和統計信息

    Args:
        img: 影像
        stats: 統計字典
        position: 位置 ('top_right', 'top_left', etc.)

    Returns:
        img_with_legend: 添加圖例後的影像
    """
    h, w = img.shape[:2]

    # 創建圖例區域
    legend_width = 350
    legend_height = 300
    legend_bg = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

    # 位置計算
    if position == 'top_right':
        y_offset = 10
        x_offset = w - legend_width - 10
    elif position == 'top_left':
        y_offset = 10
        x_offset = 10
    else:
        y_offset = 10
        x_offset = w - legend_width - 10

    # 繪製標題
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    y = 25
    cv2.putText(legend_bg, "Connection Types:", (10, y), font, font_scale, (0, 0, 0), font_thickness)

    # 成功連接
    y += 30
    cv2.line(legend_bg, (10, y), (40, y), COLORS['successful'], 2)
    cv2.putText(legend_bg, f"Successful: {stats.get('num_successful', 0)}",
                (50, y+5), font, font_scale, (0, 0, 0), font_thickness)

    # 拒絕原因
    y += 25
    cv2.putText(legend_bg, "Rejected:", (10, y), font, font_scale, (0, 0, 0), font_thickness)

    for reason in ['distance_too_far', 'cost_exceeds_threshold', 'no_valid_path', 'no_seeds']:
        y += 25
        color = COLORS[reason]

        # 虛線樣式
        draw_dashed_line(legend_bg, (10, y), (40, y), color, 2, 5)

        count = stats.get(f'rejected_{reason}', 0)
        name = REJECTION_REASON_NAMES[reason]
        cv2.putText(legend_bg, f"{name}: {count}",
                    (50, y+5), font, 0.4, (0, 0, 0), font_thickness)

    # 總計
    y += 30
    cv2.line(legend_bg, (10, y), (legend_width - 10, y), (0, 0, 0), 1)
    y += 20
    cv2.putText(legend_bg, f"Total Components: {stats.get('num_components', 0)}",
                (10, y), font, font_scale, (0, 0, 0), font_thickness)
    y += 20
    cv2.putText(legend_bg, f"Total Pairs Analyzed: {stats.get('num_pairs_analyzed', 0)}",
                (10, y), font, font_scale, (0, 0, 0), font_thickness)

    # 添加邊框
    cv2.rectangle(legend_bg, (0, 0), (legend_width-1, legend_height-1), (0, 0, 0), 2)

    # 疊加到影像上
    result = img.copy()
    roi = result[y_offset:y_offset+legend_height, x_offset:x_offset+legend_width]

    # Alpha 混合
    alpha = 0.9
    cv2.addWeighted(legend_bg, alpha, roi, 1-alpha, 0, roi)

    return result


# ============================================================================
# Main Visualization Function
# ============================================================================

def visualize_component_pairing_cv2(
    pairing_results: Dict,
    components_data: List[Dict],
    green_channel: np.ndarray,
    output_path: str,
    show_successful: bool = True,
    show_rejected: bool = True,
    show_components: bool = True,
    show_seeds: bool = True,
    show_labels: bool = True,
    show_legend: bool = True,
    component_alpha: float = 0.3,
    line_thickness: int = 2,
    seed_radius: int = 5
) -> None:
    """
    使用 OpenCV 視覺化組件配對結果

    Args:
        pairing_results: 配對分析結果字典
        components_data: 組件數據列表
        green_channel: 綠色通道影像
        output_path: 輸出路徑
        show_successful: 顯示成功連接
        show_rejected: 顯示拒絕連接
        show_components: 顯示組件顏色
        show_seeds: 顯示種子點
        show_labels: 顯示組件標籤
        show_legend: 顯示圖例
        component_alpha: 組件疊加透明度
        line_thickness: 連接線粗細
        seed_radius: 種子點半徑
    """
    logger.info(f"Creating component pairing visualization...")

    # 準備底圖
    if len(green_channel.shape) == 2:
        # 灰度圖轉 BGR
        base_image = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2BGR)
    else:
        base_image = green_channel.copy()

    # 應用 viridis colormap (類似現有視覺化風格)
    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY) if len(base_image.shape) == 3 else base_image
    colored_bg = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)

    # 生成組件顏色
    num_components = pairing_results.get('num_components', len(components_data))
    component_colors = generate_component_colors(num_components)

    # 繪製組件疊加層
    if show_components:
        colored_bg = draw_components_overlay(colored_bg, components_data, component_colors, component_alpha)

    # 分類連接
    successful_connections = []
    rejected_by_reason = {
        'distance_too_far': [],
        'cost_exceeds_threshold': [],
        'no_valid_path': [],
        'no_seeds': []
    }

    all_results = pairing_results.get('all_pair_results', [])

    for result in all_results:
        if result['should_connect']:
            successful_connections.append(result)
        else:
            reason = result.get('skipped_reason')
            if reason in rejected_by_reason:
                rejected_by_reason[reason].append(result)

    # 繪製被拒絕的連接(先畫,在底層)
    if show_rejected:
        for reason, connections in rejected_by_reason.items():
            if connections:
                logger.info(f"Drawing {len(connections)} rejected connections: {reason}")
                draw_connections(colored_bg, connections, reason, line_thickness)

    # 繪製成功的連接(後畫,在上層)
    if show_successful:
        logger.info(f"Drawing {len(successful_connections)} successful connections")
        draw_connections(colored_bg, successful_connections, 'successful', line_thickness)

    # 繪製種子點
    if show_seeds:
        seed_pairs = []
        for conn in (successful_connections if show_successful else []) + \
                    (sum(rejected_by_reason.values(), []) if show_rejected else []):
            pair = conn.get('seed_pair')
            if pair:
                seed_pairs.append(pair)

        if seed_pairs:
            logger.info(f"Drawing {len(seed_pairs)} seed pairs")
            draw_seeds(colored_bg, seed_pairs, seed_radius)

    # 繪製組件標籤
    if show_labels:
        draw_component_labels(colored_bg, components_data, component_colors)

    # 添加圖例和統計
    if show_legend:
        stats = {
            'num_components': num_components,
            'num_pairs_analyzed': pairing_results.get('num_pairs_analyzed', 0),
            'num_successful': len(successful_connections),
            'rejected_distance_too_far': len(rejected_by_reason['distance_too_far']),
            'rejected_cost_exceeds_threshold': len(rejected_by_reason['cost_exceeds_threshold']),
            'rejected_no_valid_path': len(rejected_by_reason['no_valid_path']),
            'rejected_no_seeds': len(rejected_by_reason['no_seeds']),
        }
        colored_bg = add_legend_and_stats(colored_bg, stats)

    # 保存影像
    cv2.imwrite(output_path, colored_bg)
    logger.info(f"Visualization saved to: {output_path}")


# ============================================================================
# Statistical Charts
# ============================================================================

def create_pairing_statistics(
    pairing_results: Dict,
    output_path: str
) -> None:
    """
    創建配對統計圖表

    Args:
        pairing_results: 配對分析結果
        output_path: 輸出路徑
    """
    logger.info("Creating pairing statistics charts...")

    all_results = pairing_results.get('all_pair_results', [])

    # 分類統計
    successful = [r for r in all_results if r['should_connect']]
    rejected = [r for r in all_results if not r['should_connect']]

    # 按拒絕原因分組
    rejection_counts = {}
    for r in rejected:
        reason = r.get('skipped_reason', 'unknown')
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    # 創建圖表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Component Pairing Analysis Statistics', fontsize=16, fontweight='bold')

    # 1. 成功/拒絕比例餅圖
    ax = axes[0, 0]
    labels = ['Successful', 'Rejected']
    sizes = [len(successful), len(rejected)]
    colors_pie = ['#90EE90', '#FFB3B3']
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Connection Success Rate\n(Total: {len(all_results)} pairs)')

    # 2. 拒絕原因分布
    ax = axes[0, 1]
    if rejection_counts:
        reasons = list(rejection_counts.keys())
        counts = list(rejection_counts.values())
        reason_labels = [REJECTION_REASON_NAMES.get(r, r) for r in reasons]
        colors_bar = [COLORS.get(r, (128, 128, 128)) for r in reasons]
        colors_bar_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors_bar]  # BGR to RGB

        ax.bar(range(len(reasons)), counts, color=colors_bar_rgb)
        ax.set_xticks(range(len(reasons)))
        ax.set_xticklabels(reason_labels, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Rejection Reason Distribution')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No rejections', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Rejection Reason Distribution')

    # 3. 成本分布直方圖
    ax = axes[0, 2]
    # 過濾掉 None 和 inf 值
    costs = [r['cost'] for r in all_results if r.get('cost') is not None and r['cost'] != float('inf') and np.isfinite(r['cost'])]
    if costs:
        ax.hist(costs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(costs), color='red', linestyle='--', label=f'Mean: {np.mean(costs):.2f}')
        ax.set_xlabel('Path Cost')
        ax.set_ylabel('Frequency')
        ax.set_title('Path Cost Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No cost data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Path Cost Distribution')

    # 4. 距離分布直方圖
    ax = axes[1, 0]
    # 過濾掉 None 和 inf 值
    distances = [r['distance'] for r in all_results if r.get('distance') is not None and r['distance'] != float('inf') and np.isfinite(r['distance'])]
    if distances:
        ax.hist(distances, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(distances), color='blue', linestyle='--', label=f'Mean: {np.mean(distances):.2f}')
        ax.set_xlabel('Distance (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Component Distance Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No distance data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Component Distance Distribution')

    # 5. 成本 vs 距離散點圖
    ax = axes[1, 1]
    # 過濾掉 None 和 inf 值 - 必須同時檢查 cost 和 distance
    successful_pairs = [(r['distance'], r['cost']) for r in successful
                       if r.get('cost') is not None and r['cost'] != float('inf') and np.isfinite(r['cost'])
                       and r.get('distance') is not None and r['distance'] != float('inf') and np.isfinite(r['distance'])]
    rejected_pairs = [(r['distance'], r['cost']) for r in rejected
                     if r.get('cost') is not None and r['cost'] != float('inf') and np.isfinite(r['cost'])
                     and r.get('distance') is not None and r['distance'] != float('inf') and np.isfinite(r['distance'])]

    if successful_pairs:
        successful_dists, successful_costs = zip(*successful_pairs)
        ax.scatter(successful_dists, successful_costs, c='green', alpha=0.6, label='Successful', s=50)
    if rejected_pairs:
        rejected_dists, rejected_costs = zip(*rejected_pairs)
        ax.scatter(rejected_dists, rejected_costs, c='red', alpha=0.6, label='Rejected', s=50)

    ax.set_xlabel('Distance (pixels)')
    ax.set_ylabel('Path Cost')
    ax.set_title('Cost vs Distance Relationship')
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. 統計摘要文本
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
    Component Pairing Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━

    Total Components: {pairing_results.get('num_components', 0)}
    Total Pairs Analyzed: {len(all_results)}

    ✓ Successful Connections: {len(successful)}
    ✗ Rejected Connections: {len(rejected)}

    Rejection Breakdown:
    """

    for reason, count in rejection_counts.items():
        name = REJECTION_REASON_NAMES.get(reason, reason)
        summary_text += f"  • {name}: {count}\n"

    if costs:
        summary_text += f"\nCost Statistics:\n"
        summary_text += f"  Mean: {np.mean(costs):.2f}\n"
        summary_text += f"  Median: {np.median(costs):.2f}\n"
        summary_text += f"  Std Dev: {np.std(costs):.2f}\n"

    if distances:
        summary_text += f"\nDistance Statistics:\n"
        summary_text += f"  Mean: {np.mean(distances):.2f} px\n"
        summary_text += f"  Median: {np.median(distances):.2f} px\n"
        summary_text += f"  Std Dev: {np.std(distances):.2f} px\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Statistics chart saved to: {output_path}")


# ============================================================================
# Pipeline Integration - Main Visualization Function
# ============================================================================

def visualize_component_pairing_from_images(
    annotation_path: str,
    green_channel_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    show_successful: bool = True,
    show_rejected: bool = True,
    show_components: bool = True,
    show_seeds: bool = True,
    show_labels: bool = True,
    show_legend: bool = True,
    component_alpha: float = 0.3,
    line_thickness: int = 2,
    seed_radius: int = 5
) -> None:
    """
    從原始影像自動運行完整 pipeline 並視覺化組件配對結果

    Args:
        annotation_path: 標註影像路徑 (二值 mask)
        green_channel_path: 綠色通道影像路徑
        output_dir: 輸出目錄
        config_path: 配置檔路徑 (若為 None 則使用預設配置)
        show_successful: 是否顯示成功連接
        show_rejected: 是否顯示拒絕連接
        show_components: 是否顯示組件疊加層
        show_seeds: 是否顯示種子點
        show_labels: 是否顯示組件標籤
        show_legend: 是否顯示圖例
        component_alpha: 組件疊加層透明度 (0.0 到 1.0)
        line_thickness: 連接線粗細
        seed_radius: 種子點半徑
    """
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # 載入配置
    logger.info("\n" + "="*70)
    logger.info("Loading Configuration")
    logger.info("="*70)

    config = load_config(config_path) if config_path else load_config()
    logger.info(f"Configuration loaded:")
    logger.info(f"  Connectivity: {config.connected_components.connectivity}")
    logger.info(f"  Min area: {config.connected_components.min_area}")
    logger.info(f"  Base segment length: {config.seed_extraction.base_segment_length}")

    # 初始化 pipeline
    logger.info("\n" + "="*70)
    logger.info("Initializing Neural Reconstruction Pipeline")
    logger.info("="*70)

    pipeline = NeuralReconstructionPipeline(config=config)

    # 運行完整 pipeline
    logger.info("\n" + "="*70)
    logger.info("Running Complete Pipeline")
    logger.info("="*70)

    results = pipeline.run(
        input_image_path=annotation_path,
        green_channel_path=green_channel_path,
        output_dir=None,  # 不保存中間檔案
        save_intermediates=False
    )

    # 提取配對結果
    logger.info("\n" + "="*70)
    logger.info("Extracting Results from Pipeline")
    logger.info("="*70)

    pairing_stage = results['stages']['component_pairing']
    pairing_results = {
        'num_components': pairing_stage['num_components'],
        'num_pairs_analyzed': pairing_stage['num_pairs_analyzed'],
        'num_connections': pairing_stage['num_connections'],
        'connections': pairing_stage['connections'],
        'all_pair_results': pairing_stage['all_pair_results']
    }

    logger.info(f"Pairing results extracted:")
    logger.info(f"  Components: {pairing_results['num_components']}")
    logger.info(f"  Pairs analyzed: {pairing_results['num_pairs_analyzed']}")
    logger.info(f"  Successful connections: {pairing_results['num_connections']}")

    # 提取組件數據
    skeleton_stage = results['stages']['skeletonization']
    topology_stage = results['stages']['topology_and_seeds']
    cc_stage = results['stages']['connected_components']

    # 從綠色通道影像獲取尺寸
    annotation_img = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    if annotation_img is None:
        raise ValueError(f"Failed to load annotation image: {annotation_path}")
    image_shape = annotation_img.shape

    # 建構 components_data 列表
    components_data = []
    for skeleton_data in skeleton_stage['skeleton_data']:
        region = skeleton_data['region']
        component_id = region.label

        # 找到這個組件的種子點
        component_seeds = [
            s for s in topology_stage['seeds']
            if s.get('component_id') == component_id
        ]

        # 從 region 創建 mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        coords = region.coords
        mask[coords[:, 0], coords[:, 1]] = 255

        components_data.append({
            'id': component_id,
            'seeds': component_seeds,
            'mask': mask
        })

    logger.info(f"Extracted {len(components_data)} components with masks and seeds")

    # 載入綠色通道影像
    green_channel = cv2.imread(green_channel_path, cv2.IMREAD_GRAYSCALE)
    if green_channel is None:
        raise ValueError(f"Failed to load green channel image: {green_channel_path}")

    # 生成視覺化
    logger.info("\n" + "="*70)
    logger.info("Generating Visualizations")
    logger.info("="*70)

    # 主視覺化 (overview)
    logger.info("Creating pairing overview...")
    visualize_component_pairing_cv2(
        pairing_results=pairing_results,
        components_data=components_data,
        green_channel=green_channel,
        output_path=str(output_path / 'pairing_overview.png'),
        show_successful=show_successful,
        show_rejected=show_rejected,
        show_components=show_components,
        show_seeds=show_seeds,
        show_labels=show_labels,
        show_legend=show_legend,
        component_alpha=component_alpha,
        line_thickness=line_thickness,
        seed_radius=seed_radius
    )

    # 僅成功連接
    if show_successful:
        logger.info("Creating successful connections visualization...")
        visualize_component_pairing_cv2(
            pairing_results=pairing_results,
            components_data=components_data,
            green_channel=green_channel,
            output_path=str(output_path / 'successful_connections.png'),
            show_successful=True,
            show_rejected=False,
            show_components=show_components,
            show_seeds=show_seeds,
            show_labels=show_labels,
            show_legend=True,
            component_alpha=component_alpha,
            line_thickness=line_thickness,
            seed_radius=seed_radius
        )

    # 僅拒絕連接
    if show_rejected:
        logger.info("Creating rejected connections visualization...")
        visualize_component_pairing_cv2(
            pairing_results=pairing_results,
            components_data=components_data,
            green_channel=green_channel,
            output_path=str(output_path / 'rejected_connections.png'),
            show_successful=False,
            show_rejected=True,
            show_components=show_components,
            show_seeds=show_seeds,
            show_labels=show_labels,
            show_legend=True,
            component_alpha=component_alpha,
            line_thickness=line_thickness,
            seed_radius=seed_radius
        )

    # 統計圖表
    logger.info("Creating statistics charts...")
    create_pairing_statistics(
        pairing_results=pairing_results,
        output_path=str(output_path / 'pairing_statistics.png')
    )

    # 保存結果到 JSON
    logger.info("\n" + "="*70)
    logger.info("Saving Results")
    logger.info("="*70)

    # 轉換為 JSON 可序列化格式
    json_results = {
        'num_components': int(pairing_results['num_components']),
        'num_pairs_analyzed': int(pairing_results['num_pairs_analyzed']),
        'num_connections': int(pairing_results['num_connections']),
        'connections': [{
            'component_a_id': int(c['component_a_id']),
            'component_b_id': int(c['component_b_id']),
            'cost': float(c['cost']) if c['cost'] != float('inf') else 'inf',
            'distance': float(c['distance']) if c['distance'] != float('inf') else 'inf',
            'path': [(int(p[0]), int(p[1])) for p in c['path']] if c.get('path') else None
        } for c in pairing_results['connections']],
        'all_pair_results': [{
            'component_a_id': int(r['component_a_id']),
            'component_b_id': int(r['component_b_id']),
            'should_connect': bool(r['should_connect']),
            'cost': float(r['cost']) if r['cost'] != float('inf') else 'inf',
            'distance': float(r['distance']) if r['distance'] != float('inf') else 'inf',
            'skipped_reason': r.get('skipped_reason'),
            'path': [(int(p[0]), int(p[1])) for p in r['path']] if r.get('path') else None
        } for r in pairing_results['all_pair_results']]
    }

    with open(output_path / 'pairing_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Saved pairing results: {output_path / 'pairing_results.json'}")

    logger.info("\n" + "="*70)
    logger.info("Visualization Complete")
    logger.info("="*70)
    logger.info(f"All outputs saved to: {output_path}")
    logger.info("\nGenerated files:")
    logger.info(f"  - pairing_overview.png")
    if show_successful:
        logger.info(f"  - successful_connections.png")
    if show_rejected:
        logger.info(f"  - rejected_connections.png")
    logger.info(f"  - pairing_statistics.png")
    logger.info(f"  - pairing_results.json")

if __name__ == '__main__':
    # Default paths - modify these for your use case
    annotation_path = "/Users/ponywen/projects/ienf_q/output/preprocessing/final_label.png"
    green_channel_path = "/Users/ponywen/projects/ienf_q/output/preprocessing/roi_image.png"
    output_dir = "output/component_pairing_visualization"

    # Run visualization with default settings
    visualize_component_pairing_from_images(
        annotation_path=annotation_path,
        green_channel_path=green_channel_path,
        output_dir=output_dir,
        config_path=None,  # Uses default config
        show_successful=True,
        show_rejected=True,
        show_components=True,
        show_seeds=False,
        show_labels=False,
        show_legend=False,
        component_alpha=0.3,
        line_thickness=1,
        seed_radius=1
    )
