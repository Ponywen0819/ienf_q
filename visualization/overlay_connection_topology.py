#!/usr/bin/env python3
"""
連接拓樸視覺化工具 (Connection Topology Overlay Visualizer)

直接執行 pipeline 並視覺化第四階段（元件配對與連接拓樸建構）的結果，
將重建的連接拓樸疊加在原始影像上。

功能：
1. 直接執行神經重建 pipeline
2. 繪製元件之間的連接路徑
3. 標示種子點位置
4. 用不同顏色區分不同的連接
5. 顯示連接成本資訊
6. 生成統計圖表

使用範例:
    from visualization.overlay_connection_topology import run_and_visualize
    
    run_and_visualize(
        annotation_path='data/Label/dermis_annotation.png',
        green_channel_path='data/Original/dermis_green_channel.png',
        output_dir='output/visualization',
        show_seeds=True,
        show_cost=True
    )

作者: Generated with Claude Code
日期: 2025-11-17
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nueral_reconstruction.pipeline import NeuralReconstructionPipeline

# 設定 logger
logger = logging.getLogger(__name__)


def overlay_connection_topology_cv2(
    pipeline_results: Dict[str, Any],
    base_image_path: str,
    output_path: str,
    show_seeds: bool = True,
    show_cost: bool = False,  # OpenCV 文字渲染較複雜，預設關閉
    show_components: bool = True,
    connection_color: str = 'rainbow',  # 'rainbow', 'single', 'by_cost'
    line_thickness: int = 2,
    seed_radius: int = 5
) -> None:
    """
    使用 OpenCV 將元件配對連接拓樸直接繪製在基礎影像上
    這個方法保持原始影像解析度，適合需要放大查看的情況
    
    Args:
        pipeline_results: Pipeline 執行結果字典
        base_image_path: 基礎影像路徑（通常是綠色通道影像）
        output_path: 輸出影像路徑
        show_seeds: 是否顯示種子點
        show_cost: 是否在連接上標示成本（OpenCV 文字渲染較複雜）
        show_components: 是否顯示元件編號
        connection_color: 連接顏色模式
            - 'rainbow': 每個連接使用不同顏色
            - 'single': 所有連接使用同一顏色 (青色)
            - 'by_cost': 根據成本使用顏色映射
        line_thickness: 線條粗細（像素）
        seed_radius: 種子點半徑（像素）
    """
    logger.info("=" * 70)
    logger.info("開始生成連接拓樸視覺化 (OpenCV 模式)")
    logger.info("=" * 70)
    
    # ========== 載入基礎影像 ==========
    base_image = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)
    if base_image is None:
        raise FileNotFoundError(f"無法載入基礎影像: {base_image_path}")
    
    # 將灰階影像轉換為 viridis 色彩映射
    # 正規化到 0-255
    normalized = cv2.normalize(base_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # 應用 viridis 色彩映射
    output_image = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    
    logger.info(f"基礎影像大小: {output_image.shape}")
    logger.info(f"已應用 viridis 色彩映射")
    
    # ========== 提取階段資料 ==========
    if 'component_pairing' not in pipeline_results['stages']:
        raise ValueError("Pipeline 結果中找不到 'component_pairing' 階段資料")
    
    pairing_data = pipeline_results['stages']['component_pairing']
    connections = pairing_data['connections']
    num_connections = pairing_data['num_connections']
    
    logger.info(f"連接數量: {num_connections}")
    
    if num_connections == 0:
        logger.warning("⚠️  沒有找到任何連接，將只保存基礎影像")
        cv2.imwrite(output_path, output_image)
        logger.info(f"✓ 影像已保存至: {output_path}")
        return
    
    # ========== 準備顏色映射 ==========
    def get_bgr_colors(num_colors, mode='rainbow', costs=None):
        """生成 BGR 格式的顏色列表（OpenCV 使用 BGR）"""
        if mode == 'rainbow':
            # 使用 HSV 色彩空間生成彩虹色
            colors_bgr = []
            for i in range(num_colors):
                hue = int(180 * i / num_colors)  # OpenCV HSV: H 範圍 0-180
                hsv_color = np.uint8([[[hue, 255, 255]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                colors_bgr.append(tuple(map(int, bgr_color)))
            return colors_bgr
        elif mode == 'single':
            # 青色 (BGR)
            return [(255, 255, 0)] * num_colors
        elif mode == 'by_cost':
            # 根據成本使用 viridis 色彩映射
            if costs is None:
                raise ValueError("costs 參數在 'by_cost' 模式下必須提供")
            min_cost, max_cost = min(costs), max(costs)
            colors_bgr = []
            for cost in costs:
                # 正規化成本到 0-1
                normalized = (cost - min_cost) / (max_cost - min_cost + 1e-6)
                # 使用簡單的藍-綠-黃映射
                # 低成本 -> 藍色，高成本 -> 黃色
                if normalized < 0.5:
                    # 藍 -> 綠
                    b = int(255 * (1 - 2 * normalized))
                    g = int(255 * 2 * normalized)
                    r = 0
                else:
                    # 綠 -> 黃
                    b = 0
                    g = 255
                    r = int(255 * 2 * (normalized - 0.5))
                colors_bgr.append((b, g, r))
            return colors_bgr
        else:
            raise ValueError(f"未知的顏色模式: {mode}")
    
    # 獲取顏色列表
    costs = [conn['cost'] for conn in connections] if connection_color == 'by_cost' else None
    colors = get_bgr_colors(num_connections, connection_color, costs)
    
    # ========== 繪製連接路徑 ==========
    logger.info(f"\n繪製 {num_connections} 條連接路徑...")
    
    for i, (conn, color) in enumerate(zip(connections, colors)):
        path = conn['path']
        cost = conn['cost']
        seed_pair = conn['seed_pair']
        
        # 繪製路徑
        if path and len(path) > 1:
            # 將路徑轉換為 OpenCV 格式 [(x, y), ...]
            # path 格式是 (row, col)，OpenCV 需要 (x, y) = (col, row)
            path_points = [(int(pt[1]), int(pt[0])) for pt in path]
            
            # 繪製連續的線段
            for j in range(len(path_points) - 1):
                cv2.line(output_image, path_points[j], path_points[j+1], 
                        color, line_thickness, cv2.LINE_AA)
            
            # 標示成本（在路徑中點）
            if show_cost:
                mid_idx = len(path_points) // 2
                mid_point = path_points[mid_idx]
                text = f'{cost:.1f}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # 獲取文字大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                
                # 繪製文字背景
                bg_pt1 = (mid_point[0] - text_width//2 - 2, 
                         mid_point[1] - text_height//2 - 2)
                bg_pt2 = (mid_point[0] + text_width//2 + 2, 
                         mid_point[1] + text_height//2 + baseline + 2)
                cv2.rectangle(output_image, bg_pt1, bg_pt2, color, -1)
                
                # 繪製文字
                text_org = (mid_point[0] - text_width//2, 
                           mid_point[1] + text_height//2)
                cv2.putText(output_image, text, text_org, font, 
                           font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # 繪製種子點
        if show_seeds and seed_pair:
            seed_a, seed_b = seed_pair
            pos_a = seed_a['position']
            pos_b = seed_b['position']
            
            # 起點（綠色圓圈）
            pt_a = (int(pos_a[1]), int(pos_a[0]))  # (col, row) -> (x, y)
            cv2.circle(output_image, pt_a, seed_radius, (0, 255, 0), -1)  # 填充
            cv2.circle(output_image, pt_a, seed_radius, (255, 255, 255), 2)  # 白色邊框
            
            # 終點（紅色圓圈）
            pt_b = (int(pos_b[1]), int(pos_b[0]))
            cv2.circle(output_image, pt_b, seed_radius, (0, 0, 255), -1)  # BGR: 紅色
            cv2.circle(output_image, pt_b, seed_radius, (255, 255, 255), 2)  # 白色邊框
    
    logger.info(f"✓ 已繪製 {num_connections} 條連接路徑")
    
    # ========== 繪製元件編號 ==========
    if show_components and 'topology_and_seeds' in pipeline_results['stages']:
        logger.info("\n繪製元件編號...")
        
        # 從種子資料中計算每個元件的中心位置
        all_seeds = pipeline_results['stages']['topology_and_seeds']['seeds']
        
        # 按元件 ID 分組種子
        component_seeds = {}
        for seed in all_seeds:
            comp_id = seed['component_id']
            if comp_id not in component_seeds:
                component_seeds[comp_id] = []
            component_seeds[comp_id].append(seed['position'])
        
        # 為每個元件繪製編號
        for component_id, seed_positions in component_seeds.items():
            if seed_positions:
                # 計算元件中心（所有種子的平均位置）
                positions = np.array(seed_positions)
                center = positions.mean(axis=0)
                center_pt = (int(center[1]), int(center[0]))  # (x, y) = (col, row)
                
                # 繪製元件編號
                text = f'C{component_id}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                
                # 獲取文字大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                
                # 繪製黑色背景
                bg_pt1 = (center_pt[0] - text_width//2 - 5, 
                         center_pt[1] - text_height//2 - 5)
                bg_pt2 = (center_pt[0] + text_width//2 + 5, 
                         center_pt[1] + text_height//2 + baseline + 5)
                cv2.rectangle(output_image, bg_pt1, bg_pt2, (0, 0, 0), -1)
                
                # 繪製黃色文字
                text_org = (center_pt[0] - text_width//2, 
                           center_pt[1] + text_height//2)
                cv2.putText(output_image, text, text_org, font, 
                           font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        
        logger.info(f"✓ 已繪製 {len(component_seeds)} 個元件編號")
    
    # ========== 保存圖片 ==========
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), output_image)
    
    logger.info(f"\n✓ 視覺化結果已保存至: {output_path}")
    logger.info(f"  影像大小: {output_image.shape}")
    logger.info("=" * 70)


def overlay_connection_topology(
    pipeline_results: Dict[str, Any],
    base_image_path: str,
    output_path: str,
    show_seeds: bool = True,
    show_cost: bool = True,
    show_components: bool = True,
    connection_color: str = 'rainbow',  # 'rainbow', 'single', 'by_cost'
    figsize: Tuple[int, int] = (20, 20),
    dpi: int = 150
) -> None:
    """
    將元件配對連接拓樸疊加在基礎影像上
    
    Args:
        pipeline_results: Pipeline 執行結果字典
        base_image_path: 基礎影像路徑（通常是綠色通道影像）
        output_path: 輸出影像路徑
        show_seeds: 是否顯示種子點
        show_cost: 是否在連接上標示成本
        show_components: 是否顯示元件編號
        connection_color: 連接顏色模式
            - 'rainbow': 每個連接使用不同顏色
            - 'single': 所有連接使用同一顏色
            - 'by_cost': 根據成本使用顏色映射
        figsize: 圖片大小
        dpi: 圖片解析度
    """
    logger.info("=" * 70)
    logger.info("開始生成連接拓樸視覺化")
    logger.info("=" * 70)
    
    # ========== 載入基礎影像 ==========
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        raise FileNotFoundError(f"無法載入基礎影像: {base_image_path}")
    
    # 轉換為 RGB（OpenCV 預設是 BGR）
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    logger.info(f"基礎影像大小: {base_image.shape}")
    
    # ========== 提取階段資料 ==========
    if 'component_pairing' not in pipeline_results['stages']:
        raise ValueError("Pipeline 結果中找不到 'component_pairing' 階段資料")
    
    pairing_data = pipeline_results['stages']['component_pairing']
    connections = pairing_data['connections']
    num_connections = pairing_data['num_connections']
    
    logger.info(f"連接數量: {num_connections}")
    
    if num_connections == 0:
        logger.warning("⚠️  沒有找到任何連接，將只繪製基礎影像")
    
    # ========== 建立圖表 ==========
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(base_image)
    ax.axis('off')
    
    # ========== 繪製連接路徑 ==========
    if num_connections > 0:
        logger.info(f"\n繪製 {num_connections} 條連接路徑...")
        
        # 準備顏色映射
        if connection_color == 'rainbow':
            colors = plt.cm.rainbow(np.linspace(0, 1, num_connections))
        elif connection_color == 'single':
            colors = ['cyan'] * num_connections
        elif connection_color == 'by_cost':
            costs = [conn['cost'] for conn in connections]
            min_cost, max_cost = min(costs), max(costs)
            normalized_costs = [(c - min_cost) / (max_cost - min_cost + 1e-6) for c in costs]
            colors = plt.cm.viridis(normalized_costs)
        else:
            raise ValueError(f"未知的顏色模式: {connection_color}")
        
        # 繪製每條連接
        for i, (conn, color) in enumerate(zip(connections, colors)):
            path = conn['path']
            cost = conn['cost']
            seed_pair = conn['seed_pair']
            
            # 繪製路徑
            if path and len(path) > 1:
                path_array = np.array(path)
                # 注意：path 的格式是 (row, col)，matplotlib 需要 (x, y) = (col, row)
                ax.plot(path_array[:, 1], path_array[:, 0], 
                       color=color, linewidth=2, alpha=0.8, zorder=2)
                
                # 標示成本（在路徑中點）
                if show_cost:
                    mid_idx = len(path) // 2
                    mid_point = path[mid_idx]
                    ax.text(mid_point[1], mid_point[0], f'{cost:.1f}',
                           fontsize=8, color='white', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                           ha='center', va='center', zorder=3)
            
            # 繪製種子點
            if show_seeds and seed_pair:
                seed_a, seed_b = seed_pair
                pos_a = seed_a['position']
                pos_b = seed_b['position']
                
                # 起點（綠色圓圈）
                ax.plot(pos_a[1], pos_a[0], 'go', markersize=8, 
                       markeredgecolor='white', markeredgewidth=1.5, zorder=4)
                
                # 終點（紅色圓圈）
                ax.plot(pos_b[1], pos_b[0], 'ro', markersize=8,
                       markeredgecolor='white', markeredgewidth=1.5, zorder=4)
        
        logger.info(f"✓ 已繪製 {num_connections} 條連接路徑")
    
    # ========== 繪製元件編號 ==========
    if show_components and 'topology_and_seeds' in pipeline_results['stages']:
        logger.info("\n繪製元件編號...")
        topologies = pipeline_results['stages']['topology_and_seeds']['topologies']
        
        for topo_data in topologies:
            component_id = topo_data['component_id']
            topology = topo_data['topology']
            nodes = topology['nodes']
            
            if nodes:
                # 計算元件中心（所有節點的平均位置）
                positions = np.array([node['position'] for node in nodes])
                center = positions.mean(axis=0)
                
                # 繪製元件編號
                ax.text(center[1], center[0], f'C{component_id}',
                       fontsize=12, color='yellow', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6),
                       ha='center', va='center', zorder=5)
        
        logger.info(f"✓ 已繪製 {len(topologies)} 個元件編號")
    
    # ========== 添加圖例 ==========
    legend_elements = []
    
    if show_seeds and num_connections > 0:
        legend_elements.append(
            mpatches.Patch(facecolor='green', edgecolor='white', label='Start Seed')
        )
        legend_elements.append(
            mpatches.Patch(facecolor='red', edgecolor='white', label='Target Seed')
        )
    
    if connection_color == 'by_cost' and num_connections > 0:
        legend_elements.append(
            mpatches.Patch(facecolor='purple', label=f'Connection Path (Colored by Cost)')
        )
    elif num_connections > 0:
        legend_elements.append(
            mpatches.Patch(facecolor='cyan', label=f'Connection Paths ({num_connections})')
        )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=12, framealpha=0.8)
    
    # ========== 添加標題 ==========
    title = "Component Connection Topology Visualization\n"
    title += f"Connections: {num_connections}"
    if num_connections > 0:
        costs = [conn['cost'] for conn in connections]
        title += f" | Avg Cost: {np.mean(costs):.2f}"
        title += f" | Cost Range: [{min(costs):.2f}, {max(costs):.2f}]"
    
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    
    # ========== 保存圖片 ==========
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\n✓ 視覺化結果已保存至: {output_path}")
    logger.info("=" * 70)


def visualize_connection_statistics(
    pipeline_results: Dict[str, Any],
    output_dir: str,
    dpi: int = 150
) -> None:
    """
    生成連接統計視覺化圖表
    
    Args:
        pipeline_results: Pipeline 執行結果
        output_dir: 輸出目錄
        dpi: 圖片解析度
    """
    logger.info("=" * 70)
    logger.info("開始生成連接統計圖表")
    logger.info("=" * 70)
    
    pairing_data = pipeline_results['stages']['component_pairing']
    connections = pairing_data['connections']
    all_pairs = pairing_data['all_pair_results']
    
    if not connections:
        logger.warning("⚠️  沒有連接資料，跳過統計圖表生成")
        return
    
    # 提取統計資料
    costs = [conn['cost'] for conn in connections]
    distances = [conn['distance'] for conn in connections]
    path_lengths = [len(conn['path']) for conn in connections]
    
    # 所有配對的距離分布（包含未連接的）
    all_distances = [p['distance'] for p in all_pairs if p['distance'] != float('inf')]
    
    # 建立子圖
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=dpi)
    
    # ========== 圖1: 成本分布 ==========
    ax = axes[0, 0]
    ax.hist(costs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(costs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(costs):.2f}')
    ax.axvline(np.median(costs), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(costs):.2f}')
    ax.set_xlabel('Connection Cost', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Connection Cost Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========== 圖2: 距離分布對比 ==========
    ax = axes[0, 1]
    ax.hist([all_distances, distances], bins=30, 
           color=['lightgray', 'coral'], 
           label=['All Pairs', 'Connected Pairs'],
           edgecolor='black', alpha=0.7)
    ax.set_xlabel('Euclidean Distance (pixels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Pair Distance Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========== 圖3: 路徑長度分布 ==========
    ax = axes[1, 0]
    ax.hist(path_lengths, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(path_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(path_lengths):.1f}')
    ax.set_xlabel('Path Length (pixels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Connection Path Length Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========== 圖4: 成本 vs 距離散點圖 ==========
    ax = axes[1, 1]
    scatter = ax.scatter(distances, costs, c=path_lengths, cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=1)
    ax.set_xlabel('Euclidean Distance (pixels)', fontsize=12)
    ax.set_ylabel('Connection Cost', fontsize=12)
    ax.set_title('Cost vs Distance Relationship', fontsize=14, weight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Path Length', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加相關係數
    correlation = np.corrcoef(distances, costs)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== 整體標題 ==========
    fig.suptitle(f'Component Pairing Connection Statistics ({len(connections)} connections)', 
                fontsize=16, weight='bold', y=0.995)
    
    # ========== 保存圖表 ==========
    output_path = Path(output_dir) / 'connection_statistics.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ 統計圖表已保存至: {output_path}")
    logger.info("=" * 70)


def run_and_visualize(
    annotation_path: str,
    green_channel_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    connectivity: int = 8,
    min_area: int = 50,
    show_seeds: bool = True,
    show_cost: bool = False,  # OpenCV 模式預設不顯示成本文字
    show_components: bool = True,
    connection_color: str = 'rainbow',
    generate_stats: bool = True,
    line_thickness: int = 2,
    seed_radius: int = 5,
    use_opencv: bool = True  # 預設使用 OpenCV 模式以保持原始解析度
) -> Dict[str, Any]:
    """
    執行神經重建 pipeline 並立即視覺化連接拓樸
    
    Args:
        annotation_path: 標註影像路徑
        green_channel_path: 綠色通道影像路徑
        output_dir: 輸出目錄
        config_path: 配置文件路徑（可選）
        connectivity: 連通性 (4 或 8)
        min_area: 最小元件面積
        show_seeds: 是否顯示種子點
        show_cost: 是否顯示成本（OpenCV 模式下較複雜）
        show_components: 是否顯示元件編號
        connection_color: 連接顏色模式 ('rainbow', 'single', 'by_cost')
        generate_stats: 是否生成統計圖表
        line_thickness: 線條粗細（OpenCV 模式）
        seed_radius: 種子點半徑（OpenCV 模式）
        use_opencv: 使用 OpenCV 直接繪製（保持原始解析度）或 matplotlib（可調整 DPI）
        
    Returns:
        pipeline 執行結果
    """
    logger.info("\n" + "=" * 70)
    logger.info("連接拓樸視覺化工具")
    logger.info("=" * 70)
    logger.info("此工具將：")
    logger.info("  1. 執行神經重建 pipeline")
    logger.info("  2. 提取元件配對連接拓樸")
    logger.info("  3. 生成視覺化圖片")
    if generate_stats:
        logger.info("  4. 生成統計圖表")
    logger.info("=" * 70)
    
    # ========== 步驟 1: 執行 Pipeline ==========
    logger.info("\n" + "=" * 70)
    logger.info("步驟 1/3: 執行神經重建 Pipeline")
    logger.info("=" * 70)
    
    # 建立 pipeline
    if config_path:
        pipeline = NeuralReconstructionPipeline(
            connectivity=connectivity,
            min_area=min_area,
            config_path=config_path
        )
    else:
        pipeline = NeuralReconstructionPipeline(
            connectivity=connectivity,
            min_area=min_area,
            config_path='config/default.yaml'
        )
    
    # 執行 pipeline
    results = pipeline.run(
        input_image_path=annotation_path,
        green_channel_path=green_channel_path,
        output_dir=output_dir,
        save_intermediates=True
    )
    
    # ========== 步驟 2: 生成視覺化 ==========
    logger.info("\n" + "=" * 70)
    logger.info("步驟 2/3: 生成連接拓樸視覺化")
    logger.info("=" * 70)
    
    # 生成主視覺化圖片
    vis_output = Path(output_dir) / f'connection_topology_{connection_color}.png'
    
    if use_opencv:
        # 使用 OpenCV 模式（保持原始解析度）
        overlay_connection_topology_cv2(
            pipeline_results=results,
            base_image_path=green_channel_path,
            output_path=str(vis_output),
            show_seeds=show_seeds,
            show_cost=show_cost,
            show_components=show_components,
            connection_color=connection_color,
            line_thickness=line_thickness,
            seed_radius=seed_radius
        )
    else:
        # 使用 matplotlib 模式（可調整 DPI）
        overlay_connection_topology(
            pipeline_results=results,
            base_image_path=green_channel_path,
            output_path=str(vis_output),
            show_seeds=show_seeds,
            show_cost=show_cost,
            show_components=show_components,
            connection_color=connection_color,
            figsize=(20, 20),
            dpi=150
        )
    
    # ========== 步驟 3: 生成統計圖表（可選）==========
    if generate_stats:
        logger.info("\n" + "=" * 70)
        logger.info("步驟 3/3: 生成統計圖表")
        logger.info("=" * 70)
        
        visualize_connection_statistics(
            pipeline_results=results,
            output_dir=output_dir,
            dpi=150
        )
    
    # ========== 完成 ==========
    logger.info("\n" + "=" * 70)
    logger.info("視覺化完成！")
    logger.info("=" * 70)
    
    # 顯示結果摘要
    pairing_data = results['stages']['component_pairing']
    logger.info(f"\n結果摘要:")
    logger.info(f"  元件數: {pairing_data['num_components']}")
    logger.info(f"  分析配對數: {pairing_data['num_pairs_analyzed']}")
    logger.info(f"  建議連接數: {pairing_data['num_connections']}")
    
    if pairing_data['num_connections'] > 0:
        costs = [c['cost'] for c in pairing_data['connections']]
        logger.info(f"  平均連接成本: {np.mean(costs):.2f}")
        logger.info(f"  成本範圍: [{min(costs):.2f}, {max(costs):.2f}]")
    
    logger.info(f"\n輸出檔案:")
    logger.info(f"  視覺化圖片: {vis_output}")
    if generate_stats:
        logger.info(f"  統計圖表: {output_dir}/connection_statistics.png")
    logger.info("=" * 70)
    
    return results


def main():
    """測試主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='連接拓樸視覺化工具 - 直接執行 pipeline 並視覺化結果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用（執行 pipeline 並視覺化）
  python -m visualization.overlay_connection_topology \\
      --annotation data/Label/dermis_annotation.png \\
      --green-channel data/Original/dermis_green_channel.png \\
      --output-dir output/visualization
  
  # 自訂視覺化選項
  python -m visualization.overlay_connection_topology \\
      --annotation data/Label/dermis_annotation.png \\
      --green-channel data/Original/dermis_green_channel.png \\
      --output-dir output/visualization \\
      --color-mode by_cost \\
      --no-seeds \\
      --with-stats \\
      --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--annotation',
        type=str,
        required=True,
        help='標註影像路徑'
    )
    
    parser.add_argument(
        '--green-channel',
        type=str,
        required=True,
        help='綠色通道影像路徑'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='輸出目錄'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路徑（預設使用 config/default.yaml）'
    )
    
    parser.add_argument(
        '--color-mode',
        type=str,
        default='rainbow',
        choices=['rainbow', 'single', 'by_cost'],
        help='連接顏色模式（預設: rainbow）'
    )
    
    parser.add_argument(
        '--no-seeds',
        action='store_true',
        help='不顯示種子點'
    )
    
    parser.add_argument(
        '--no-cost',
        action='store_true',
        help='不顯示成本標籤'
    )
    
    parser.add_argument(
        '--no-components',
        action='store_true',
        help='不顯示元件編號'
    )
    
    parser.add_argument(
        '--with-stats',
        action='store_true',
        help='同時生成統計圖表'
    )
    
    parser.add_argument(
        '--connectivity',
        type=int,
        default=8,
        choices=[4, 8],
        help='連通性 (4 或 8，預設: 8)'
    )
    
    parser.add_argument(
        '--min-area',
        type=int,
        default=50,
        help='最小元件面積（像素，預設: 50）'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='輸出解析度（預設: 150）'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日誌級別（預設: INFO）'
    )
    
    args = parser.parse_args()
    
    # 設定 logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(message)s'
    )
    
    # 執行 pipeline 並視覺化
    run_and_visualize(
        annotation_path=args.annotation,
        green_channel_path=args.green_channel,
        output_dir=args.output_dir,
        config_path=args.config,
        connectivity=args.connectivity,
        min_area=args.min_area,
        show_seeds=not args.no_seeds,
        show_cost=not args.no_cost,
        show_components=not args.no_components,
        connection_color=args.color_mode,
        generate_stats=args.with_stats,
        dpi=args.dpi
    )


if __name__ == '__main__':
    main()
