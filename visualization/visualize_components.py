#!/usr/bin/env python3
"""
連通元件視覺化工具 (Connected Components Visualizer)

視覺化 ConnectedComponentsAnalyzer 辨認出的所有連通元件，
為每個 component 填上不同顏色，並提供統計資訊。

功能：
1. 載入 binary annotation 影像
2. 執行連通元件分析
3. 為每個 component 分配唯一顏色
4. 生成高解析度彩色視覺化
5. 可選疊加原始影像背景
6. 生成統計圖表和資訊
7. 支援顯示 component ID 標籤

使用範例:
    from visualization.visualize_components import visualize_components

    visualize_components(
        annotation_path='data/Label/S163-2_a.tif',
        output_dir='output/components_visualization',
        show_labels=True,
        show_statistics=True
    )

作者: Generated with Claude Code
日期: 2025-11-19
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nueral_reconstruction.connected_components import ConnectedComponentsAnalyzer
from src.nueral_reconstruction.config_loader import load_config, IENFConfig

# 設定 logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_component_colors(num_components: int, colormap: str = 'auto') -> np.ndarray:
    """
    為 components 生成顏色

    Args:
        num_components: Component 數量
        colormap: 配色方案 ('auto', 'tab20', 'hsv', 'rainbow')

    Returns:
        顏色陣列 (N, 3) BGR 格式，範圍 0-255
    """
    if colormap == 'auto':
        # 自動選擇配色
        if num_components <= 20:
            colormap = 'tab20'
        else:
            colormap = 'hsv'

    # 生成顏色
    if colormap == 'tab20':
        cmap = cm.get_cmap('tab20')
    elif colormap == 'hsv':
        cmap = cm.get_cmap('hsv')
    elif colormap == 'rainbow':
        cmap = cm.get_cmap('rainbow')
    else:
        cmap = cm.get_cmap(colormap)

    colors = []
    for i in range(num_components):
        # 獲取 RGBA 顏色 (0-1 範圍)
        rgba = cmap(i / max(num_components, 1))
        # 轉換為 BGR (0-255 範圍)
        bgr = np.array([rgba[2], rgba[1], rgba[0]]) * 255
        colors.append(bgr.astype(np.uint8))

    return np.array(colors)


def create_colored_components_image(
    binary_image: np.ndarray,
    components_list: List[Any],
    colors: np.ndarray,
    background_image: Optional[np.ndarray] = None,
    alpha: float = 0.7
) -> np.ndarray:
    """
    創建彩色 components 視覺化影像

    Args:
        binary_image: 原始二值影像
        components_list: RegionProperties 列表
        colors: Component 顏色陣列 (N, 3) BGR
        background_image: 可選的背景影像（灰階）
        alpha: Component 顏色的不透明度 (0-1)

    Returns:
        彩色視覺化影像 (BGR)
    """
    h, w = binary_image.shape

    # 創建輸出影像
    if background_image is not None:
        # 將背景影像轉換為 BGR
        if len(background_image.shape) == 2:
            output = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
        else:
            output = background_image.copy()
        output = output.astype(np.float32)
    else:
        # 純黑背景
        output = np.zeros((h, w, 3), dtype=np.float32)

    # 為每個 component 填充顏色
    for idx, region in enumerate(components_list):
        color = colors[idx].astype(np.float32)

        # 獲取 component 的所有像素座標
        coords = region.coords  # (N, 2) array of (row, col)

        # 創建遮罩
        mask = np.zeros((h, w), dtype=np.float32)
        mask[coords[:, 0], coords[:, 1]] = 1.0

        # 混合顏色
        for c in range(3):
            output[:, :, c] = output[:, :, c] * (1 - mask * alpha) + color[c] * mask * alpha

    return output.astype(np.uint8)


def add_component_labels(
    image: np.ndarray,
    components_list: List[Any],
    colors: np.ndarray,
    font_scale: float = 0.5,
    thickness: int = 2
) -> np.ndarray:
    """
    在影像上添加 component ID 標籤

    Args:
        image: 輸入影像 (BGR)
        components_list: RegionProperties 列表
        colors: Component 顏色陣列
        font_scale: 字體大小
        thickness: 字體粗細

    Returns:
        標註後的影像
    """
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, region in enumerate(components_list):
        # 獲取質心位置
        centroid = region.centroid  # (row, col)
        center = (int(centroid[1]), int(centroid[0]))  # (x, y)

        # 標籤文字
        label = f"{idx + 1}"

        # 計算文字大小
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 繪製白色背景矩形
        padding = 2
        cv2.rectangle(
            output,
            (center[0] - text_w // 2 - padding, center[1] - text_h // 2 - padding),
            (center[0] + text_w // 2 + padding, center[1] + text_h // 2 + padding + baseline),
            (255, 255, 255),
            -1
        )

        # 繪製黑色文字
        cv2.putText(
            output,
            label,
            (center[0] - text_w // 2, center[1] + text_h // 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    return output


def create_statistics_plot(
    components_list: List[Any],
    output_path: str
) -> None:
    """
    創建 component 統計圖表

    Args:
        components_list: RegionProperties 列表
        output_path: 輸出路徑
    """
    # 提取面積資訊
    areas = [region.area for region in components_list]

    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Connected Components Statistics', fontsize=16, fontweight='bold')

    # 1. 面積分布直方圖
    ax1 = axes[0, 0]
    ax1.hist(areas, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Component Area (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Component Area Distribution')
    ax1.grid(True, alpha=0.3)

    # 2. 面積 Box Plot
    ax2 = axes[0, 1]
    bp = ax2.boxplot(areas, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    ax2.set_ylabel('Component Area (pixels)')
    ax2.set_title('Component Area Box Plot')
    ax2.grid(True, alpha=0.3)

    # 3. 統計摘要文字
    ax3 = axes[1, 0]
    ax3.axis('off')

    total_components = len(components_list)
    total_area = sum(areas)
    avg_area = np.mean(areas)
    std_area = np.std(areas)
    min_area = min(areas)
    max_area = max(areas)
    median_area = np.median(areas)

    stats_text = f"""
    Component Statistics Summary
    {'=' * 40}

    Total Components:     {total_components}
    Total Area:           {total_area:,} pixels

    Area Statistics:
      • Average:          {avg_area:.1f} pixels
      • Std Dev:          {std_area:.1f} pixels
      • Median:           {median_area:.1f} pixels
      • Minimum:          {min_area} pixels
      • Maximum:          {max_area:,} pixels
      • Range:            {max_area - min_area:,} pixels

    Size Categories:
      • Small (< 100px):   {sum(1 for a in areas if a < 100)}
      • Medium (100-500):  {sum(1 for a in areas if 100 <= a < 500)}
      • Large (≥ 500px):   {sum(1 for a in areas if a >= 500)}
    """

    ax3.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
             verticalalignment='center', transform=ax3.transAxes)

    # 4. Top 10 最大 components
    ax4 = axes[1, 1]
    sorted_areas = sorted(enumerate(areas, 1), key=lambda x: x[1], reverse=True)[:10]
    component_ids = [f"C{comp_id}" for comp_id, _ in sorted_areas]
    component_areas = [area for _, area in sorted_areas]

    bars = ax4.barh(component_ids, component_areas, color='coral', edgecolor='black')
    ax4.set_xlabel('Area (pixels)')
    ax4.set_title('Top 10 Largest Components')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()

    # 添加數值標籤
    for bar, area in zip(bars, component_areas):
        ax4.text(area, bar.get_y() + bar.get_height()/2,
                f' {area:,}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 統計圖表已儲存: {output_path}")


def save_components_info(
    components_list: List[Any],
    output_path: str
) -> None:
    """
    儲存 component 詳細資訊為 JSON

    Args:
        components_list: RegionProperties 列表
        output_path: 輸出路徑
    """
    components_info = []

    for idx, region in enumerate(components_list, 1):
        info = {
            'component_id': idx,
            'area': int(region.area),
            'centroid': {
                'row': float(region.centroid[0]),
                'col': float(region.centroid[1])
            },
            'bbox': {
                'min_row': int(region.bbox[0]),
                'min_col': int(region.bbox[1]),
                'max_row': int(region.bbox[2]),
                'max_col': int(region.bbox[3])
            },
            'perimeter': float(region.perimeter),
            'eccentricity': float(region.eccentricity),
            'solidity': float(region.solidity)
        }
        components_info.append(info)

    # 儲存 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(components_info, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Component 資訊已儲存: {output_path}")


def visualize_components(
    annotation_path: str,
    output_dir: str,
    green_channel_path: Optional[str] = None,
    show_labels: bool = True,
    show_statistics: bool = True,
    alpha: float = 0.7,
    colormap: str = 'auto',
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    完整的 component 視覺化流程

    Args:
        annotation_path: Binary annotation 影像路徑
        output_dir: 輸出目錄
        green_channel_path: 可選的 green channel 背景影像路徑
        show_labels: 是否顯示 component ID 標籤
        show_statistics: 是否生成統計圖表
        alpha: Component 顏色不透明度 (0-1)
        colormap: 配色方案 ('auto', 'tab20', 'hsv', 'rainbow')
        config_path: 配置文件路徑（可選，預設使用 config/default.yaml）

    Returns:
        包含視覺化結果資訊的字典
    """
    logger.info("=" * 60)
    logger.info("開始 Connected Components 視覺化")
    logger.info("=" * 60)

    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 載入配置
    if config_path is not None:
        config = load_config(config_path)
        logger.info(f"✓ 已載入配置文件: {config_path}")
    else:
        try:
            config = load_config()  # 載入 config/default.yaml
            logger.info("✓ 已載入預設配置文件 (config/default.yaml)")
        except FileNotFoundError:
            logger.warning("⚠️  未找到配置文件，使用內建預設值")
            config = IENFConfig()

    # 從配置取得參數
    connectivity = config.connected_components.connectivity
    min_area = config.connected_components.min_area
    logger.info(f"  連通性: {connectivity}-連通")
    logger.info(f"  最小面積: {min_area} 像素")

    # 1. 執行連通元件分析
    logger.info("\n步驟 1: 執行連通元件分析")
    analyzer = ConnectedComponentsAnalyzer(
        connectivity=connectivity,
        min_area=min_area
    )

    components_list = analyzer.process(annotation_path)
    num_components = len(components_list)

    logger.info(f"✓ 偵測到 {num_components} 個連通元件")

    # 2. 生成顏色
    logger.info(f"\n步驟 2: 生成配色方案 ({colormap})")
    colors = get_component_colors(num_components, colormap)
    logger.info(f"✓ 已生成 {num_components} 個顏色")

    # 3. 載入背景影像（如果提供）
    background_image = None
    if green_channel_path:
        logger.info(f"\n步驟 3: 載入背景影像")
        background_image = cv2.imread(green_channel_path, cv2.IMREAD_GRAYSCALE)
        if background_image is not None:
            logger.info(f"✓ 背景影像已載入: {background_image.shape}")
        else:
            logger.warning(f"無法載入背景影像: {green_channel_path}")
    else:
        logger.info("\n步驟 3: 使用純黑背景")

    # 4. 創建彩色 components 視覺化
    logger.info("\n步驟 4: 創建彩色視覺化")
    binary_image = analyzer.load_binary_image(annotation_path)

    colored_image = create_colored_components_image(
        binary_image=binary_image,
        components_list=components_list,
        colors=colors,
        background_image=background_image,
        alpha=alpha
    )

    logger.info(f"✓ 彩色視覺化已創建")

    # 5. 添加標籤（如果需要）
    if show_labels:
        logger.info("\n步驟 5: 添加 Component ID 標籤")
        colored_image = add_component_labels(
            image=colored_image,
            components_list=components_list,
            colors=colors
        )
        logger.info(f"✓ 已添加 {num_components} 個標籤")
    else:
        logger.info("\n步驟 5: 跳過標籤添加")

    # 6. 儲存主視覺化影像
    logger.info("\n步驟 6: 儲存視覺化影像")
    main_output_path = output_path / "components_colored.png"
    cv2.imwrite(str(main_output_path), colored_image)
    logger.info(f"✓ 主視覺化已儲存: {main_output_path}")

    # 7. 生成統計圖表（如果需要）
    if show_statistics:
        logger.info("\n步驟 7: 生成統計圖表")
        stats_output_path = output_path / "components_statistics.png"
        create_statistics_plot(components_list, str(stats_output_path))

    # 8. 儲存 component 資訊
    logger.info("\n步驟 8: 儲存 Component 資訊")
    info_output_path = output_path / "components_info.json"
    save_components_info(components_list, str(info_output_path))

    # 9. 生成顏色圖例
    logger.info("\n步驟 9: 生成顏色圖例")
    create_color_legend(colors, num_components, str(output_path / "color_legend.png"))

    logger.info("\n" + "=" * 60)
    logger.info("✓ 視覺化完成！")
    logger.info("=" * 60)
    logger.info(f"\n輸出檔案：")
    logger.info(f"  • 主視覺化: {main_output_path}")
    if show_statistics:
        logger.info(f"  • 統計圖表: {stats_output_path}")
    logger.info(f"  • Component 資訊: {info_output_path}")
    logger.info(f"  • 顏色圖例: {output_path / 'color_legend.png'}")

    return {
        'num_components': num_components,
        'output_dir': str(output_path),
        'main_image': str(main_output_path),
        'components_list': components_list
    }


def create_color_legend(
    colors: np.ndarray,
    num_components: int,
    output_path: str,
    cols: int = 10
) -> None:
    """
    創建顏色圖例

    Args:
        colors: 顏色陣列 (N, 3) BGR
        num_components: Component 數量
        output_path: 輸出路徑
        cols: 每行顯示的顏色數量
    """
    rows = (num_components + cols - 1) // cols

    fig, ax = plt.subplots(figsize=(12, rows * 0.5 + 1))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis('off')
    ax.set_aspect('equal')

    for idx in range(num_components):
        row = idx // cols
        col = idx % cols

        # BGR to RGB
        color_rgb = colors[idx][[2, 1, 0]] / 255.0

        # 繪製色塊
        rect = mpatches.Rectangle(
            (col, rows - row - 1),
            0.9, 0.9,
            facecolor=color_rgb,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)

        # 添加 ID 標籤
        ax.text(
            col + 0.45, rows - row - 0.5,
            str(idx + 1),
            ha='center', va='center',
            fontsize=8,
            fontweight='bold',
            color='white' if sum(color_rgb) < 1.5 else 'black'
        )

    plt.title(f'Component Color Legend (Total: {num_components})',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 顏色圖例已儲存: {output_path}")


# ============================================================================
# 主程式與使用範例
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：視覺化 components，使用配置系統
    """

    # 範例 1: 使用預設配置 (config/default.yaml)
    logger.info("\n範例 1: 使用預設配置視覺化")
    visualize_components(
        annotation_path='closing_5.png',
        output_dir='output/components_visualization/default',
        green_channel_path='split/S163-2_a_epidermis_correct_12.png',
        show_labels=False,  # 不顯示編號，避免遮擋 components
        show_statistics=True,
        alpha=0.7
        # connectivity 和 min_area 從 config/default.yaml 自動讀取
    )

    # 範例 2: 使用 fast 配置
    # logger.info("\n範例 2: 使用 fast 配置視覺化")
    # visualize_components(
    #     annotation_path='closing_5.png',
    #     output_dir='output/components_visualization/fast',
    #     green_channel_path='split/S163-2_a_epidermis_correct_12.png',
    #     show_labels=False,
    #     show_statistics=True,
    #     alpha=0.7,
    #     config_path='config/fast.yaml'  # 使用 fast 配置
    # )

    # 範例 3: 使用 high_quality 配置
    # logger.info("\n範例 3: 使用 high_quality 配置視覺化")
    # visualize_components(
    #     annotation_path='closing_5.png',
    #     output_dir='output/components_visualization/high_quality',
    #     green_channel_path='split/S163-2_a_epidermis_correct_12.png',
    #     show_labels=False,
    #     show_statistics=True,
    #     alpha=0.7,
    #     config_path='config/high_quality.yaml'  # 使用 high_quality 配置
    # )

    logger.info("\n所有視覺化已完成！")
