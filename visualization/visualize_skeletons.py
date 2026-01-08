#!/usr/bin/env python3
"""
骨架視覺化工具 (Skeleton Visualizer)

視覺化 SkeletonAnalyzer 分析的骨架結構，包括端點和分支點標記，
並提供統計資訊。整合 config 系統進行參數管理。

功能：
1. 骨架疊加視覺化
2. 端點標記（紅色圓點）
3. 分支點標記（藍色方框）
4. 可選 component ID 標籤
5. 統計圖表生成
6. 使用 config 系統管理參數

使用範例:
    from visualization.visualize_skeletons import visualize_skeletons

    visualize_skeletons(
        annotation_path='closing_5.png',
        output_dir='output/skeleton_visualization',
        show_endpoints=True,
        show_branchpoints=True
    )

作者: Generated with Claude Code
日期: 2025-11-19
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nueral_reconstruction.connected_components import ConnectedComponentsAnalyzer
from src.nueral_reconstruction.skeletonization import SkeletonAnalyzer
from src.nueral_reconstruction.config_loader import load_config, IENFConfig

# 設定 logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def draw_skeleton_overlay(
    base_image: np.ndarray,
    skeleton_results: List[Dict],
    label_mask: Optional[np.ndarray] = None,
    show_endpoints: bool = True,
    show_branchpoints: bool = True,
    skeleton_color: tuple = (0, 255, 255),  # BGR: 黃色
    endpoint_color: tuple = (0, 255, 0),     # BGR: 綠色
    branchpoint_color: tuple = (0, 255, 0),  # BGR: 綠色
    label_color: tuple = (0, 0, 255),        # BGR: 紅色 (不透明)
    alpha: float = 0.7
) -> np.ndarray:
    """
    在基礎影像上繪製骨架和標記點

    Args:
        base_image: 基礎影像 (灰階或 BGR)
        skeleton_results: 骨架分析結果列表
        label_mask: 原始標註遮罩 (可選)
        show_endpoints: 是否顯示端點
        show_branchpoints: 是否顯示分支點
        skeleton_color: 骨架顏色 (BGR)
        endpoint_color: 端點顏色 (BGR)
        branchpoint_color: 分支點顏色 (BGR)
        label_color: 標註遮罩顏色 (BGR)
        alpha: 骨架混合不透明度

    Returns:
        疊加後的影像 (BGR)
    """
    # 確保是 BGR 格式
    if len(base_image.shape) == 2:
        output = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        output = base_image.copy()

    output = output.astype(np.float32)
    h, w = output.shape[:2]

    # 繪製標註遮罩 (如果提供) - 不透明
    if label_mask is not None:
        mask_indices = label_mask > 0
        output[mask_indices] = label_color

    # 創建骨架疊加層
    skeleton_overlay = np.zeros((h, w, 3), dtype=np.float32)

    for skeleton_info in skeleton_results:
        region = skeleton_info['region']
        skeleton = skeleton_info['skeleton']

        # 獲取 bbox 偏移
        min_row, min_col, max_row, max_col = region.bbox

        # 將骨架繪製到全圖座標
        skeleton_binary = (skeleton > 0)
        for i in range(skeleton.shape[0]):
            for j in range(skeleton.shape[1]):
                if skeleton_binary[i, j]:
                    global_y = min_row + i
                    global_x = min_col + j
                    if 0 <= global_y < h and 0 <= global_x < w:
                        skeleton_overlay[global_y, global_x] = skeleton_color

    # 混合骨架層
    skeleton_mask = (skeleton_overlay.sum(axis=2) > 0).astype(np.float32)
    for c in range(3):
        output[:, :, c] = output[:, :, c] * (1 - skeleton_mask * alpha) + \
                         skeleton_overlay[:, :, c] * alpha

    output = output.astype(np.uint8)

    # 繪製端點 (單像素)
    if show_endpoints:
        for skeleton_info in skeleton_results:
            region = skeleton_info['region']
            min_row, min_col = region.bbox[0], region.bbox[1]

            for endpoint in skeleton_info['endpoints']:
                global_x = min_col + endpoint['x']
                global_y = min_row + endpoint['y']
                if 0 <= global_y < h and 0 <= global_x < w:
                    output[global_y, global_x] = endpoint_color

    # 繪製分支點 (單像素)
    if show_branchpoints:
        for skeleton_info in skeleton_results:
            region = skeleton_info['region']
            min_row, min_col = region.bbox[0], region.bbox[1]

            for branchpoint in skeleton_info['branchpoints']:
                global_x = min_col + branchpoint['x']
                global_y = min_row + branchpoint['y']
                if 0 <= global_y < h and 0 <= global_x < w:
                    output[global_y, global_x] = branchpoint_color

    return output


def add_skeleton_labels(
    image: np.ndarray,
    skeleton_results: List[Dict],
    font_scale: float = 0.4,
    thickness: int = 1
) -> np.ndarray:
    """
    添加 component ID 標籤

    Args:
        image: 輸入影像 (BGR)
        skeleton_results: 骨架分析結果列表
        font_scale: 字體大小
        thickness: 字體粗細

    Returns:
        標註後的影像
    """
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for skeleton_info in skeleton_results:
        region = skeleton_info['region']
        # 使用質心位置
        centroid_y, centroid_x = region.centroid
        center = (int(centroid_x), int(centroid_y))

        # 標籤文字
        label = f"{region.label}"

        # 計算文字大小
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 繪製白色背景
        cv2.rectangle(
            output,
            (center[0] - text_w // 2 - 2, center[1] - text_h // 2 - 2),
            (center[0] + text_w // 2 + 2, center[1] + text_h // 2 + baseline + 2),
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


def create_skeleton_statistics(
    skeleton_results: List[Dict],
    output_path: str
) -> None:
    """
    創建骨架統計圖表

    Args:
        skeleton_results: 骨架分析結果列表
        output_path: 輸出路徑
    """
    # 提取統計數據
    num_endpoints_list = [s['num_endpoints'] for s in skeleton_results]
    num_branchpoints_list = [s['num_branchpoints'] for s in skeleton_results]
    complexity = [s['num_endpoints'] + s['num_branchpoints'] for s in skeleton_results]

    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Skeleton Analysis Statistics', fontsize=16, fontweight='bold')

    # 1. 端點數量分布
    ax1 = axes[0, 0]
    if num_endpoints_list:
        ax1.hist(num_endpoints_list, bins=max(10, max(num_endpoints_list)//2),
                color='salmon', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Endpoints')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Endpoint Distribution')
    ax1.grid(True, alpha=0.3)

    # 2. 分支點數量分布
    ax2 = axes[0, 1]
    if num_branchpoints_list:
        ax2.hist(num_branchpoints_list, bins=max(10, max(num_branchpoints_list)//2) if max(num_branchpoints_list) > 0 else 5,
                color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Number of Branchpoints')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Branchpoint Distribution')
    ax2.grid(True, alpha=0.3)

    # 3. 統計摘要
    ax3 = axes[1, 0]
    ax3.axis('off')

    total_components = len(skeleton_results)
    total_endpoints = sum(num_endpoints_list)
    total_branchpoints = sum(num_branchpoints_list)
    avg_endpoints = np.mean(num_endpoints_list) if num_endpoints_list else 0
    avg_branchpoints = np.mean(num_branchpoints_list) if num_branchpoints_list else 0

    stats_text = f"""
    Skeleton Statistics Summary
    {'=' * 40}

    Total Components:        {total_components}

    Endpoints:
      • Total:               {total_endpoints}
      • Average per comp:    {avg_endpoints:.1f}
      • Max:                 {max(num_endpoints_list) if num_endpoints_list else 0}
      • Components w/ 0:     {sum(1 for x in num_endpoints_list if x == 0)}
      • Components w/ 1-2:   {sum(1 for x in num_endpoints_list if 1 <= x <= 2)}
      • Components w/ >2:    {sum(1 for x in num_endpoints_list if x > 2)}

    Branchpoints:
      • Total:               {total_branchpoints}
      • Average per comp:    {avg_branchpoints:.1f}
      • Max:                 {max(num_branchpoints_list) if num_branchpoints_list else 0}
      • Components w/ 0:     {sum(1 for x in num_branchpoints_list if x == 0)}
      • Components w/ ≥1:    {sum(1 for x in num_branchpoints_list if x >= 1)}
    """

    ax3.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=9,
             verticalalignment='center', transform=ax3.transAxes)

    # 4. Top 10 最複雜的 components
    ax4 = axes[1, 1]
    if skeleton_results:
        sorted_complexity = sorted(
            enumerate(complexity, 1),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        component_ids = [f"C{comp_id}" for comp_id, _ in sorted_complexity]
        complexity_values = [val for _, val in sorted_complexity]

        bars = ax4.barh(component_ids, complexity_values,
                       color='lightcoral', edgecolor='black')
        ax4.set_xlabel('Total Points (Endpoints + Branchpoints)')
        ax4.set_title('Top 10 Most Complex Components')
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.invert_yaxis()

        # 添加數值標籤
        for bar, val in zip(bars, complexity_values):
            ax4.text(val, bar.get_y() + bar.get_height()/2,
                    f' {val}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 統計圖表已儲存: {output_path}")


def save_skeleton_info(
    skeleton_results: List[Dict],
    output_path: str
) -> None:
    """
    儲存骨架詳細資訊為 JSON

    Args:
        skeleton_results: 骨架分析結果列表
        output_path: 輸出路徑
    """
    skeletons_info = []

    for skeleton_info in skeleton_results:
        region = skeleton_info['region']
        info = {
            'component_id': int(region.label),
            'area': int(region.area),
            'num_endpoints': skeleton_info['num_endpoints'],
            'num_branchpoints': skeleton_info['num_branchpoints'],
            'endpoints': skeleton_info['endpoints'],
            'branchpoints': skeleton_info['branchpoints'],
            'centroid': {
                'row': float(region.centroid[0]),
                'col': float(region.centroid[1])
            }
        }
        skeletons_info.append(info)

    # 儲存 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(skeletons_info, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ 骨架資訊已儲存: {output_path}")


def visualize_skeletons(
    annotation_path: str,
    output_dir: str,
    green_channel_path: Optional[str] = None,
    show_endpoints: bool = True,
    show_branchpoints: bool = True,
    show_labels: bool = False,
    show_statistics: bool = True,
    alpha: float = 0.7,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    完整的骨架視覺化流程

    Args:
        annotation_path: Binary annotation 影像路徑
        output_dir: 輸出目錄
        green_channel_path: 可選的 green channel 背景影像路徑
        show_endpoints: 是否顯示端點標記
        show_branchpoints: 是否顯示分支點標記
        show_labels: 是否顯示 component ID 標籤
        show_statistics: 是否生成統計圖表
        alpha: 骨架混合不透明度 (0-1)
        config_path: 配置文件路徑（可選，預設使用 config/default.yaml）

    Returns:
        包含視覺化結果資訊的字典
    """
    logger.info("=" * 60)
    logger.info("開始 Skeleton 視覺化")
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
    cc_analyzer = ConnectedComponentsAnalyzer(
        connectivity=connectivity,
        min_area=min_area
    )
    regions = cc_analyzer.analyze(cc_analyzer.load_binary_image(annotation_path))
    logger.info(f"✓ 偵測到 {len(regions)} 個連通元件")

    # 2. 執行骨架化分析
    logger.info("\n步驟 2: 執行骨架化分析")
    skeleton_analyzer = SkeletonAnalyzer()
    skeleton_results = skeleton_analyzer.batch_process(regions)

    total_endpoints = sum(s['num_endpoints'] for s in skeleton_results)
    total_branchpoints = sum(s['num_branchpoints'] for s in skeleton_results)
    logger.info(f"✓ 骨架化完成")
    logger.info(f"  總端點數: {total_endpoints}")
    logger.info(f"  總分支點數: {total_branchpoints}")

    # 3. 載入背景影像
    logger.info("\n步驟 3: 準備背景影像")
    if green_channel_path:
        background_image = cv2.imread(green_channel_path, cv2.IMREAD_GRAYSCALE)
        if background_image is not None:
            logger.info(f"✓ 背景影像已載入: {background_image.shape}")
        else:
            logger.warning(f"無法載入背景影像: {green_channel_path}，使用純黑背景")
            # 載入 annotation 取得尺寸
            annotation_img = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            background_image = np.zeros_like(annotation_img)
    else:
        logger.info("使用純黑背景")
        annotation_img = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        background_image = np.zeros_like(annotation_img)

    # 4. 繪製骨架疊加圖
    logger.info("\n步驟 4: 繪製骨架疊加圖")
    
    # 載入標註遮罩以供顯示
    label_mask_img = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    
    skeleton_overlay = draw_skeleton_overlay(
        base_image=background_image,
        skeleton_results=skeleton_results,
        label_mask=label_mask_img,
        show_endpoints=show_endpoints,
        show_branchpoints=show_branchpoints,
        alpha=alpha
    )
    logger.info("✓ 骨架疊加圖已創建")

    # 5. 添加標籤（如果需要）
    if show_labels:
        logger.info("\n步驟 5: 添加 Component ID 標籤")
        skeleton_overlay = add_skeleton_labels(
            image=skeleton_overlay,
            skeleton_results=skeleton_results
        )
        logger.info(f"✓ 已添加 {len(skeleton_results)} 個標籤")
    else:
        logger.info("\n步驟 5: 跳過標籤添加")

    # 6. 儲存主視覺化影像
    logger.info("\n步驟 6: 儲存視覺化影像")
    main_output_path = output_path / "skeletons_overlay.png"
    cv2.imwrite(str(main_output_path), skeleton_overlay)
    logger.info(f"✓ 主視覺化已儲存: {main_output_path}")

    # 儲存裁切版本 (Focused View)
    crop_x, crop_y = 1700, 360
    crop_w, crop_h = 400, 300
    h, w = skeleton_overlay.shape[:2]
    
    # 簡單邊界檢查
    y2 = min(crop_y + crop_h, h)
    x2 = min(crop_x + crop_w, w)
    
    if crop_y < h and crop_x < w:
        cropped_overlay = skeleton_overlay[crop_y:y2, crop_x:x2]
        cropped_output_path = output_path / "skeletons_overlay_cropped.png"
        cv2.imwrite(str(cropped_output_path), cropped_overlay)
        logger.info(f"✓ 裁切視覺化已儲存: {cropped_output_path}")

    # 7. 生成統計圖表（如果需要）
    if show_statistics:
        logger.info("\n步驟 7: 生成統計圖表")
        stats_output_path = output_path / "skeletons_statistics.png"
        create_skeleton_statistics(skeleton_results, str(stats_output_path))

    # 8. 儲存骨架資訊
    logger.info("\n步驟 8: 儲存骨架資訊")
    info_output_path = output_path / "skeletons_info.json"
    save_skeleton_info(skeleton_results, str(info_output_path))

    logger.info("\n" + "=" * 60)
    logger.info("✓ 視覺化完成！")
    logger.info("=" * 60)
    logger.info(f"\n輸出檔案：")
    logger.info(f"  • 主視覺化: {main_output_path}")
    if show_statistics:
        logger.info(f"  • 統計圖表: {stats_output_path}")
    logger.info(f"  • 骨架資訊: {info_output_path}")

    return {
        'num_components': len(skeleton_results),
        'total_endpoints': total_endpoints,
        'total_branchpoints': total_branchpoints,
        'output_dir': str(output_path),
        'main_image': str(main_output_path),
        'skeleton_results': skeleton_results
    }


# ============================================================================
# 主程式與使用範例
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：視覺化 skeletons，使用配置系統
    """

    # 範例 1: 使用預設配置
    logger.info("\n範例 1: 使用預設配置視覺化骨架")
    visualize_skeletons(
        annotation_path='output/preprocessing_normalization/final_label.png',
        output_dir='output/skeleton_visualization/default',
        green_channel_path='output/preprocessing_normalization/roi_image.png',
        show_endpoints=True,
        show_branchpoints=True,
        show_labels=False,
        show_statistics=True,
        alpha=1
        # connectivity 和 min_area 從 config/default.yaml 自動讀取
    )

    logger.info("\n所有視覺化已完成！")
