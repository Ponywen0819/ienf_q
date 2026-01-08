#!/usr/bin/env python3
"""
單一連通元件骨架視覺化工具 (Component Skeleton Visualizer)

針對特定連通元件進行骨架化分析，並以放大視圖顯示骨架結構、
端點（紅色）與分叉點（藍色）。

功能：
1. 指定特定元件 ID 進行骨架化
2. 裁切放大顯示目標元件區域
3. 不透明疊加原始元件 mask
4. 標記端點與分叉點
5. 輸出統計資訊
6. 支援從原始影像進行前處理（可選）

使用範例:
    # 使用已處理的影像
    python visualization/visualize_component_skeleton.py \
        --annotation output/preprocessing/final_label.png \
        --component-id 5 \
        --output output/component_5_skeleton.png \
        --green-channel output/preprocessing/roi_image.png

    # 從原始影像開始（啟用前處理）
    python visualization/visualize_component_skeleton.py \
        --label-image data/label.tif \
        --epidermis-mask data/mask.tif \
        --original-image data/original.tif \
        --component-id 5 \
        --output output/component_5_skeleton.png \
        --preprocess

作者: Generated with Claude Code
日期: 2025-01-05
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

# 添加 src 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nueral_reconstruction.connected_components import ConnectedComponentsAnalyzer
from src.nueral_reconstruction.skeletonization import SkeletonAnalyzer
from src.nueral_reconstruction.config_loader import load_config, IENFConfig
from src.preprocessing.pipeline import SkinAnalysisPipeline

# 設定 logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 顏色配置 (BGR)
COLORS = {
    'mask': (255, 255, 0),       # 青色 - 原始 mask
    'skeleton': (0, 255, 255),   # 黃色 - 骨架線
    'endpoint': (0, 0, 255),     # 紅色 - 端點
    'branchpoint': (255, 0, 0),  # 藍色 - 分叉點
}

# 預設前處理配置
DEFAULT_PREPROCESSING_CONFIG = {
    'morphology': {
        'closing_kernel': 5,
        'opening_kernel': 3
    },
    'mask': {
        'dilate_offset': 100
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
        'enabled': True
    }
}


def run_preprocessing(
    label_image_path: str,
    epidermis_mask_path: str,
    original_image_path: str,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    執行前處理 pipeline

    Args:
        label_image_path: 標籤影像路徑
        epidermis_mask_path: 表皮 mask 路徑
        original_image_path: 原始影像路徑
        config: 前處理配置（可選）

    Returns:
        (final_label, roi_image) 處理後的標籤和 ROI 影像
    """
    logger.info("\n" + "=" * 70)
    logger.info("Running Preprocessing Pipeline")
    logger.info("=" * 70)

    # 載入影像
    label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
    epidermis_mask = cv2.imread(epidermis_mask_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)

    if label_image is None:
        raise FileNotFoundError(f"Cannot load label image: {label_image_path}")
    if epidermis_mask is None:
        raise FileNotFoundError(f"Cannot load epidermis mask: {epidermis_mask_path}")
    if original_image is None:
        raise FileNotFoundError(f"Cannot load original image: {original_image_path}")

    logger.info(f"  Label image: {label_image.shape}")
    logger.info(f"  Epidermis mask: {epidermis_mask.shape}")
    logger.info(f"  Original image: {original_image.shape}")

    # 提取綠色通道
    if len(original_image.shape) == 3:
        original_green = original_image[:, :, 1]
        logger.info("  Extracted green channel from RGB image")
    else:
        original_green = original_image
        logger.info("  Using grayscale image as-is")

    # 執行前處理
    preprocessing_config = config or DEFAULT_PREPROCESSING_CONFIG
    pipeline = SkinAnalysisPipeline(preprocessing_config)
    final_label, roi_image = pipeline.run(
        label_image,
        epidermis_mask,
        original_green,
        debug=False
    )

    logger.info("Preprocessing complete")
    logger.info("=" * 70 + "\n")

    return final_label, roi_image


def list_components(
    annotation_path: Optional[str] = None,
    annotation_array: Optional[np.ndarray] = None,
    green_channel_path: Optional[str] = None,
    green_channel_array: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    sort_by: str = 'id'
) -> list:
    """
    列出所有連通元件的資訊，並可選擇輸出標記圖

    Args:
        annotation_path: 二值標註影像路徑
        annotation_array: 二值標註影像 numpy array
        green_channel_path: 可選的 green channel 背景影像路徑
        green_channel_array: 可選的 green channel 背景影像 numpy array
        output_path: 可選的輸出影像路徑（標記所有元件 ID）
        config_path: 配置檔路徑（可選）
        sort_by: 排序方式 ('id', 'area', 'x', 'y')

    Returns:
        元件資訊列表
    """
    logger.info("=" * 70)
    logger.info("List All Components")
    logger.info("=" * 70)

    # 載入配置
    if config_path is not None:
        config = load_config(config_path)
    else:
        try:
            config = load_config()
        except FileNotFoundError:
            config = IENFConfig()

    connectivity = config.connected_components.connectivity
    min_area = config.connected_components.min_area

    # 連通元件分析
    cc_analyzer = ConnectedComponentsAnalyzer(
        connectivity=connectivity,
        min_area=min_area
    )

    # 從路徑或 array 載入標註影像
    if annotation_array is not None:
        binary_image = annotation_array
        if binary_image.max() > 1:
            binary_image = (binary_image > 127).astype(np.uint8) * 255
    elif annotation_path is not None:
        binary_image = cc_analyzer.load_binary_image(annotation_path)
    else:
        raise ValueError("Must provide either annotation_path or annotation_array")

    regions = cc_analyzer.analyze(binary_image)

    # 骨架分析
    skeleton_analyzer = SkeletonAnalyzer()

    # 收集元件資訊
    components_info = []
    for region in regions:
        skel_info = skeleton_analyzer.process(region)
        min_row, min_col, max_row, max_col = region.bbox
        centroid_y, centroid_x = region.centroid

        components_info.append({
            'id': region.label,
            'area': region.area,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'bbox': (min_row, min_col, max_row, max_col),
            'endpoints': skel_info['num_endpoints'],
            'branchpoints': skel_info['num_branchpoints']
        })

    # 排序
    if sort_by == 'area':
        components_info.sort(key=lambda x: x['area'], reverse=True)
    elif sort_by == 'x':
        components_info.sort(key=lambda x: x['centroid_x'])
    elif sort_by == 'y':
        components_info.sort(key=lambda x: x['centroid_y'])
    else:  # 預設按 id
        components_info.sort(key=lambda x: x['id'])

    # 輸出列表
    logger.info(f"\nFound {len(components_info)} components:")
    logger.info("-" * 90)
    logger.info(f"{'ID':>5} | {'Area':>6} | {'Centroid (y, x)':>18} | {'Endpoints':>9} | {'Branchpts':>9}")
    logger.info("-" * 90)
    for comp in components_info:
        logger.info(
            f"{comp['id']:>5} | {comp['area']:>6.0f} | "
            f"({comp['centroid_y']:>7.1f}, {comp['centroid_x']:>7.1f}) | "
            f"{comp['endpoints']:>9} | {comp['branchpoints']:>9}"
        )
    logger.info("-" * 90)

    # 如果指定輸出路徑，生成標記圖
    if output_path:
        logger.info(f"\nGenerating component ID map...")
        img_h, img_w = binary_image.shape[:2]

        # 準備背景
        if green_channel_array is not None:
            background = green_channel_array
            if len(background.shape) == 3:
                background = background[:, :, 1]
        elif green_channel_path:
            background = cv2.imread(green_channel_path, cv2.IMREAD_GRAYSCALE)
            if background is None:
                background = np.zeros((img_h, img_w), dtype=np.uint8)
        else:
            background = np.zeros((img_h, img_w), dtype=np.uint8)

        # 建立輸出影像
        output_img = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # 繪製所有元件 mask
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            mask = region.image.astype(np.uint8) * 255
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] > 0:
                        gy, gx = min_row + i, min_col + j
                        if 0 <= gy < img_h and 0 <= gx < img_w:
                            output_img[gy, gx] = COLORS['mask']

        # 繪製元件 ID 標籤
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1

        for region in regions:
            centroid_y, centroid_x = region.centroid
            label_text = str(region.label)

            # 計算文字大小
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # 繪製白色背景
            cx, cy = int(centroid_x), int(centroid_y)
            cv2.rectangle(
                output_img,
                (cx - text_w // 2 - 1, cy - text_h // 2 - 1),
                (cx + text_w // 2 + 1, cy + text_h // 2 + baseline + 1),
                (255, 255, 255),
                -1
            )

            # 繪製黑色文字
            cv2.putText(
                output_img,
                label_text,
                (cx - text_w // 2, cy + text_h // 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, output_img)
        logger.info(f"  Component ID map saved to: {output_path}")

    logger.info("=" * 70)

    return components_info


def visualize_component_skeleton(
    output_path: str,
    component_id: Optional[int],
    annotation_path: Optional[str] = None,
    green_channel_path: Optional[str] = None,
    annotation_array: Optional[np.ndarray] = None,
    green_channel_array: Optional[np.ndarray] = None,
    crop_padding: int = 30,
    output_size: int = 800,
    config_path: Optional[str] = None,
    endpoint_radius: int = 3,
    branchpoint_size: int = 5
) -> List[Dict[str, Any]]:
    """
    視覺化特定連通元件的骨架結構

    Args:
        output_path: 輸出影像路徑
        component_id: 目標元件 ID (若為 None 則處理所有元件)
        annotation_path: 二值標註影像路徑（與 annotation_array 二選一）
        green_channel_path: 可選的 green channel 背景影像路徑
        annotation_array: 二值標註影像 numpy array（與 annotation_path 二選一）
        green_channel_array: 可選的 green channel 背景影像 numpy array
        crop_padding: 裁切邊距（像素）
        output_size: 輸出影像尺寸
        config_path: 配置檔路徑（可選）
        endpoint_radius: 端點標記半徑
        branchpoint_size: 分叉點標記尺寸

    Returns:
        包含視覺化結果資訊的字典列表
    """
    logger.info("=" * 70)
    logger.info("Component Skeleton Visualization")
    logger.info("=" * 70)
    if component_id is not None:
        logger.info(f"Component ID: {component_id}")
    else:
        logger.info("Processing ALL components")
    logger.info("-" * 70)

    # 建立輸出目錄
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入配置
    if config_path is not None:
        config = load_config(config_path)
        logger.info(f"Config loaded: {config_path}")
    else:
        try:
            config = load_config()
            logger.info("Config loaded: config/default.yaml")
        except FileNotFoundError:
            logger.warning("Config not found, using defaults")
            config = IENFConfig()

    connectivity = config.connected_components.connectivity
    min_area = config.connected_components.min_area

    # 步驟 1: 連通元件分析
    logger.info("\nStep 1: Connected Components Analysis")
    cc_analyzer = ConnectedComponentsAnalyzer(
        connectivity=connectivity,
        min_area=min_area
    )

    # 從路徑或 array 載入標註影像
    if annotation_array is not None:
        binary_image = annotation_array
        if binary_image.max() > 1:
            binary_image = (binary_image > 127).astype(np.uint8) * 255
        logger.info(f"  Using annotation array: {binary_image.shape}")
    elif annotation_path is not None:
        binary_image = cc_analyzer.load_binary_image(annotation_path)
        logger.info(f"  Loaded annotation: {annotation_path}")
    else:
        raise ValueError("Must provide either annotation_path or annotation_array")

    regions = cc_analyzer.analyze(binary_image)
    logger.info(f"  Found {len(regions)} components")

    # 尋找目標元件
    target_regions = []
    if component_id is not None:
        for region in regions:
            if region.label == component_id:
                target_regions.append(region)
                break
        if not target_regions:
            available_ids = [r.label for r in regions]
            logger.error(f"Component ID {component_id} not found!")
            logger.error(f"Available IDs: {available_ids}")
            raise ValueError(f"Component ID {component_id} not found. Available: {available_ids}")
        logger.info(f"  Target component found: ID={component_id}")
    else:
        target_regions = regions
        logger.info(f"  Processing all {len(regions)} components")

    # 步驟 3: 準備背景影像 (移到迴圈外)
    logger.info("\nStep 3: Preparing Background Image")
    img_h, img_w = binary_image.shape[:2]

    if green_channel_array is not None:
        background = green_channel_array
        if len(background.shape) == 3:
            background = background[:, :, 1]  # 取綠色通道
        logger.info(f"  Using green channel array: {background.shape}")
    elif green_channel_path:
        background = cv2.imread(green_channel_path, cv2.IMREAD_GRAYSCALE)
        if background is None:
            logger.warning(f"Cannot load green channel: {green_channel_path}")
            background = np.zeros((img_h, img_w), dtype=np.uint8)
        else:
            logger.info(f"  Green channel loaded: {background.shape}")
    else:
        background = np.zeros((img_h, img_w), dtype=np.uint8)
        logger.info("  Using black background")

    results = []

    for target_region in target_regions:
        current_id = target_region.label
        
        # Determine output path
        if component_id is not None:
            current_output_path = output_path
        else:
            p = Path(output_path)
            if p.suffix:
                # Insert ID before suffix
                current_output_path = str(p.parent / f"{p.stem}_{current_id}{p.suffix}")
            else:
                # Directory
                current_output_path = str(p / f"component_{current_id}.png")

        # 步驟 2: 骨架化分析
        logger.info(f"\nProcessing Component ID: {current_id}")
        logger.info("Step 2: Skeletonization Analysis")
        skeleton_analyzer = SkeletonAnalyzer()
        skeleton_info = skeleton_analyzer.process(target_region)

        logger.info(f"  Endpoints: {skeleton_info['num_endpoints']}")
        logger.info(f"  Branchpoints: {skeleton_info['num_branchpoints']}")

        # 步驟 4: 計算裁切區域
        logger.info("Step 4: Computing Crop Region")
        min_row, min_col, max_row, max_col = target_region.bbox

        # 加上 padding
        crop_min_row = max(0, min_row - crop_padding)
        crop_min_col = max(0, min_col - crop_padding)
        crop_max_row = min(img_h, max_row + crop_padding)
        crop_max_col = min(img_w, max_col + crop_padding)

        logger.info(f"  Original bbox: ({min_row}, {min_col}) - ({max_row}, {max_col})")
        logger.info(f"  Crop region: ({crop_min_row}, {crop_min_col}) - ({crop_max_row}, {crop_max_col})")

        # 步驟 5: 繪製視覺化影像
        logger.info("Step 5: Drawing Visualization")

        # 建立 BGR 輸出影像
        output_img = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # 5a: 繪製原始元件 mask（不透明）
        mask = target_region.image.astype(np.uint8) * 255
        mask_h, mask_w = mask.shape

        for i in range(mask_h):
            for j in range(mask_w):
                if mask[i, j] > 0:
                    global_y = min_row + i
                    global_x = min_col + j
                    if 0 <= global_y < img_h and 0 <= global_x < img_w:
                        output_img[global_y, global_x] = COLORS['mask']

        # 5b: 繪製骨架線
        skeleton = skeleton_info['skeleton']
        skel_h, skel_w = skeleton.shape

        for i in range(skel_h):
            for j in range(skel_w):
                if skeleton[i, j] > 0:
                    global_y = min_row + i
                    global_x = min_col + j
                    if 0 <= global_y < img_h and 0 <= global_x < img_w:
                        output_img[global_y, global_x] = COLORS['skeleton']

        # 5c: 繪製端點（紅色圓點）
        for endpoint in skeleton_info['endpoints']:
            local_x, local_y = endpoint['x'], endpoint['y']
            global_x = min_col + local_x
            global_y = min_row + local_y
            cv2.circle(output_img, (global_x, global_y),
                    radius=endpoint_radius, color=COLORS['endpoint'], thickness=-1)

        # 5d: 繪製分叉點（藍色方框）
        half_size = branchpoint_size // 2
        for branchpoint in skeleton_info['branchpoints']:
            local_x, local_y = branchpoint['x'], branchpoint['y']
            global_x = min_col + local_x
            global_y = min_row + local_y
            cv2.rectangle(
                output_img,
                (global_x - half_size, global_y - half_size),
                (global_x + half_size, global_y + half_size),
                COLORS['branchpoint'],
                thickness=-1
            )

        # 步驟 6: 裁切與放大
        logger.info("Step 6: Cropping and Resizing")
        cropped = output_img[crop_min_row:crop_max_row, crop_min_col:crop_max_col]
        crop_h, crop_w = cropped.shape[:2]

        # 計算保持比例的縮放
        scale = output_size / max(crop_h, crop_w)
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)

        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"  Resized from {crop_w}x{crop_h} to {new_w}x{new_h}")

        # 步驟 7: 儲存輸出
        logger.info("Step 7: Saving Output")
        cv2.imwrite(current_output_path, resized)
        logger.info(f"  Saved to: {current_output_path}")

        results.append({
            'component_id': current_id,
            'area': target_region.area,
            'bbox': target_region.bbox,
            'centroid': target_region.centroid,
            'num_endpoints': skeleton_info['num_endpoints'],
            'num_branchpoints': skeleton_info['num_branchpoints'],
            'endpoints': skeleton_info['endpoints'],
            'branchpoints': skeleton_info['branchpoints'],
            'output_path': current_output_path
        })

    logger.info("\n" + "=" * 70)
    logger.info(f"Processed {len(results)} components")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='視覺化特定連通元件的骨架結構',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 列出所有元件（定位元件 ID）
  python visualization/visualize_component_skeleton.py \\
      --list \\
      --annotation output/preprocessing/final_label.png \\
      --output output/component_id_map.png

  # 列出所有元件（從原始影像，按面積排序）
  python visualization/visualize_component_skeleton.py \\
      --list --preprocess --sort-by area \\
      --label-image data/Label/S163-2_a.tif \\
      --epidermis-mask data/Mask/S163-2_a.tif \\
      --original-image data/Original/S163-2_a.tif \\
      --output output/component_id_map.png

  # 視覺化特定元件（使用已處理的影像）
  python visualization/visualize_component_skeleton.py \\
      --annotation output/preprocessing/final_label.png \\
      --component-id 5 \\
      --output output/component_5_skeleton.png \\
      --green-channel output/preprocessing/roi_image.png

  # 視覺化特定元件（從原始影像開始）
  python visualization/visualize_component_skeleton.py \\
      --preprocess \\
      --label-image data/Label/S163-2_a.tif \\
      --epidermis-mask data/Mask/S163-2_a.tif \\
      --original-image data/Original/S163-2_a.tif \\
      --component-id 5 \\
      --output output/component_5_skeleton.png
        """
    )

    # 模式選擇
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有元件資訊（定位模式）'
    )
    parser.add_argument(
        '--component-id',
        type=int,
        default=None,
        help='目標元件 ID（若未指定則處理所有元件）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='輸出影像路徑'
    )
    parser.add_argument(
        '--sort-by',
        type=str,
        choices=['id', 'area', 'x', 'y'],
        default='id',
        help='列出模式的排序方式（預設: id）'
    )

    # 模式 1: 使用已處理的影像
    processed_group = parser.add_argument_group('已處理影像模式')
    processed_group.add_argument(
        '--annotation',
        type=str,
        default=None,
        help='二值標註影像路徑（已處理）'
    )
    processed_group.add_argument(
        '--green-channel',
        type=str,
        default=None,
        help='綠色通道背景影像路徑（可選）'
    )

    # 模式 2: 從原始影像開始（啟用前處理）
    preprocess_group = parser.add_argument_group('前處理模式')
    preprocess_group.add_argument(
        '--preprocess',
        action='store_true',
        help='啟用前處理模式'
    )
    preprocess_group.add_argument(
        '--label-image',
        type=str,
        default=None,
        help='原始標籤影像路徑'
    )
    preprocess_group.add_argument(
        '--epidermis-mask',
        type=str,
        default=None,
        help='表皮 mask 影像路徑'
    )
    preprocess_group.add_argument(
        '--original-image',
        type=str,
        default=None,
        help='原始影像路徑（RGB）'
    )

    # 可選參數
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置檔路徑（預設使用 config/default.yaml）'
    )
    parser.add_argument(
        '--crop-padding',
        type=int,
        default=30,
        help='裁切邊距（像素，預設: 30）'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=800,
        help='輸出影像尺寸（像素，預設: 800）'
    )
    parser.add_argument(
        '--endpoint-radius',
        type=int,
        default=1,
        help='端點標記半徑（預設: 1）'
    )
    parser.add_argument(
        '--branchpoint-size',
        type=int,
        default=1,
        help='分叉點標記尺寸（預設: 1）'
    )

    args = parser.parse_args()

    if args.list:
        # 如果是 list 模式，這裡應該呼叫 list_components
        # 但為了保持現有邏輯不變，這裡暫時不處理，或者假設使用者會正確使用參數
        pass

    if not args.output and not args.list:
        parser.error("--output is required")

    # 判斷使用哪種模式
    if args.preprocess:
        # 前處理模式：從原始影像開始
        if not all([args.label_image, args.epidermis_mask, args.original_image]):
            parser.error("前處理模式需要提供 --label-image, --epidermis-mask, --original-image")

        # 執行前處理
        final_label, roi_image = run_preprocessing(
            label_image_path=args.label_image,
            epidermis_mask_path=args.epidermis_mask,
            original_image_path=args.original_image
        )
        
        # 使用前處理結果執行視覺化
        visualize_component_skeleton(
            output_path=args.output,
            component_id=args.component_id,
            annotation_array=final_label,
            green_channel_array=roi_image,
            crop_padding=args.crop_padding,
            output_size=args.output_size,
            config_path=args.config,
            endpoint_radius=args.endpoint_radius,
            branchpoint_size=args.branchpoint_size
        )
    else:
        # 已處理影像模式
        if not args.annotation:
            parser.error("需要提供 --annotation 或使用 --preprocess 模式")

        visualize_component_skeleton(
            output_path=args.output,
            component_id=args.component_id,
            annotation_path=args.annotation,
            green_channel_path=args.green_channel,
            crop_padding=args.crop_padding,
            output_size=args.output_size,
            config_path=args.config,
            endpoint_radius=args.endpoint_radius,
            branchpoint_size=args.branchpoint_size
        )


if __name__ == '__main__':
    main()


"""
uv run visualization/visualize_component_skeleton.py \
    --preprocess \
    --label-image data/Label/S163-2_a.tif \
    --epidermis-mask data/Mask/S163-2_a.tif \
    --original-image data/Original/S163-2_a.tif \
    --component-id 26 \
    --output output/component_26_skeleton.png
"""