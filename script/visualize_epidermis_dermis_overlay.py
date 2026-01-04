"""
可視化表皮與真皮層標記並重疊在影像上

此腳本使用 SkinAnalysisPipeline 處理影像，並分別提取：
1. 表皮層標記 (processed_label) -> 紅色
2. 真皮層標記 (pseudo_label) -> 藍色

將其重疊在原始影像與處理後影像上以進行檢查。
"""
import cv2
import numpy as np
import os
from preprocessing import SkinAnalysisPipeline

# ========================================
# 裁切區域設定 (可自行調整)
# ========================================
CROP_X = 1700      # 左上角 x 座標
CROP_Y = 360      # 左上角 y 座標
CROP_WIDTH = 400  # 裁切寬度
CROP_HEIGHT = 300 # 裁切高度

def crop_region(image: np.ndarray) -> np.ndarray:
    """裁切指定區域"""
    h, w = image.shape[:2]
    x1 = min(CROP_X, w - 1)
    y1 = min(CROP_Y, h - 1)
    x2 = min(CROP_X + CROP_WIDTH, w)
    y2 = min(CROP_Y + CROP_HEIGHT, h)
    return image[y1:y2, x1:x2]

def create_dual_visualization(
    image: np.ndarray, 
    epidermis_label: np.ndarray, 
    dermis_label: np.ndarray, 
    roi_mask: np.ndarray
) -> np.ndarray:
    """
    建立雙色可視化結果：
    1. 表皮層標記 (紅色，不透明)
    2. 真皮層標記 (藍色，不透明)
    3. 真皮層 ROI 範圍 (綠色輪廓)
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
        
    # 1. 繪製表皮層標記 (紅色: BGR 0, 0, 255)
    if epidermis_label is not None:
        vis[epidermis_label > 0] = (0, 0, 255)
        
    # 2. 繪製真皮層標記 (藍色: BGR 255, 0, 0)
    if dermis_label is not None:
        vis[dermis_label > 0] = (255, 0, 0)
    
    # 3. 繪製真皮層 ROI 範圍 (綠色輪廓)
    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    
    return vis

if __name__ == "__main__":
    # ========================================
    # 配置
    # ========================================
    config = {
        'morphology': {'closing_kernel': 5, 'opening_kernel': 3},
        'mask': {'dilate_offset': 40},
        'background': {
            'method': 'rolling_ball',
            'radius': 2,
            'light_background': False
        },
        'threshold': {'method': 'binary'},
        'normalization': {'enabled': True}
    }

    # ========================================
    # 載入測試影像
    # ========================================
    IMAGE_ID = 'S163-2_a'
    LABEL_IMAGE_PATH = f'data/Label/{IMAGE_ID}.tif'
    EPIDERMIS_MASK_PATH = f'data/Mask/{IMAGE_ID}.tif'
    ORIGINAL_IMAGE_PATH = f'data/Original/{IMAGE_ID}.tif'
    
    print(f"載入影像: {IMAGE_ID}")
    label_image = cv2.imread(LABEL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    epidermis_mask = cv2.imread(EPIDERMIS_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    
    if original_image is None:
        print(f"錯誤: 無法讀取影像 {ORIGINAL_IMAGE_PATH}")
        exit(1)
        
    # 提取綠色通道
    if len(original_image.shape) == 3:
        original_green_image = original_image[:, :, 1]
    else:
        original_green_image = original_image

    # ========================================
    # 執行 Pipeline (開啟 debug 模式)
    # ========================================
    print("執行 SkinAnalysisPipeline...")
    pipeline = SkinAnalysisPipeline(config)
    # 使用 debug=True 來獲取中間結果
    final_label, roi_image, debug_output = pipeline.run(
        label_image, 
        epidermis_mask, 
        original_green_image, 
        debug=True
    )

    # 獲取標記
    epidermis_label = debug_output.processed_label # 表皮層 (來自輸入 Label)
    dermis_label = debug_output.pseudo_label       # 真皮層 (Pipeline 生成)
    dermis_roi_mask = debug_output.dermis_roi_mask
    processed_image = debug_output.background_corrected

    # ========================================
    # 儲存結果
    # ========================================
    output_dir = 'output/epidermis_dermis_visualization'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 建立可視化圖 (全圖) - 原始影像
    vis_full = create_dual_visualization(original_green_image, epidermis_label, dermis_label, dermis_roi_mask)
    cv2.imwrite(f'{output_dir}/dual_vis_full.png', vis_full)

    # 2. 建立可視化圖 (全圖) - 處理後影像
    vis_processed_full = create_dual_visualization(processed_image, epidermis_label, dermis_label, dermis_roi_mask)
    cv2.imwrite(f'{output_dir}/dual_vis_processed_full.png', vis_processed_full)

    # 3. 裁切區域檢視
    print(f"裁切區域: x={CROP_X}, y={CROP_Y}, w={CROP_WIDTH}, h={CROP_HEIGHT}")
    
    crop_original = crop_region(original_green_image)
    crop_processed = crop_region(processed_image)
    crop_epidermis_label = crop_region(epidermis_label)
    crop_dermis_label = crop_region(dermis_label)
    crop_dermis_roi = crop_region(dermis_roi_mask)
    
    vis_crop = create_dual_visualization(crop_original, crop_epidermis_label, crop_dermis_label, crop_dermis_roi)
    cv2.imwrite(f'{output_dir}/dual_vis_crop.png', vis_crop)

    vis_processed_crop = create_dual_visualization(crop_processed, crop_epidermis_label, crop_dermis_label, crop_dermis_roi)
    cv2.imwrite(f'{output_dir}/dual_vis_processed_crop.png', vis_processed_crop)

    print(f"\n結果已儲存至 {output_dir}/")
    print(f"  - dual_vis_full.png: 雙色標記疊加 (原始影像)")
    print(f"  - dual_vis_processed_full.png: 雙色標記疊加 (處理後影像)")
    print(f"  - dual_vis_crop.png: 雙色標記疊加 (原始影像裁切)")
    print(f"  - dual_vis_processed_crop.png: 雙色標記疊加 (處理後影像裁切)")
    print(f"  顏色說明: 表皮層=紅色, 真皮層=藍色, 真皮範圍=綠框")
