"""
分析 data/Label 中所有標註圖片的聯通元件厚度（粗細）。
使用 Distance Transform (距離變換) 方法，這比最小外接矩形更適合彎曲的長條型結構。

原理：
1. 對每個聯通元件計算距離變換 (Distance Transform)。
2. 找出該元件內部的最大距離值 (即最大半徑)。
3. 厚度 (Thickness) ≈ 2 * 最大半徑。
"""

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def analyze_labels_thickness(data_dir):
    # 取得所有 .tif 檔案
    image_paths = glob.glob(os.path.join(data_dir, "*.tif"))
    
    if not image_paths:
        print(f"在 {data_dir} 找不到任何 .tif 檔案")
        return

    print(f"找到 {len(image_paths)} 張圖片，開始使用距離變換分析厚度...")

    max_thickness = 0.0
    max_thickness_file = ""
    max_thickness_contour_idx = -1
    
    all_thicknesses = []

    for img_path in tqdm(image_paths):
        # 讀取圖片 (灰階)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"無法讀取圖片: {img_path}")
            continue

        # 二值化
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 尋找聯通元件
        # 使用 connectedComponentsWithStats 可以一次取得所有元件的 mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # labels: 標記圖，背景為 0，元件為 1, 2, ...
        # num_labels: 元件數量 + 1 (背景)
        
        # 為了效率，我們只對有足夠面積的元件做分析
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 5: # 忽略極小雜訊
                continue

            # 建立該元件的 Mask
            # 這裡使用一個小技巧：只在該元件的 bounding box 內做計算，加速處理
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 取出 ROI
            component_roi = labels[y:y+h, x:x+w]
            
            # 建立二值 mask (只包含當前元件 i)
            # 為了避免元件填滿整個 ROI 導致 distanceTransform 找不到 0 (背景)，我們需要做 padding
            mask_roi = (component_roi == i).astype(np.uint8) * 255
            mask = cv2.copyMakeBorder(mask_roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            
            # 計算距離變換 (L2 distance)
            # dist_map 的每個數值代表該像素離最近背景的距離
            dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            # 該元件的最大半徑
            max_radius = float(np.max(dist_map))
            
            # 厚度 = 直徑 = 2 * 半徑
            thickness = 2.0 * max_radius
            
            if np.isinf(thickness) or np.isnan(thickness):
                continue

            all_thicknesses.append(thickness)

            if thickness > max_thickness:
                max_thickness = thickness
                max_thickness_file = img_path
                max_thickness_contour_idx = i

    print("\n" + "="*50)
    print("分析結果 (Distance Transform Method)")
    print("="*50)
    print(f"分析圖片數量: {len(image_paths)}")
    print(f"總聯通元件數量: {len(all_thicknesses)}")
    
    if all_thicknesses:
        print(f"所有聯通元件中最大厚度: {max_thickness:.4f} pixels")
        print(f"發生在檔案: {max_thickness_file}")
        print(f"平均厚度: {np.mean(all_thicknesses):.4f} pixels")
        print(f"中位數厚度: {np.median(all_thicknesses):.4f} pixels")
        print(f"99分位數厚度 (排除極端值參考): {np.percentile(all_thicknesses, 99):.4f} pixels")
    else:
        print("未發現任何有效的聯通元件。")

if __name__ == "__main__":
    # 設定資料夾路徑
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(project_root, "data", "Label")
    
    print(f"Data directory: {DATA_DIR}")

    analyze_labels_thickness(DATA_DIR)
