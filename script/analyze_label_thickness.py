"""
分析 data/Label 中所有標註圖片的聯通元件最短邊（寬度/厚度）。
找出所有圖片中，聯通元件最短邊的最大值。
"""

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def analyze_labels(data_dir):
    # 取得所有 .tif 檔案
    image_paths = glob.glob(os.path.join(data_dir, "*.tif"))
    
    if not image_paths:
        print(f"在 {data_dir} 找不到任何 .tif 檔案")
        return

    print(f"找到 {len(image_paths)} 張圖片，開始分析...")

    max_shortest_side = 0.0
    max_shortest_side_file = ""
    max_shortest_side_contour_idx = -1
    
    all_shortest_sides = []

    for img_path in tqdm(image_paths):
        # 讀取圖片 (灰階)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"無法讀取圖片: {img_path}")
            continue

        # 二值化 (確保是 0 或 255)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 尋找輪廓 (聯通元件)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            # 忽略太小的雜訊 (可選，這裡設為面積 > 0)
            if cv2.contourArea(contour) < 1:
                continue

            # 取得最小外接矩形
            # rect = ((center_x, center_y), (width, height), angle)
            rect = cv2.minAreaRect(contour)
            (width, height) = rect[1]

            # 最短邊
            shortest_side = min(width, height)
            
            # 紀錄
            all_shortest_sides.append(shortest_side)

            if shortest_side > max_shortest_side:
                max_shortest_side = shortest_side
                max_shortest_side_file = img_path
                max_shortest_side_contour_idx = i

    print("\n" + "="*50)
    print("分析結果")
    print("="*50)
    print(f"分析圖片數量: {len(image_paths)}")
    print(f"總聯通元件數量: {len(all_shortest_sides)}")
    
    if all_shortest_sides:
        print(f"所有聯通元件中最短邊的最大值: {max_shortest_side:.4f} pixels")
        print(f"發生在檔案: {max_shortest_side_file}")
        print(f"平均最短邊: {np.mean(all_shortest_sides):.4f} pixels")
        print(f"中位數最短邊: {np.median(all_shortest_sides):.4f} pixels")
    else:
        print("未發現任何有效的聯通元件。")

if __name__ == "__main__":
    # 設定資料夾路徑
    # 假設腳本在 script/ 下，data 在專案根目錄下
    # 使用絕對路徑確保正確
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(project_root, "data", "Label")
    
    print(f"Data directory: {DATA_DIR}")

    analyze_labels(DATA_DIR)
