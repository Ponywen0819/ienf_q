#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
圖片處理腳本：
1. 使用原始 RGB 圖片
2. 找到遮罩的下邊界作為中線
3. 膨脹遮罩20像素
4. 將新遮罩應用到原始圖片上
5. 儲存結果
"""

import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from pathlib import Path
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ThreadSafeCounter:
    """執行緒安全的計數器"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

def process_image_task(original_file, mask_path, label_path, output_path, expansion_pixels, success_counter):
    """單個圖片處理任務（用於多執行緒）"""
    # 尋找對應的遮罩檔案
    mask_file = mask_path / original_file.name
    label_file = label_path / original_file.name

    if not mask_file.exists() or not label_file.exists():
        print(f"警告: 找不到對應的遮罩檔案: {mask_file} 或標記檔案: {label_file}")
        return False
    
    # 設定輸出檔案路徑
    output_file = output_path / f"processed_{original_file.name}"
    
    # 處理圖片
    result = process_single_image(str(original_file), str(mask_file), 
                                str(label_file), str(output_file), expansion_pixels)

    if result:
        success_counter.increment()
    
    print()  # 空行分隔
    return result

def load_image(image_path):
    """載入圖片"""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        print(f"載入圖片失敗 {image_path}: {e}")
        return None

def prepare_image_for_processing(image_array):
    """準備圖片進行處理（保持原始 RGB 格式）"""
    if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
        # RGB圖片，保持原始格式
        return image_array
    elif len(image_array.shape) == 2:
        # 如果是灰階圖片，轉換為 RGB 格式
        return np.stack([image_array, image_array, image_array], axis=2)
    else:
        raise ValueError("不支援的圖片格式")


def _get_longest_8connected_path(delete_upper_y_contour, debug=False):
    """
    找到最長的8聯通路徑（處理多個分離區域的情況）
    """
    if len(delete_upper_y_contour) == 0:
        return np.array([])
    
    point_set = set(tuple(point) for point in delete_upper_y_contour)
    visited_global = set()
    all_paths = []
    
    directions = [(1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
    
    # 找到所有連通分量
    for point in delete_upper_y_contour:
        point_tuple = tuple(point)
        
        if point_tuple in visited_global:
            continue
        
        # 從這個點開始追蹤一條路徑
        path = []
        visited_local = set()
        current = point_tuple
        
        # 使用栈進行深度優先搜索
        stack = [current]
        
        while stack:
            current = stack.pop()
            
            if current in visited_local:
                continue
                
            visited_local.add(current)
            visited_global.add(current)
            path.append(current)
            
            # 找鄰居
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if (neighbor in point_set and 
                    neighbor not in visited_local and 
                    neighbor not in visited_global):
                    stack.append(neighbor)
        
        if len(path) > 1:  # 只保留有意義的路徑
            all_paths.append(path)
    
    if not all_paths:
        return np.array([])
    
    # 返回最長的路徑
    longest_path = max(all_paths, key=len)
    
    if debug:
        print(f"找到 {len(all_paths)} 個連通分量")
        print(f"最長路徑長度: {len(longest_path)}")
    
    return np.array(longest_path)

def _delete_upper_x(contour, mask):
	new_contour = []
	points = contour.reshape(-1, 2)  # 關鍵：先reshape

	h, w = mask.shape[:2]
	for x in range(w):
		current_x_points = points[points[:, 0] == x]
		if len(current_x_points) == 0:
			continue
		# print(f"x={x}, current_x_points={current_x_points}")
		
		max_y_idx = np.argmin(current_x_points[:, 1])
		max_y_index = current_x_points[:, 1] != current_x_points[max_y_idx, 1]

		max_y_point = current_x_points[max_y_index]

		new_contour.extend(max_y_point)
	return np.array(new_contour)

def find_mask_bottom_boundary(mask_array):
    """找到遮罩的下邊界"""
    height, width = mask_array.shape[:2]
    print(f"  圖片尺寸: {width} x {height}")
    
    # 將遮罩轉為二值圖片
    if len(mask_array.shape) == 3:
        mask_gray = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask_array.copy()
    
    # 二值化處理
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    largest_contour = max(contours, key=cv2.contourArea)
    
    largest_contour = _delete_upper_x(largest_contour, binary_mask)

    largest_contour = _get_longest_8connected_path(largest_contour, debug=False)

    print(f"  底部非零像素數: {len(largest_contour)}")

    return np.array(largest_contour)

def create_expanded_mask(original_shape, bottom_boundary_points, expansion_pixels=20):
    """創建膨脹後的遮罩"""
    height, width = original_shape[:2]
    
    print(f"  原始圖片尺寸: {width} x {height}")
    print(f"  下邊界 y 座標: {bottom_boundary_points[:, 1]}")
    print(f"  膨脹像素數: {expansion_pixels}")
    
    # 驗證輸入參數
    if bottom_boundary_points.shape[0] == 0:
        print(f"錯誤: 找不到有效的下邊界")
        return None
    
    # 創建新的遮罩
    new_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 計算新的遮罩範圍（以底部邊界為中線，向上下各擴展
    for point in bottom_boundary_points:
        # upper = max(0, point[1] - expansion_pixels)
        # lower = min(height - 1, point[1] + expansion_pixels)
        # new_mask[upper:lower + 1, point[0]] = 255
        cv2.circle(new_mask, (point[0], point[1]), expansion_pixels, 255, -1)

    print(f"  新遮罩非零像素數: {np.count_nonzero(new_mask)}")
    
    return new_mask

def apply_mask_to_image(image_array, mask_array):
    """將遮罩應用到圖片上"""
    if len(image_array.shape) == 3:
        # 彩色圖片
        result = image_array.copy()
        for channel in range(image_array.shape[2]):
            result[:, :, channel] = np.where(mask_array > 127, 
                                           image_array[:, :, channel], 
                                           0)
    else:
        # 灰階圖片
        result = np.where(mask_array > 127, image_array, 0)
    
    return result

def process_single_image(original_path, mask_path, label_path, output_path, expansion_pixels=20):
    """處理單張圖片"""
    print(f"處理圖片: {os.path.basename(original_path)}")
    
    # 1. 載入圖片
    original_img = load_image(original_path)
    mask_img = load_image(mask_path)
    label_img = load_image(label_path)

    if original_img is None or mask_img is None or label_img is None:
        print(f"跳過: 載入失敗")
        return False
    
    # 2. 準備原始 RGB 圖片
    processed_img = prepare_image_for_processing(original_img)
    print(f"  ✓ 準備 RGB 圖片: {processed_img.shape}")
    
    # 3. 找到遮罩的下邊界
    boundary_info = find_mask_bottom_boundary(mask_img)
    if boundary_info is None:
        print(f"  ✗ 找不到遮罩邊界")
        return False

    print(f"  ✓ 找到下邊界:  {len(boundary_info)} 個點")
    
    # 4. 創建膨脹後的遮罩
    expanded_mask = create_expanded_mask(original_img.shape, boundary_info, expansion_pixels)
    if expanded_mask is None:
        print(f"  ✗ 創建膨脹遮罩失敗")
        return False
    print(f"  ✓ 創建膨脹遮罩: 膨脹 {expansion_pixels} 像素")
    
    # 5. 將遮罩應用到 RGB 圖片
    masked_rgb = apply_mask_to_image(processed_img, expanded_mask)
    print(f"  ✓ 應用遮罩到 RGB 圖片")


    # 6. 將遮罩套用到標記圖片
    masked_label = apply_mask_to_image(label_img, expanded_mask)
    print(f"  ✓ 應用遮罩到標記圖片")

    # 7. 儲存結果
    try:
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 儲存處理後的圖片
        result_img = Image.fromarray(masked_rgb.astype(np.uint8), mode='RGB')
        result_img.save(output_path)
        print(f"  ✓ 儲存至: {output_path}")
        
        # 也儲存膨脹後的遮罩供檢視
        mask_output_path = output_path.replace('.tif', '_mask.tif')
        mask_result_img = Image.fromarray(expanded_mask, mode='L')
        mask_result_img.save(mask_output_path)
        print(f"  ✓ 儲存遮罩至: {mask_output_path}")

        # 儲存處理後的標記圖片
        label_output_path = output_path.replace('.tif', '_label.tif')
        label_result_img = Image.fromarray(masked_label.astype(np.uint8), mode='L')
        label_result_img.save(label_output_path)
        print(f"  ✓ 儲存標記至: {label_output_path}")

        # 儲存中間線（用於膨脹的邊界線）
        boundary_output_path = output_path.replace('.tif', '_boundary.tif')
        boundary_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
        for point in boundary_info:
            if 0 <= point[1] < boundary_mask.shape[0] and 0 <= point[0] < boundary_mask.shape[1]:
                boundary_mask[point[1], point[0]] = 255
        boundary_result_img = Image.fromarray(boundary_mask, mode='L')
        boundary_result_img.save(boundary_output_path)
        print(f"  ✓ 儲存中間線至: {boundary_output_path}")

        return True
        
    except Exception as e:
        print(f"  ✗ 儲存失敗: {e}")
        return False

def process_all_images(original_dir, mask_dir, label_dir, output_dir, expansion_pixels=20):
    """批量處理所有圖片"""
    original_path = Path(original_dir)
    mask_path = Path(mask_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)
    
    # 檢查輸入目錄
    if not original_path.exists():
        print(f"錯誤: 原始圖片目錄不存在: {original_dir}")
        return
    
    if not mask_path.exists():
        print(f"錯誤: 遮罩目錄不存在: {mask_dir}")
        return

    if not label_path.exists():
        print(f"錯誤: 標記目錄不存在: {label_dir}")
        return

    # 創建輸出目錄
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有.tif檔案
    original_files = list(original_path.glob('*.tif'))
    original_files.sort()
    
    print(f"找到 {len(original_files)} 個原始圖片檔案")
    
    success_counter = ThreadSafeCounter()
    total_count = len(original_files)
    
    # 使用多執行緒處理
    max_workers = min(12, os.cpu_count())  # 限制最大執行緒數量
    print(f"使用 {max_workers} 個執行緒進行並行處理")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任務
        future_to_file = {
            executor.submit(process_image_task, original_file, mask_path, label_path, output_path, expansion_pixels, success_counter): original_file
            for original_file in original_files
        }
        
        # 等待所有任務完成
        for future in as_completed(future_to_file):
            original_file = future_to_file[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'處理圖片 {original_file.name} 時發生錯誤: {exc}')
    
    print(f"批量處理完成: {success_counter.value}/{total_count} 成功")

def main():
    parser = argparse.ArgumentParser(description='圖片處理工具：使用原始 RGB 圖片並應用膨脹遮罩')
    
    parser.add_argument('--original', '-o', required=True,
                       help='原始圖片目錄或單個檔案路徑')
    parser.add_argument('--mask', '-m', required=True,
                       help='遮罩圖片目錄或單個檔案路徑')
    parser.add_argument('--output', '-out', required=True,
                       help='輸出目錄或檔案路徑')
    parser.add_argument('--expansion', '-e', type=int, default=40,
                       help='膨脹像素數量 (預設: 20)')
    parser.add_argument('--single', '-s', action='store_true',
                       help='處理單個檔案而非批量處理')
    
    args = parser.parse_args()
    
    print("=== 圖片處理工具 ===")
    print(f"膨脹像素: {args.expansion}")
    print()
    
    if args.single:
        # 單個檔案處理
        success = process_single_image(args.original, args.mask, 
                                     args.output, args.expansion)
        if success:
            print("處理完成!")
        else:
            print("處理失敗!")
            sys.exit(1)
    else:
        # 批量處理
        process_all_images(args.original, args.mask, args.output, args.expansion)

if __name__ == "__main__":
    # 如果沒有命令列參數，使用預設路徑
    if len(sys.argv) == 1:
        print("使用預設路徑進行批量處理...")
        current_dir = Path(__file__).parent
        process_all_images(
            original_dir=current_dir / "Original",
            mask_dir=current_dir / "Mask",
            label_dir=current_dir / "Ground_Truth",
            output_dir=current_dir / "Centered",
            expansion_pixels=40
        )
    else:
        main()