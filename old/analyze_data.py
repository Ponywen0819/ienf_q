import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_tiff_files():
    data_dir = Path("/home/bl515-ml/Documents/shaio_jie/ienf_q/Centered")
    
    # 找一個範例檔案組
    sample_base = "processed_S1037-2_a"
    
    # 加載不同類型的檔案
    image_path = data_dir / f"{sample_base}.tif"
    label_path = data_dir / f"{sample_base}_label.tif"
    mask_path = data_dir / f"{sample_base}_mask.tif"
    boundary_path = data_dir / f"{sample_base}_boundary.tif"
    
    # 讀取圖像
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    boundary = cv2.imread(str(boundary_path), cv2.IMREAD_UNCHANGED)
    
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Label shape: {label.shape}, dtype: {label.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Boundary shape: {boundary.shape}, dtype: {boundary.dtype}")
    
    # 檢查 label 中的唯一值
    unique_label_values = np.unique(label)
    print(f"Unique values in label: {unique_label_values}")
    
    # 檢查 255 值的分布
    neural_pixels = np.sum(label == 255)
    total_pixels = label.size
    print(f"Neural pixels (255): {neural_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"Neural pixel ratio: {neural_pixels/total_pixels:.6f}")
    
    # 檢查 mask 的值
    unique_mask_values = np.unique(mask)
    print(f"Unique values in mask: {unique_mask_values}")
    
    # 尋找中心線（通常mask中會有特殊標記）
    mask_shape = mask.shape
    center_y = mask_shape[0] // 2
    print(f"Image dimensions: {mask_shape}")
    print(f"Estimated center line at y={center_y}")
    
    # 檢查上半部分的mask區域
    upper_half = mask[:center_y]
    upper_valid_pixels = np.sum(upper_half > 0)
    print(f"Valid pixels in upper half: {upper_valid_pixels}")
    
    return {
        'image': image,
        'label': label,
        'mask': mask,
        'boundary': boundary,
        'neural_pixels': neural_pixels,
        'center_y': center_y,
        'upper_valid_pixels': upper_valid_pixels
    }

if __name__ == "__main__":
    result = analyze_tiff_files()