#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
螢光染料分割工具：使用 ICA (獨立成分分析) 分離兩種螢光染料頻道

主要功能：
1. 載入含有兩種螢光染料的圖像
2. 使用 ICA 演算法分離獨立的螢光染料信號
3. 後處理和去噪
4. 視覺化分離結果
5. 批量處理多個圖像
"""

import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class FluorescenceICAProcessor:
    """螢光染料 ICA 分離處理器"""
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        """
        初始化處理器
        
        Args:
            n_components: ICA 成分數量（預設2，對應兩種染料）
            random_state: 隨機種子，確保結果可重現
        """
        self.n_components = n_components
        self.random_state = random_state
        self.ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
        self.scaler = StandardScaler()
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        載入圖像
        
        Args:
            image_path: 圖像文件路徑
            
        Returns:
            圖像數組或None（載入失敗時）
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # 確保是彩色圖像
            if len(img_array.shape) == 2:
                # 如果是灰階圖，轉為RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
                # 如果有alpha通道，只取RGB
                img_array = img_array[:, :, :3]
                
            print(f"✓ 載入圖像: {os.path.basename(image_path)} - 尺寸: {img_array.shape}")
            return img_array
            
        except Exception as e:
            print(f"✗ 載入圖像失敗 {image_path}: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        圖像預處理
        
        Args:
            image: 輸入圖像
            
        Returns:
            預處理後的圖像
        """
        # 轉換為float類型並正規化到[0,1]
        processed = image.astype(np.float64) / 255.0
        
        # 高斯濾波去噪
        for i in range(processed.shape[2]):
            processed[:, :, i] = ndimage.gaussian_filter(processed[:, :, i], sigma=0.5)
        
        return processed
    
    def prepare_data_for_ica(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        為ICA準備數據格式
        
        Args:
            image: 預處理後的圖像 (H, W, C)
            
        Returns:
            重塑後的數據 (n_pixels, n_channels) 和原始形狀
        """
        height, width, channels = image.shape
        # 重塑為 (n_pixels, n_channels)
        data = image.reshape(-1, channels)
        
        # 移除全零像素（背景）
        non_zero_mask = np.any(data > 0.01, axis=1)  # 閾值設定為0.01
        filtered_data = data[non_zero_mask]
        
        print(f"  有效像素數: {len(filtered_data)} / {len(data)} ({len(filtered_data)/len(data)*100:.1f}%)")
        
        return filtered_data, (height, width), non_zero_mask
    
    def perform_ica_separation(self, data: np.ndarray) -> np.ndarray:
        """
        執行ICA分離
        
        Args:
            data: 準備好的數據 (n_pixels, n_channels)
            
        Returns:
            分離後的獨立成分 (n_pixels, n_components)
        """
        print("  執行ICA分離...")
        
        # 標準化數據
        data_scaled = self.scaler.fit_transform(data)
        
        # 執行ICA
        separated_sources = self.ica.fit_transform(data_scaled)
        
        # 正規化分離結果到[0,1]
        for i in range(separated_sources.shape[1]):
            source = separated_sources[:, i]
            source_min, source_max = source.min(), source.max()
            if source_max > source_min:
                separated_sources[:, i] = (source - source_min) / (source_max - source_min)
            else:
                separated_sources[:, i] = 0
        
        print(f"  ✓ ICA分離完成，獲得 {separated_sources.shape[1]} 個獨立成分")
        return separated_sources
    
    def reconstruct_images(self, separated_sources: np.ndarray, 
                          original_shape: Tuple[int, int], 
                          non_zero_mask: np.ndarray) -> List[np.ndarray]:
        """
        重建分離後的圖像
        
        Args:
            separated_sources: ICA分離結果
            original_shape: 原始圖像形狀
            non_zero_mask: 有效像素掩碼
            
        Returns:
            重建的圖像列表
        """
        height, width = original_shape
        total_pixels = height * width
        reconstructed_images = []
        
        for i in range(separated_sources.shape[1]):
            # 創建空白圖像
            full_image = np.zeros(total_pixels)
            
            # 將分離的成分放回原位置
            full_image[non_zero_mask] = separated_sources[:, i]
            
            # 重塑為原始形狀
            reconstructed = full_image.reshape(height, width)
            reconstructed_images.append(reconstructed)
        
        return reconstructed_images
    
    def post_process_component(self, component: np.ndarray, 
                             median_filter_size: int = 3,
                             threshold: float = 0.1) -> np.ndarray:
        """
        對分離出的成分進行後處理
        
        Args:
            component: 分離出的成分
            median_filter_size: 中值濾波器大小
            threshold: 二值化閾值
            
        Returns:
            後處理的成分
        """
        # 中值濾波去噪
        filtered = ndimage.median_filter(component, size=median_filter_size)
        
        # 形態學操作
        kernel = np.ones((3,3), np.uint8)
        
        # 轉換為uint8進行形態學操作
        filtered_uint8 = (filtered * 255).astype(np.uint8)
        
        # 開操作（去除小噪點）
        opened = cv2.morphologyEx(filtered_uint8, cv2.MORPH_OPEN, kernel)
        
        # 閉操作（填充小孔洞）
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # 轉換回float
        processed = closed.astype(np.float64) / 255.0
        
        return processed
    
    def process_image(self, image_path: str, 
                     output_dir: Optional[str] = None,
                     save_intermediate: bool = True) -> Optional[List[np.ndarray]]:
        """
        處理單張圖像的完整流程
        
        Args:
            image_path: 輸入圖像路徑
            output_dir: 輸出目錄
            save_intermediate: 是否保存中間結果
            
        Returns:
            分離的螢光染料成分列表
        """
        print(f"\n處理圖像: {os.path.basename(image_path)}")
        
        # 1. 載入圖像
        raw_image = self.load_image(image_path)
        if raw_image is None:
            return None
        
        # 2. 預處理
        processed_image = self.preprocess_image(raw_image)
        print("  ✓ 圖像預處理完成")
        
        # 3. 準備ICA數據
        ica_data, original_shape, non_zero_mask = self.prepare_data_for_ica(processed_image)
        if len(ica_data) < 100:  # 至少需要100個有效像素
            print("  ✗ 有效像素太少，無法進行ICA分析")
            return None
        
        # 4. ICA分離
        separated_sources = self.perform_ica_separation(ica_data)
        
        # 5. 重建圖像
        separated_components = self.reconstruct_images(separated_sources, original_shape, non_zero_mask)
        
        # 6. 後處理
        processed_components = []
        for i, component in enumerate(separated_components):
            processed_comp = self.post_process_component(component)
            processed_components.append(processed_comp)
            print(f"  ✓ 成分 {i+1} 後處理完成")
        
        # 7. 保存結果
        if output_dir and save_intermediate:
            self.save_results(image_path, raw_image, processed_components, output_dir)
        
        return processed_components
    
    def save_results(self, original_path: str, original_image: np.ndarray,
                    separated_components: List[np.ndarray], output_dir: str):
        """
        保存分離結果
        
        Args:
            original_path: 原始圖像路徑
            original_image: 原始圖像
            separated_components: 分離的成分
            output_dir: 輸出目錄
        """
        # 創建輸出目錄
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        base_name = Path(original_path).stem
        
        # 保存各個分離成分
        for i, component in enumerate(separated_components):
            component_path = os.path.join(output_dir, f"{base_name}_component_{i+1}.tif")
            component_uint8 = (component * 255).astype(np.uint8)
            Image.fromarray(component_uint8, mode='L').save(component_path)
            print(f"  ✓ 保存成分 {i+1}: {component_path}")
        
        # 保存組合視覺化
        self.save_visualization(base_name, original_image, separated_components, output_dir)
    
    def save_visualization(self, base_name: str, original_image: np.ndarray,
                          separated_components: List[np.ndarray], output_dir: str):
        """
        保存視覺化結果
        
        Args:
            base_name: 基本文件名
            original_image: 原始圖像
            separated_components: 分離的成分
            output_dir: 輸出目錄
        """
        n_components = len(separated_components)
        fig, axes = plt.subplots(1, n_components + 1, figsize=(15, 5))
        
        # 顯示原始圖像
        axes[0].imshow(original_image)
        axes[0].set_title('原始圖像')
        axes[0].axis('off')
        
        # 顯示各個分離成分
        colors = ['Reds', 'Greens', 'Blues']
        for i, component in enumerate(separated_components):
            axes[i+1].imshow(component, cmap=colors[i % len(colors)])
            axes[i+1].set_title(f'螢光染料 {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        
        # 保存視覺化圖像
        viz_path = os.path.join(output_dir, f"{base_name}_separation_result.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 保存視覺化: {viz_path}")

def process_single_image(input_path: str, output_dir: str, n_components: int = 2):
    """處理單張圖像"""
    processor = FluorescenceICAProcessor(n_components=n_components)
    results = processor.process_image(input_path, output_dir)
    return results is not None

def process_batch(input_dir: str, output_dir: str, n_components: int = 2, 
                 file_extensions: List[str] = ['.tif', '.tiff', '.jpg', '.png']):
    """批量處理圖像"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"錯誤: 輸入目錄不存在: {input_dir}")
        return
    
    # 獲取所有圖像文件
    image_files = []
    for ext in file_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
    
    if not image_files:
        print(f"警告: 在 {input_dir} 中未找到圖像文件")
        return
    
    print(f"找到 {len(image_files)} 個圖像文件")
    
    processor = FluorescenceICAProcessor(n_components=n_components)
    success_count = 0
    
    for img_file in image_files:
        try:
            result = processor.process_image(str(img_file), output_dir)
            if result is not None:
                success_count += 1
        except Exception as e:
            print(f"處理 {img_file.name} 時發生錯誤: {e}")
    
    print(f"\n批量處理完成: {success_count}/{len(image_files)} 成功")

def main():
    parser = argparse.ArgumentParser(description='螢光染料ICA分離工具')
    parser.add_argument('--input', '-i', required=True,
                       help='輸入圖像文件或目錄路徑')
    parser.add_argument('--output', '-o', required=True,
                       help='輸出目錄')
    parser.add_argument('--components', '-c', type=int, default=2,
                       help='螢光染料成分數量 (預設: 2)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='批量處理模式')
    
    args = parser.parse_args()
    
    print("=== 螢光染料 ICA 分離工具 ===")
    print(f"成分數量: {args.components}")
    print(f"輸入: {args.input}")
    print(f"輸出: {args.output}")
    print()
    
    if args.batch or os.path.isdir(args.input):
        # 批量處理
        process_batch(args.input, args.output, args.components)
    else:
        # 單文件處理
        success = process_single_image(args.input, args.output, args.components)
        if success:
            print("處理完成!")
        else:
            print("處理失敗!")
            sys.exit(1)

if __name__ == "__main__":
    # 如果沒有命令列參數，顯示使用說明
    if len(sys.argv) == 1:
        print("螢光染料 ICA 分離工具")
        print("使用方法:")
        print("  單文件: python ica_fluorescence_separation.py -i input.tif -o output/")
        print("  批量:   python ica_fluorescence_separation.py -i input_dir/ -o output/ -b")
        print("  設定成分數: python ica_fluorescence_separation.py -i input.tif -o output/ -c 3")
    else:
        main()