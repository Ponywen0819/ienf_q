"""
神經影像亮點區域生長系統
使用8聯通鄰域檢測，基於亮點相似度進行區域生長
"""

import numpy as np
import cv2
from skimage import io, morphology
from collections import deque
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

class NeuralImageRegionGrowing:
    """神經影像亮點區域生長分析器"""
    
    def __init__(self, strictness=0.8, max_growth_ratio=5.0, min_component_size=3):
        """
        初始化區域生長參數
        
        Args:
            strictness (float): 生長的嚴格程度 (0-1，越高越嚴格)
            max_growth_ratio (float): 最大生長倍數（相對於原始標籤大小）
            min_component_size (int): 最小連通區域大小
        """
        self.strictness = strictness
        self.max_growth_ratio = max_growth_ratio
        self.min_component_size = min_component_size
        
        # 8聯通鄰域方向
        self.directions_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                            (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def load_neural_image_data(self, base_path):
        """
        載入神經影像數據（原始圖片、標籤、遮罩）
        
        Args:
            base_path (str): 文件基礎路徑（不包括副檔名）
        
        Returns:
            tuple: (original, label, mask)
        """
        try:
            original = io.imread(f"{base_path}.tif")
            label = io.imread(f"{base_path}_label.tif")
            mask = io.imread(f"{base_path}_mask.tif")
            
            print(f"Successfully loaded: {Path(base_path).name}")
            print(f"  Image shape: {original.shape}")
            print(f"  Bright spots: {np.sum(label > 0)} pixels")
            
            return original, label, mask
            
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            return None, None, None
    
    def analyze_bright_spot_characteristics(self, original_image, label_image):
        """
        分析神經影像中亮點的特徵
        
        Args:
            original_image: 原始神經影像
            label_image: 亮點標籤圖片
            
        Returns:
            dict: 亮點特徵統計資訊
        """
        bright_spots = label_image > 0
        background = label_image == 0
        
        if np.sum(bright_spots) == 0:
            print("Warning: No bright spots found in label image")
            return None
        
        spot_pixels = original_image[bright_spots]
        bg_pixels = original_image[background]
        
        characteristics = {
            'spot_count': np.sum(bright_spots),
            'spot_mean': np.mean(spot_pixels),
            'spot_std': np.std(spot_pixels),
            'spot_range': (np.min(spot_pixels), np.max(spot_pixels)),
            'bg_mean': np.mean(bg_pixels),
            'bg_std': np.std(bg_pixels),
            'bg_range': (np.min(bg_pixels), np.max(bg_pixels)),
            'separation': np.mean(spot_pixels) - np.mean(bg_pixels)
        }
        
        return characteristics
    
    def extract_bright_spot_seeds(self, label_image, original_image):
        """
        從標籤圖片中提取亮點種子區域
        
        Args:
            label_image: 亮點標籤圖片
            original_image: 原始圖片
            
        Returns:
            list: 種子區域資訊列表
        """
        bright_spots = label_image > 0
        labeled_components, num_components = morphology.label(bright_spots, return_num=True)
        
        seed_regions = []
        
        for i in range(1, num_components + 1):
            region_coords = np.where(labeled_components == i)
            
            if len(region_coords[0]) > 0:
                region_pixels = original_image[region_coords]
                
                region_info = {
                    'id': i,
                    'coords': region_coords,
                    'centroid': (int(np.mean(region_coords[0])), int(np.mean(region_coords[1]))),
                    'size': len(region_pixels),
                    'intensity_mean': np.mean(region_pixels),
                    'intensity_std': np.std(region_pixels),
                    'intensity_range': (np.min(region_pixels), np.max(region_pixels))
                }
                
                seed_regions.append(region_info)
        
        return seed_regions
    
    def region_growing_8_connected(self, image, seed_regions):
        """
        基於8聯通的神經影像亮點區域生長
        
        Args:
            image: 原始神經影像
            seed_regions: 種子區域列表
            
        Returns:
            numpy.ndarray: 區域生長結果的二值遮罩
        """
        height, width = image.shape[:2]
        result_mask = np.zeros((height, width), dtype=np.uint8)
        visited = np.zeros((height, width), dtype=bool)
        
        for region in seed_regions:
            centroid_x, centroid_y = region['centroid']
            
            # 檢查種子點是否有效
            if not (0 <= centroid_x < height and 0 <= centroid_y < width):
                continue
            
            if visited[centroid_x, centroid_y]:
                continue
            
            # 計算該區域的動態閾值
            seed_intensity_mean = region['intensity_mean']
            seed_intensity_std = region['intensity_std']
            
            # 基於標準差和嚴格程度計算閾值
            base_threshold = min(seed_intensity_std * 2, 24)  # 限制最大基礎閾值
            dynamic_threshold = base_threshold * (1 - self.strictness) + 5 * self.strictness
            
            # 初始化區域生長
            growth_queue = deque([region['centroid']])
            visited[centroid_x, centroid_y] = True
            result_mask[centroid_x, centroid_y] = 255
            
            grown_pixel_count = 1
            max_growth_limit = int(region['size'] * self.max_growth_ratio)
            
            # 開始8聯通區域生長
            while growth_queue and grown_pixel_count < max_growth_limit:
                current_x, current_y = growth_queue.popleft()
                current_intensity = float(image[current_x, current_y])
                
                # 檢查8聯通鄰域
                for dx, dy in self.directions_8:
                    neighbor_x, neighbor_y = current_x + dx, current_y + dy
                    
                    # 邊界檢查
                    if not (0 <= neighbor_x < height and 0 <= neighbor_y < width):
                        continue
                    
                    # 已訪問檢查
                    if visited[neighbor_x, neighbor_y]:
                        continue
                    
                    neighbor_intensity = float(image[neighbor_x, neighbor_y])
                    
                    # 多重相似度判斷
                    # 1. 與種子區域平均亮度的差異
                    diff_to_seed = abs(neighbor_intensity - seed_intensity_mean)
                    
                    # 2. 與當前像素的差異
                    diff_to_current = abs(neighbor_intensity - current_intensity)
                    
                    # 3. 是否在種子區域合理範圍內
                    min_intensity, max_intensity = region['intensity_range']
                    within_range = (min_intensity - dynamic_threshold <= neighbor_intensity <= 
                                  max_intensity + dynamic_threshold)
                    
                    # 綜合判斷是否加入生長區域
                    if (diff_to_seed <= dynamic_threshold and 
                        diff_to_current <= dynamic_threshold * 0.8 and
                        within_range):
                        
                        visited[neighbor_x, neighbor_y] = True
                        result_mask[neighbor_x, neighbor_y] = 255
                        growth_queue.append((neighbor_x, neighbor_y))
                        grown_pixel_count += 1
                        
                        # 避免過度生長
                        if grown_pixel_count >= max_growth_limit:
                            break
        
        return result_mask
    
    def post_process_result(self, grown_mask):
        """
        對區域生長結果進行後處理
        
        Args:
            grown_mask: 區域生長結果遮罩
            
        Returns:
            numpy.ndarray: 後處理後的結果
        """
        # 1. 移除過小的連通區域
        labeled_mask, num_labels = morphology.label(grown_mask, return_num=True)
        filtered_mask = np.zeros_like(grown_mask)
        
        for i in range(1, num_labels + 1):
            component = labeled_mask == i
            if np.sum(component) >= self.min_component_size:
                filtered_mask[component] = 255
        
        # 2. 輕微的形態學操作
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # refined_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
        
        return filtered_mask
    
    def evaluate_performance(self, ground_truth, predicted):
        """
        評估區域生長性能
        
        Args:
            ground_truth: 真實標籤
            predicted: 預測結果
            
        Returns:
            dict: 性能指標
        """
        # 計算混淆矩陣
        tp = np.sum((ground_truth > 0) & (predicted > 0))
        fp = np.sum((ground_truth == 0) & (predicted > 0))
        fn = np.sum((ground_truth > 0) & (predicted == 0))
        tn = np.sum((ground_truth == 0) & (predicted == 0))
        
        # 計算性能指標
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        
        performance = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'original_pixels': np.sum(ground_truth > 0),
            'grown_pixels': np.sum(predicted > 0),
            'growth_ratio': np.sum(predicted > 0) / np.sum(ground_truth > 0) if np.sum(ground_truth > 0) > 0 else 0
        }
        
        return performance
    
    def visualize_results(self, original, label, grown_result, seed_regions, 
                         performance_metrics, save_path=None):
        """
        可視化區域生長結果
        
        Args:
            original: 原始圖片
            label: 真實標籤
            grown_result: 區域生長結果
            seed_regions: 種子區域
            performance_metrics: 性能指標
            save_path: 保存路徑
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Neural Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(label, cmap='gray')
        axes[0, 1].set_title('Ground Truth Labels')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(grown_result, cmap='gray')
        axes[0, 2].set_title('Region Growing Result')
        axes[0, 2].axis('off')
        
        # 第二行
        # 種子點顯示
        axes[1, 0].imshow(original, cmap='gray')
        for i, region in enumerate(seed_regions):
            cx, cy = region['centroid']
            intensity = region['intensity_mean']
            size = region['size']
            
            # 根據亮度選擇顏色
            if intensity > 60:
                color = 'red'
            elif intensity > 40:
                color = 'yellow'
            else:
                color = 'cyan'
            
            axes[1, 0].plot(cy, cx, 'o', color=color, markersize=max(3, min(8, size//2)))
        
        axes[1, 0].set_title('Bright Spot Seeds\n(Color by intensity)')
        axes[1, 0].axis('off')
        
        # 比較顯示
        comparison = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)
        comparison[:,:,0] = original
        comparison[:,:,1] = original
        comparison[:,:,2] = original
        
        # 顏色編碼
        comparison[label > 0] = [255, 255, 0]  # 黃色 - 真實標籤
        comparison[grown_result > 0] = [255, 0, 0]  # 紅色 - 生長結果
        comparison[(label > 0) & (grown_result > 0)] = [0, 255, 0]  # 綠色 - 匹配區域
        
        axes[1, 1].imshow(comparison)
        axes[1, 1].set_title('Comparison Overlay\n(Yellow:GT, Red:Grown, Green:Match)')
        axes[1, 1].axis('off')
        
        # 性能統計
        stats_text = f"""Performance Metrics:

Precision:  {performance_metrics['precision']:.3f}
Recall:     {performance_metrics['recall']:.3f}  
F1-Score:   {performance_metrics['f1_score']:.3f}
Accuracy:   {performance_metrics['accuracy']:.3f}

Original:   {performance_metrics['original_pixels']} pixels
Grown:      {performance_metrics['grown_pixels']} pixels  
Ratio:      {performance_metrics['growth_ratio']:.2f}x

Seeds:      {len(seed_regions)} regions
Max Growth: {self.max_growth_ratio}x
Strictness: {self.strictness}"""
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Analysis Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def process_single_image(self, base_path, save_results=True):
        """
        處理單個神經影像
        
        Args:
            base_path: 文件基礎路徑
            save_results: 是否保存結果
            
        Returns:
            dict: 處理結果和性能指標
        """
        print(f"\n{'='*60}")
        print(f"Processing: {Path(base_path).name}")
        print(f"{'='*60}")
        
        # 1. 載入數據
        original, label, mask = self.load_neural_image_data(base_path)
        if original is None:
            return None
        
        # 2. 分析亮點特徵
        characteristics = self.analyze_bright_spot_characteristics(original, label)
        if characteristics is None:
            return None
        
        print(f"Bright spot characteristics:")
        print(f"  Count: {characteristics['spot_count']}")
        print(f"  Intensity: {characteristics['spot_mean']:.1f} ± {characteristics['spot_std']:.1f}")
        print(f"  Separation from background: {characteristics['separation']:.1f}")
        
        # 3. 提取種子區域
        seed_regions = self.extract_bright_spot_seeds(label, original)
        print(f"Extracted {len(seed_regions)} seed regions")
        
        # 4. 套用高斯模糊處理
        if mask is not None:
            # original = cv2.GaussianBlur(original, (11, 11), 0, mask=mask)
            print("Applied Gaussian blur to original image")

        # #  套用最小值濾波
        # original = cv2.medianBlur(original, 3)
        # print("Applied Median filter to original image")

        # 5. 執行區域生長
        print("Performing 8-connected region growing...")
        grown_mask = self.region_growing_8_connected(original, seed_regions)
        
        # 6. 後處理
        final_result = self.post_process_result(grown_mask)
        
        # 7. 評估性能
        performance = self.evaluate_performance(label, final_result)
        
        print(f"\nPerformance Results:")
        print(f"  Precision: {performance['precision']:.3f}")
        print(f"  Recall:    {performance['recall']:.3f}")
        print(f"  F1-Score:  {performance['f1_score']:.3f}")
        
        # 7. 可視化和保存
        if save_results:
            sample_name = Path(base_path).name
            vis_path = f"{sample_name}_region_growing_analysis.png"
            result_path = f"{sample_name}_grown_result.tif"
            
            self.visualize_results(original, label, final_result, seed_regions, 
                                 performance, vis_path)
            cv2.imwrite(result_path, final_result)
            print(f"Results saved: {result_path}")
        
        return {
            'sample_name': Path(base_path).name,
            'characteristics': characteristics,
            'seed_regions': len(seed_regions), 
            'performance': performance,
            'final_result': final_result
        }

def main():
    """主程序：演示神經影像區域生長系統"""
    
    # 初始化區域生長系統
    region_grower = NeuralImageRegionGrowing(
        strictness=0.85,      # 較高的嚴格度
        max_growth_ratio=3.0,  # 限制生長倍數
        min_component_size=3   # 過濾小區域
    )
    
    # 處理樣本
    sample_path = "Centered/processed_S1037-2_a"
    
    print("神經影像亮點區域生長分析系統")
    print("使用8聯通鄰域進行基於相似度的區域生長")
    
    result = region_grower.process_single_image(sample_path, save_results=True)
    
    if result:
        print(f"\n處理完成！")
        print(f"樣本: {result['sample_name']}")
        print(f"檢測的種子區域: {result['seed_regions']}")
        print(f"F1分數: {result['performance']['f1_score']:.3f}")
    else:
        print("處理失敗！")

if __name__ == "__main__":
    main()