import numpy as np
import cv2
from pathlib import Path
import pickle
import json
from tqdm import tqdm

def preprocess_sampling_regions(data_dir, output_dir=None):
    """
    預處理所有圖像，計算抽樣範圍並保存到文件
    
    Args:
        data_dir: 包含所有圖像文件的目錄
        output_dir: 輸出預處理結果的目錄，如果為None則使用data_dir
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir / "preprocessed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # 找到所有完整的圖像組
    image_files = []
    for img_file in data_dir.glob("*.tif"):
        # 跳過帶有後綴的檔案
        if any(suffix in img_file.name for suffix in ['_label', '_mask', '_boundary']):
            continue
            
        base_name = img_file.stem
        label_file = data_dir / f"{base_name}_label.tif"
        mask_file = data_dir / f"{base_name}_mask.tif"
        boundary_file = data_dir / f"{base_name}_boundary.tif"
        
        if label_file.exists() and mask_file.exists() and boundary_file.exists():
            image_files.append({
                'base_name': base_name,
                'image': img_file,
                'label': label_file,
                'mask': mask_file,
                'boundary': boundary_file
            })
    
    print(f"Found {len(image_files)} complete image sets")
    
    # 處理每個圖像組
    sampling_data = {}
    
    for file_info in tqdm(image_files, desc="Processing images"):
        base_name = file_info['base_name']
        
        try:
            # 讀取圖像
            label = cv2.imread(str(file_info['label']), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(file_info['mask']), cv2.IMREAD_GRAYSCALE)
            boundary = cv2.imread(str(file_info['boundary']), cv2.IMREAD_GRAYSCALE)
            
            # 計算抽樣區域
            sampling_info = calculate_sampling_region(label, mask, boundary)
            
            # 保存到字典
            sampling_data[base_name] = sampling_info
            
            # 另外保存為獨立的 numpy 文件（便於快速加載）
            np.savez_compressed(
                output_dir / f"{base_name}_sampling.npz",
                **sampling_info
            )
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue
    
    # 保存整體統計信息
    stats = calculate_dataset_statistics(sampling_data)
    
    # 保存所有抽樣數據
    with open(output_dir / "sampling_data.pkl", 'wb') as f:
        pickle.dump(sampling_data, f)
    
    # 保存統計信息（JSON格式便於查看）
    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Preprocessing completed. Results saved to {output_dir}")
    print(f"Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return sampling_data, stats

def create_upper_region_mask(boundary, mask):
    """
    創建中心線以上的區域遮罩
    
    Args:
        boundary: 邊界圖像 (255 為中心線)
        mask: 原始遮罩圖像 (255 為有效區域)
    
    Returns:
        upper_region: 中心線以上的有效區域遮罩
    """
    h, w = boundary.shape
    upper_region = np.zeros_like(mask)
    
    # 找到中心線像素
    boundary_pixels = (boundary == 255)
    
    # 對每一列，找到最上方的邊界像素
    for col in range(w):
        boundary_rows_in_col = np.where(boundary_pixels[:, col])[0]
        
        if len(boundary_rows_in_col) > 0:
            # 使用最上方的邊界像素作為該列的分界點
            top_boundary = np.min(boundary_rows_in_col)
            # 將該分界點以上的像素標記為有效
            upper_region[:top_boundary, col] = mask[:top_boundary, col]
        else:
            # 如果該列沒有邊界像素，使用整列的上半部分
            mid_point = h // 2
            upper_region[:mid_point, col] = mask[:mid_point, col]
    
    return upper_region

def calculate_sampling_region(label, mask, boundary):
    """
    計算單個圖像的抽樣區域信息（動態抽樣版本）
    只保存有效抽樣範圍，不預先計算像素座標
    
    Args:
        label: 標籤圖像 (255 為神經亮點)
        mask: 遮罩圖像 (255 為有效區域)  
        boundary: 邊界圖像 (255 為中心線)
    
    Returns:
        dict: 包含抽樣範圍信息的字典
    """
    h, w = label.shape
    
    # 創建中心線以上的區域遮罩
    upper_region = create_upper_region_mask(boundary, mask)
    
    # 計算統計信息（用於驗證和統計）
    neural_count = np.sum(label == 255)
    non_neural_count = np.sum((label == 0) & (upper_region == 255))
    max_sample_count = min(neural_count, non_neural_count)
    boundary_pixel_count = np.sum(boundary == 255)
    
    # 只保存用於動態抽樣的必要信息
    sampling_info = {
        'upper_region': upper_region.astype(np.uint8),
        'neural_count': int(neural_count),
        'non_neural_count': int(non_neural_count), 
        'max_sample_count': int(max_sample_count),
        'boundary_pixel_count': int(boundary_pixel_count),
        'image_shape': (h, w)
    }
    
    return sampling_info

def calculate_dataset_statistics(sampling_data):
    """計算整個數據集的統計信息"""
    stats = {
        'total_images': len(sampling_data),
        'neural_counts': [],
        'non_neural_counts': [],
        'max_sample_counts': [],
        'boundary_pixel_counts': [],
        'upper_region_pixel_counts': [],
        'image_shapes': []
    }
    
    for base_name, info in sampling_data.items():
        stats['neural_counts'].append(info['neural_count'])
        stats['non_neural_counts'].append(info['non_neural_count'])
        stats['max_sample_counts'].append(info['max_sample_count'])
        stats['boundary_pixel_counts'].append(info['boundary_pixel_count'])
        stats['upper_region_pixel_counts'].append(np.sum(info['upper_region'] == 255))
        stats['image_shapes'].append(info['image_shape'])
    
    # 計算統計值
    def calc_stats(values):
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': int(np.min(values)),
            'max': int(np.max(values)),
            'median': float(np.median(values))
        }
    
    summary_stats = {
        'total_images': stats['total_images'],
        'neural_count_stats': calc_stats(stats['neural_counts']),
        'non_neural_count_stats': calc_stats(stats['non_neural_counts']),
        'max_sample_count_stats': calc_stats(stats['max_sample_counts']),
        'boundary_pixel_count_stats': calc_stats(stats['boundary_pixel_counts']),
        'upper_region_pixel_count_stats': calc_stats(stats['upper_region_pixel_counts']),
        'common_image_shapes': list(set(stats['image_shapes'])),
        'sampling_feasibility': {
            'images_with_samples': sum(1 for count in stats['max_sample_counts'] if count > 0),
            'images_without_samples': sum(1 for count in stats['max_sample_counts'] if count == 0)
        }
    }
    
    return summary_stats

def load_sampling_data(preprocessed_dir, base_name):
    """
    加載預處理的抽樣數據
    
    Args:
        preprocessed_dir: 預處理數據目錄
        base_name: 圖像基礎名稱
    
    Returns:
        dict: 抽樣信息
    """
    preprocessed_dir = Path(preprocessed_dir)
    
    # 嘗試加載 npz 文件
    npz_file = preprocessed_dir / f"{base_name}_sampling.npz"
    if npz_file.exists():
        data = np.load(npz_file)
        return {key: data[key] for key in data.keys()}
    
    # 如果沒有 npz 文件，嘗試從 pickle 文件加載
    pkl_file = preprocessed_dir / "sampling_data.pkl"
    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            sampling_data = pickle.load(f)
        return sampling_data.get(base_name, None)
    
    return None

def verify_preprocessing(data_dir, output_dir=None):
    """驗證預處理結果的正確性"""
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir / "preprocessed"
    else:
        output_dir = Path(output_dir)
    
    # 隨機選擇一個圖像進行驗證
    all_tif_files = list(data_dir.glob("*.tif"))
    base_images = [f for f in all_tif_files if not any(suffix in f.name for suffix in ['_label', '_mask', '_boundary'])]
    sample_file = base_images[0] if base_images else None
    
    if sample_file is None:
        print("❌ No valid image files found")
        return False
    base_name = sample_file.stem
    
    print(f"Verifying preprocessing for: {base_name}")
    
    # 加載原始數據
    label = cv2.imread(str(data_dir / f"{base_name}_label.tif"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(data_dir / f"{base_name}_mask.tif"), cv2.IMREAD_GRAYSCALE)
    boundary = cv2.imread(str(data_dir / f"{base_name}_boundary.tif"), cv2.IMREAD_GRAYSCALE)
    
    # 重新計算
    fresh_calc = calculate_sampling_region(label, mask, boundary)
    
    # 加載預處理結果
    loaded_data = load_sampling_data(output_dir, base_name)
    
    if loaded_data is None:
        print("❌ Failed to load preprocessed data")
        return False
    
    # 比較結果（動態抽樣版本）
    checks = [
        ('neural_count', fresh_calc['neural_count'] == loaded_data['neural_count']),
        ('non_neural_count', fresh_calc['non_neural_count'] == loaded_data['non_neural_count']),
        ('max_sample_count', fresh_calc['max_sample_count'] == loaded_data['max_sample_count']),
        ('boundary_pixel_count', fresh_calc['boundary_pixel_count'] == loaded_data['boundary_pixel_count']),
        ('upper_region', np.array_equal(fresh_calc['upper_region'], loaded_data['upper_region']))
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {passed}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("🎉 All verification checks passed!")
    else:
        print("⚠️  Some verification checks failed!")
    
    # 顯示一些統計信息
    print(f"\nSample statistics:")
    print(f"  Neural pixels: {fresh_calc['neural_count']}")
    print(f"  Non-neural candidates: {fresh_calc['non_neural_count']}")
    print(f"  Max sample count: {fresh_calc['max_sample_count']}")
    print(f"  Boundary pixels: {fresh_calc['boundary_pixel_count']}")
    print(f"  Upper region pixels: {np.sum(fresh_calc['upper_region'] == 255)}")
    
    return all_passed

if __name__ == "__main__":
    data_dir = "/home/bl515-ml/Documents/shaio_jie/ienf_q/Centered"
    
    print("Starting preprocessing...")
    sampling_data, stats = preprocess_sampling_regions(data_dir)
    
    print("\nVerifying preprocessing results...")
    verify_preprocessing(data_dir)