#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
神經標記處理和連接模組：
1. 從Ground_Truth圖片中檢測神經標記
2. 使用DBSCAN對太靠近的神經標記進行聚類
3. 計算群組中心點
4. 使用BFS搜索連接不同的神經標記群組
"""

import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from pathlib import Path
from collections import deque
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import json


def load_ground_truth_markers(gt_path):
    """從Ground_Truth圖片中檢測神經標記"""
    try:
        img = Image.open(gt_path)
        gt_array = np.array(img)
        
        # 轉為灰度圖片（如果是彩色）
        if len(gt_array.shape) == 3:
            gt_gray = cv2.cvtColor(gt_array, cv2.COLOR_RGB2GRAY)
        else:
            gt_gray = gt_array.copy()
        
        # 二值化處理，找到白色標記
        _, binary = cv2.threshold(gt_gray, 127, 255, cv2.THRESH_BINARY)
        
        # 找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 提取標記中心點
        markers = []
        for contour in contours:
            if cv2.contourArea(contour) > 5:  # 過濾掉太小的噪點
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    markers.append([cx, cy])
        
        return np.array(markers)
        
    except Exception as e:
        print(f"載入Ground_Truth標記失敗 {gt_path}: {e}")
        return np.array([])


def cluster_markers_dbscan(markers, eps=10, min_samples=1):
    """使用DBSCAN對太靠近的神經標記進行聚類"""
    if len(markers) == 0:
        return [], []
    
    # 應用DBSCAN聚類
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(markers)
    
    # 計算每個聚類的中心點
    cluster_centers = []
    clustered_markers = []
    
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label == -1:  # 噪音點
            continue
        
        cluster_points = markers[cluster_labels == label]
        center = np.mean(cluster_points, axis=0).astype(int)
        cluster_centers.append(center)
        clustered_markers.append(cluster_points)
    
    return cluster_centers, clustered_markers


def intensity_weighted_bfs(processed_img, mask_img, start_point, target_markers, max_radius=20):
    """
    基於像素強度差異的BFS搜索算法（限制在遮罩區域內）
    
    Args:
        processed_img: 處理後的圖像（綠色通道）
        mask_img: 遮罩圖像
        start_point: 起始神經標記位置 [x, y]
        target_markers: 目標神經標記列表
        max_radius: 最大搜索半徑
    
    Returns:
        連接路徑和目標標記（如果找到）
    """
    height, width = processed_img.shape
    start_x, start_y = start_point
    
    # 檢查起始點是否在圖像範圍內
    if not (0 <= start_x < width and 0 <= start_y < height):
        return None, None
    
    # 初始化
    visited = set()
    queue = deque([(start_x, start_y, 0, [])])  # (x, y, distance, path)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    start_intensity = processed_img[start_y, start_x]
    
    while queue:
        x, y, distance, path = queue.popleft()
        
        # 檢查是否超出最大搜索半徑
        if distance > max_radius:
            continue
            
        # 檢查是否已訪問
        if (x, y) in visited:
            continue
        
        visited.add((x, y))
        current_path = path + [(x, y)]
        
        # 檢查是否到達目標標記
        for target in target_markers:
            target_x, target_y = target
            if abs(x - target_x) <= 2 and abs(y - target_y) <= 2:  # 允許小誤差
                return current_path, target
        
        # 探索鄰近像素
        current_intensity = processed_img[y, x]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            # 檢查邊界
            if not (0 <= new_x < width and 0 <= new_y < height):
                continue
            
            if (new_x, new_y) in visited:
                continue
            
            # 檢查是否在遮罩區域內
            if mask_img[new_y, new_x] <= 127:  # 不在遮罩白色區域內
                continue
            
            # 計算強度差異作為權重
            neighbor_intensity = processed_img[new_y, new_x]
            intensity_diff = abs(int(current_intensity) - int(neighbor_intensity))
            
            # 如果強度差異太大，跳過（可調整閾值）
            if intensity_diff > 50:  # 可調整的閾值
                continue
            
            queue.append((new_x, new_y, distance + 1, current_path))
    
    return None, None


def connect_neural_markers(processed_img_path, mask_path, gt_path, eps=10, max_radius=20):
    """
    連接神經標記的主要函數
    
    Args:
        processed_img_path: 處理後圖像路徑（綠色通道）
        mask_path: 對應的遮罩圖像路徑
        gt_path: Ground_Truth圖像路徑
        eps: DBSCAN聚類距離參數
        max_radius: BFS搜索最大半徑
    
    Returns:
        連接結果字典
    """
    print(f"處理神經連接: {os.path.basename(processed_img_path)}")
    
    # 1. 載入處理後的圖像（綠色通道）
    try:
        processed_img = Image.open(processed_img_path)
        processed_array = np.array(processed_img)
    except Exception as e:
        print(f"載入處理後圖像失敗: {e}")
        return None
    
    # 2. 載入遮罩圖像
    try:
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)
    except Exception as e:
        print(f"載入遮罩圖像失敗: {e}")
        return None
    
    # 3. 載入Ground_Truth標記
    markers = load_ground_truth_markers(gt_path)
    if len(markers) == 0:
        print("沒有找到神經標記")
        return None
    
    print(f"找到 {len(markers)} 個神經標記")
    
    # 4. 過濾標記：只保留在遮罩範圍內的標記
    valid_markers = []
    for marker in markers:
        x, y = marker
        if (0 <= x < mask_array.shape[1] and 0 <= y < mask_array.shape[0] and 
            mask_array[y, x] > 127):  # 在遮罩白色區域內
            valid_markers.append(marker)
    
    markers = np.array(valid_markers)
    print(f"遮罩範圍內的有效標記: {len(markers)} 個")
    
    # 3. 使用DBSCAN聚類
    cluster_centers, clustered_markers = cluster_markers_dbscan(markers, eps=eps)
    print(f"聚類後得到 {len(cluster_centers)} 個群組")
    
    if len(cluster_centers) < 2:
        print("群組數量不足，無法建立連接")
        return {
            'markers': markers.tolist(),
            'cluster_centers': cluster_centers,
            'connections': []
        }
    
    # 4. 為每個群組中心尋找連接
    connections = []
    
    for i, center in enumerate(cluster_centers):
        # 獲取其他群組中心作為目標
        other_centers = [c for j, c in enumerate(cluster_centers) if j != i]
        
        # 使用BFS搜索連接（限制在遮罩區域內）
        path, target = intensity_weighted_bfs(
            processed_array, mask_array, center, other_centers, max_radius
        )
        
        if path is not None and target is not None:
            connections.append({
                'from': [int(center[0]), int(center[1])],
                'to': [int(target[0]), int(target[1])],
                'path': [[int(p[0]), int(p[1])] for p in path],
                'length': len(path)
            })
            print(f"找到連接: {center} -> {target}, 路徑長度: {len(path)}")
    
    # 確保所有數據都是JSON可序列化的
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        return obj
    
    result = {
        'markers': convert_to_serializable(markers),
        'cluster_centers': convert_to_serializable(cluster_centers),
        'clustered_markers': convert_to_serializable(clustered_markers),
        'connections': convert_to_serializable(connections),
        'image_shape': list(processed_array.shape)
    }
    
    print(f"總共建立了 {len(connections)} 個神經連接")
    return result


def visualize_neural_connections(processed_img_path, mask_path, connection_result, output_path):
    """可視化神經連接結果"""
    if connection_result is None:
        print("沒有連接結果可視化")
        return
    
    # 載入處理後的圖像
    processed_img = Image.open(processed_img_path)
    processed_array = np.array(processed_img)
    
    # 載入遮罩圖像
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)
    
    # 創建可視化圖像
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 疊加顯示：綠色通道圖像為基礎，遮罩區域加上半透明覆蓋
    ax.imshow(processed_array, cmap='gray', alpha=0.8)
    ax.imshow(mask_array, cmap='Reds', alpha=0.3)  # 遮罩區域用紅色半透明覆蓋
    
    # 顯示原始標記
    markers = np.array(connection_result['markers'])
    # if len(markers) > 0:
    #     ax.scatter(markers[:, 0], markers[:, 1], c='red', s=20, alpha=0.6, label='Original Centers')

    # 顯示群組中心
    centers = np.array(connection_result['cluster_centers'])
    # if len(centers) > 0:
    #     ax.scatter(centers[:, 0], centers[:, 1], c='blue', s=50, marker='x', label='Cluster Centers')
    
    # 顯示連接路徑
    connections = connection_result['connections']
    for i, conn in enumerate(connections):
        path = np.array(conn['path'])
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], 'green', linewidth=1, alpha=0.7)
            # 標記起點和終點
            ax.scatter(conn['from'][0], conn['from'][1], c='yellow', s=1, marker='s')
            ax.scatter(conn['to'][0], conn['to'][1], c='orange', s=1, marker='s')
    
    ax.set_title(f'Neural Connections - {len(connections)} connections found')
    ax.legend()
    ax.set_xlim(0, processed_array.shape[1])
    ax.set_ylim(processed_array.shape[0], 0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=800, bbox_inches='tight')
    plt.close()
    
    print(f"可視化結果儲存至: {output_path}")


def process_all_neural_connections(processed_dir, gt_dir, output_dir, eps=10, max_radius=20):
    """批量處理所有神經連接"""
    processed_path = Path(processed_dir)
    gt_path = Path(gt_dir)
    output_path = Path(output_dir)
    
    # 創建輸出目錄
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有處理後的圖片（非遮罩）
    processed_files = [f for f in processed_path.glob('processed_*.tif') 
                      if not f.name.endswith('_mask.tif')]
    
    results = {}
    
    for processed_file in processed_files:
        # 提取原始檔名
        base_name = processed_file.name.replace('processed_', '')
        gt_file = gt_path / base_name
        
        # 找對應的遮罩檔案
        mask_file_name = processed_file.name.replace('.tif', '_mask.tif')
        mask_file = processed_path / mask_file_name
        
        if not gt_file.exists():
            print(f"找不到對應的Ground_Truth檔案: {gt_file}")
            continue
            
        if not mask_file.exists():
            print(f"找不到對應的遮罩檔案: {mask_file}")
            continue
        
        # 處理神經連接
        connection_result = connect_neural_markers(
            str(processed_file), str(mask_file), str(gt_file), eps, max_radius
        )
        
        if connection_result is not None:
            results[base_name] = connection_result
            
            # 儲存結果到JSON檔案
            json_output = output_path / f"{base_name.replace('.tif', '_connections.json')}"
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(connection_result, f, indent=2, ensure_ascii=False)
            
            # 創建可視化
            viz_output = output_path / f"{base_name.replace('.tif', '_visualization.png')}"
            visualize_neural_connections(str(processed_file), str(mask_file), connection_result, str(viz_output))
        
        print()  # 空行分隔
        break  # For testing, remove this line to process all files
    
    print(f"批量神經連接處理完成: {len(results)} 個檔案成功處理")
    return results


if __name__ == "__main__":
    # 預設路徑處理
    current_dir = Path(__file__).parent
    
    # 處理所有神經連接
    results = process_all_neural_connections(
        processed_dir=current_dir / "Centered",
        gt_dir=current_dir / "Ground_Truth",
        output_dir=current_dir / "Neural_Connections",
        eps=5,
        max_radius=40
    )
    
    print(f"處理完成！總共處理了 {len(results)} 個檔案")