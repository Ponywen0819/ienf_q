import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.morphology import skeletonize
from skimage.measure import label as connected_components

from scipy.ndimage import distance_transform_edt

class DistanceWeightedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=2.0):
        super().__init__()
        self.alpha = alpha  # 距離權重係數
        self.beta = beta    # 距離衰減指數
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def compute_distance_weights(self, bright_points_mask):
        """計算每個像素到最近亮點的距離權重"""
        # bright_points_mask: (B, H, W) 亮點的二值遮罩
        batch_size = bright_points_mask.shape[0]
        weights = torch.zeros_like(bright_points_mask, dtype=torch.float32)
        
        for b in range(batch_size):
            # 計算到亮點的距離變換
            bright_points = bright_points_mask[b].cpu().numpy()
            if bright_points.sum() > 0:
                distance_map = distance_transform_edt(1 - bright_points)
                # 距離越近權重越高
                weight_map = np.exp(-distance_map / self.beta)
                weights[b] = torch.from_numpy(weight_map)
        
        return weights.to(bright_points_mask.device)
    
    def forward(self, predictions, bright_points_mask, background_mask):
        # predictions: (B, num_classes, H, W)
        # 創建目標：亮點=1, 潛在連結=1, 背景=0
        targets = bright_points_mask.long()
        
        # 計算距離權重
        distance_weights = self.compute_distance_weights(bright_points_mask)
        
        # 標準交叉熵損失
        ce_loss = self.ce_loss(predictions, targets)
        
        # 應用距離權重：鼓勵亮點附近區域預測為前景
        weighted_loss = ce_loss * (1 + self.alpha * distance_weights)
        
        return weighted_loss.mean()
    


class ConnectivityLoss(nn.Module):
    def __init__(self, connectivity_weight=2.0, topology_weight=1.0):
        super().__init__()
        self.connectivity_weight = connectivity_weight
        self.topology_weight = topology_weight
        
    def compute_connectivity_target(self, bright_points, max_distance=50):
        """基於亮點生成連通性目標"""
        batch_size = bright_points.shape[0]
        targets = torch.zeros_like(bright_points, dtype=torch.float32)
        
        for b in range(batch_size):
            points = bright_points[b].cpu().numpy()
            if points.sum() < 2:  # 需要至少2個點
                continue
                
            # 使用距離變換創建潛在連結
            distance_map = distance_transform_edt(1 - points)
            
            # 創建連結遮罩：距離在閾值內的區域
            connection_mask = distance_map <= max_distance
            
            # 使用形態學操作連接鄰近點
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(points, iterations=max_distance//4)
            
            # 骨架化以獲得連結路徑
            skeleton = skeletonize(dilated)
            
            # 組合亮點和骨架作為目標
            target = np.logical_or(points, skeleton).astype(np.float32)
            targets[b] = torch.from_numpy(target)
        
        return targets.to(bright_points.device)
    
    def forward(self, predictions, bright_points_mask):
        # 生成連通性目標
        connectivity_targets = self.compute_connectivity_target(bright_points_mask)
        
        # Soft Dice Loss for connectivity
        predictions_soft = F.softmax(predictions, dim=1)[:, 1]  # 前景概率
        
        # Dice係數
        intersection = (predictions_soft * connectivity_targets).sum(dim=[1,2])
        union = predictions_soft.sum(dim=[1,2]) + connectivity_targets.sum(dim=[1,2])
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1 - dice.mean()
        
        # 確保亮點區域預測正確
        bright_point_loss = F.binary_cross_entropy(
            predictions_soft, bright_points_mask.float()
        )
        
        total_loss = (self.connectivity_weight * dice_loss + 
                     self.topology_weight * bright_point_loss)
        
        return total_loss
    
class PathGuidedLoss(nn.Module):
    def __init__(self, path_weight=3.0, background_weight=0.5):
        super().__init__()
        self.path_weight = path_weight
        self.background_weight = background_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def create_path_guidance(self, bright_points, image):
        """使用傳統演算法創建路徑指導"""
        batch_size = bright_points.shape[0]
        path_masks = torch.zeros_like(bright_points, dtype=torch.float32)
        
        for b in range(batch_size):
            points = bright_points[b].cpu().numpy()
            img = image[b].cpu().numpy()
            
            if points.sum() < 2:
                continue
            
            # 找到所有亮點位置
            point_coords = np.where(points > 0)
            point_list = list(zip(point_coords[0], point_coords[1]))
            
            if len(point_list) < 2:
                continue
            
            # 使用你之前的 Dijkstra 演算法連接點
            path_mask = np.zeros_like(points)
            
            for i in range(len(point_list)-1):
                start = point_list[i]
                end = point_list[i+1]
                
                # 這裡調用你的路徑搜尋演算法
                path, _ = self.find_optimal_path(img, start, end)
                
                # 將路徑標記在遮罩上
                for point in path:
                    if 0 <= point[0] < path_mask.shape[0] and 0 <= point[1] < path_mask.shape[1]:
                        path_mask[point[0], point[1]] = 1
            
            path_masks[b] = torch.from_numpy(path_mask)
        
        return path_masks.to(bright_points.device)
    
    def find_optimal_path(self, image, start, end):
        # 這裡插入你的標準差基路徑搜尋演算法
        # 返回路徑和成本
        pass
    
    def forward(self, predictions, bright_points_mask, image):
        # 創建路徑指導
        path_guidance = self.create_path_guidance(bright_points_mask, image)
        
        # 組合目標：亮點 + 路徑 = 前景
        combined_targets = torch.clamp(bright_points_mask + path_guidance, 0, 1)
        
        # 創建加權遮罩
        weights = torch.ones_like(combined_targets)
        weights[combined_targets > 0] = self.path_weight  # 路徑和亮點更重要
        weights[combined_targets == 0] = self.background_weight  # 背景較不重要
        
        # 計算損失
        ce_loss = self.ce_loss(predictions, combined_targets.long())
        weighted_loss = ce_loss * weights
        
        return weighted_loss.mean()