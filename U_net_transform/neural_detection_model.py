import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess_sampling_regions import load_sampling_data
import wandb

class NeuralDetectionCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(NeuralDetectionCNN, self).__init__()
        
        # 編碼器部分
        self.encoder = nn.Sequential(
            # 第一層
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二層
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三層
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第四層
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 解碼器部分
        self.decoder = nn.Sequential(
            # 上採樣和卷積
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 最終輸出層
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
        
    def forward(self, x):
        # 編碼
        encoded = self.encoder(x)
        
        # 解碼
        decoded = self.decoder(encoded)
        
        return decoded

class SpecialLoss(nn.Module):
    def __init__(self):
        super(SpecialLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predictions, labels, sampling_data_batch):
        """
        動態抽樣的 loss 計算
        
        Args:
            predictions: 模型預測 [B, C, H, W]
            labels: ground truth labels [B, H, W] (255 為神經亮點)
            sampling_data_batch: 包含抽樣範圍信息的列表
        """
        batch_size = predictions.size(0)
        device = predictions.device
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            if sampling_data_batch[b] is None:
                continue
                
            pred_b = predictions[b]  # [C, H, W]
            label_b = labels[b]      # [H, W]
            sampling_data = sampling_data_batch[b]
            
            # 動態計算神經像素座標
            neural_mask = (label_b == 255)
            neural_coords = torch.nonzero(neural_mask, as_tuple=False)
            
            if len(neural_coords) == 0:
                continue
                
            neural_rows = neural_coords[:, 0]
            neural_cols = neural_coords[:, 1]
            neural_count = len(neural_rows)
            
            # 動態計算非神經像素座標（在有效抽樣範圍內）
            upper_region = torch.from_numpy(sampling_data['upper_region']).to(device)
            non_neural_mask = ((label_b == 0) & (upper_region == 255))
            non_neural_coords = torch.nonzero(non_neural_mask, as_tuple=False)
            
            if len(non_neural_coords) == 0:
                continue
                
            non_neural_rows = non_neural_coords[:, 0]
            non_neural_cols = non_neural_coords[:, 1]
            non_neural_count = len(non_neural_rows)
            
            # 隨機抽樣相同數量的非神經像素
            sample_count = min(neural_count, non_neural_count)
            
            if sample_count > 0:
                # 隨機抽樣非神經像素
                sample_indices = torch.randperm(non_neural_count, device=device)[:sample_count]
                sampled_non_neural_rows = non_neural_rows[sample_indices]
                sampled_non_neural_cols = non_neural_cols[sample_indices]
                
                # 合併所有訓練像素的座標
                train_rows = torch.cat([neural_rows, sampled_non_neural_rows])
                train_cols = torch.cat([neural_cols, sampled_non_neural_cols])
                
                # 提取對應的預測和標籤
                train_predictions = pred_b[:, train_rows, train_cols].transpose(0, 1)  # [N, C]
                train_labels = label_b[train_rows, train_cols]  # [N]
                
                # 將 255 轉換為類別 1，0 保持為類別 0
                train_labels = (train_labels == 255).long()
                
                # 計算 cross-entropy loss
                loss_b = self.ce_loss(train_predictions, train_labels)
                total_loss += torch.mean(loss_b)
                valid_batches += 1
        
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

class NeuralDataset(Dataset):
    def __init__(self, data_dir, preprocessed_dir=None, patch_size=256, transform=None, 
                 split='train', train_ratio=0.7, val_ratio=0.2, random_seed=42):
        """
        Args:
            data_dir: 數據目錄
            preprocessed_dir: 預處理數據目錄
            patch_size: patch大小
            transform: 數據變換
            split: 'train', 'val', 或 'test'
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例 (測試集比例 = 1 - train_ratio - val_ratio)
            random_seed: 隨機種子，確保可重現的分割
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.split = split
        
        if preprocessed_dir is None:
            self.preprocessed_dir = self.data_dir / "preprocessed"
        else:
            self.preprocessed_dir = Path(preprocessed_dir)
        
        # 找到所有圖像檔案
        all_image_files = []
        for img_file in self.data_dir.glob("*.tif"):
            # 跳過帶有後綴的檔案
            if any(suffix in img_file.name for suffix in ['_label', '_mask', '_boundary']):
                continue
                
            base_name = img_file.stem
            label_file = self.data_dir / f"{base_name}_label.tif"
            mask_file = self.data_dir / f"{base_name}_mask.tif"
            boundary_file = self.data_dir / f"{base_name}_boundary.tif"
            
            if label_file.exists() and mask_file.exists() and boundary_file.exists():
                all_image_files.append({
                    'base_name': base_name,
                    'image': img_file,
                    'label': label_file,
                    'mask': mask_file,
                    'boundary': boundary_file
                })
        
        # 按名稱排序確保一致性
        all_image_files.sort(key=lambda x: x['base_name'])
        
        # 使用固定隨機種子分割數據集
        np.random.seed(random_seed)
        indices = np.random.permutation(len(all_image_files))
        
        # 計算各集合的大小
        n_total = len(all_image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        # 分割索引
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # 根據split選擇對應的圖像文件
        if split == 'train':
            self.image_files = [all_image_files[i] for i in train_indices]
        elif split == 'val':
            self.image_files = [all_image_files[i] for i in val_indices]
        elif split == 'test':
            self.image_files = [all_image_files[i] for i in test_indices]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        
        print(f"Dataset split '{split}': {len(self.image_files)} image sets")
        print(f"Total images - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    def __len__(self):
        return len(self.image_files) * 10  # 每個圖像生成多個patches
    
    def __getitem__(self, idx):
        file_idx = idx % len(self.image_files)
        files = self.image_files[file_idx]
        base_name = files['base_name']
        
        # 讀取圖像
        image = cv2.imread(str(files['image']), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(files['label']), cv2.IMREAD_GRAYSCALE)
        
        # 加載預處理的抽樣數據
        sampling_data = load_sampling_data(self.preprocessed_dir, base_name)
        
        # 以神經亮點為中心的裁剪
        h, w = image.shape[:2]
        top, left = 0, 0
        
        if h > self.patch_size and w > self.patch_size:
            # 尋找神經亮點位置
            neural_positions = np.where(label == 255)
            
            if len(neural_positions[0]) > 0:
                # 隨機選擇一個神經亮點作為中心
                center_idx = random.randint(0, len(neural_positions[0]) - 1)
                center_y = neural_positions[0][center_idx]
                center_x = neural_positions[1][center_idx]
                
                # 計算patch的左上角座標，使神經亮點盡量在中心
                half_patch = self.patch_size // 2
                top = max(0, min(center_y - half_patch, h - self.patch_size))
                left = max(0, min(center_x - half_patch, w - self.patch_size))
            else:
                # 如果沒有神經亮點，回退到隨機裁切
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
            
            image = image[top:top+self.patch_size, left:left+self.patch_size]
            label = label[top:top+self.patch_size, left:left+self.patch_size]
            
            # 調整預處理數據以匹配 patch
            if sampling_data is not None:
                sampling_data = self.adjust_sampling_data_for_patch(
                    sampling_data, top, left, self.patch_size, self.patch_size
                )
        
        # 轉換為 tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        label = torch.from_numpy(label).long()
        
        return image, label, sampling_data
    
    def adjust_sampling_data_for_patch(self, sampling_data, top, left, patch_h, patch_w):
        """調整預處理數據以匹配裁剪的 patch（動態抽樣版本）"""
        if sampling_data is None:
            return None
        
        adjusted_data = {}
        
        # 調整 upper_region 遮罩
        upper_region = sampling_data['upper_region']
        adjusted_data['upper_region'] = upper_region[top:top+patch_h, left:left+patch_w]
        
        # 保留統計信息（這些將在動態抽樣時重新計算）
        adjusted_data['neural_count'] = sampling_data['neural_count']
        adjusted_data['non_neural_count'] = sampling_data['non_neural_count']
        adjusted_data['max_sample_count'] = sampling_data['max_sample_count']
        adjusted_data['boundary_pixel_count'] = sampling_data['boundary_pixel_count']
        adjusted_data['image_shape'] = (patch_h, patch_w)
        
        return adjusted_data

def collate_fn(batch):
    """自定義 collate 函數來處理 sampling_data"""
    images, labels, sampling_data_list = zip(*batch)
    
    # 將 images 和 labels 堆疊
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    
    return images, labels, list(sampling_data_list)

def evaluate_model(model, dataloader, criterion, device):
    """評估模型性能"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels, sampling_data_list in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, labels, sampling_data_list)
            
            total_loss += loss.item()
            total_samples += 1
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss

def train_model(data_dir, num_epochs=50, batch_size=64, learning_rate=0.001, patch_size=256, use_wandb=True, project_name="neural-detection"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化wandb
    if use_wandb:
        wandb.init(
            project=project_name,
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "patch_size": patch_size,
                "device": str(device),
                "architecture": "NeuralDetectionCNN",
                "loss_function": "SpecialLoss"
            }
        )
    
    # 創建數據集和數據加載器
    train_dataset = NeuralDataset(data_dir, patch_size=patch_size, split='train')
    val_dataset = NeuralDataset(data_dir, patch_size=patch_size, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, collate_fn=collate_fn)
    
    # 初始化模型
    model = NeuralDetectionCNN(input_channels=3, num_classes=2).to(device)
    criterion = SpecialLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 記錄模型到wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
    
    # 訓練循環
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels, sampling_data_list) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向傳播
            predictions = model(images)
            
            # 計算損失
            loss = criterion(predictions, labels, sampling_data_list)
            
            # 反向傳播
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 記錄batch級別的指標到wandb
            if use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch": epoch * len(train_loader) + batch_idx,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        # 驗證評估
        val_loss = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Validation Loss: {val_loss:.4f}')
        
        # 記錄epoch級別的指標到wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr
            })
        
        scheduler.step()
        
        # 每 10 個 epoch 保存模型
        if (epoch + 1) % 10 == 0:
            model_path = f'neural_detection_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            
            # 記錄模型檢查點到wandb
            if use_wandb:
                wandb.save(model_path)
                wandb.log({"model_checkpoint": epoch + 1})
    
    # 保存最終模型
    final_model_path = 'neural_detection_model_final.pth'
    torch.save(model.state_dict(), final_model_path)
    
    # 最終評估測試集
    print("\nEvaluating on test set...")
    test_dataset = NeuralDataset(data_dir, patch_size=patch_size, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)
    
    test_loss = evaluate_model(model, test_loader, criterion, device)
    print(f'Final Test Loss: {test_loss:.4f}')
    
    if use_wandb:
        wandb.log({"final_test_loss": test_loss})
        wandb.save(final_model_path)
        wandb.finish()
    
    print("Training completed!")
    
    return model

if __name__ == "__main__":
    data_dir = "/home/bl515-ml/Documents/shaio_jie/ienf_q/Centered"
    model = train_model(
        data_dir=data_dir, 
        num_epochs=50, 
        batch_size=16, 
        patch_size=512,
        use_wandb=True,  # 設為False可禁用wandb記錄
        project_name="neural-detection-dynamic-sampling"
    )