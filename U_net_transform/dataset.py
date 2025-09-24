import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class IENFDataset(Dataset):
    """IENF數據集類"""
    def __init__(self, 
                 data_path: str, 
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = False,
                 augment_config: Dict = None):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.augment = augment
        self.augment_config = augment_config or {}
        
        # 載入數據列表
        self.data_list = self._load_data_list()
        
        # 基本轉換
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 遮罩轉換
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # 數據增強
        if augment:
            self.augment_transform = self._get_augment_transform()
    
    def _load_data_list(self):
        """載入數據列表"""
        data_list = []
        
        # 假設數據結構：
        # data_path/
        #   ├── images/
        #   │   ├── image1.png
        #   │   └── image2.png
        #   ├── bright_points/
        #   │   ├── image1.png
        #   │   └── image2.png
        #   └── metadata.json (可選)
        
        image_dir = self.data_path / "images"
        bright_points_dir = self.data_path / "bright_points"
        
        if not image_dir.exists():
            raise ValueError(f"圖像目錄不存在: {image_dir}")
        if not bright_points_dir.exists():
            raise ValueError(f"亮點遮罩目錄不存在: {bright_points_dir}")
        
        for image_path in sorted(image_dir.glob("*.png")):
            bright_point_path = bright_points_dir / image_path.name
            
            if bright_point_path.exists():
                data_list.append({
                    'image': str(image_path),
                    'bright_points': str(bright_point_path),
                    'image_id': image_path.stem
                })
        
        print(f"載入 {len(data_list)} 個樣本")
        return data_list
    
    def _get_augment_transform(self):
        """獲取數據增強變換"""
        augment_list = []
        
        if self.augment_config.get('rotation_range', 0) > 0:
            rotation_range = self.augment_config['rotation_range']
            augment_list.append(
                transforms.RandomRotation((-rotation_range, rotation_range))
            )
        
        if self.augment_config.get('flip_horizontal', False):
            augment_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
        if self.augment_config.get('flip_vertical', False):
            augment_list.append(transforms.RandomVerticalFlip(p=0.5))
        
        if self.augment_config.get('brightness_range', 0) > 0:
            brightness = self.augment_config['brightness_range']
            augment_list.append(
                transforms.ColorJitter(brightness=brightness)
            )
        
        if self.augment_config.get('contrast_range', 0) > 0:
            contrast = self.augment_config['contrast_range']
            augment_list.append(
                transforms.ColorJitter(contrast=contrast)
            )
        
        return transforms.Compose(augment_list) if augment_list else None
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        data_info = self.data_list[idx]
        
        # 載入圖像
        image = Image.open(data_info['image']).convert('RGB')
        bright_points = Image.open(data_info['bright_points']).convert('L')
        
        # 轉換為numpy進行一致性增強
        image_np = np.array(image)
        bright_points_np = np.array(bright_points)
        
        # 數據增強（保持圖像和遮罩一致性）
        if self.augment and self.augment_transform is not None:
            # 使用相同的隨機種子確保一致性
            seed = np.random.randint(0, 2**32)
            
            # 增強圖像
            np.random.seed(seed)
            torch.manual_seed(seed)
            image = self.augment_transform(Image.fromarray(image_np))
            
            # 增強遮罩（使用相同種子）
            np.random.seed(seed)
            torch.manual_seed(seed)
            bright_points = self.augment_transform(Image.fromarray(bright_points_np))
        else:
            image = Image.fromarray(image_np)
            bright_points = Image.fromarray(bright_points_np)
        
        # 應用基本變換
        image = self.base_transform(image)
        bright_points = self.mask_transform(bright_points)
        
        # 將遮罩轉換為二值
        bright_points = (bright_points > 0.5).float()
        
        # 創建背景遮罩
        background_mask = (bright_points == 0).float()
        
        return {
            'image': image,
            'bright_points': bright_points.squeeze(0),  # 移除通道維度
            'background_mask': background_mask.squeeze(0),
            'image_id': data_info['image_id']
        }


class IENFDataModule:
    """數據模塊，管理訓練和驗證數據加載器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.augment_config = config.get('augmentation', {})
        
    def get_dataloader(self, split: str) -> DataLoader:
        """獲取指定split的數據加載器"""
        if split == 'train':
            data_path = self.data_config['train_data_path']
            shuffle = True
            augment = self.augment_config.get('enable', False)
        elif split == 'val':
            data_path = self.data_config['val_data_path']
            shuffle = False
            augment = False
        else:
            raise ValueError(f"未知的split: {split}")
        
        dataset = IENFDataset(
            data_path=data_path,
            image_size=tuple(self.data_config['image_size']),
            augment=augment,
            augment_config=self.augment_config
        )
        
        return DataLoader(
            dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=shuffle,
            num_workers=self.data_config['num_workers'],
            pin_memory=True,
            drop_last=True if split == 'train' else False
        )
    
    def setup(self):
        """設置數據模塊"""
        self.train_dataloader = self.get_dataloader('train')
        self.val_dataloader = self.get_dataloader('val')
        
        print(f"訓練集大小: {len(self.train_dataloader.dataset)}")
        print(f"驗證集大小: {len(self.val_dataloader.dataset)}")
        
        return self.train_dataloader, self.val_dataloader


def test_dataset():
    """測試數據集載入"""
    config = {
        'data': {
            'train_data_path': 'data/train',
            'val_data_path': 'data/val',
            'batch_size': 2,
            'num_workers': 0,
            'image_size': [512, 512]
        },
        'augmentation': {
            'enable': True,
            'rotation_range': 15,
            'brightness_range': 0.2,
            'flip_horizontal': True,
            'flip_vertical': True
        }
    }
    
    try:
        data_module = IENFDataModule(config)
        train_loader, val_loader = data_module.setup()
        
        # 測試一個批次
        for batch in train_loader:
            print("批次形狀:")
            print(f"圖像: {batch['image'].shape}")
            print(f"亮點遮罩: {batch['bright_points'].shape}")
            print(f"背景遮罩: {batch['background_mask'].shape}")
            break
            
    except Exception as e:
        print(f"數據集測試失敗: {e}")


if __name__ == "__main__":
    test_dataset()