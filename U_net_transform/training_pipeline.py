import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import wandb
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

from loss import DistanceWeightedLoss, ConnectivityLoss, PathGuidedLoss


class TrainingStage:
    """訓練階段基類"""
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.epochs = config.get('epochs', 50)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        
    def get_loss_function(self) -> nn.Module:
        """獲取該階段的損失函數"""
        raise NotImplementedError
        
    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """獲取該階段的優化器"""
        return optim.Adam(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def should_freeze_encoder(self) -> bool:
        """是否凍結編碼器"""
        return False
        
    def get_stage_specific_metrics(self) -> Dict[str, Any]:
        """獲取階段特定的評估指標"""
        return {}


class BasicStage(TrainingStage):
    """基礎階段：學習基本的前景/背景分割"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("basic", config)
        
    def get_loss_function(self) -> nn.Module:
        return DistanceWeightedLoss(
            alpha=self.config.get('distance_alpha', 1.0),
            beta=self.config.get('distance_beta', 2.0)
        )


class ConnectivityStage(TrainingStage):
    """連接階段：學習亮點間的連通性"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("connectivity", config)
        
    def get_loss_function(self) -> nn.Module:
        return ConnectivityLoss(
            connectivity_weight=self.config.get('connectivity_weight', 2.0),
            topology_weight=self.config.get('topology_weight', 1.0)
        )
        
    def should_freeze_encoder(self) -> bool:
        return self.config.get('freeze_encoder', True)


class PathStage(TrainingStage):
    """路徑階段：學習最優路徑預測"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("path", config)
        
    def get_loss_function(self) -> nn.Module:
        return PathGuidedLoss(
            path_weight=self.config.get('path_weight', 3.0),
            background_weight=self.config.get('background_weight', 0.5)
        )
        
    def should_freeze_encoder(self) -> bool:
        return self.config.get('freeze_encoder', True)


class nnUNetBasedModel(nn.Module):
    """基於nnU-Net的模型"""
    def __init__(self, nnunet_checkpoint_path: str, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        
        # 載入預訓練的nnU-Net模型
        self.nnunet_model = self._load_nnunet_model(nnunet_checkpoint_path)
        
        # 凍結nnU-Net的參數（初始階段）
        self._freeze_nnunet(freeze=True)
        
        # 添加自定義頭部用於微調
        self.custom_head = nn.Sequential(
            nn.Conv2d(self.nnunet_model.num_classes, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def _load_nnunet_model(self, checkpoint_path: str):
        """載入nnU-Net模型"""
        # 這裡需要根據您的nnU-Net版本調整
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        
        # 載入預訓練權重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = checkpoint['network']
        
        return model
    
    def _freeze_nnunet(self, freeze: bool = True):
        """凍結/解凍nnU-Net參數"""
        for param in self.nnunet_model.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_encoder(self):
        """解凍編碼器進行微調"""
        self._freeze_nnunet(freeze=False)
        
    def forward(self, x):
        # nnU-Net特徵提取
        features = self.nnunet_model(x)
        
        # 自定義頭部
        output = self.custom_head(features)
        
        return output


class MultiStageTrainer:
    """多階段訓練器"""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = nnUNetBasedModel(
            nnunet_checkpoint_path=self.config['model']['nnunet_checkpoint'],
            num_classes=self.config['model']['num_classes']
        ).to(self.device)
        
        # 初始化訓練階段
        self.stages = self._initialize_stages()
        
        # 初始化數據加載器
        self.train_loader = self._get_dataloader('train')
        self.val_loader = self._get_dataloader('val')
        
        # 初始化wandb
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config['wandb']['project'],
                config=self.config,
                name=f"multistage_training_{self.config['experiment_name']}"
            )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """載入配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_stages(self) -> Dict[str, TrainingStage]:
        """初始化所有訓練階段"""
        stages = {}
        
        if 'basic' in self.config['stages']:
            stages['basic'] = BasicStage(self.config['stages']['basic'])
            
        if 'connectivity' in self.config['stages']:
            stages['connectivity'] = ConnectivityStage(self.config['stages']['connectivity'])
            
        if 'path' in self.config['stages']:
            stages['path'] = PathStage(self.config['stages']['path'])
            
        return stages
    
    def _get_dataloader(self, split: str) -> DataLoader:
        """獲取數據加載器"""
        # 這裡需要根據您的數據集實現
        # 返回適當的DataLoader
        pass
    
    def train_stage(self, stage: TrainingStage):
        """訓練單個階段"""
        print(f"\n開始訓練階段: {stage.name}")
        print(f"訓練輪數: {stage.epochs}")
        print(f"學習率: {stage.learning_rate}")
        
        # 設置模型狀態
        if stage.should_freeze_encoder():
            print("凍結編碼器參數")
        else:
            print("解凍編碼器參數")
            self.model.unfreeze_encoder()
        
        # 獲取損失函數和優化器
        criterion = stage.get_loss_function()
        optimizer = stage.get_optimizer(self.model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(stage.epochs):
            # 訓練階段
            train_loss = self._train_epoch(criterion, optimizer, stage.name)
            
            # 驗證階段
            val_loss, val_metrics = self._validate_epoch(criterion, stage.name)
            
            # 更新學習率
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{stage.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 記錄到wandb
            if self.config.get('use_wandb', False):
                log_dict = {
                    f"{stage.name}/train_loss": train_loss,
                    f"{stage.name}/val_loss": val_loss,
                    f"{stage.name}/learning_rate": optimizer.param_groups[0]['lr'],
                    f"{stage.name}/epoch": epoch
                }
                log_dict.update({f"{stage.name}/{k}": v for k, v in val_metrics.items()})
                wandb.log(log_dict)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(stage.name, epoch, val_loss)
    
    def _train_epoch(self, criterion: nn.Module, optimizer: optim.Optimizer, stage_name: str) -> float:
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 獲取批次數據
            images = batch['image'].to(self.device)
            bright_points = batch['bright_points'].to(self.device)
            background_mask = batch.get('background_mask', None)
            
            if background_mask is not None:
                background_mask = background_mask.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向傳播
            predictions = self.model(images)
            
            # 計算損失（根據不同階段調用不同參數）
            if stage_name == 'basic':
                loss = criterion(predictions, bright_points, background_mask)
            elif stage_name == 'connectivity':
                loss = criterion(predictions, bright_points)
            elif stage_name == 'path':
                loss = criterion(predictions, bright_points, images)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def _validate_epoch(self, criterion: nn.Module, stage_name: str):
        """驗證一個epoch"""
        self.model.eval()
        total_loss = 0.0
        metrics = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                bright_points = batch['bright_points'].to(self.device)
                background_mask = batch.get('background_mask', None)
                
                if background_mask is not None:
                    background_mask = background_mask.to(self.device)
                
                predictions = self.model(images)
                
                # 計算損失
                if stage_name == 'basic':
                    loss = criterion(predictions, bright_points, background_mask)
                elif stage_name == 'connectivity':
                    loss = criterion(predictions, bright_points)
                elif stage_name == 'path':
                    loss = criterion(predictions, bright_points, images)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics
    
    def _save_checkpoint(self, stage_name: str, epoch: int, val_loss: float):
        """保存檢查點"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"best_{stage_name}_model.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'stage': stage_name,
            'config': self.config
        }, checkpoint_path)
        
        print(f"保存檢查點: {checkpoint_path}")
    
    def run_training(self):
        """執行完整的多階段訓練"""
        print("開始多階段訓練流程")
        print(f"訓練階段順序: {list(self.stages.keys())}")
        
        for stage_name, stage in self.stages.items():
            self.train_stage(stage)
            print(f"完成階段: {stage_name}")
        
        print("所有訓練階段完成！")
        
        if self.config.get('use_wandb', False):
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='多階段U-Net訓練')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--stage', type=str, choices=['basic', 'connectivity', 'path', 'all'],
                       default='all', help='指定訓練階段')
    
    args = parser.parse_args()
    
    # 創建訓練器
    trainer = MultiStageTrainer(args.config)
    
    if args.stage == 'all':
        trainer.run_training()
    else:
        # 訓練指定階段
        if args.stage in trainer.stages:
            trainer.train_stage(trainer.stages[args.stage])
        else:
            print(f"階段 {args.stage} 未在配置中定義")


if __name__ == "__main__":
    main()