# IENF 多階段 U-Net 訓練系統

基於 nnU-Net 的多階段訓練系統，用於 IENF (Intraepidermal Nerve Fiber) 分割和路徑預測。

## 系統架構

### 三個訓練階段

1. **基礎階段 (Basic Stage)**
   - 學習基本的前景/背景分割
   - 使用距離權重損失函數
   - 重點：準確檢測神經纖維和亮點

2. **連接階段 (Connectivity Stage)**  
   - 學習亮點間的連通性
   - 使用連接性損失函數
   - 重點：建立亮點間的拓撲關係

3. **路徑階段 (Path Stage)**
   - 學習最優路徑預測
   - 結合傳統演算法指導
   - 重點：生成符合生物學的連接路徑

## 文件結構

```
U_net_transform/
├── training_pipeline.py    # 主要訓練流程
├── loss.py                # 損失函數定義
├── dataset.py             # 數據載入和處理
├── model.py               # 模型定義（待實現）
├── config.yaml            # 配置文件範例
└── README.md              # 本文件
```

## 快速開始

### 1. 準備數據

數據目錄結構：
```
data/
├── train/
│   ├── images/           # 原始圖像
│   └── bright_points/    # 亮點遮罩
└── val/
    ├── images/
    └── bright_points/
```

### 2. 配置設定

編輯 `config.yaml` 文件：
```yaml
# 更新以下路徑
model:
  nnunet_checkpoint: "path/to/your/nnunet/checkpoint.pth"

data:
  train_data_path: "path/to/train/data"
  val_data_path: "path/to/val/data"
```

### 3. 執行訓練

```bash
# 執行完整的三階段訓練
python training_pipeline.py --config config.yaml --stage all

# 執行單一階段
python training_pipeline.py --config config.yaml --stage basic
python training_pipeline.py --config config.yaml --stage connectivity  
python training_pipeline.py --config config.yaml --stage path
```

## 配置說明

### 模型配置
- `nnunet_checkpoint`: 預訓練 nnU-Net 檢查點路徑
- `num_classes`: 分類數量（通常為2：前景/背景）

### 訓練階段配置
每個階段可獨立配置：
- `epochs`: 訓練輪數
- `learning_rate`: 學習率
- `weight_decay`: 權重衰減
- `freeze_encoder`: 是否凍結編碼器

### 數據增強
- `rotation_range`: 旋轉角度範圍
- `brightness_range`: 亮度變化範圍
- `flip_horizontal/vertical`: 翻轉增強

## 損失函數

### DistanceWeightedLoss
- 基於到亮點距離的加權交叉熵
- 鼓勵亮點附近區域預測為前景

### ConnectivityLoss
- Soft Dice Loss + 亮點損失
- 基於骨架化建立連通性目標

### PathGuidedLoss
- 結合傳統路徑演算法的指導損失
- 使用加權交叉熵強化路徑預測

## 輸出文件

- `checkpoints/`: 每階段最佳模型檢查點
- `wandb/`: 訓練記錄和可視化
- 控制台輸出：實時訓練進度

## 自定義化

### 添加新的訓練階段
1. 繼承 `TrainingStage` 類
2. 實現 `get_loss_function()` 方法
3. 在配置中添加階段配置
4. 在 `MultiStageTrainer._initialize_stages()` 中註冊

### 修改模型架構
編輯 `nnUNetBasedModel` 類來調整：
- 自定義頭部結構
- 特徵融合方式  
- 輸出層配置

## 依賴需求

```bash
pip install torch torchvision
pip install wandb
pip install PyYAML
pip install scikit-image
pip install scipy
pip install Pillow
pip install nnunetv2  # 根據您的nnU-Net版本
```

## 注意事項

1. **nnU-Net 版本**: 確保 nnU-Net 檢查點與代碼版本兼容
2. **記憶體使用**: 根據GPU記憶體調整批次大小
3. **數據格式**: 確保圖像和遮罩尺寸一致
4. **路徑演算法**: 在 `PathGuidedLoss` 中實現您的特定路徑搜尋演算法

## 故障排除

### 常見問題
1. **找不到 nnU-Net 模型**: 檢查檢查點路徑是否正確
2. **記憶體不足**: 減少批次大小或圖像尺寸
3. **數據載入錯誤**: 確保數據目錄結構正確

### 調試模式
```bash
# 測試數據載入
python dataset.py

# 測試單一批次
python -c "from dataset import test_dataset; test_dataset()"
```