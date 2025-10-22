# IENF 前處理模組

神經纖維影像的前處理流程，用於改善影像品質並為後續分析做準備。

## 功能概述

### 處理流程

#### 0.1 通道去耦合
**問題**: 紅色通道（其他染料）干擾綠色通道（神經纖維）

**方法**: 線性校正
```
Green_corrected = Green_original - α × Red

參數 α:
- 基礎值: 0.2
- 表皮下方紅色密度高時: 可能需要 0.3-0.5
- 需要根據實際影像調整
```

**自適應策略**（可選）:
- 根據區域紅色密度動態調整 α
- 表皮上方: α 較小
- 表皮下方: α 較大
- 使用平滑過渡避免邊界偽影

#### 0.2 背景不均校正
**方法**: Rolling Ball 背景扣除

**原理**:
- 模擬球體在影像強度表面下方滾動
- 球接觸的表面視為背景
- 從原影像扣除背景

**參數**:
- ball_radius: 40-60 像素
- 應大於神經纖維最大寬度

#### 0.3 局部對比度增強
**方法**: CLAHE (對比度受限自適應直方圖均衡化)

**參數**:
- tile_size: 8×8 或 16×16
- clip_limit: 2.0-4.0

**效果**:
- 平衡不同區域亮度
- 增強局部對比度
- 使表皮上下方的綠色通道品質更接近

#### 0.4 標準化
**方法**: 百分位數標準化
- 計算 1% 和 99% 百分位數
- 裁剪極端值
- 映射到 [0, 255]

**輸出**: 校正後的綠色通道影像，用於所有後續步驟

## 安裝

```bash
# 安裝依賴
pip install numpy pillow opencv-python scikit-image
```

## 使用方法

### 1. Python API

```python
from preprocessing import PreprocessingPipeline, PreprocessingConfig

# 使用默認配置
pipeline = PreprocessingPipeline()
result = pipeline.process_file("input.tif", "output.tif")

# 使用自定義配置
config = PreprocessingConfig()
config.channel_decoupling.alpha = 0.3
config.background_correction.ball_radius = 60
config.contrast_enhancement.clip_limit = 3.0

pipeline = PreprocessingPipeline(config)
result = pipeline.process_file("input.tif", "output.tif")

# 批量處理
pipeline.process_batch(
    input_dir="./raw_images",
    output_dir="./processed_images",
    num_workers=4
)
```

### 2. 命令列工具

```bash
# 處理單個文件
python -m preprocessing.cli process-single input.tif output.tif

# 使用配置文件
python -m preprocessing.cli process-single input.tif output.tif --config config.yaml

# 批量處理
python -m preprocessing.cli process-batch ./input_dir ./output_dir

# 批量處理（自定義參數）
python -m preprocessing.cli process-batch ./input_dir ./output_dir --workers 8

# 生成默認配置文件
python -m preprocessing.cli create-config config.yaml
```

### 3. 配置文件

配置文件使用 YAML 格式：

```yaml
# 通道去耦合
channel_decoupling:
  alpha: 0.2
  adaptive: false
  alpha_min: 0.2
  alpha_max: 0.5

# 背景校正
background_correction:
  ball_radius: 50
  method: rolling_ball

# 對比度增強
contrast_enhancement:
  tile_size: [8, 8]
  clip_limit: 2.0

# 標準化
normalization:
  lower_percentile: 1.0
  upper_percentile: 99.0
  output_range: [0, 255]

# 流程控制
enable_channel_decoupling: true
enable_background_correction: true
enable_contrast_enhancement: true
enable_normalization: true

# 輸出控制
save_intermediate: false
verbose: true
```

## 模組結構

```
preprocessing/
├── __init__.py                    # 套件初始化
├── image_loader.py                # 圖像載入
├── channel_decoupling.py          # 通道去耦合
├── background_correction.py       # 背景校正
├── contrast_enhancement.py        # 對比度增強
├── normalization.py               # 標準化
├── pipeline.py                    # 處理流程
├── config.py                      # 配置管理
├── cli.py                         # 命令列介面
└── README.md                      # 文檔
```

## 參數調整建議

### 通道去耦合 (alpha)
- 紅色干擾輕微: α = 0.1-0.2
- 紅色干擾中等: α = 0.2-0.3
- 紅色干擾嚴重: α = 0.3-0.5

### Rolling Ball 半徑
- 神經纖維較細: radius = 40
- 神經纖維較粗: radius = 60

### CLAHE 參數
- 對比度不足: clip_limit = 3.0-4.0
- 對比度過強: clip_limit = 1.5-2.0
- tile_size 越小，局部增強越明顯

## 輸出

- **主要輸出**: 處理後的綠色通道圖像
- **中間輸出**（可選）:
  - `*_decoupled.tif`: 通道去耦合後
  - `*_background_corrected.tif`: 背景校正後
  - `*_contrast_enhanced.tif`: 對比度增強後
  - `*_normalized.tif`: 標準化後（最終輸出）

## 故障排除

### 圖像過暗
- 增加 CLAHE 的 clip_limit
- 檢查百分位數標準化的參數

### 圖像過亮或過度增強
- 降低 CLAHE 的 clip_limit
- 調整 tile_size

### 背景校正不足
- 增加 ball_radius
- 確保 radius 大於神經纖維寬度

### 紅色通道干擾仍存在
- 增加 alpha 值
- 考慮啟用自適應策略

## 開發狀態

當前版本: 1.0.0

- [x] 模組架構設計
- [ ] 圖像載入功能
- [ ] 通道去耦合實現
- [ ] 背景校正實現
- [ ] 對比度增強實現
- [ ] 標準化實現
- [ ] 流程整合
- [ ] CLI 工具
- [ ] 單元測試
- [ ] 性能優化
