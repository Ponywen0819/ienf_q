# IENF-Q: 神經纖維量化與重建系統

IENF-Q (Intra-Epidermal Nerve Fiber Quantification) 是一個全自動的分析流程，旨在從顯微鏡影像和稀疏的手動標註中重建完整的神經纖維網絡。本系統採用傳統電腦視覺演算法，確保了結果的可解釋性、可控性和穩定性。

## 🚀 功能特性

- **完整 Pipeline**: 整合前處理與重建，提供統一的 API 介面
- **極簡輸入**: 僅需原始影像、表皮 mask 和標註影像即可開始
- **全自動化**: 從影像前處理到神經網路重建，無需人工干預
- **演算法驅動**: 基於骨架分析、A\* 路徑尋找和最小生成樹 (MST) 等可解釋的演算法
- **曲率感知**: 獨特的種子點提取策略，能準確保留神經纖維的彎曲和分支特徵
- **高度可配置**: 所有關鍵參數均可透過程式化配置進行調整
- **靈活輸入**: 支援 RGB 或灰階影像，自動萃取綠色通道

## 📂 專案結構

```
.
├── src/neural_reconstruction/
│   ├── common/                      # 共用資料型別
│   ├── core/
│   │   ├── preprocessing/           # 影像前處理 pipeline
│   │   ├── construction/            # 神經網路重建
│   │   │   ├── component_analyzer/  # 骨架化、拓樸、種子萃取
│   │   │   ├── connection_graph_builder/  # A* 路徑尋找
│   │   │   ├── backbone_extractor/  # MST 萃取
│   │   │   └── main.py              # 重建主入口
│   │   └── crosses_detection/       # 表皮交叉偵測
│   └── ui/
│       └── main_pipeline.py         # 完整 pipeline 整合
├── data/
│   ├── Original/                    # 原始影像
│   ├── Label/                       # 標註影像
│   └── Mask/                        # 表皮 mask
├── test/                            # 單元測試
├── examples/                        # 使用範例
└── tools/                           # 輔助工具
```

## 🛠️ 安裝

本專案使用 `uv` 作為套件管理工具。

```bash
# 安裝依賴
uv sync
```

**需求**: Python >= 3.10

## 🏃‍♂️ 快速開始

### 方法 1: 使用完整 Pipeline (推薦)

```python
from neural_reconstruction.ui.main_pipeline import NeuralReconstructionPipeline

# 建立 pipeline (使用預設配置)
pipeline = NeuralReconstructionPipeline()

# 從檔案路徑執行
result = pipeline.run_from_files(
    label_path="data/Label/S163-2_a.tif",
    mask_path="data/Mask/S163-2_a.tif",
    image_path="data/Original/S163-2_a.tif"
)

# 取得結果
print(f"重建完成: {result.num_nodes} 節點, {result.num_edges} 條邊")
print(f"連通分量: {result.num_components}")

# 存取 NetworkX 圖
mst_forest = result.mst_forest
final_label = result.final_label
roi_image = result.roi_image
```

### 方法 2: 使用 NumPy 陣列

```python
import numpy as np
from PIL import Image
from neural_reconstruction.ui.main_pipeline import NeuralReconstructionPipeline

# 載入影像
label_image = np.array(Image.open("data/Label/S163-2_a.tif"))
mask_image = np.array(Image.open("data/Mask/S163-2_a.tif"))
original_image = np.array(Image.open("data/Original/S163-2_a.tif"))

# 執行 pipeline
pipeline = NeuralReconstructionPipeline()
result = pipeline.run(
    label_image=label_image,
    mask_image=mask_image,
    original_image=original_image  # 支援 RGB (H,W,3) 或灰階 (H,W)
)
```

### 執行測試腳本

```bash
# 執行完整 pipeline 測試
python test_pipeline.py

# 使用 uv
uv run python test_pipeline.py

# 執行範例程式
python examples/pipeline_usage.py
```

## ⚙️ 自訂配置

### 前處理配置

```python
preprocessing_config = {
    'morphology': {
        'closing_kernel': 5,        # 形態學 closing 核心大小
        'opening_kernel': 3         # 形態學 opening 核心大小
    },
    'mask': {
        'dilate_offset': 100        # 垂直擴張偏移量（像素）
    },
    'background': {
        'method': 'rolling_ball',   # 背景校正方法
        'radius': 20,               # Rolling ball 半徑
        'light_background': True    # 背景是否為亮色
    },
    'threshold': {
        'method': 'binary'          # 閾值方法
    },
    'normalization': {
        'enabled': True             # 啟用區域正規化
    }
}
```

### 重建配置

```python
reconstruction_config = {
    'connectivity': 4,              # 連通性 (4 或 8)
    'min_area': 30,                 # 最小元件面積（像素）
    'segment_length': 3.0,          # 種子間隔長度（像素）
    'search_radius': 100.0,         # 搜尋半徑（像素）
    'max_cost_threshold': 0.95,     # 最大成本閾值 (0-1)
    'intensity_weight': 0.7,        # 強度權重
    'shape_weight': 0.3             # 形狀權重
}

# 建立自訂 pipeline
pipeline = NeuralReconstructionPipeline(
    preprocessing_config=preprocessing_config,
    reconstruction_config=reconstruction_config
)
```

## 🔬 演算法流程

系統的重建過程主要分為以下階段：

### 階段 1: 影像前處理

1. **綠色通道萃取**: 從 RGB 影像自動萃取綠色通道（神經組織訊號最強）
2. **形態學處理**: Closing 填補空隙，Opening 移除雜訊
3. **背景校正**: Rolling ball 或形態學方法校正不均勻背景
4. **ROI 萃取**: 使用表皮 mask 提取感興趣區域
5. **閾值化**: Otsu 方法產生 pseudo-label

### 階段 2: 神經網路重建

1. **連通元件分析**: 識別離散的纖維段落
2. **骨架化與拓樸分析**: 使用 Zhang-Suen 演算法提取骨架，建立拓樸結構
3. **種子點萃取**: 沿骨架以曲率感知方式放置種子點
4. **連接圖建構**: 使用 A\* 演算法尋找元件間的最佳連接路徑
5. **骨架萃取**: 使用 MST (Minimum Spanning Tree) 萃取最優神經網路

## 📊 輸出結果

Pipeline 執行後會返回 `PipelineResult` 物件，包含：

- `mst_forest`: NetworkX Graph - 重建的神經網路骨架
  - 節點: 種子點座標 (y, x)
  - 邊: 包含 'path' 屬性的連接
- `final_label`: NumPy 陣列 - 前處理後的最終標註
- `roi_image`: NumPy 陣列 - ROI 區域影像
- `num_nodes`: 節點數
- `num_edges`: 邊數
- `num_components`: 連通分量數

### 儲存結果範例

```python
import networkx as nx
from PIL import Image

# 執行 pipeline
result = pipeline.run_from_files(...)

# 儲存影像
Image.fromarray(result.final_label).save("output/final_label.png")
Image.fromarray(result.roi_image).save("output/roi_image.png")

# 儲存圖結構
nx.write_graphml(result.mst_forest, "output/mst_forest.graphml")

# 儲存統計資訊
with open("output/summary.txt", 'w') as f:
    f.write(f"節點數: {result.num_nodes}\n")
    f.write(f"邊數: {result.num_edges}\n")
    f.write(f"連通分量數: {result.num_components}\n")
```

## 🧪 測試

```bash
# 執行所有測試
pytest test/

# 執行特定模組測試
pytest test/construction/component_analyzer/
pytest test/construction/connection_graph_builder/

# 執行單一測試檔案
pytest test/construction/component_analyzer/test_analyzer.py
```

## 📚 更多範例

查看 [examples/pipeline_usage.py](examples/pipeline_usage.py) 了解更多使用範例：

- 使用預設配置
- 自訂配置
- 直接使用 NumPy 陣列
- 儲存結果

## 🔧 進階使用

### 僅使用重建模組（不含前處理）

```python
from neural_reconstruction.core.construction.main import build_neural_network

# 假設已有前處理後的資料
mst_forest = build_neural_network(
    label_image=binary_label,       # (H, W) 二值影像
    green_channel=green_channel,    # (H, W) uint8
    connectivity=4,
    min_area=50,
    segment_length=5.0,
    search_radius=50.0,
    max_cost_threshold=0.98,
    intensity_weight=0.6,
    shape_weight=0.4
)
```

### 僅使用前處理模組

```python
from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline

config = {
    'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
    'background': {'radius': 12, 'light_background': True},
    'mask': {'dilate_offset': 50},
    'threshold': {'method': 'binary'}
}

pipeline = SkinAnalysisPipeline(config)
final_label, roi_image = pipeline.run(
    label_image, mask_image, original_image, debug=False
)
```

## 📝 重要說明

- **綠色通道**: 系統會自動從 RGB 影像萃取綠色通道，因為神經組織在該通道有最強訊號
- **連通性參數**: API 使用 4 和 8 表示連通性，內部會自動轉換為 scikit-image 的格式（1 和 2）
- **無邊結果**: 若 pipeline 執行後 `num_edges=0`，這是正常的，表示：
  - 元件之間距離超過 `search_radius`
  - 路徑成本超過 `max_cost_threshold`
  - 可嘗試增大 `search_radius` 或降低 `max_cost_threshold`

## 🤝 貢獻

歡迎提交 Pull Requests 或回報 Issues。

## 📄 授權

本專案採用 MIT 授權。

## 📖 文件

詳細的開發文件請參考 [CLAUDE.md](CLAUDE.md)，包含：

- 完整的 API 說明
- 詳細的架構說明
- 配置參數說明
- 程式碼慣例
