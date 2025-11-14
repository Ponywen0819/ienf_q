# IENF 神經纖維重建 Pipeline

> 從兩張圖片到完整重建 - 全自動 IENF 量化分析流程

## 🚀 快速開始

```bash
python run_pipeline.py \
    --image data/original_image.png \
    --annotation data/manual_annotation.png \
    --output output/reconstruction
```

**就這麼簡單！** 只需要：
1. 原始顯微鏡影像（RGB 或灰階）
2. 手動標註影像（二值化）

Pipeline 自動完成 6 個步驟的完整流程！

---

## 📋 Pipeline 流程

```
輸入兩張圖片
├── 原始影像 (RGB/灰階)
└── 手動標註 (二值化)
    ↓
[步驟 1] 影像前處理 - 提取綠色通道
    ↓
[步驟 2] 標註處理 - 連通元件分析
    ↓
[步驟 3] 骨架化處理 - Zhang-Suen 演算法
    ↓
[步驟 4] 種子提取 - 曲率感知分段
    ↓
[步驟 5] 網路建構 - 多因子成本計算 + A* 尋路
    ↓
[步驟 6] 神經重建 - 約束 MST 森林
    ↓
輸出最終結果
├── MST 森林 (GraphML + JSON)
├── 視覺化圖像
└── 統計資訊
```

---

## 💡 主要特點

✅ **極簡輸入** - 只需兩張圖片
✅ **全自動處理** - 6 步驟一次完成
✅ **乾淨輸出** - 只有最終結果，無雜亂中間檔
✅ **彈性配置** - 3 種預設配置 + YAML + CLI 覆蓋
✅ **中間產物管理** - 預設自動清理，需要時可保存
✅ **參數可調** - 22 個超參數完全可配置

---

## 📦 配置選項

### 預設配置（平衡）

```bash
python run_pipeline.py \
    --image IMAGE --annotation ANNO --output OUT
```

- 曲率閾值: 30°
- 影像權重: 0.9
- 適合日常分析

### 高品質配置（論文用）

```bash
python run_pipeline.py \
    --image IMAGE --annotation ANNO --output OUT \
    --config config/high_quality.yaml
```

- 曲率閾值: 25° (更敏感)
- 影像權重: 0.92 (更貼合)
- 自動保存中間產物
- 適合論文發表

### 快速配置（測試用）

```bash
python run_pipeline.py \
    --image IMAGE --annotation ANNO --output OUT \
    --config config/fast.yaml
```

- 處理速度優先
- 適合快速測試

---

## 🔧 參數調整

### 常用參數覆蓋

```bash
# 調整曲率敏感度
python run_pipeline.py --image IMAGE --annotation ANNO --output OUT \
    --curvature-threshold 25

# 調整連接嚴格度
python run_pipeline.py --image IMAGE --annotation ANNO --output OUT \
    --max-edge-cost 160

# 增強影像引導
python run_pipeline.py --image IMAGE --annotation ANNO --output OUT \
    --beta 0.95 --alpha 0.03
```

### 可調參數總覽

| 參數 | 預設 | 說明 |
|------|------|------|
| `--curvature-threshold` | 30 | 曲率閾值（度），越小越敏感 |
| `--max-edge-cost` | 150 | 最大邊成本，越大越寬鬆 |
| `--alpha` | 0.05 | 幾何成本權重 |
| `--beta` | 0.9 | 影像成本權重 |
| `--gamma` | 0.05 | 曲率成本權重 |

---

## 📂 輸出結構

### 預設（不保存中間產物）

```
output/
└── reconstruction/
    ├── mst_forest.graphml      # MST 森林圖
    ├── mst_forest.json         # 詳細資訊
    ├── visualization_mst.png   # MST 視覺化
    └── statistics.json         # 統計摘要
```

### 保存中間產物時

```
output/
├── intermediates/              # 所有中間步驟
│   ├── green_channel.png      # 步驟 1
│   ├── components/            # 步驟 2
│   ├── skeletons/             # 步驟 3
│   ├── seeds/                 # 步驟 4
│   └── network/               # 步驟 5
└── reconstruction/             # 步驟 6: 最終結果
```

使用 `--save-intermediates` 來保存中間產物（用於除錯）。

---

## 🎯 使用範例

### 範例 1: 日常分析

```bash
python run_pipeline.py \
    --image samples/patient01.png \
    --annotation samples/patient01_anno.png \
    --output results/patient01
```

### 範例 2: 論文品質

```bash
python run_pipeline.py \
    --image samples/patient01.png \
    --annotation samples/patient01_anno.png \
    --output results/patient01 \
    --config config/high_quality.yaml
```

### 範例 3: 批次處理

```bash
for sample in sample01 sample02 sample03; do
    python run_pipeline.py \
        --image data/${sample}.png \
        --annotation data/${sample}_anno.png \
        --output output/${sample}
done
```

### 範例 4: 參數微調

```bash
# 如果連接過多（假陽性）
python run_pipeline.py --image IMAGE --annotation ANNO --output OUT \
    --max-edge-cost 130

# 如果遺漏纖維段（假陰性）
python run_pipeline.py --image IMAGE --annotation ANNO --output OUT \
    --curvature-threshold 25 --max-edge-cost 160
```

---

## 📝 輸入圖片要求

### 原始影像 (`--image`)
- 格式: PNG, JPG, TIFF
- 類型: RGB 彩色或灰階
- 內容: IENF 染色顯微鏡影像
- 推薦: RGB 影像（自動提取綠色通道）

### 手動標註 (`--annotation`)
- 格式: PNG, JPG
- 類型: 二值化影像
- 內容:
  - 神經纖維 = 白色 (255)
  - 背景 = 黑色 (0)
- 可使用 ImageJ、GIMP 或 Photoshop 製作

---

## 🛠 疑難排解

### 連接過多？
```bash
--max-edge-cost 130  # 降低閾值
```

### 遺漏纖維段？
```bash
--curvature-threshold 25 --max-edge-cost 160  # 提高靈敏度
```

### 重建過於直線？
```bash
--alpha 0.03 --beta 0.95  # 增強影像引導
```

### 需要檢查中間結果？
```bash
--save-intermediates  # 保存所有步驟
```

---

## 📚 完整文檔

- [USAGE_GUIDE.md](USAGE_GUIDE.md) - 詳細使用指南
- [config/README.md](config/README.md) - 配置參數說明
- [PIPELINE_INTEGRATION_SUMMARY.md](PIPELINE_INTEGRATION_SUMMARY.md) - 技術實作細節

---

## 🎓 演算法概述

1. **前處理** - 綠色通道提取
2. **連通元件** - 標註分離
3. **骨架化** - Zhang-Suen 細化
4. **種子提取** - 曲率感知自適應分段
5. **網路建構** - 多因子成本 + A* 尋路
   - α × 幾何成本 + β × 影像成本 + γ × 曲率成本
6. **神經重建** - 約束 MST 森林（Kruskal's 演算法）

---

## 💻 系統需求

- Python 3.8+
- OpenCV
- scikit-image
- NetworkX
- NumPy
- Matplotlib
- Pydantic (用於配置)

---

## 📄 授權

由 Claude Code 生成並整合。

---

**輕鬆兩步驟，完成 IENF 量化！** 🎉
