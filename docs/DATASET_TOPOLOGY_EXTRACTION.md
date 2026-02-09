# 數據集拓樸提取工具使用指南

## 概述

`tools/extract_dataset_topologies.py` 是一個自動化工具,用於從數據集中批量提取 Ground Truth 拓樸並儲存為 Pickle 格式。

## 主要特點

- ✅ **自動掃描**: 自動掃描 `data/` 資料夾中的所有樣本
- ✅ **批量處理**: 一次性處理所有包含 `label.png` 的樣本
- ✅ **GT 拓樸提取**: 使用 `TopologyExtractor` 從 label.png 提取完整拓樸
- ✅ **Pickle 格式**: 儲存為高效的 `.pkl` 格式,包含節點和邊路徑點
- ✅ **統計報告**: 提供詳細的提取統計和失敗報告

## 使用方法

### 基本用法

```bash
# 使用默認設置 (data/ -> output/topologies/)
uv run python tools/extract_dataset_topologies.py
```

### 自定義路徑

```bash
# 指定輸入輸出目錄
uv run python tools/extract_dataset_topologies.py \
    --data-dir data \
    --output-dir output/gt_topologies
```

### 詳細日志模式

```bash
# 查看詳細的處理過程
uv run python tools/extract_dataset_topologies.py --verbose
```

## 數據集結構

工具期望的數據集結構:

```
data/
├── S1140-2_a/
│   ├── image.png
│   ├── mask.png
│   ├── annotation.png
│   └── label.png          # 必須存在才會處理
├── S1585-2_a/
│   ├── image.png
│   ├── mask.png
│   ├── annotation.png
│   └── label.png
└── ...
```

**重要**: 只有包含 `label.png` 的樣本才會被處理。

## 輸出格式

### 文件命名

生成的拓樸文件命名格式: `{sample_id}_gt.pkl`

示例:
- `S1140-2_a/label.png` → `S1140-2_a_gt.pkl`
- `S1585-2_a/label.png` → `S1585-2_a_gt.pkl`

### 拓樸內容

每個 `.pkl` 文件包含一個 NetworkX Graph:
- **節點**: (y, x) 座標
- **邊**: 包含 `path-coordinates` 屬性,存儲完整的 skeleton 路徑點

### 輸出示例

```
================================================================================
提取摘要
================================================================================
總樣本數: 17
成功: 16
失敗: 1
成功率: 94.1%

拓樸統計:
  平均節點數: 357.8
  平均邊數: 291.6
  總節點數: 5724
  總邊數: 4665

失敗樣本:
  - S341-2_b: 拓樸提取失敗或圖為空

輸出目錄: output/topologies
================================================================================
```

## 使用提取的拓樸

### 1. 載入拓樸

```python
from pathlib import Path
from tools.compare_topologies import TopologyLoader

loader = TopologyLoader()
graph = loader.load(Path("output/topologies/S1585-2_a_gt.pkl"))

print(f"節點數: {graph.number_of_nodes()}")
print(f"邊數: {graph.number_of_edges()}")
```

### 2. 比對拓樸

```bash
# 比對兩個 GT 拓樸
uv run python tools/compare_topologies.py \
    --topology1 output/topologies/S1585-2_a_gt.pkl \
    --topology2 output/topologies/S1585-2_b_gt.pkl
```

輸出:
```
================================================================================
比對結果
================================================================================
拓樸 1: S1585-2_a_gt
  節點數: 322
  邊數: 248
  總點數: 5171

拓樸 2: S1585-2_b_gt
  節點數: 374
  邊數: 297
  總點數: 6245

平均 Hausdorff 距離: 36.4518 像素
================================================================================
```

### 3. 批量比對

如果你有預測拓樸和 GT 拓樸,可以批量比對:

```bash
# 假設你有:
# - output/predictions/S1585-2_a_pred.pkl
# - output/topologies/S1585-2_a_gt.pkl

uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir output/predictions \
    --gt-dir output/topologies \
    --output results.csv
```

## 工作流程示例

### 完整評測流程

```bash
# 步驟 1: 從數據集提取所有 GT 拓樸
uv run python tools/extract_dataset_topologies.py \
    --output-dir output/gt_topologies

# 步驟 2: 運行 Pipeline 生成預測拓樸 (需要修改 Pipeline 以保存拓樸)
# python run_pipeline.py --save-topology

# 步驟 3: 批量比對預測與 GT
uv run python tools/compare_topologies.py \
    --batch \
    --pred-dir output/predictions \
    --gt-dir output/gt_topologies \
    --output evaluation_results.csv

# 步驟 4: 分析結果
# python analyze_results.py evaluation_results.csv
```

## 技術細節

### 拓樸提取過程

1. **載入 label.png**: 讀取 GT 標註圖像
2. **連通分量檢測**: 使用 `skimage.measure.label` 識別所有連通分量
3. **骨架化**: 對每個分量進行 Zhang-Suen 骨架提取
4. **拓樸建構**: 使用 `skan` 庫建構圖結構
5. **合併**: 將所有分量的拓樸合併為單一圖
6. **儲存**: 以 Pickle 格式儲存完整圖結構

### 為什麼使用 Pickle 格式?

| 優點 | 說明 |
|------|------|
| **完整性** | 保留所有 NetworkX 圖屬性,包括複雜的邊路徑點 |
| **效率** | 讀寫速度快,文件大小適中 |
| **可靠性** | Python 原生格式,無轉換損失 |
| **兼容性** | 與 `compare_topologies.py` 工具完美兼容 |

### 邊路徑點說明

GT 拓樸的邊使用 `path-coordinates` 屬性存儲完整的 skeleton 路徑點:

```python
# 邊的格式
{
    "path-coordinates": [(y1, x1), (y2, x2), ..., (yn, xn)]
}
```

這些路徑點在平均 Hausdorff 距離計算中會被包含,提供更準確的相似度評估。

## 故障排除

### 問題: 某些樣本提取失敗

**原因**: label.png 可能為空或格式錯誤

**解決方案**:
1. 檢查 label.png 是否為有效的二值圖像
2. 使用詳細模式查看錯誤: `--verbose`
3. 手動檢查失敗樣本的 label.png

### 問題: 內存不足

**原因**: 同時處理大量大型圖像

**解決方案**:
- 分批處理數據集
- 增加系統可用內存
- 減少並行處理數量(當前是串行處理,不會有此問題)

### 問題: 輸出目錄權限錯誤

**原因**: 沒有寫入權限

**解決方案**:
```bash
# 檢查目錄權限
ls -la output/

# 創建目錄並設置權限
mkdir -p output/topologies
chmod 755 output/topologies
```

## 性能數據

基於測試數據集 (17 個樣本):

| 指標 | 數值 |
|------|------|
| 處理速度 | ~5-10 秒/樣本 |
| 成功率 | 94.1% (16/17) |
| 平均文件大小 | ~380 KB/樣本 |
| 總輸出大小 | ~6.1 MB (16 個文件) |
| 平均節點數 | 357.8 節點/樣本 |
| 平均邊數 | 291.6 邊/樣本 |

## 相關工具

- [tools/compare_topologies.py](../tools/compare_topologies.py) - 拓樸比對工具
- [tools/evaluate_dataset.py](../tools/evaluate_dataset.py) - 完整評測工具
- [docs/TOPOLOGY_COMPARISON.md](TOPOLOGY_COMPARISON.md) - 拓樸比對文檔
- [QUICK_REFERENCE.md](../QUICK_REFERENCE.md) - 快速參考

## 總結

`extract_dataset_topologies.py` 提供了一個**自動化、可靠、高效**的解決方案,用於:

- ✅ 批量提取數據集中的 GT 拓樸
- ✅ 統一拓樸格式以便後續比對
- ✅ 分離拓樸提取和圖像處理流程
- ✅ 加速評測和實驗迭代

推薦將提取的 GT 拓樸儲存在版本控制之外,作為評測基準。

---

**創建日期**: 2026-02-09
**版本**: 1.0
