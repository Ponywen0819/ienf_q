# 階段四：MST 神經纖維重建 - 實作計劃

## 概述

本階段實現 README 第 307-355 行描述的「最小生成樹重建」功能，從階段三建構的連接圖中提取最優神經網絡拓撲。

**核心目標**：

- 從候選連接圖構建約束 MST 森林
- 允許多個獨立連通分量（不強制單一樹）
- 驗證拓撲合理性（交叉邊、銳角分支）
- 評估路徑質量

---

## 模組架構

```
src/04_nueral_reconstruction/
├── PROCESS.md                  # 本文件
├── __init__.py                 # 模組初始化
├── mst_builder.py              # MST/森林構建器
├── topology_validator.py       # 拓撲驗證（交叉邊、銳角）
├── path_quality_checker.py     # 路徑質量評估
├── reconstruction_runner.py    # 主流程協調器
├── visualization.py            # MST 專用視覺化
└── run_reconstruction.py       # CLI 執行腳本
```

---

## 各模組功能說明

### 1. `mst_builder.py` - MST/森林構建器

**核心功能**：約束 MST（README 312-330 行）

**類別**：`MSTBuilder`

**方法**：

- `build_constrained_mst_forest(G: nx.Graph) -> nx.Graph`

  - 構建約束 MST 森林
  - 過濾高成本邊（cost >= max_edge_cost）
  - 對每個連通分量獨立構建 MST
  - 返回可能包含多個連通分量的森林

- `get_forest_statistics(forest: nx.Graph) -> dict`
  - 統計森林資訊
  - 連通分量數、總節點數、總邊數
  - 每個分量的詳細資訊

**實現細節**：

- 使用 `nx.connected_components()` 找出所有連通分量
- 對每個分量使用 `nx.minimum_spanning_tree()`
- 合併所有分量的 MST 形成森林

---

### 2. `topology_validator.py` - 拓撲驗證

**核心功能**：檢測拓撲問題（README 334-341 行）

**類別**：`TopologyValidator`

**方法**：

- `detect_crossing_edges(forest: nx.Graph) -> List[Tuple[tuple, tuple]]`

  - 檢測交叉邊
  - 使用線段相交算法
  - 返回交叉的邊對列表

- `detect_sharp_branches(forest: nx.Graph, min_angle: float = 30) -> List[dict]`

  - 檢測銳角分支
  - 找出度數 >= 3 的節點（分支點）
  - 計算相鄰邊之間的夾角
  - 標記 < min_angle 的分支

- `validate_topology(forest: nx.Graph) -> dict`
  - 完整拓撲驗證
  - 返回所有問題列表和驗證狀態

**線段相交算法**：

```python
def segments_intersect(p1, p2, p3, p4):
    """
    檢查線段 (p1-p2) 和 (p3-p4) 是否相交
    使用向量叉積方法
    """
```

**角度計算**：

```python
def calculate_angle(v1, v2):
    """
    計算兩個向量的夾角（度數）
    """
```

---

### 3. `path_quality_checker.py` - 路徑質量評估

**核心功能**：評估路徑質量（README 345-348 行）

**類別**：`PathQualityChecker`

**方法**：

- `assess_edge_quality(path: List[Tuple[int, int]]) -> dict`

  - 評估單條邊的路徑質量
  - 計算路徑上綠色強度統計
  - 返回質量分數和低質量標記

- `assess_forest_quality(forest: nx.Graph) -> dict`
  - 評估整個森林的質量
  - 統計低質量邊的比例
  - 計算平均路徑質量

**質量指標**：

- mean_intensity: 路徑平均綠色強度
- min_intensity: 路徑最小強度
- std_intensity: 強度標準差
- quality_score: 0-1 的質量分數
- is_low_quality: bool（強度 < 閾值）

---

### 4. `reconstruction_runner.py` - 主流程協調器

**核心功能**：整合所有模組的主流程

**類別**：`ReconstructionRunner`

**配置參數**：

```python
config = {
    'max_edge_cost': 150,        # MST 邊成本閾值
    'min_branch_angle': 30,      # 銳角分支閾值（度）
    'min_quality_threshold': 80  # 路徑質量閾值
}
```

**主方法**：

```python
def run(
    graph_path: str,          # output/network/network.graphml
    seeds_path: str,          # output/seeds/seeds.json
    green_channel_path: str,  # 綠色通道影像
    output_dir: str
) -> dict
```

**執行流程**：

1. 載入圖、種子、影像
2. 構建 MST 森林
3. 拓撲驗證
4. 路徑質量評估
5. 保存結果（GraphML, JSON）
6. 生成視覺化
7. 生成統計報告

---

### 5. `visualization.py` - MST 專用視覺化

**核心功能**：MST 森林視覺化

**類別**：`MSTVisualizer`

**方法**：

- `visualize_mst_forest()` - 完整森林視覺化

  - 不同連通分量用不同顏色
  - 標記分支點（度數 >= 3）
  - 邊的粗細反映權重

- `visualize_validation_report()` - 驗證報告視覺化

  - 高亮交叉邊（紅色）
  - 標記銳角分支（黃色警告）
  - 標示低質量邊（虛線）

- `visualize_component_breakdown()` - 分量分解圖

  - 選擇最大的 N 個分量
  - 每個分量獨立顯示
  - 標註統計資訊

- `visualize_quality_heatmap()` - 路徑質量熱力圖
  - 邊的顏色反映路徑質量
  - colorbar 顯示質量分數

---

### 6. `run_reconstruction.py` - CLI 執行腳本

**命令行參數**：

```bash
uv run  src/04_nueral_reconstruction/run_reconstruction.py \
    --graph output/network/network.graphml \
    --seeds output/seeds/seeds.json \
    --image data/Original/S163-2_a_corrected_normalized.tif \
    --output output/reconstruction \
    --max-cost 150
```

**參數說明**：

- `--graph`: 網路圖檔案路徑（來自階段三）
- `--seeds`: 種子檔案路徑
- `--image`: 綠色通道影像路徑
- `--output`: 輸出目錄
- `--max-cost`: 最大邊成本（可選，預設 150）

---

## 輸出檔案規劃

**output/reconstruction/**

```
├── mst_forest.graphml              # MST 森林圖結構
├── mst_forest_with_paths.json     # 包含路徑的森林資料
├── validation_report.json         # 驗證報告
├── quality_assessment.json        # 質量評估報告
├── reconstruction_summary.txt     # 統計摘要
│
├── mst_forest_full.png            # 完整森林視覺化
├── mst_forest_zoomed.png          # 局部放大圖
├── validation_issues.png          # 問題標記圖
├── component_breakdown.png        # 分量分解圖
└── quality_heatmap.png            # 路徑質量熱力圖
```

---

## 實現順序

### 階段一：核心 MST 功能 ✓

**目標**：能夠從 network.graphml 生成 MST 森林

**步驟**：

1. ✓ 創建 PROCESS.md
2. [ ] 實作 `__init__.py`
3. [ ] 實作 `mst_builder.py`
4. [ ] 實作 `reconstruction_runner.py`（基本版）
5. [ ] 實作 `run_reconstruction.py`
6. [ ] 測試：執行並生成 MST 森林

**驗收標準**：

- 能載入 network.graphml
- 能構建 MST 森林並保存
- 輸出 mst_forest.graphml 和統計摘要

---

### 階段二：驗證與評估

**目標**：添加拓撲驗證和質量評估

**步驟**： 7. [ ] 實作 `topology_validator.py`

- 線段相交檢測
- 銳角分支檢測
- 完整驗證流程

8. [ ] 實作 `path_quality_checker.py`

   - 單邊質量評估
   - 森林整體質量評估

9. [ ] 整合到 `reconstruction_runner.py`
   - 調用驗證器
   - 調用質量檢查器
   - 保存驗證和質量報告

**驗收標準**：

- 輸出 validation_report.json
- 輸出 quality_assessment.json
- 統計摘要包含問題數量

---

### 階段三：視覺化

**目標**：完整的視覺化輸出

**步驟**： 10. [ ] 實作 `visualization.py` - MST 森林視覺化 - 驗證報告視覺化 - 分量分解視覺化 - 質量熱力圖

11. [ ] 整合到 `reconstruction_runner.py`

    - 生成所有視覺化圖片
    - 處理大型網絡（降採樣、分頁）

12. [ ] 完整測試
    - 端到端測試
    - 檢視所有輸出
    - 調整參數優化結果

**驗收標準**：

- 所有規劃的視覺化圖片生成
- 圖片清晰可讀，標註完整
- 能處理不同大小的網絡

---

## 關鍵設計決策

### 1. 為什麼允許森林而非單一樹？

- README 324-329 行明確指出：神經可能有多個分離網絡
- 強制單一樹會產生不合理的長距離連接
- 森林結構更符合真實神經分布

### 2. MST 的輸入是什麼？

- **輸入**：階段三建構的完整圖（已過濾高成本邊）
- **輸出**：精簡後的 MST 森林（每個連通分量的最小生成樹）

### 3. 與階段三的關係？

- **階段三**：建構候選連接圖（所有可能的連接）
- **階段四**：從候選中選擇最優子集（MST）
- MST 是原圖的子圖（節點相同，邊數減少）

### 4. 如何處理路徑資訊？

- 從 edges_with_paths.json 讀取完整路徑
- MST 選擇的邊保留其路徑資訊
- 路徑用於質量評估和視覺化

---

## 參數調整指南

### MST 構建參數

- `max_edge_cost`: 150（預設）
  - 降低：更嚴格，可能產生更多分量
  - 提高：更寬鬆，可能連接不相關的神經

### 驗證參數

- `min_branch_angle`: 30 度（預設）
  - 降低：更寬容銳角分支
  - 提高：更嚴格檢測銳角

### 質量評估參數

- `min_quality_threshold`: 80（預設，0-255 綠色強度）
  - 降低：更寬鬆的質量要求
  - 提高：更嚴格的質量要求

---

## 測試策略

### 單元測試

- MST 構建正確性（子圖、無環）
- 線段相交算法正確性
- 角度計算正確性
- 路徑質量計算正確性

### 整合測試

- 端到端流程（載入 -> MST -> 驗證 -> 保存）
- 使用現有的 network.graphml 測試
- 檢查輸出檔案完整性

### 視覺驗證

- 比較 MST 前後的網絡結構
- 確認移除的邊是冗余的
- 確認保留的拓撲合理

---

## 預期效果（README 461-478 行）

### 連通性

- 同一神經的區塊正確連接
- 不同神經不錯誤連接
- 目標準確率：> 90%

### 幾何保真度

- 保留彎折和分支結構
- 分支點正確識別
- 路徑平滑且自然

### 處理效能

- MST 計算：< 5 秒（圖通常 < 1000 節點）
- 完整流程：< 30 秒

### 輸出品質

- 森林結構合理（自動識別獨立網絡）
- 驗證報告完整（標記所有問題）
- 視覺化清晰可讀

---

## 後續改進方向

### 短期改進

- 交互式編輯工具（手動調整連接）
- 參數自動調優（基於驗證結果）
- 更智能的質量評估（考慮路徑連續性）

### 中期改進

- 批次處理多張影像
- 統計分析工具
- 導出標準格式（SWC, NeuroML）

### 長期改進

- 3D 重建（多切片整合）
- 機器學習輔助拓撲選擇
- GPU 加速大規模網絡處理

---

## 實作進度追蹤

### 階段一：核心 MST 功能（已完成 ✓）

- [x] 創建 PROCESS.md
- [x] **init**.py
- [x] mst_builder.py
- [x] reconstruction_runner.py（基本版）
- [x] run_reconstruction.py
- [ ] 測試基本功能

### 階段二：驗證與評估（待實作）

- [ ] topology_validator.py
- [ ] path_quality_checker.py
- [ ] 整合驗證和評估到 reconstruction_runner.py

### 階段三：視覺化（待實作）

- [ ] visualization.py
- [ ] 整合視覺化到 reconstruction_runner.py
- [ ] 完整整合與測試

**最後更新**: 2025-01-02（階段一完成）
