# 演算法比較：階層式片段連接 vs 純 MST

本文件詳細說明兩種神經纖維重建演算法的每個環節，並比較其設計差異。

---

## 概覽

兩種演算法的目標相同：從顯微影像上的稀疏手工標註，重建完整的神經纖維網路。差異在於**如何建立斷開的標註片段之間的連接**。

| 面向 | 純 MST | 階層式片段連接 |
|---|---|---|
| 連接策略 | 單次全域 MST | 兩階段：端點延伸 + MST |
| 方向性約束 | 無 | 兩階段皆有角度過濾 |
| 成本模型 | 僅影像強度 | 強度 + 角度懲罰 + 距離懲罰 |
| MST 執行次數 | 1 次 | 2 次（階段1後、階段2後各一次） |
| 參數數量 | 6 個 | 14 個 |

---

## 共用流程

兩種演算法共用相同的前處理與後處理階段：

### 1. 前處理

**目的**：擷取感興趣區域（ROI）並增強神經纖維的可見性。

| 步驟 | 操作 | 目的 |
|---|---|---|
| 1a | 綠色通道擷取 | 神經組織在綠色通道有最強信號 |
| 1b | 表皮遮罩垂直膨脹 (`dilate_epidermis_vertically`) | 向上下擴展遮罩以定義 ROI 範圍 |
| 1c | 背景減除（形態學開運算） | 移除不均勻照明與大尺度強度變化 |
| 1d | ROI 遮罩運算 (`bitwise_and`) | 將處理範圍限制在膨脹後的表皮區域 |
| 1e | 對標註做形態學開運算 | 移除手工標註中的小雜訊點 |
| 1f | 對標註做形態學閉運算 | 填補標註片段內的小孔洞 |

### 2. 骨架拓撲構建

**目的**：將二值化的標註區塊轉換為沿骨架中心線連接的種子點圖。

兩種演算法皆使用 `TopologyBuilder.build_seed_graph()`，其流程為：
1. 對二值標註進行骨架化（Zhang-Suen 細化演算法）
2. 使用 `skan` 函式庫建立拓撲圖
3. 沿骨架以 `segment_length` 間距放置種子點
4. 曲率感知的間距調整（彎曲/分支處放置更多種子）

產出的**種子圖**包含：
- **節點**：種子點的 (y, x) 像素座標
- **邊**：沿骨架的連接，帶有 `path` 屬性儲存像素級路徑
- **邊權重**：設為 `1e-5`（近零），以強烈偏好保留現有標註結構

### 3. 後處理：移除小連通分量

**目的**：過濾掉總路徑長度低於 `min_component_length` 的連通分量，去除重建雜訊。

### 4. 交叉分析

**目的**：計算重建的神經纖維穿越表皮邊界的數量（臨床上的 IENF 密度指標）。

兩種演算法執行相同步驟：
1. **分段偵測** (`SegmentDetector`)：識別圖中的拓撲分段
2. **短殘段修剪**：移除長度 < 5 像素的懸掛分段
3. **重新偵測**：修剪後重新執行分段偵測
4. **區域標記** (`RegionLabeler`)：根據遮罩將邊分類為表皮/真皮
5. **穿越計數** (`CrossingCounter`)：計算有效的表皮穿越數

---

## 純 MST 演算法

**原始碼**：[`src/neural_reconstruction/algorithms/pure_mst/linker.py`](../src/neural_reconstruction/algorithms/pure_mst/linker.py)

### 設計理念

直觀的方法：透過 A* 路徑搜尋找出所有可能的片段間連接，再由 MST 選出全域最優子集。

### 流程圖

```
前處理
  │
  v
種子圖構建 (TopologyBuilder)
  │
  v
連通分量標記 (skimage.label, 8-connectivity)
  │
  v
成本地圖構建
  │
  v
A* 路徑搜尋（搜尋半徑內所有跨元件種子對）
  │
  v
將候選邊加入種子圖
  │
  v
全域 MST (Kruskal 演算法)
  │
  v
移除小連通分量
  │
  v
交叉分析
```

### 各步驟詳解

#### 成本地圖構建

```
cost_map = ((255 - intensity) / 255) ^ intensity_weight
```

- 暗像素（低強度，遠離神經組織）→ 高成本
- 亮像素（神經組織）→ 低成本
- `intensity_weight`（預設：2.0）控制非線性程度：值越大，偏離纖維的路徑懲罰越重

#### 路徑搜尋

使用 `PathFinder.find_paths_from_seeds()`：
1. 從所有種子點建立 KDTree
2. 對每個種子找出 `search_radius` 內的鄰居
3. 對跨元件的配對執行 A*（同元件配對透過 `label_img` 排除）
4. 回傳查找字典：`(source, target) -> (path, cost)`

所有發現的跨元件路徑都作為候選邊加入圖中。

#### MST 萃取

單次呼叫 `nx.minimum_spanning_tree()` 選出連接所有可達元件的最小成本邊子集。

### 參數表

| 參數 | 預設值 | 用途 |
|---|---|---|
| `offset_px` | 50 | 表皮遮罩垂直膨脹量（像素） |
| `rolling_ball_radius` | 50 | 背景減除核心半徑 |
| `opening_kernel_size` | 3 | 形態學雜訊移除核心大小 |
| `segment_length` | 5.0 | 沿骨架的種子間距（像素） |
| `search_radius` | 50.0 | 跨元件路徑搜尋最大距離 |
| `intensity_weight` | 2.0 | 成本地圖非線性指數 |
| `min_component_length` | 10.0 | 保留的最小連通分量長度 |

### 優缺點

**優點**：
- 簡單，需要調整的參數少
- 在給定成本函數下，MST 結果是全域最優的
- 無方向偏好 — 純粹基於路徑成本連接片段

**缺點**：
- 缺乏方向感知：可能在連接點產生不自然的急轉彎
- 對所有連接使用同一套門檻：無法區分高信心與推測性連結
- 所有候選邊在 MST 中平等競爭，不考慮幾何合理性

---

## 階層式片段連接演算法

**原始碼**：[`src/neural_reconstruction/algorithms/fragment_linking/linker.py`](../src/neural_reconstruction/algorithms/fragment_linking/linker.py)

### 設計理念

利用神經纖維幾何的生物學先驗知識：纖維傾向沿大致相同的方向延續。透過將連接分為兩個嚴格程度不同的階段，先鎖定高信心連接，再填補剩餘間隙。

### 流程圖

```
前處理
  │
  v
種子圖構建 (TopologyBuilder)
  │
  v
成本地圖 & 種子地圖構建
  │
  v
連通分量標記 (skimage.label, 8-connectivity)
  │
  v
A* 路徑搜尋（搜尋半徑內所有跨元件配對）
  │
  v
階段1：端點延伸（嚴格約束）
  │    - 僅處理端點（degree == 1）
  │    - 嚴格角度約束（75°）
  │    - 方向去重
  │    - 每個端點僅選最佳候選
  │    - 被接受的邊享有權重折扣
  │
  v
第1次 MST（鎖定階段1連接）
  │
  v
階段2：MST 候選邊生成（寬鬆約束）
  │    - 更大搜尋半徑（50 px）
  │    - 更寬鬆的角度約束（90°）
  │    - 角度 + 距離懲罰
  │    - 成本門檻過濾
  │    - 每個端點可生成多條候選
  │    - 孤立節點特殊處理
  │
  v
第2次 MST（最終全域優化）
  │
  v
移除小連通分量
  │
  v
交叉分析
```

### 階段1：端點延伸

**原始碼**：[`endpoint_extension.py`](../src/neural_reconstruction/algorithms/fragment_linking/endpoint_extension.py)

**目的**：沿端點的自然延伸方向，以嚴格的幾何約束進行延伸。此階段針對最明顯、最高信心的連接。

**每個端點的處理流程**：

1. **計算延伸方向**：從端點的唯一鄰居指向端點的向量 (`endpoint - neighbor`)，代表纖維「指向」的方向。

2. **KDTree 查詢**：找出 `search_radius_endpoint_extension`（預設 20 px）內的所有種子點。

3. **候選按距離排序**（由近到遠）。

4. **逐一評估候選**，依序套用過濾條件：
   - 跳過自身、鄰居、已連接的節點
   - **方向去重**：若候選方向與已通過篩選的候選方向夾角 < `direction_threshold`（預設 5°），則跳過。避免同一方向上的多個候選互相競爭。
   - **角度檢查**：計算延伸向量與候選向量的夾角，若 > `max_angle_endpoint_extension`（預設 75°）則拒絕。
   - **帶懲罰的成本**：`final_cost = base_cost * (1 + penalty)`，其中 `penalty = max_angle_penalty * (angle / max_angle)`。

5. **選取最佳**：每個端點僅保留成本最低的一個候選。

6. **權重折扣**：被接受的階段1邊，其權重乘以 `endpoint_extension_weight_discount`（預設 0.5），使其在後續 MST 中被優先保留。

**階段1結束後**：執行第1次 MST，將高信心延伸鎖定，再進入階段2。

### 階段2：MST 候選邊生成

**原始碼**：[`mst_candidates.py`](../src/neural_reconstruction/algorithms/fragment_linking/mst_candidates.py)

**目的**：以較寬鬆的約束生成更多候選邊，填補剩餘的間隙。與階段1不同，此階段每個端點可產生**多條候選邊**，交由 MST 選擇最佳方案。

**端點處理（degree == 1）**：

1. 計算延伸方向（與階段1相同）
2. KDTree 查詢 `search_radius_mst`（預設 50 px）內的候選
3. 對每個候選：
   - 角度檢查：若 > `max_angle_mst`（預設 90°）則拒絕
   - 計算綜合成本：
     ```
     distance_penalty = distance_weight * (distance / search_radius)
     angle_penalty = angle_penalty_weight * (angle / max_angle)
     final_cost = base_cost * (1 + angle_penalty) * (1 + distance_penalty)
     ```
   - 接受條件：`final_cost <= max_cost_threshold * path_length`

**孤立節點處理（degree == 0）**：

- 無方向資訊，故不做角度過濾
- 搜尋半徑減半 (`search_radius / 2`) 以保守處理
- 僅套用距離懲罰

### 最終 MST

將階段2所有候選邊加入圖中（已包含第1次 MST 鎖定的階段1邊），執行第2次 MST 萃取最終最優網路。

### 參數表

| 參數 | 預設值 | 階段 | 用途 |
|---|---|---|---|
| `offset_px` | 50 | 前處理 | 表皮遮罩垂直膨脹量 |
| `rolling_ball_radius` | 50 | 前處理 | 背景減除核心半徑 |
| `opening_kernel_size` | 3 | 前處理 | 形態學雜訊移除核心大小 |
| `segment_length` | 3.0 | 拓撲 | 沿骨架的種子間距 |
| `intensity_power` | 2.0 | 成本地圖 | 非線性指數 |
| `search_radius_endpoint_extension` | 20.0 | 階段1 | 最大搜尋距離 |
| `max_angle_endpoint_extension` | 75.0 | 階段1 | 最大允許角度（度） |
| `angle_penalty_endpoint_extension` | 0.5 | 階段1 | 角度懲罰權重 |
| `direction_threshold_endpoint_extension` | 5.0 | 階段1 | 方向去重門檻（度） |
| `search_radius_mst` | 50.0 | 階段2 | 最大搜尋距離 |
| `max_angle_mst` | 90.0 | 階段2 | 最大允許角度（度） |
| `angle_penalty_mst` | 0.5 | 階段2 | 角度懲罰權重 |
| `distance_weight_mst` | 0.2 | 階段2 | 距離懲罰權重 |
| `max_cost_threshold_mst` | 0.75 | 階段2 | 成本接受門檻 |
| `endpoint_extension_weight_discount` | 0.5 | MST | 階段1邊的權重折扣因子 |
| `min_component_length` | 10.0 | 後處理 | 保留的最小連通分量長度 |

### 優缺點

**優點**：
- 方向感知使連接更符合解剖學合理性
- 兩階段機制：高信心連結先鎖定，推測性連結後處理
- 權重折扣機制確保階段1連接被保留
- 方向去重避免冗餘候選
- 孤立節點有專門的保守處理策略

**缺點**：
- 參數多，調參成本較高
- 階段1的嚴格約束可能遺漏不符合預期方向但實際有效的連接
- 兩次 MST 增加計算開銷
- 基於角度的過濾假設纖維平滑延續，在分支點可能不成立

---

## 核心差異總結

### 1. 連接策略

| | 純 MST | 階層式 |
|---|---|---|
| 方法 | 全配對搜尋 + 單次 MST | 兩階段：嚴格延伸 + 寬鬆 MST |
| 方向過濾 | 無 | 兩階段皆有角度約束 |
| 每端點候選數 | 半徑內全部 | 階段1：僅最佳；階段2：多個 |

### 2. 成本函數

**純 MST**：
```
cost = A* 路徑成本（僅基於影像強度）
```

**階層式 - 階段1**：
```
cost = base_cost * (1 + angle_penalty) * weight_discount
```

**階層式 - 階段2**：
```
cost = base_cost * (1 + angle_penalty) * (1 + distance_penalty)
```

### 3. 種子密度

- 純 MST：`segment_length = 5.0`（較稀疏）
- 階層式：`segment_length = 3.0`（較密集，以利更精確的方向估計）

### 4. 搜尋半徑

- 純 MST：所有連接統一使用 50 px
- 階層式：階段1 用 20 px（保守），階段2 用 50 px（寬鬆），孤立節點減半

### 5. MST 執行次數

- 純 MST：1 次
- 階層式：2 次（階段1後鎖定延伸結果，階段2後做最終優化）

---

## 使用建議

| 場景 | 建議演算法 |
|---|---|
| 密集標註、間隙小 | 純 MST（簡單且足夠） |
| 稀疏標註、間隙大 | 階層式（方向引導有助於填補） |
| 希望最少參數調整 | 純 MST |
| 重視連接的解剖學合理性 | 階層式 |
| 快速原型 / 基線比較 | 純 MST |
| 正式 IENF 量化 | 階層式（更穩健） |
