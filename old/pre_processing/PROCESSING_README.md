# 圖片處理工具

這個工具可以自動處理圖片，執行以下操作：
1. 抽取原始圖片的綠色頻道
2. 找到遮罩的下邊界作為中線
3. 以下邊界為中心膨脹指定像素數（預設20px）
4. 將新遮罩應用到綠色頻道上
5. 儲存處理結果

## 快速開始

### 方法1：自動處理（推薦）

```bash
python3 run_process.py
```

這個腳本會：
- 自動安裝所需依賴（numpy, Pillow, opencv-python）
- 檢查目錄結構
- 批量處理所有圖片

### 方法2：手動執行

1. 安裝依賴：
```bash
pip install numpy Pillow opencv-python
```

2. 批量處理（使用預設路徑）：
```bash
python3 process_images.py
```

3. 指定路徑批量處理：
```bash
python3 process_images.py --original ./Original --mask ./Mask --output ./Processed --expansion 20
```

4. 處理單個檔案：
```bash
python3 process_images.py --single --original original.tif --mask mask.tif --output result.tif --expansion 20
```

## 參數說明

- `--original, -o`: 原始圖片目錄或檔案路徑
- `--mask, -m`: 遮罩圖片目錄或檔案路徑  
- `--output, -out`: 輸出目錄或檔案路徑
- `--expansion, -e`: 膨脹像素數量（預設：20）
- `--single, -s`: 單檔案模式

## 輸出檔案

每個處理的圖片會產生兩個輸出檔案：
- `processed_原檔名.tif`: 處理後的綠色頻道圖片
- `processed_原檔名_mask.tif`: 膨脹後的遮罩（供檢視用）

## 處理流程詳解

### 1. 綠色頻道抽取
從RGB圖片中提取綠色頻道（G通道），如果輸入已經是灰階圖片則直接使用。

### 2. 遮罩邊界檢測
- 將遮罩轉為二值圖片
- 使用OpenCV找到輪廓
- 識別最大輪廓作為主要遮罩區域
- 找到輪廓的最下方點作為邊界

### 3. 遮罩膨脹
- 以下邊界為中心線
- 向上下各擴展指定的像素數量
- 創建新的矩形遮罩區域

### 4. 遮罩應用
- 將膨脹後的遮罩應用到綠色頻道上
- 遮罩外的區域設為0（黑色）
- 遮罩內的區域保持原始像素值

## 目錄結構

```
ienf_q/
├── Original/          # 原始圖片
├── Mask/             # 遮罩圖片
├── Processed/        # 處理結果（自動創建）
├── process_images.py # 主處理腳本
├── run_process.py    # 啟動腳本
└── PROCESSING_README.md
```

## 故障排除

### 依賴安裝問題
```bash
# 更新pip
python3 -m pip install --upgrade pip

# 手動安裝依賴
pip3 install numpy Pillow opencv-python
```

### 圖片格式問題
- 確保圖片格式為TIFF（.tif）
- 檢查原始圖片和遮罩圖片檔名是否一致

### 記憶體不足
對於大尺寸圖片，可能需要：
- 增加系統記憶體
- 分批處理較少的圖片

### 遮罩檢測失敗
如果出現"找不到遮罩輪廓"錯誤：
- 檢查遮罩是否為有效的二值圖片
- 確認遮罩中有明確的白色區域

## 範例

處理單個檔案：
```bash
python3 process_images.py --single \
  --original Original/S1037-2_a.tif \
  --mask Mask/S1037-2_a.tif \
  --output result.tif \
  --expansion 25
```

批量處理並自定義膨脹：
```bash
python3 process_images.py \
  --original ./Original \
  --mask ./Mask \
  --output ./MyResults \
  --expansion 30
```