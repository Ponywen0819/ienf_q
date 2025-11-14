# IENF Pipeline 使用指南

## 快速開始

### 最簡單的使用方式

```bash
python run_pipeline.py \
    --image data/original_image.png \
    --annotation data/manual_annotation.png \
    --output output/reconstruction
```

就這麼簡單！只需要兩張圖片：
- **原始影像** - RGB 顯微鏡影像（或灰階）
- **手動標註** - 標記的神經纖維（二值化，神經纖維為白色）

Pipeline 會自動執行完整的 6 步驟流程，從前處理到最終重建！

## 完整說明

請參考：[USAGE_GUIDE.md](USAGE_GUIDE.md)（完整版文檔）
