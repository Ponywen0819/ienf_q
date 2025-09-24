#!/bin/bash
# IENF 多階段訓練啟動腳本

# 設置環境變數
export CUDA_VISIBLE_DEVICES=0  # 指定GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 配置文件路徑
CONFIG_FILE="config.yaml"

# 檢查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "錯誤: 找不到配置文件 $CONFIG_FILE"
    echo "請確保配置文件存在並且路徑正確"
    exit 1
fi

# 創建檢查點目錄
mkdir -p checkpoints

# 選擇訓練模式
echo "IENF 多階段訓練系統"
echo "==================="
echo "1. 執行完整三階段訓練"
echo "2. 只執行基礎階段"
echo "3. 只執行連接階段"
echo "4. 只執行路徑階段"
echo "5. 自定義階段組合"
echo ""

read -p "請選擇訓練模式 (1-5): " choice

case $choice in
    1)
        echo "開始執行完整三階段訓練..."
        python training_pipeline.py --config $CONFIG_FILE --stage all
        ;;
    2)
        echo "開始執行基礎階段訓練..."
        python training_pipeline.py --config $CONFIG_FILE --stage basic
        ;;
    3)
        echo "開始執行連接階段訓練..."
        python training_pipeline.py --config $CONFIG_FILE --stage connectivity
        ;;
    4)
        echo "開始執行路徑階段訓練..."
        python training_pipeline.py --config $CONFIG_FILE --stage path
        ;;
    5)
        echo "可用階段: basic, connectivity, path"
        read -p "請輸入要執行的階段 (例如: basic): " stage
        echo "開始執行 $stage 階段訓練..."
        python training_pipeline.py --config $CONFIG_FILE --stage $stage
        ;;
    *)
        echo "無效選擇，退出"
        exit 1
        ;;
esac

echo ""
echo "訓練完成！"
echo "檢查點保存在: checkpoints/"
echo "Wandb 記錄可在網頁界面查看"