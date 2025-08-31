#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
圖片處理啟動腳本
自動安裝依賴並執行圖片處理
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """安裝必要的依賴"""
    dependencies = [
        'numpy',
        'Pillow', 
        'opencv-python'
    ]
    
    print("檢查並安裝依賴...")
    for dep in dependencies:
        try:
            if dep == 'Pillow':
                import PIL
                print(f"✓ {dep} 已安裝")
            elif dep == 'numpy':
                import numpy
                print(f"✓ {dep} 已安裝")
            elif dep == 'opencv-python':
                import cv2
                print(f"✓ {dep} 已安裝")
        except ImportError:
            print(f"正在安裝 {dep}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✓ {dep} 安裝完成")
            except subprocess.CalledProcessError:
                print(f"✗ {dep} 安裝失敗")
                return False
    
    return True

def check_directories():
    """檢查必要的目錄"""
    base_dir = Path(__file__).parent
    required_dirs = {
        'Original': base_dir / 'Original',
        'Mask': base_dir / 'Mask'
    }
    
    print("\n檢查目錄結構...")
    for name, path in required_dirs.items():
        if path.exists():
            tif_count = len(list(path.glob('*.tif')))
            print(f"✓ {name}: {tif_count} 個 .tif 檔案")
        else:
            print(f"✗ {name}: 目錄不存在")
            return False
    
    return True

def main():
    print("=== 圖片處理工具啟動程式 ===\n")
    
    # 1. 安裝依賴
    if not install_dependencies():
        print("\n錯誤: 依賴安裝失敗")
        return
    
    # 2. 檢查目錄
    if not check_directories():
        print("\n錯誤: 請確保 Original 和 Mask 目錄存在且包含 .tif 檔案")
        return
    
    # 3. 執行圖片處理
    print("\n開始處理圖片...")
    try:
        from process_images import process_all_images
        
        current_dir = Path(__file__).parent
        process_all_images(
            original_dir=current_dir / "Original",
            mask_dir=current_dir / "Mask",
            output_dir=current_dir / "Processed",
            expansion_pixels=40
        )
        
        print("\n處理完成! 結果儲存在 'Processed' 資料夾中")
        
    except Exception as e:
        print(f"\n處理過程中發生錯誤: {e}")
        print("請檢查圖片格式和路徑是否正確")

if __name__ == "__main__":
    main()