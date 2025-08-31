#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
測試圖片處理功能
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path

def test_single_file():
    """測試單個檔案處理"""
    print("=== 測試單個檔案處理 ===\n")
    
    # 找到第一個可用的檔案進行測試
    base_dir = Path(__file__).parent
    original_dir = base_dir / "Original"
    mask_dir = base_dir / "Mask"
    
    if not original_dir.exists() or not mask_dir.exists():
        print("錯誤: Original 或 Mask 目錄不存在")
        return False
    
    # 獲取第一個檔案
    original_files = list(original_dir.glob('*.tif'))
    if not original_files:
        print("錯誤: Original 目錄中沒有 .tif 檔案")
        return False
    
    test_file = original_files[0]
    mask_file = mask_dir / test_file.name
    
    if not mask_file.exists():
        print(f"錯誤: 找不到對應的遮罩檔案: {mask_file}")
        return False
    
    print(f"測試檔案: {test_file.name}")
    
    try:
        from process_images import process_single_image
        
        output_dir = base_dir / "TestOutput"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"test_{test_file.name}"
        
        success = process_single_image(
            str(test_file), 
            str(mask_file), 
            str(output_file), 
            expansion_pixels=40
        )
        
        if success:
            print("\n✓ 測試成功！")
            return True
        else:
            print("\n✗ 測試失敗！")
            return False
            
    except Exception as e:
        print(f"\n✗ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("圖片處理功能測試\n")
    
    # 檢查依賴
    try:
        import numpy
        import cv2
        from PIL import Image
        print("✓ 所有依賴都已安裝")
    except ImportError as e:
        print(f"✗ 缺少依賴: {e}")
        return
    
    # 測試單個檔案
    test_single_file()

if __name__ == "__main__":
    main()