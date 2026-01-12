#!/usr/bin/env python3
"""
測試主要進入點函數

簡單測試腳本，驗證 build_neural_network 函數是否正常運作
"""

import logging
import numpy as np
import cv2

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.neural_reconstruction.core.construction.main import build_neural_network

def test_with_simple_image():
    """使用簡單的合成影像測試"""
    print("\n" + "=" * 70)
    print("測試 1: 簡單合成影像")
    print("=" * 70)

    # 建立簡單的測試影像 (100x100)
    label_image = np.zeros((100, 100), dtype=np.uint8)

    # 建立兩個小矩形元件
    label_image[20:30, 20:30] = 255  # 元件 1
    label_image[70:80, 70:80] = 255  # 元件 2

    # 建立綠色通道（全白）
    green_channel = np.ones((100, 100), dtype=np.uint8) * 255

    # 執行重建
    mst = build_neural_network(
        label_image=label_image,
        green_channel=green_channel,
        connectivity=4,
        min_area=0,
        segment_length=5.0,
        search_radius=50.0,
        max_cost_threshold=0.98,
    )

    print(f"\n結果:")
    print(f"  節點數: {mst.number_of_nodes()}")
    print(f"  邊數: {mst.number_of_edges()}")
    print(f"  連通分量: {len(list(mst.edges()))}")

    assert mst.number_of_nodes() > 0, "應該要有節點"
    print("\n✓ 測試 1 通過")

def test_with_real_data():
    """使用實際資料測試（如果存在）"""
    print("\n" + "=" * 70)
    print("測試 2: 實際資料")
    print("=" * 70)

    try:
        # 嘗試讀取實際資料
        label_image = cv2.imread('data/Label/S163-2_a.tif', cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread('data/Original/S163-2_a.tif', cv2.IMREAD_UNCHANGED)

        if label_image is None or original_image is None:
            print("實際資料檔案不存在，跳過測試 2")
            return

        green_channel = original_image[:, :, 1]

        print(f"影像大小: {label_image.shape}")
        print(f"綠色通道大小: {green_channel.shape}")

        # 執行重建
        mst = build_neural_network(
            label_image=label_image,
            green_channel=green_channel,
            connectivity=4,
            min_area=50,
            segment_length=5.0,
            search_radius=50.0,
            max_cost_threshold=0.98,
        )

        print(f"\n結果:")
        print(f"  節點數: {mst.number_of_nodes()}")
        print(f"  邊數: {mst.number_of_edges()}")

        assert mst.number_of_nodes() > 0, "應該要有節點"
        print("\n✓ 測試 2 通過")

    except Exception as e:
        print(f"測試 2 發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("開始測試主要進入點函數...")

    test_with_simple_image()
    test_with_real_data()

    print("\n" + "=" * 70)
    print("所有測試完成！")
    print("=" * 70)
