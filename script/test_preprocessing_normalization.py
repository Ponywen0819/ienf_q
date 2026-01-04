"""
測試前處理 pipeline 的區域正規化功能
"""
import cv2
import os
from preprocessing import SkinAnalysisPipeline


if __name__ == "__main__":
    # 載入測試影像
    label_image = cv2.imread('data/Label/S163-2_a.tif', cv2.IMREAD_GRAYSCALE)
    epidermis_mask = cv2.imread('data/Mask/S163-2_a.tif', cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread('data/Original/S163-2_a.tif', cv2.IMREAD_UNCHANGED)
    original_green_image = original_image[:, :, 1]

    # 啟用區域正規化的配置
    config = {
        'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
        'mask': {'dilate_offset': 100},
        'background': {
            'radius': 12,
            'light_background': False
        },
        'threshold': {'method': 'binary'},
        'normalization': {'enabled': True}  # 啟用區域正規化
    }

    # 執行 pipeline
    pipeline = SkinAnalysisPipeline(config)
    final_label, roi_image = pipeline.run(label_image, epidermis_mask, original_green_image)

    # 儲存結果
    output_dir = 'output/preprocessing_normalization'
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/roi_image.png', roi_image)
    cv2.imwrite(f'{output_dir}/final_label.png', final_label)

    print(f"結果已儲存至 {output_dir}/")
    print(f"  - roi_image.png: 正規化後的 ROI 影像")
    print(f"  - final_label.png: 最終標籤影像")
