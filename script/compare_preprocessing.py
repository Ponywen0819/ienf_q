"""
比對兩種不同前處理配置的結果

修改 config_a 和 config_b 即可比對任意配置差異
目前預設：比對有/無背景移除的差異
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessing import SkinAnalysisPipeline


# ========================================
# 裁切區域設定 (可自行調整)
# ========================================
CROP_X = 1700      # 左上角 x 座標
CROP_Y = 360      # 左上角 y 座標
CROP_WIDTH = 400  # 裁切寬度
CROP_HEIGHT = 300 # 裁切高度


def crop_region(image: np.ndarray) -> np.ndarray:
    """裁切指定區域"""
    h, w = image.shape[:2]
    x1 = min(CROP_X, w - 1)
    y1 = min(CROP_Y, h - 1)
    x2 = min(CROP_X + CROP_WIDTH, w)
    y2 = min(CROP_Y + CROP_HEIGHT, h)
    return image[y1:y2, x1:x2]


def create_viridis_comparison_with_colorbar(
    img_a: np.ndarray,
    img_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: str
) -> None:
    """建立帶有 colorbar 的 viridis 對比圖"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 圖 A
    im_a = axes[0].imshow(img_a, cmap='viridis', vmin=0, vmax=255)
    axes[0].set_title(label_a, fontsize=10)
    axes[0].axis('off')

    # 圖 B
    im_b = axes[1].imshow(img_b, cmap='viridis', vmin=0, vmax=255)
    axes[1].set_title(label_b, fontsize=10)
    axes[1].axis('off')

    # 調整子圖間距，為 colorbar 留空間
    plt.subplots_adjust(right=0.85, wspace=0.1)

    # 在右側添加 colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im_b, cax=cbar_ax)
    cbar.set_label('Pixel Intensity', fontsize=10)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_overlay(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    """建立標籤疊加圖 (紅色半透明)"""
    if len(image.shape) == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()
        
    overlay = base.copy()
    overlay[label > 0] = (0, 0, 255) # Red
    
    return cv2.addWeighted(base, 0, overlay, 1, 0)


def add_label_to_image(image: np.ndarray, label: str) -> np.ndarray:
    """在圖片上方加上標題"""
    label_height = 40

    # 確保是 BGR 格式
    if len(image.shape) == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image.copy()

    labeled = np.zeros((img_bgr.shape[0] + label_height, img_bgr.shape[1], 3), dtype=np.uint8)
    labeled[label_height:, :, :] = img_bgr

    # 加上白色文字
    cv2.putText(labeled, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return labeled


def create_comparison(img_a: np.ndarray, img_b: np.ndarray,
                      label_a: str, label_b: str) -> np.ndarray:
    """建立並排對比圖"""
    labeled_a = add_label_to_image(img_a, label_a)
    labeled_b = add_label_to_image(img_b, label_b)

    # 水平拼接
    return np.hstack([labeled_a, labeled_b])


if __name__ == "__main__":
    # ========================================
    # 配置 A: Rolling Ball
    # ========================================
    config_a = {
        'morphology': {'closing_kernel': 0, 'opening_kernel': 0},
        'mask': {'dilate_offset': 100},
        'background': {
            'method': 'rolling_ball',
            'radius': 2,
            'light_background': False
        },
        'threshold': {'method': 'binary'},
        'normalization': {'enabled': True}
    }
    label_a = "Rolling Ball (r=6)"

    # ========================================
    # 配置 B: Gaussian Blur
    # ========================================
    config_b = {
        'morphology': {'closing_kernel': 5, 'opening_kernel': 3},
        'mask': {'dilate_offset': 100},
        'background': {
            # 'method': 'gaussian',
            # 'sigma': 12.0,
            'method': 'rolling_ball',
            'radius': 2,
            'light_background': False
        },
        'threshold': {'method': 'binary'},
        'normalization': {'enabled': True}
    }
    label_b = "Rolling Ball (r=3)"

    # ========================================
    # 載入測試影像
    # ========================================
    IMAGE_ID = 'S163-2_a'
    LABEL_IMAGE_PATH = f'data/Label/{IMAGE_ID}.tif'
    EPIDERMIS_MASK_PATH = f'data/Mask/{IMAGE_ID}.tif'
    ORIGINAL_IMAGE_PATH = f'data/Original/{IMAGE_ID}.tif'
    label_image = cv2.imread(LABEL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    epidermis_mask = cv2.imread(EPIDERMIS_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    original_green_image = original_image[:, :, 1]

    # ========================================
    # 執行兩組 pipeline
    # ========================================
    print("執行配置 A...")
    pipeline_a = SkinAnalysisPipeline(config_a)
    final_label_a, roi_image_a = pipeline_a.run(label_image, epidermis_mask, original_green_image)

    print("執行配置 B...")
    pipeline_b = SkinAnalysisPipeline(config_b)
    final_label_b, roi_image_b = pipeline_b.run(label_image, epidermis_mask, original_green_image)

    # ========================================
    # 儲存結果
    # ========================================
    output_dir = 'output/preprocessing_compare'

    # 分開儲存
    os.makedirs(f'{output_dir}/config_a', exist_ok=True)
    os.makedirs(f'{output_dir}/config_b', exist_ok=True)
    os.makedirs(f'{output_dir}/comparison', exist_ok=True)

    cv2.imwrite(f'{output_dir}/config_a/roi_image.png', roi_image_a)
    cv2.imwrite(f'{output_dir}/config_a/final_label.png', final_label_a)
    cv2.imwrite(f'{output_dir}/config_b/roi_image.png', roi_image_b)
    cv2.imwrite(f'{output_dir}/config_b/final_label.png', final_label_b)

    # ========================================
    # 裁切小範圍區域
    # ========================================
    roi_crop_a = crop_region(roi_image_a)
    roi_crop_b = crop_region(roi_image_b)
    label_crop_a = crop_region(final_label_a)
    label_crop_b = crop_region(final_label_b)
    original_crop = crop_region(original_green_image)

    # ========================================
    # 並排對比圖 (裁切版)
    # ========================================
    roi_comparison = create_comparison(roi_crop_a, roi_crop_b, label_a, label_b)
    label_comparison = create_comparison(label_crop_a, label_crop_b, label_a, label_b)

    # 疊加圖
    # overlay_a = create_overlay(original_crop, label_crop_a)
    # overlay_b = create_overlay(original_crop, label_crop_b)
    overlay_a = create_overlay(roi_crop_a, label_crop_a)
    overlay_b = create_overlay(roi_crop_b, label_crop_b)
    overlay_comparison = create_comparison(overlay_a, overlay_b, label_a, label_b)

    cv2.imwrite(f'{output_dir}/comparison/roi_comparison.png', roi_comparison)
    cv2.imwrite(f'{output_dir}/comparison/label_comparison.png', label_comparison)
    cv2.imwrite(f'{output_dir}/comparison/overlay_comparison.png', overlay_comparison)

    # ========================================
    # Viridis colormap 對比圖 (帶 colorbar)
    # ========================================
    create_viridis_comparison_with_colorbar(
        roi_crop_a, roi_crop_b,
        label_a, label_b,
        f'{output_dir}/comparison/roi_viridis_comparison.png'
    )

    print(f"\n結果已儲存至 {output_dir}/")
    print(f"  - config_a/: {label_a}")
    print(f"  - config_b/: {label_b}")
    print(f"  - comparison/:")
    print(f"      - roi_comparison.png (裁切區域對比)")
    print(f"      - label_comparison.png (標籤對比)")
    print(f"      - overlay_comparison.png (標籤疊加對比)")
    print(f"      - roi_viridis_comparison.png (Viridis colormap)")
    print(f"\n裁切區域: x={CROP_X}, y={CROP_Y}, w={CROP_WIDTH}, h={CROP_HEIGHT}")
