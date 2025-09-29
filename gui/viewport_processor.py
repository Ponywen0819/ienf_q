#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
視窗區域處理器
只處理當前可見區域的圖片，大幅提升大圖和高縮放比例下的性能
"""

import numpy as np
import cv2
from PIL import Image


class ViewportProcessor:
    """視窗區域圖片處理器"""

    def __init__(self):
        # 視窗參數
        self.viewport_x = 0          # 視窗左上角X座標（相對於原圖）
        self.viewport_y = 0          # 視窗左上角Y座標（相對於原圖）
        self.viewport_width = 800    # 視窗寬度（畫布尺寸）
        self.viewport_height = 600   # 視窗高度（畫布尺寸）
        self.zoom_factor = 1.0       # 縮放係數

        # 圖片數據
        self.original_image = None
        self.neural_spots_image = None
        self.center_line_image = None

        # 顯示設定
        self.show_original = True
        self.show_neural = True
        self.show_center = True
        self.channel = "all"
        self.neural_opacity = 0.5
        self.center_opacity = 0.75

        # 緩存
        self.cached_viewport_img = None
        self.cached_settings = None
        self.cache_viewport_bounds = None

    def set_images(self, original=None, neural=None, center=None):
        """設定圖片數據"""
        if original is not None:
            self.original_image = original
        if neural is not None:
            self.neural_spots_image = neural
        if center is not None:
            self.center_line_image = center
        self._clear_cache()

    def set_viewport(self, x, y, width, height, zoom):
        """設定視窗區域參數"""
        self.viewport_x = max(0, int(x))
        self.viewport_y = max(0, int(y))
        self.viewport_width = int(width)
        self.viewport_height = int(height)
        self.zoom_factor = zoom
        self._clear_cache()

    def set_display_settings(self, show_original=None, show_neural=None, show_center=None,
                           channel=None, neural_opacity=None, center_opacity=None):
        """設定顯示參數"""
        if show_original is not None:
            self.show_original = show_original
        if show_neural is not None:
            self.show_neural = show_neural
        if show_center is not None:
            self.show_center = show_center
        if channel is not None:
            self.channel = channel
        if neural_opacity is not None:
            self.neural_opacity = neural_opacity
        if center_opacity is not None:
            self.center_opacity = center_opacity
        self._clear_cache()

    def get_viewport_image(self):
        """
        獲取當前視窗區域的處理後圖片
        """
        if not self._has_any_image():
            return None

        # 檢查緩存是否有效
        current_settings = self._get_current_settings()
        current_bounds = (self.viewport_x, self.viewport_y,
                         self.viewport_width, self.viewport_height, self.zoom_factor)

        if (self.cached_viewport_img is not None and
            self.cached_settings == current_settings and
            self.cache_viewport_bounds == current_bounds):
            return self.cached_viewport_img

        # 計算需要處理的原圖區域
        crop_region = self._calculate_source_region()
        if crop_region is None:
            return None

        # 只裁切和處理需要的區域
        viewport_img = self._process_viewport_region(crop_region)

        # 更新緩存
        self.cached_viewport_img = viewport_img
        self.cached_settings = current_settings
        self.cache_viewport_bounds = current_bounds

        return viewport_img

    def _calculate_source_region(self):
        """計算需要從原圖裁切的區域"""
        if not self._has_any_image():
            return None

        # 獲取基準圖片尺寸（通常是最大的那張）
        base_img = self._get_base_image()
        if base_img is None:
            return None

        img_height, img_width = base_img.shape[:2]

        # 計算在原圖中需要裁切的區域（考慮縮放）
        # 視窗區域對應的原圖區域
        source_x = int(self.viewport_x / self.zoom_factor)
        source_y = int(self.viewport_y / self.zoom_factor)
        source_width = int(self.viewport_width / self.zoom_factor)
        source_height = int(self.viewport_height / self.zoom_factor)

        # 確保不超出圖片邊界
        source_x = max(0, min(source_x, img_width))
        source_y = max(0, min(source_y, img_height))
        source_width = min(source_width, img_width - source_x)
        source_height = min(source_height, img_height - source_y)

        # 如果區域太小，返回None
        if source_width <= 0 or source_height <= 0:
            return None

        return {
            'x': source_x,
            'y': source_y,
            'width': source_width,
            'height': source_height
        }

    def _process_viewport_region(self, crop_region):
        """處理視窗區域的圖片"""
        x, y = crop_region['x'], crop_region['y']
        w, h = crop_region['width'], crop_region['height']

        # 初始化結果圖片
        result_img = None

        # 處理原始圖片
        if self.original_image is not None and self.show_original:
            # 只裁切需要的區域
            cropped = self.original_image[y:y+h, x:x+w].copy()
            result_img = cropped

        # 處理神經亮點圖片
        if self.neural_spots_image is not None and self.show_neural:
            # 裁切對應區域
            neural_cropped = self._crop_and_resize_to_match(
                self.neural_spots_image, x, y, w, h, result_img.shape[:2] if result_img is not None else None
            )

            if neural_cropped is not None:
                if result_img is None:
                    result_img = np.zeros_like(neural_cropped)

                # 疊加神經亮點
                result_img = cv2.addWeighted(
                    result_img.astype(np.float32), 1.0,
                    neural_cropped.astype(np.float32), self.neural_opacity, 0
                )
                result_img = np.clip(result_img, 0, 255).astype(np.uint8)

        # 處理Center Line圖片
        if self.center_line_image is not None and self.show_center:
            # 裁切對應區域
            center_cropped = self._crop_and_resize_to_match(
                self.center_line_image, x, y, w, h, result_img.shape[:2] if result_img is not None else None
            )

            if center_cropped is not None:
                if result_img is None:
                    result_img = np.zeros_like(center_cropped)

                # 疊加Center Line
                result_img = cv2.addWeighted(
                    result_img.astype(np.float32), 1.0,
                    center_cropped.astype(np.float32), self.center_opacity, 0
                )
                result_img = np.clip(result_img, 0, 255).astype(np.uint8)

        if result_img is None:
            return None

        # 應用通道過濾
        result_img = self._apply_channel_filter(result_img)

        # 縮放到顯示尺寸（只縮放裁切後的小圖）
        if self.zoom_factor != 1.0:
            target_width = int(w * self.zoom_factor)
            target_height = int(h * self.zoom_factor)
            result_img = cv2.resize(result_img, (target_width, target_height),
                                  interpolation=cv2.INTER_NEAREST)

        return result_img

    def _crop_and_resize_to_match(self, img, x, y, w, h, target_shape):
        """裁切圖片並調整到匹配的尺寸"""
        if img is None:
            return None

        img_h, img_w = img.shape[:2]

        # 確保裁切區域在圖片範圍內
        x = max(0, min(x, img_w))
        y = max(0, min(y, img_h))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return None

        # 裁切圖片
        cropped = img[y:y+h, x:x+w]

        # 如果需要調整尺寸以匹配目標
        if target_shape is not None and cropped.shape[:2] != target_shape:
            target_h, target_w = target_shape
            cropped = cv2.resize(cropped, (target_w, target_h))

        return cropped

    def _apply_channel_filter(self, img):
        """應用通道過濾"""
        if self.channel == "all":
            return img

        channel_map = {"red": 0, "green": 1, "blue": 2}
        if self.channel in channel_map:
            gray_channel = img[:, :, channel_map[self.channel]]
            return np.stack([gray_channel] * 3, axis=-1)

        return img

    def _get_base_image(self):
        """獲取基準圖片（用於確定尺寸）"""
        if self.original_image is not None:
            return self.original_image
        elif self.neural_spots_image is not None:
            return self.neural_spots_image
        elif self.center_line_image is not None:
            return self.center_line_image
        return None

    def _has_any_image(self):
        """檢查是否有任何圖片"""
        return any([
            self.original_image is not None,
            self.neural_spots_image is not None,
            self.center_line_image is not None
        ])

    def _get_current_settings(self):
        """獲取當前設定（用於緩存檢查）"""
        return (
            self.show_original, self.show_neural, self.show_center,
            self.channel, self.neural_opacity, self.center_opacity
        )

    def _clear_cache(self):
        """清除緩存"""
        self.cached_viewport_img = None
        self.cached_settings = None
        self.cache_viewport_bounds = None

    def get_full_image_size(self):
        """獲取完整圖片尺寸"""
        base_img = self._get_base_image()
        if base_img is not None:
            return base_img.shape[1], base_img.shape[0]  # width, height
        return 0, 0

    def pan_viewport(self, dx, dy):
        """平移視窗"""
        img_width, img_height = self.get_full_image_size()
        if img_width == 0 or img_height == 0:
            return

        # 計算新的視窗位置
        new_x = self.viewport_x + dx
        new_y = self.viewport_y + dy

        # 限制在圖片範圍內
        max_x = max(0, int(img_width * self.zoom_factor - self.viewport_width))
        max_y = max(0, int(img_height * self.zoom_factor - self.viewport_height))

        self.viewport_x = max(0, min(new_x, max_x))
        self.viewport_y = max(0, min(new_y, max_y))
        self._clear_cache()

    def zoom_at_point(self, zoom_ratio, point_x, point_y):
        """在指定點進行縮放"""
        img_width, img_height = self.get_full_image_size()
        if img_width == 0 or img_height == 0:
            return

        old_zoom = self.zoom_factor
        new_zoom = max(0.1, min(10.0, old_zoom * zoom_ratio))

        if new_zoom == old_zoom:
            return

        # 計算縮放中心在原圖中的位置
        center_x_in_image = (self.viewport_x + point_x) / old_zoom
        center_y_in_image = (self.viewport_y + point_y) / old_zoom

        # 更新縮放
        self.zoom_factor = new_zoom

        # 計算新的視窗位置，使縮放中心保持在相同位置
        new_viewport_x = center_x_in_image * new_zoom - point_x
        new_viewport_y = center_y_in_image * new_zoom - point_y

        # 限制在有效範圍內
        max_x = max(0, int(img_width * new_zoom - self.viewport_width))
        max_y = max(0, int(img_height * new_zoom - self.viewport_height))

        self.viewport_x = max(0, min(int(new_viewport_x), max_x))
        self.viewport_y = max(0, min(int(new_viewport_y), max_y))

        self._clear_cache()
        return True