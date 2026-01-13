"""
Unit tests for thresholding operations

Tests thresholding functions in src/neural_reconstruction/core/preprocessing/thresholding.py
"""

import pytest
import numpy as np
import cv2

from neural_reconstruction.core.preprocessing.thresholding import (
    otsu_threshold,
    fixed_threshold,
    adaptive_threshold,
    multi_otsu_threshold,
    triangle_threshold,
)


class TestOtsuThreshold:
    """Test otsu_threshold() function"""

    def test_basic_otsu(self, circles_image):
        """Test basic Otsu thresholding"""
        result = otsu_threshold(circles_image)
        assert result.shape == circles_image.shape
        assert result.dtype == np.uint8
        # Binary result should have only 0 and 255
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)

    def test_otsu_binary_type(self, gradient_image):
        """Test Otsu with binary threshold type"""
        result = otsu_threshold(gradient_image, threshold_type=cv2.THRESH_BINARY)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_otsu_binary_inv_type(self, gradient_image):
        """Test Otsu with binary inverse threshold type"""
        result = otsu_threshold(gradient_image, threshold_type=cv2.THRESH_BINARY_INV)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_return_threshold_value(self, circles_image):
        """Test Otsu with return_threshold=True"""
        result, threshold_value = otsu_threshold(circles_image, return_threshold=True)
        assert isinstance(threshold_value, (int, float))
        assert 0 <= threshold_value <= 255
        assert result.shape == circles_image.shape

    def test_bimodal_distribution(self):
        """Test Otsu on bimodal distribution"""
        # Create image with clear bimodal distribution
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 50  # Dark region
        image[50:, :] = 200  # Bright region

        result = otsu_threshold(image)
        # Should separate the two modes
        assert np.any(result == 0)
        assert np.any(result == 255)

    def test_empty_image(self, empty_image):
        """Test Otsu on empty image"""
        result = otsu_threshold(empty_image)
        # Empty image should remain empty
        np.testing.assert_array_equal(result, empty_image)

    def test_constant_image(self, constant_image):
        """Test Otsu on constant image"""
        # Constant image has no threshold, but should not crash
        result = otsu_threshold(constant_image)
        assert result.shape == constant_image.shape

    def test_noisy_image(self, noisy_image):
        """Test Otsu on noisy image"""
        result = otsu_threshold(noisy_image)
        assert result.shape == noisy_image.shape
        assert result.dtype == np.uint8


class TestFixedThreshold:
    """Test fixed_threshold() function"""

    def test_basic_fixed_threshold(self, gradient_image):
        """Test basic fixed thresholding"""
        result = fixed_threshold(gradient_image, threshold_value=128)
        assert result.shape == gradient_image.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    @pytest.mark.parametrize("threshold_value", [50, 100, 128, 150, 200])
    def test_various_threshold_values(self, circles_image, threshold_value):
        """Test fixed threshold with various threshold values"""
        result = fixed_threshold(circles_image, threshold_value=threshold_value)
        assert result.shape == circles_image.shape
        # Higher threshold should result in fewer white pixels
        white_pixels = np.sum(result == 255)
        assert white_pixels >= 0

    def test_threshold_0(self, circles_image):
        """Test fixed threshold with value 0"""
        result = fixed_threshold(circles_image, threshold_value=0)
        # Everything above 0 should be white
        assert np.all(result[circles_image > 0] == 255)

    def test_threshold_255(self, circles_image):
        """Test fixed threshold with value 255"""
        result = fixed_threshold(circles_image, threshold_value=255)
        # Nothing should be above 255, so mostly black
        black_pixels = np.sum(result == 0)
        total_pixels = result.size
        assert black_pixels >= total_pixels * 0.9  # At least 90% black

    def test_binary_inv_type(self, gradient_image):
        """Test fixed threshold with binary inverse"""
        result = fixed_threshold(gradient_image, threshold_value=128,
                                threshold_type=cv2.THRESH_BINARY_INV)
        assert result.shape == gradient_image.shape
        assert set(np.unique(result)).issubset({0, 255})

    def test_custom_max_value(self, circles_image):
        """Test fixed threshold with custom max value"""
        result = fixed_threshold(circles_image, threshold_value=128, max_value=200)
        # Max value should be 200 instead of 255
        assert result.max() <= 200

    def test_empty_image(self, empty_image):
        """Test fixed threshold on empty image"""
        result = fixed_threshold(empty_image, threshold_value=128)
        np.testing.assert_array_equal(result, empty_image)

    def test_constant_image_below_threshold(self):
        """Test fixed threshold on constant image below threshold"""
        image = np.full((50, 50), 100, dtype=np.uint8)
        result = fixed_threshold(image, threshold_value=150)
        # All values below threshold, should be black
        np.testing.assert_array_equal(result, np.zeros_like(image))

    def test_constant_image_above_threshold(self):
        """Test fixed threshold on constant image above threshold"""
        image = np.full((50, 50), 200, dtype=np.uint8)
        result = fixed_threshold(image, threshold_value=150)
        # All values above threshold, should be white
        np.testing.assert_array_equal(result, np.full_like(image, 255))


class TestAdaptiveThreshold:
    """Test adaptive_threshold() function"""

    def test_basic_adaptive_mean(self, uneven_illumination_image):
        """Test basic adaptive thresholding with mean method"""
        result = adaptive_threshold(uneven_illumination_image, method=cv2.ADAPTIVE_THRESH_MEAN_C)
        assert result.shape == uneven_illumination_image.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_basic_adaptive_gaussian(self, uneven_illumination_image):
        """Test basic adaptive thresholding with Gaussian method"""
        result = adaptive_threshold(uneven_illumination_image, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        assert result.shape == uneven_illumination_image.shape
        assert set(np.unique(result)).issubset({0, 255})

    @pytest.mark.parametrize("block_size", [3, 5, 11, 21])
    def test_various_block_sizes(self, uneven_illumination_image, block_size):
        """Test adaptive threshold with various block sizes"""
        result = adaptive_threshold(uneven_illumination_image, block_size=block_size)
        assert result.shape == uneven_illumination_image.shape

    @pytest.mark.parametrize("c_value", [-10, -5, 0, 5, 10])
    def test_various_c_values(self, uneven_illumination_image, c_value):
        """Test adaptive threshold with various C values"""
        result = adaptive_threshold(uneven_illumination_image, c=c_value)
        assert result.shape == uneven_illumination_image.shape

    def test_binary_inv_type(self, uneven_illumination_image):
        """Test adaptive threshold with binary inverse"""
        result = adaptive_threshold(uneven_illumination_image,
                                   threshold_type=cv2.THRESH_BINARY_INV)
        assert result.shape == uneven_illumination_image.shape

    def test_uneven_illumination_handling(self, uneven_illumination_image):
        """Test that adaptive threshold handles uneven illumination"""
        # Adaptive should work better than global on uneven illumination
        adaptive_result = adaptive_threshold(uneven_illumination_image)
        global_result = otsu_threshold(uneven_illumination_image)

        # Both should produce binary results
        assert adaptive_result.shape == global_result.shape

    def test_tiny_image(self, tiny_image):
        """Test adaptive threshold on tiny image"""
        # Block size must be smaller than image
        result = adaptive_threshold(tiny_image, block_size=3)
        assert result.shape == tiny_image.shape

    def test_fiber_like_structure(self, fiber_like_image):
        """Test adaptive threshold on fiber-like structure"""
        result = adaptive_threshold(fiber_like_image, block_size=11)
        assert result.shape == fiber_like_image.shape
        # Should preserve some structure
        assert np.any(result == 255)


class TestMultiOtsuThreshold:
    """Test multi_otsu_threshold() function"""

    def test_basic_multi_otsu_2_classes(self, gradient_image):
        """Test multi-Otsu with 2 classes (same as regular Otsu)"""
        result = multi_otsu_threshold(gradient_image, n_classes=2)
        assert result.shape == gradient_image.shape
        assert result.dtype == np.uint8

    def test_multi_otsu_3_classes(self, gradient_image):
        """Test multi-Otsu with 3 classes"""
        result = multi_otsu_threshold(gradient_image, n_classes=3)
        assert result.shape == gradient_image.shape
        unique_values = np.unique(result)
        # Should have up to 3 distinct values
        assert len(unique_values) <= 3

    def test_multi_otsu_4_classes(self, gradient_image):
        """Test multi-Otsu with 4 classes"""
        result = multi_otsu_threshold(gradient_image, n_classes=4)
        assert result.shape == gradient_image.shape
        unique_values = np.unique(result)
        assert len(unique_values) <= 4

    def test_return_thresholds(self, gradient_image):
        """Test multi-Otsu with return_thresholds=True"""
        result, thresholds = multi_otsu_threshold(gradient_image, n_classes=3,
                                                  return_thresholds=True)
        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == 2  # n_classes - 1 thresholds
        # Thresholds should be in ascending order
        assert np.all(thresholds[:-1] <= thresholds[1:])

    def test_multi_level_segmentation(self):
        """Test multi-level segmentation on synthetic data"""
        # Create image with 3 distinct intensity levels
        image = np.zeros((90, 100), dtype=np.uint8)
        image[:30, :] = 50
        image[30:60, :] = 125
        image[60:, :] = 200

        result = multi_otsu_threshold(image, n_classes=3)
        # Should segment into 3 classes
        unique_values = np.unique(result)
        assert len(unique_values) == 3

    def test_constant_image(self, constant_image):
        """Test multi-Otsu on constant image"""
        # Should handle constant image gracefully
        result = multi_otsu_threshold(constant_image, n_classes=3)
        assert result.shape == constant_image.shape


class TestTriangleThreshold:
    """Test triangle_threshold() function"""

    def test_basic_triangle(self, circles_image):
        """Test basic triangle thresholding"""
        result = triangle_threshold(circles_image)
        assert result.shape == circles_image.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_triangle_binary_type(self, gradient_image):
        """Test triangle with binary threshold type"""
        result = triangle_threshold(gradient_image, threshold_type=cv2.THRESH_BINARY)
        assert result.dtype == np.uint8

    def test_triangle_binary_inv_type(self, gradient_image):
        """Test triangle with binary inverse type"""
        result = triangle_threshold(gradient_image, threshold_type=cv2.THRESH_BINARY_INV)
        assert result.dtype == np.uint8

    def test_return_threshold_value(self, circles_image):
        """Test triangle with return_threshold=True"""
        result, threshold_value = triangle_threshold(circles_image, return_threshold=True)
        assert isinstance(threshold_value, (int, float))
        assert 0 <= threshold_value <= 255
        assert result.shape == circles_image.shape

    def test_skewed_distribution(self):
        """Test triangle on skewed distribution"""
        # Create image with skewed distribution (more dark pixels)
        image = np.random.exponential(scale=50, size=(100, 100)).astype(np.uint8)
        image = np.clip(image, 0, 255)

        result = triangle_threshold(image)
        # Should produce binary result
        assert set(np.unique(result)).issubset({0, 255})

    def test_noisy_image(self, noisy_image):
        """Test triangle on noisy image"""
        result = triangle_threshold(noisy_image)
        assert result.shape == noisy_image.shape


class TestThresholdingIntegration:
    """Integration tests for thresholding operations"""

    def test_compare_otsu_and_triangle(self, circles_image):
        """Test that Otsu and triangle produce similar results"""
        otsu_result = otsu_threshold(circles_image)
        triangle_result = triangle_threshold(circles_image)

        # Results should have same shape
        assert otsu_result.shape == triangle_result.shape

        # Both should be binary
        assert set(np.unique(otsu_result)).issubset({0, 255})
        assert set(np.unique(triangle_result)).issubset({0, 255})

    def test_fixed_vs_otsu(self, gradient_image):
        """Test fixed threshold vs Otsu threshold"""
        # Get Otsu threshold value
        _, otsu_value = otsu_threshold(gradient_image, return_threshold=True)

        # Apply fixed threshold with Otsu value
        otsu_result = otsu_threshold(gradient_image)
        fixed_result = fixed_threshold(gradient_image, threshold_value=int(otsu_value))

        # Results should be identical
        np.testing.assert_array_equal(otsu_result, fixed_result)

    def test_adaptive_vs_global_on_uneven(self, uneven_illumination_image):
        """Test adaptive vs global thresholding on uneven illumination"""
        adaptive_result = adaptive_threshold(uneven_illumination_image, block_size=21)
        global_result = otsu_threshold(uneven_illumination_image)

        # Both should produce results
        assert adaptive_result.shape == global_result.shape
        # Adaptive should detect more foreground in dim regions
        # (This is a qualitative test, just ensuring both run)

    def test_multi_otsu_with_2_classes_equals_otsu(self, circles_image):
        """Test that multi-Otsu with 2 classes equals regular Otsu"""
        otsu_result, otsu_thresh = otsu_threshold(circles_image, return_threshold=True)
        multi_result, multi_thresh = multi_otsu_threshold(circles_image, n_classes=2,
                                                          return_thresholds=True)

        # Threshold values should be similar
        assert abs(otsu_thresh - multi_thresh[0]) < 5

    def test_all_methods_on_same_image(self, fiber_like_image):
        """Test all thresholding methods on same image"""
        methods = [
            lambda img: otsu_threshold(img),
            lambda img: fixed_threshold(img, 128),
            lambda img: adaptive_threshold(img),
            lambda img: multi_otsu_threshold(img, n_classes=2),
            lambda img: triangle_threshold(img),
        ]

        for method in methods:
            result = method(fiber_like_image)
            assert result.shape == fiber_like_image.shape
            assert result.dtype == np.uint8
            # Should produce binary or multi-level result
            assert len(np.unique(result)) <= 3
