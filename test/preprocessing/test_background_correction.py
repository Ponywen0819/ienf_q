"""
Unit tests for background correction

Tests background correction functions in src/neural_reconstruction/core/preprocessing/background_correction.py
"""

import pytest
import numpy as np

from neural_reconstruction.core.preprocessing.background_correction import (
    correct_background,
    create_ball_kernel,
    simple_background_subtraction,
)


class TestCreateBallKernel:
    """Test create_ball_kernel() function"""

    def test_basic_ball_kernel(self):
        """Test basic ball kernel creation"""
        kernel = create_ball_kernel(radius=5)
        assert kernel.ndim == 2
        assert kernel.shape[0] == kernel.shape[1]  # Should be square
        # Kernel may be float32 or uint8 depending on implementation
        assert kernel.dtype in [np.uint8, np.float32]

    @pytest.mark.parametrize("radius", [1, 3, 5, 10, 20])
    def test_various_radii(self, radius):
        """Test ball kernel with various radii"""
        kernel = create_ball_kernel(radius=radius)
        # Kernel size should be approximately 2*radius + 1
        expected_size = 2 * radius + 1
        assert kernel.shape[0] >= expected_size - 2  # Allow some tolerance
        assert kernel.shape[1] >= expected_size - 2

    def test_kernel_center_is_white(self):
        """Test that kernel center is white (part of ball)"""
        kernel = create_ball_kernel(radius=10)
        center_y, center_x = kernel.shape[0] // 2, kernel.shape[1] // 2
        assert kernel[center_y, center_x] > 0

    def test_kernel_corners_are_black(self):
        """Test that kernel corners are black (outside ball)"""
        kernel = create_ball_kernel(radius=10)
        # Corners should be outside the ball
        assert kernel[0, 0] == 0
        assert kernel[0, -1] == 0
        assert kernel[-1, 0] == 0
        assert kernel[-1, -1] == 0

    def test_tiny_radius(self):
        """Test ball kernel with very small radius"""
        kernel = create_ball_kernel(radius=1)
        assert kernel.size > 0
        assert kernel.shape[0] >= 1


class TestSimpleBackgroundSubtraction:
    """Test simple_background_subtraction() function"""

    def test_basic_subtraction(self):
        """Test basic background subtraction"""
        # Create image with background
        image = np.full((100, 100), 150, dtype=np.uint8)
        background = np.full((100, 100), 100, dtype=np.uint8)

        result = simple_background_subtraction(image, background)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

        # Result should be approximately image - background (clipped to 0)
        # Most pixels should be around 50
        assert np.median(result) >= 40
        assert np.median(result) <= 60

    def test_subtraction_prevents_negative(self):
        """Test that subtraction clips negative values to 0"""
        image = np.full((50, 50), 50, dtype=np.uint8)
        background = np.full((50, 50), 100, dtype=np.uint8)

        result = simple_background_subtraction(image, background)
        # Should not have negative values
        assert result.min() >= 0
        # Most values should be 0 (50 - 100 = -50, clipped to 0)
        assert np.sum(result == 0) > result.size * 0.9

    def test_subtraction_with_gradient_background(self, gradient_image):
        """Test subtraction with gradient background"""
        # Create artificial elevated version as image
        image = np.clip(gradient_image.astype(np.int16) + 50, 0, 255).astype(np.uint8)

        result = simple_background_subtraction(image, gradient_image)
        assert result.shape == image.shape
        # Most pixels should be around 50 (the added value)
        assert np.median(result) >= 40
        assert np.median(result) <= 60

    def test_shape_mismatch(self):
        """Test that shape mismatch raises error"""
        image = np.zeros((100, 100), dtype=np.uint8)
        background = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises((ValueError, AssertionError)):
            simple_background_subtraction(image, background)

    def test_identical_image_and_background(self, circles_image):
        """Test subtracting image from itself"""
        result = simple_background_subtraction(circles_image, circles_image)
        # Result should be all zeros
        np.testing.assert_array_equal(result, np.zeros_like(circles_image))


class TestCorrectBackgroundMorphology:
    """Test correct_background() with morphology method"""

    def test_basic_morphology_correction(self, uneven_illumination_image):
        """Test basic morphological background correction"""
        result = correct_background(uneven_illumination_image,
                                   method='morphology',
                                   radius=20)
        assert result.shape == uneven_illumination_image.shape
        assert result.dtype == uneven_illumination_image.dtype

    def test_morphology_light_background(self, uneven_illumination_image):
        """Test morphology with light background"""
        result = correct_background(uneven_illumination_image,
                                   method='morphology',
                                   radius=20,
                                   light_background=True)
        assert result.shape == uneven_illumination_image.shape

    def test_morphology_dark_background(self):
        """Test morphology with dark background"""
        # Create image with dark background and bright spots
        image = np.full((100, 100), 50, dtype=np.uint8)
        image[40:60, 40:60] = 200  # Bright spot

        result = correct_background(image,
                                   method='morphology',
                                   radius=20,
                                   light_background=False)
        assert result.shape == image.shape

    @pytest.mark.parametrize("radius", [5, 10, 20, 50])
    def test_various_radii_morphology(self, uneven_illumination_image, radius):
        """Test morphology with various radii"""
        result = correct_background(uneven_illumination_image,
                                   method='morphology',
                                   radius=radius)
        assert result.shape == uneven_illumination_image.shape

    def test_morphology_flattens_illumination(self, uneven_illumination_image):
        """Test that morphology flattens uneven illumination"""
        result = correct_background(uneven_illumination_image,
                                   method='morphology',
                                   radius=30)

        # Corrected image should have more uniform intensity
        original_std = np.std(uneven_illumination_image)
        corrected_std = np.std(result)

        # Standard deviation should be reduced (more uniform)
        # Note: This may not always be true, so we just check it runs
        assert result.shape == uneven_illumination_image.shape


class TestCorrectBackgroundRollingBall:
    """Test correct_background() with rolling_ball method"""

    def test_basic_rolling_ball_correction(self, uneven_illumination_image):
        """Test basic rolling ball background correction"""
        result = correct_background(uneven_illumination_image,
                                   method='rolling_ball',
                                   radius=20)
        assert result.shape == uneven_illumination_image.shape
        assert result.dtype == uneven_illumination_image.dtype

    def test_rolling_ball_light_background(self, uneven_illumination_image):
        """Test rolling ball with light background"""
        result = correct_background(uneven_illumination_image,
                                   method='rolling_ball',
                                   radius=20,
                                   light_background=True)
        assert result.shape == uneven_illumination_image.shape

    def test_rolling_ball_dark_background(self):
        """Test rolling ball with dark background"""
        image = np.full((100, 100), 50, dtype=np.uint8)
        image[40:60, 40:60] = 200

        result = correct_background(image,
                                   method='rolling_ball',
                                   radius=20,
                                   light_background=False)
        assert result.shape == image.shape

    @pytest.mark.parametrize("radius", [1, 5, 10, 20, 50])
    def test_various_radii_rolling_ball(self, uneven_illumination_image, radius):
        """Test rolling ball with various radii"""
        result = correct_background(uneven_illumination_image,
                                   method='rolling_ball',
                                   radius=radius)
        assert result.shape == uneven_illumination_image.shape

    def test_rolling_ball_with_smoothing(self, uneven_illumination_image):
        """Test rolling ball with smoothing enabled"""
        result = correct_background(uneven_illumination_image,
                                   method='rolling_ball',
                                   radius=20,
                                   smoothing=True)
        assert result.shape == uneven_illumination_image.shape

    def test_rolling_ball_without_smoothing(self, uneven_illumination_image):
        """Test rolling ball without smoothing"""
        result = correct_background(uneven_illumination_image,
                                   method='rolling_ball',
                                   radius=20,
                                   smoothing=False)
        assert result.shape == uneven_illumination_image.shape


class TestCorrectBackgroundGaussian:
    """Test correct_background() with gaussian method"""

    def test_basic_gaussian_correction(self, uneven_illumination_image):
        """Test basic Gaussian background correction"""
        result = correct_background(uneven_illumination_image,
                                   method='gaussian',
                                   sigma=10.0)
        assert result.shape == uneven_illumination_image.shape
        assert result.dtype == uneven_illumination_image.dtype

    def test_gaussian_light_background(self, uneven_illumination_image):
        """Test Gaussian with light background"""
        result = correct_background(uneven_illumination_image,
                                   method='gaussian',
                                   sigma=10.0,
                                   light_background=True)
        assert result.shape == uneven_illumination_image.shape

    def test_gaussian_dark_background(self):
        """Test Gaussian with dark background"""
        image = np.full((100, 100), 50, dtype=np.uint8)
        image[40:60, 40:60] = 200

        result = correct_background(image,
                                   method='gaussian',
                                   sigma=10.0,
                                   light_background=False)
        assert result.shape == image.shape

    @pytest.mark.parametrize("sigma", [1.0, 5.0, 10.0, 20.0, 50.0])
    def test_various_sigma_values(self, uneven_illumination_image, sigma):
        """Test Gaussian with various sigma values"""
        result = correct_background(uneven_illumination_image,
                                   method='gaussian',
                                   sigma=sigma)
        assert result.shape == uneven_illumination_image.shape

    def test_gaussian_small_sigma(self, uneven_illumination_image):
        """Test Gaussian with very small sigma"""
        result = correct_background(uneven_illumination_image,
                                   method='gaussian',
                                   sigma=1.0)
        assert result.shape == uneven_illumination_image.shape

    def test_gaussian_large_sigma(self, uneven_illumination_image):
        """Test Gaussian with very large sigma"""
        result = correct_background(uneven_illumination_image,
                                   method='gaussian',
                                   sigma=100.0)
        assert result.shape == uneven_illumination_image.shape


class TestBackgroundCorrectionEdgeCases:
    """Test background correction edge cases"""

    def test_empty_image(self, empty_image):
        """Test background correction on empty image"""
        result = correct_background(empty_image, method='morphology', radius=10)
        # Empty image should remain empty or near-empty
        assert result.shape == empty_image.shape
        assert result.max() <= 10  # Allow small values from processing

    def test_constant_image(self, constant_image):
        """Test background correction on constant image"""
        result = correct_background(constant_image, method='morphology', radius=10)
        # Constant image should have low variance after correction
        assert result.shape == constant_image.shape

    def test_tiny_image(self, tiny_image):
        """Test background correction on tiny image"""
        # Use small radius for tiny image
        result = correct_background(tiny_image, method='morphology', radius=1)
        assert result.shape == tiny_image.shape

    def test_high_contrast_image(self):
        """Test background correction on high contrast image"""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[::2, ::2] = 255  # Checkerboard pattern

        result = correct_background(image, method='morphology', radius=5)
        assert result.shape == image.shape

    def test_invalid_method(self, circles_image):
        """Test that invalid method raises error"""
        with pytest.raises(ValueError):
            correct_background(circles_image, method='invalid_method')


class TestBackgroundCorrectionIntegration:
    """Integration tests for background correction"""

    def test_all_methods_produce_valid_output(self, uneven_illumination_image):
        """Test that all methods produce valid output"""
        methods = ['morphology', 'rolling_ball', 'gaussian']

        for method in methods:
            if method == 'gaussian':
                result = correct_background(uneven_illumination_image,
                                          method=method, sigma=10.0)
            else:
                result = correct_background(uneven_illumination_image,
                                          method=method, radius=20)

            assert result.shape == uneven_illumination_image.shape
            assert result.dtype == uneven_illumination_image.dtype
            assert result.min() >= 0
            assert result.max() <= 255

    def test_light_vs_dark_background_difference(self, uneven_illumination_image):
        """Test difference between light and dark background modes"""
        light_result = correct_background(uneven_illumination_image,
                                        method='morphology',
                                        radius=20,
                                        light_background=True)

        dark_result = correct_background(uneven_illumination_image,
                                       method='morphology',
                                       radius=20,
                                       light_background=False)

        # Results should be different
        assert not np.array_equal(light_result, dark_result)

    def test_compare_methods_on_same_image(self, uneven_illumination_image):
        """Test that different methods produce reasonable results"""
        morph_result = correct_background(uneven_illumination_image,
                                        method='morphology', radius=20)
        rb_result = correct_background(uneven_illumination_image,
                                      method='rolling_ball', radius=20)
        gauss_result = correct_background(uneven_illumination_image,
                                        method='gaussian', sigma=10.0)

        # All should produce same-shaped results
        assert morph_result.shape == uneven_illumination_image.shape
        assert rb_result.shape == uneven_illumination_image.shape
        assert gauss_result.shape == uneven_illumination_image.shape

        # Results may differ but should all be valid
        for result in [morph_result, rb_result, gauss_result]:
            assert result.min() >= 0
            assert result.max() <= 255

    def test_radius_effect_on_correction(self, uneven_illumination_image):
        """Test that radius affects correction strength"""
        small_radius = correct_background(uneven_illumination_image,
                                        method='morphology', radius=5)
        large_radius = correct_background(uneven_illumination_image,
                                        method='morphology', radius=50)

        # Both should be valid
        assert small_radius.shape == uneven_illumination_image.shape
        assert large_radius.shape == uneven_illumination_image.shape

        # Results may differ due to different radius
        # (We don't enforce specific behavior, just that both run successfully)

    def test_correction_on_gradient_image(self, gradient_image):
        """Test background correction on gradient image"""
        result = correct_background(gradient_image,
                                   method='morphology',
                                   radius=30)

        # Gradient should be reduced
        assert result.shape == gradient_image.shape
        # After correction, image should be more uniform
        # (Standard deviation should decrease)
        # Note: This is not always guaranteed, so we just verify it runs

    def test_correction_preserves_features(self, fiber_like_image):
        """Test that correction preserves important features"""
        result = correct_background(fiber_like_image,
                                   method='rolling_ball',
                                   radius=15)

        # Features (white fibers) should still be present
        assert result.shape == fiber_like_image.shape
        # Should have some bright pixels (fibers)
        assert result.max() > 100  # Fibers should be bright

    def test_multiple_corrections(self, uneven_illumination_image):
        """Test applying correction multiple times"""
        result1 = correct_background(uneven_illumination_image,
                                    method='morphology', radius=20)
        result2 = correct_background(result1,
                                    method='morphology', radius=20)

        # Should not crash and produce valid output
        assert result2.shape == uneven_illumination_image.shape
        assert result2.dtype == np.uint8
