"""
Unit tests for preprocessing utility functions

Tests utility functions in src/neural_reconstruction/core/preprocessing/utils.py
"""

import pytest
import numpy as np
import cv2

from neural_reconstruction.core.preprocessing.utils import (
    validate_image,
    normalize_image,
    denormalize_image,
    create_kernel,
    ensure_grayscale,
    clip_image,
    regional_minmax_normalize,
)


class TestValidateImage:
    """Test validate_image() function"""

    def test_valid_uint8_image(self, circles_image):
        """Test validation of valid uint8 image"""
        # Should not raise any exception
        validate_image(circles_image)

    def test_valid_float32_image(self):
        """Test validation of valid float32 image"""
        image = np.random.rand(100, 100).astype(np.float32)
        validate_image(image)

    def test_valid_3d_image(self, rgb_gradient_image):
        """Test validation of valid 3D RGB image"""
        validate_image(rgb_gradient_image)

    def test_invalid_none(self):
        """Test validation fails on None"""
        with pytest.raises((TypeError, ValueError)):
            validate_image(None)

    def test_invalid_list(self):
        """Test validation fails on list"""
        with pytest.raises((TypeError, ValueError)):
            validate_image([[1, 2], [3, 4]])

    def test_invalid_1d_array(self):
        """Test validation fails on 1D array"""
        with pytest.raises((ValueError, AssertionError)):
            validate_image(np.array([1, 2, 3, 4]))

    def test_invalid_4d_array(self):
        """Test validation fails on 4D array"""
        with pytest.raises((ValueError, AssertionError)):
            validate_image(np.random.rand(10, 10, 3, 2))


class TestNormalizeImage:
    """Test normalize_image() function"""

    def test_uint8_to_uint8(self, circles_image):
        """Test uint8 image remains uint8"""
        normalized, was_float, original_dtype = normalize_image(circles_image)
        assert normalized.dtype == np.uint8
        assert was_float is False
        assert original_dtype == np.uint8
        np.testing.assert_array_equal(normalized, circles_image)

    def test_float32_to_uint8(self):
        """Test float32 image converts to uint8"""
        float_image = np.random.rand(50, 50).astype(np.float32)
        normalized, was_float, original_dtype = normalize_image(float_image)
        assert normalized.dtype == np.uint8
        assert was_float is True
        assert original_dtype == np.float32
        assert normalized.min() >= 0
        assert normalized.max() <= 255

    def test_float64_to_uint8(self):
        """Test float64 image converts to uint8"""
        float_image = np.random.rand(50, 50).astype(np.float64)
        normalized, was_float, original_dtype = normalize_image(float_image)
        assert normalized.dtype == np.uint8
        assert was_float is True
        assert original_dtype == np.float64

    def test_value_range_preservation(self):
        """Test that value range is preserved correctly"""
        # Create float image with known range [0, 1]
        float_image = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        normalized, _, _ = normalize_image(float_image)
        assert normalized[0, 0] == 0
        assert normalized[1, 1] == 255

    def test_empty_image(self, empty_image):
        """Test normalization of empty image"""
        normalized, was_float, original_dtype = normalize_image(empty_image)
        assert normalized.dtype == np.uint8
        assert np.all(normalized == 0)


class TestDenormalizeImage:
    """Test denormalize_image() function"""

    def test_uint8_remains_uint8(self, circles_image):
        """Test uint8 image remains uint8"""
        denormalized = denormalize_image(circles_image, was_float=False, original_dtype=np.uint8)
        assert denormalized.dtype == np.uint8
        np.testing.assert_array_equal(denormalized, circles_image)

    def test_uint8_to_float32(self):
        """Test uint8 converts back to float32"""
        uint8_image = np.array([[0, 128, 255]], dtype=np.uint8)
        denormalized = denormalize_image(uint8_image, was_float=True, original_dtype=np.float32)
        assert denormalized.dtype == np.float32
        assert denormalized.min() >= 0.0
        assert denormalized.max() <= 1.0

    def test_roundtrip_float32(self):
        """Test normalize then denormalize preserves float range"""
        original = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        normalized, was_float, original_dtype = normalize_image(original)
        denormalized = denormalize_image(normalized, was_float, original_dtype)

        assert denormalized.dtype == np.float32
        np.testing.assert_allclose(denormalized, original, atol=0.01)


class TestCreateKernel:
    """Test create_kernel() function"""

    def test_rectangular_kernel(self):
        """Test creation of rectangular kernel"""
        kernel = create_kernel(size=5, shape='rect')
        assert kernel.shape == (5, 5)
        assert np.all(kernel == 1)

    def test_ellipse_kernel(self):
        """Test creation of ellipse kernel"""
        kernel = create_kernel(size=7, shape='ellipse')
        assert kernel.shape == (7, 7)
        # Check that corners are 0 (ellipse shape)
        assert kernel[0, 0] == 0
        assert kernel[0, -1] == 0

    def test_cross_kernel(self):
        """Test creation of cross kernel"""
        kernel = create_kernel(size=5, shape='cross')
        assert kernel.shape == (5, 5)
        # Check cross pattern (center row and column are 1)
        assert np.all(kernel[2, :] == 1)  # Center row
        assert np.all(kernel[:, 2] == 1)  # Center column

    def test_kernel_size_1(self):
        """Test kernel with size 1"""
        kernel = create_kernel(size=1, shape='rect')
        assert kernel.shape == (1, 1)
        assert kernel[0, 0] == 1

    def test_kernel_size_3(self):
        """Test kernel with size 3"""
        kernel = create_kernel(size=3, shape='rect')
        assert kernel.shape == (3, 3)

    @pytest.mark.parametrize("size", [5, 7, 9, 11])
    def test_various_sizes(self, size):
        """Test kernel creation with various sizes"""
        kernel = create_kernel(size=size, shape='rect')
        assert kernel.shape == (size, size)

    def test_invalid_shape(self):
        """Test invalid kernel shape"""
        with pytest.raises(ValueError):
            create_kernel(size=5, shape='invalid')


class TestEnsureGrayscale:
    """Test ensure_grayscale() function"""

    def test_grayscale_unchanged(self, circles_image):
        """Test grayscale image remains unchanged"""
        result = ensure_grayscale(circles_image)
        assert result.ndim == 2
        np.testing.assert_array_equal(result, circles_image)

    def test_rgb_to_grayscale_default(self, rgb_gradient_image):
        """Test RGB converts to grayscale (default: green channel)"""
        result = ensure_grayscale(rgb_gradient_image)
        assert result.ndim == 2
        assert result.shape == rgb_gradient_image.shape[:2]
        # Should extract green channel by default
        np.testing.assert_array_equal(result, rgb_gradient_image[:, :, 1])

    def test_rgb_green_channel(self, rgb_fiber_image):
        """Test explicit green channel extraction"""
        result = ensure_grayscale(rgb_fiber_image, extract_channel='green')
        assert result.ndim == 2
        np.testing.assert_array_equal(result, rgb_fiber_image[:, :, 1])

    def test_rgb_red_channel(self, rgb_gradient_image):
        """Test red channel extraction"""
        result = ensure_grayscale(rgb_gradient_image, extract_channel='red')
        assert result.ndim == 2
        np.testing.assert_array_equal(result, rgb_gradient_image[:, :, 0])

    def test_rgb_blue_channel(self, rgb_gradient_image):
        """Test blue channel extraction"""
        result = ensure_grayscale(rgb_gradient_image, extract_channel='blue')
        assert result.ndim == 2
        np.testing.assert_array_equal(result, rgb_gradient_image[:, :, 2])

    def test_rgb_to_gray_conversion(self, rgb_gradient_image):
        """Test RGB to grayscale conversion"""
        result = ensure_grayscale(rgb_gradient_image, extract_channel='gray')
        assert result.ndim == 2
        assert result.dtype == rgb_gradient_image.dtype

    def test_invalid_channel(self, rgb_gradient_image):
        """Test invalid channel name"""
        with pytest.raises(ValueError):
            ensure_grayscale(rgb_gradient_image, extract_channel='invalid')

    def test_empty_image(self, empty_image):
        """Test empty grayscale image"""
        result = ensure_grayscale(empty_image)
        assert result.ndim == 2
        np.testing.assert_array_equal(result, empty_image)


class TestClipImage:
    """Test clip_image() function"""

    def test_uint8_no_clipping_needed(self, circles_image):
        """Test uint8 image within valid range"""
        result = clip_image(circles_image)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, circles_image)

    def test_uint8_out_of_range(self):
        """Test uint8 image with invalid values gets clipped"""
        # Create image with out-of-range values (using int16 temporarily)
        image_int16 = np.array([[-10, 128, 300]], dtype=np.int16)
        # Convert to uint8 (will overflow)
        image = image_int16.astype(np.uint8)
        result = clip_image(image)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_float32_clipping(self):
        """Test float32 image gets clipped to [0, 1]"""
        image = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32)
        result = clip_image(image)
        assert result.dtype == np.float32
        assert result.min() == 0.0
        assert result.max() == 1.0
        np.testing.assert_array_equal(result, [[0.0, 0.5, 1.0]])

    def test_float64_clipping(self):
        """Test float64 image gets clipped to [0, 1]"""
        image = np.array([[-1.0, 0.5, 2.0]], dtype=np.float64)
        result = clip_image(image)
        assert result.dtype == np.float64
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_empty_image(self, empty_image):
        """Test clipping empty image"""
        result = clip_image(empty_image)
        np.testing.assert_array_equal(result, empty_image)

    def test_constant_image(self, constant_image):
        """Test clipping constant value image"""
        result = clip_image(constant_image)
        np.testing.assert_array_equal(result, constant_image)


class TestRegionalMinmaxNormalize:
    """Test regional_minmax_normalize() function"""

    def test_basic_normalization(self):
        """Test basic regional normalization"""
        image = np.array([[0, 50, 100], [50, 100, 150], [100, 150, 200]], dtype=np.uint8)
        epidermis_mask = np.zeros_like(image)
        epidermis_mask[:1, :] = 255
        dermis_mask = np.zeros_like(image)
        dermis_mask[1:, :] = 255

        result = regional_minmax_normalize(image, epidermis_mask, dermis_mask)
        assert result.dtype == np.uint8
        assert result.shape == image.shape

    def test_uniform_region(self, constant_image):
        """Test normalization of uniform region"""
        epidermis_mask = np.full(constant_image.shape, 255, dtype=np.uint8)
        dermis_mask = np.zeros_like(constant_image)

        result = regional_minmax_normalize(constant_image, epidermis_mask, dermis_mask)
        assert result.dtype == np.uint8
        assert result.shape == constant_image.shape

    def test_gradient_normalization(self, gradient_image):
        """Test normalization of gradient image"""
        h, w = gradient_image.shape
        epidermis_mask = np.zeros_like(gradient_image)
        epidermis_mask[:h//2, :] = 255
        dermis_mask = np.zeros_like(gradient_image)
        dermis_mask[h//2:, :] = 255

        result = regional_minmax_normalize(gradient_image, epidermis_mask, dermis_mask)
        assert result.dtype == np.uint8
        assert result.shape == gradient_image.shape

    def test_empty_image(self, empty_image):
        """Test normalization of empty image"""
        epidermis_mask = np.full(empty_image.shape, 255, dtype=np.uint8)
        dermis_mask = np.zeros_like(empty_image)

        result = regional_minmax_normalize(empty_image, epidermis_mask, dermis_mask)
        assert result.dtype == np.uint8

    def test_tiny_image(self, tiny_image):
        """Test normalization of tiny image"""
        epidermis_mask = np.full(tiny_image.shape, 255, dtype=np.uint8)
        dermis_mask = np.zeros_like(tiny_image)

        result = regional_minmax_normalize(tiny_image, epidermis_mask, dermis_mask)
        assert result.dtype == np.uint8
        assert result.shape == tiny_image.shape


class TestUtilsIntegration:
    """Integration tests for utility functions"""

    def test_normalize_denormalize_roundtrip(self):
        """Test normalize then denormalize preserves image"""
        original = np.random.rand(50, 50).astype(np.float32)
        normalized, was_float, original_dtype = normalize_image(original)
        denormalized = denormalize_image(normalized, was_float, original_dtype)

        assert denormalized.dtype == original.dtype
        np.testing.assert_allclose(denormalized, original, atol=0.01)

    def test_rgb_to_grayscale_then_clip(self, rgb_gradient_image):
        """Test RGB conversion followed by clipping"""
        grayscale = ensure_grayscale(rgb_gradient_image)
        clipped = clip_image(grayscale)
        assert clipped.ndim == 2
        assert clipped.dtype == grayscale.dtype
        assert clipped.min() >= 0
        assert clipped.max() <= 255

    def test_full_preprocessing_chain(self, rgb_fiber_image):
        """Test full utility preprocessing chain"""
        # Convert to grayscale
        gray = ensure_grayscale(rgb_fiber_image, extract_channel='green')

        # Normalize
        normalized, was_float, original_dtype = normalize_image(gray)

        # Clip
        clipped = clip_image(normalized)

        # Regional normalization
        epidermis_mask = np.zeros_like(clipped)
        epidermis_mask[:clipped.shape[0]//2, :] = 255
        dermis_mask = np.zeros_like(clipped)
        dermis_mask[clipped.shape[0]//2:, :] = 255
        regional = regional_minmax_normalize(clipped, epidermis_mask, dermis_mask)

        # Validate final result
        assert regional.dtype == np.uint8
        assert regional.shape == rgb_fiber_image.shape[:2]
        assert regional.min() >= 0
        assert regional.max() <= 255
