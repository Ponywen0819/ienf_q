"""
Unit tests for mask operations

Tests mask manipulation functions in src/neural_reconstruction/core/preprocessing/mask_operations.py
"""

import pytest
import numpy as np

from neural_reconstruction.core.preprocessing.mask_operations import (
    dilate_epidermis_vertically,
    apply_mask,
    invert_mask,
    combine_masks_or,
)


class TestDilateEpidermisVertically:
    """Test dilate_epidermis_vertically() function"""

    def test_basic_vertical_dilation(self, epidermis_mask):
        """Test basic vertical dilation"""
        result = dilate_epidermis_vertically(epidermis_mask, offset_px=10)
        assert result.shape == epidermis_mask.shape
        assert result.dtype == epidermis_mask.dtype

        # Dilated region should be larger
        original_white = np.sum(epidermis_mask == 255)
        dilated_white = np.sum(result == 255)
        assert dilated_white >= original_white

    @pytest.mark.parametrize("offset", [0, 1, 10, 50, 100])
    def test_various_offset_values(self, epidermis_mask, offset):
        """Test dilation with various offset values"""
        result = dilate_epidermis_vertically(epidermis_mask, offset_px=offset)
        assert result.shape == epidermis_mask.shape

        if offset == 0:
            # No dilation
            np.testing.assert_array_equal(result, epidermis_mask)
        else:
            # Should dilate
            dilated_white = np.sum(result == 255)
            original_white = np.sum(epidermis_mask == 255)
            assert dilated_white >= original_white

    def test_large_offset(self, epidermis_mask):
        """Test dilation with very large offset"""
        # Large offset might cover entire image
        result = dilate_epidermis_vertically(epidermis_mask, offset_px=500)
        assert result.shape == epidermis_mask.shape
        # Should have significant white region
        assert np.sum(result == 255) > epidermis_mask.size * 0.5

    def test_vertical_only_dilation(self):
        """Test that dilation is vertical only (not horizontal)"""
        # Create horizontal line
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 40:60] = 255  # Horizontal line in center

        result = dilate_epidermis_vertically(mask, offset_px=10)

        # Check vertical dilation occurred
        assert np.any(result[40, 40:60] == 255)  # Above line
        assert np.any(result[60, 40:60] == 255)  # Below line

        # Check horizontal extent remains same
        # Left and right of the line should remain black
        assert np.all(result[50, :30] == 0)  # Far left
        assert np.all(result[50, 70:] == 0)  # Far right

    def test_boundary_handling_top(self):
        """Test dilation near top boundary"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5, :] = 255  # Line near top

        result = dilate_epidermis_vertically(mask, offset_px=10)
        # Should not crash and handle top boundary
        assert result.shape == mask.shape
        # Top rows should have dilation
        assert np.any(result[0, :] == 255)

    def test_boundary_handling_bottom(self):
        """Test dilation near bottom boundary"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[95, :] = 255  # Line near bottom

        result = dilate_epidermis_vertically(mask, offset_px=10)
        # Should not crash and handle bottom boundary
        assert result.shape == mask.shape
        # Bottom rows should have dilation
        assert np.any(result[-1, :] == 255)

    def test_empty_mask(self, empty_image):
        """Test dilation on empty mask"""
        result = dilate_epidermis_vertically(empty_image, offset_px=10)
        # Empty mask should remain empty
        np.testing.assert_array_equal(result, empty_image)

    def test_full_mask(self):
        """Test dilation on full white mask"""
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = dilate_epidermis_vertically(mask, offset_px=10)
        # Full mask should remain full
        np.testing.assert_array_equal(result, mask)

    def test_irregular_mask(self, irregular_mask):
        """Test dilation on irregular mask"""
        result = dilate_epidermis_vertically(irregular_mask, offset_px=20)
        assert result.shape == irregular_mask.shape
        # Should have more white pixels
        assert np.sum(result == 255) >= np.sum(irregular_mask == 255)


class TestApplyMask:
    """Test apply_mask() function"""

    def test_basic_mask_application(self, circles_image, simple_binary_mask):
        """Test basic mask application"""
        result = apply_mask(circles_image, simple_binary_mask)
        assert result.shape == circles_image.shape
        assert result.dtype == circles_image.dtype

        # Masked region (black in mask) should be black in result
        assert np.all(result[simple_binary_mask == 0] == 0)

    def test_mask_preserves_unmasked_region(self, circles_image, simple_binary_mask):
        """Test that unmasked region is preserved"""
        result = apply_mask(circles_image, simple_binary_mask)

        # Unmasked region (white in mask) should be unchanged
        unmasked_indices = simple_binary_mask == 255
        np.testing.assert_array_equal(result[unmasked_indices],
                                     circles_image[unmasked_indices])

    def test_full_mask(self, circles_image):
        """Test applying full white mask (no masking)"""
        mask = np.full(circles_image.shape, 255, dtype=np.uint8)
        result = apply_mask(circles_image, mask)
        # Should be identical to original
        np.testing.assert_array_equal(result, circles_image)

    def test_empty_mask(self, circles_image, empty_image):
        """Test applying empty mask (mask everything)"""
        result = apply_mask(circles_image, empty_image)
        # Should be all black
        np.testing.assert_array_equal(result, empty_image)

    def test_irregular_mask(self, fiber_like_image, irregular_mask):
        """Test applying irregular mask"""
        result = apply_mask(fiber_like_image, irregular_mask)
        assert result.shape == fiber_like_image.shape

        # Masked areas should be black
        assert np.all(result[irregular_mask == 0] == 0)

    def test_mask_on_gradient(self, gradient_image, epidermis_mask):
        """Test mask on gradient image"""
        result = apply_mask(gradient_image, epidermis_mask)
        assert result.shape == gradient_image.shape

    def test_shape_mismatch(self, circles_image):
        """Test that shape mismatch raises error"""
        wrong_mask = np.ones((50, 50), dtype=np.uint8) * 255
        with pytest.raises((ValueError, AssertionError)):
            apply_mask(circles_image, wrong_mask)

    def test_empty_image_with_mask(self, empty_image, simple_binary_mask):
        """Test applying mask to empty image"""
        result = apply_mask(empty_image, simple_binary_mask)
        # Should remain empty
        np.testing.assert_array_equal(result, empty_image)


class TestInvertMask:
    """Test invert_mask() function"""

    def test_basic_inversion(self, simple_binary_mask):
        """Test basic mask inversion"""
        result = invert_mask(simple_binary_mask)
        assert result.shape == simple_binary_mask.shape
        assert result.dtype == simple_binary_mask.dtype

        # Black should become white and vice versa
        assert np.all(result[simple_binary_mask == 0] == 255)
        assert np.all(result[simple_binary_mask == 255] == 0)

    def test_double_inversion(self, epidermis_mask):
        """Test that double inversion returns to original"""
        inverted_once = invert_mask(epidermis_mask)
        inverted_twice = invert_mask(inverted_once)

        # Should be equal to original
        np.testing.assert_array_equal(inverted_twice, epidermis_mask)

    def test_empty_mask_inversion(self, empty_image):
        """Test inversion of empty mask"""
        result = invert_mask(empty_image)
        # Should become all white
        expected = np.full(empty_image.shape, 255, dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_full_mask_inversion(self):
        """Test inversion of full white mask"""
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = invert_mask(mask)
        # Should become all black
        expected = np.zeros((100, 100), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_irregular_mask_inversion(self, irregular_mask):
        """Test inversion of irregular mask"""
        result = invert_mask(irregular_mask)
        assert result.shape == irregular_mask.shape

        # Count inversions
        original_white = np.sum(irregular_mask == 255)
        result_black = np.sum(result == 0)
        assert original_white == result_black

    def test_multi_region_mask_inversion(self, multi_region_mask):
        """Test inversion of multi-region mask"""
        result = invert_mask(multi_region_mask)
        assert result.shape == multi_region_mask.shape

        # White regions should become black
        assert np.all(result[multi_region_mask == 255] == 0)
        # Black regions should become white
        assert np.all(result[multi_region_mask == 0] == 255)


class TestCombineMasksOr:
    """Test combine_masks_or() function"""

    def test_basic_or_combination(self, simple_binary_mask):
        """Test basic OR combination of two masks"""
        # Create second mask with different pattern
        mask2 = np.zeros_like(simple_binary_mask)
        mask2[50:, :] = 255  # Bottom half

        result = combine_masks_or(simple_binary_mask, mask2)
        assert result.shape == simple_binary_mask.shape
        assert result.dtype == np.uint8

        # Result should have white pixels where either mask is white
        assert np.all(result[simple_binary_mask == 255] == 255)
        assert np.all(result[mask2 == 255] == 255)

    def test_or_with_identical_masks(self, epidermis_mask):
        """Test OR combination with identical masks"""
        result = combine_masks_or(epidermis_mask, epidermis_mask)
        # Should be identical to original
        np.testing.assert_array_equal(result, epidermis_mask)

    def test_or_with_empty_mask(self, circles_image, empty_image):
        """Test OR combination with empty mask"""
        result = combine_masks_or(circles_image, empty_image)
        # Should preserve first mask
        np.testing.assert_array_equal(result, circles_image)

        # Test reverse order
        result2 = combine_masks_or(empty_image, circles_image)
        np.testing.assert_array_equal(result2, circles_image)

    def test_or_with_full_masks(self):
        """Test OR combination with full white masks"""
        mask1 = np.full((100, 100), 255, dtype=np.uint8)
        mask2 = np.full((100, 100), 255, dtype=np.uint8)

        result = combine_masks_or(mask1, mask2)
        # Should be all white
        np.testing.assert_array_equal(result, mask1)

    def test_or_non_overlapping_masks(self):
        """Test OR combination of non-overlapping masks"""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[:50, :] = 255  # Top half

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[50:, :] = 255  # Bottom half

        result = combine_masks_or(mask1, mask2)
        # Should cover entire image
        assert np.all(result == 255)

    def test_or_overlapping_masks(self):
        """Test OR combination of overlapping masks"""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:60, 20:60] = 255  # Center square

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[40:80, 40:80] = 255  # Offset square

        result = combine_masks_or(mask1, mask2)
        # Union should contain both squares
        assert np.any(result[30, 30] == 255)  # In mask1 only
        assert np.any(result[50, 50] == 255)  # In both
        assert np.any(result[70, 70] == 255)  # In mask2 only

    def test_or_irregular_masks(self, irregular_mask, multi_region_mask):
        """Test OR combination of irregular masks"""
        result = combine_masks_or(irregular_mask, multi_region_mask)
        assert result.shape == irregular_mask.shape

        # Should have at least as many white pixels as either mask
        result_white = np.sum(result == 255)
        mask1_white = np.sum(irregular_mask == 255)
        mask2_white = np.sum(multi_region_mask == 255)

        assert result_white >= mask1_white
        assert result_white >= mask2_white

    def test_shape_mismatch(self, simple_binary_mask):
        """Test that shape mismatch raises error"""
        wrong_mask = np.ones((50, 50), dtype=np.uint8) * 255
        with pytest.raises((ValueError, AssertionError)):
            combine_masks_or(simple_binary_mask, wrong_mask)

    def test_or_commutative(self, epidermis_mask, irregular_mask):
        """Test that OR operation is commutative"""
        result1 = combine_masks_or(epidermis_mask, irregular_mask)
        result2 = combine_masks_or(irregular_mask, epidermis_mask)

        # Results should be identical regardless of order
        np.testing.assert_array_equal(result1, result2)


class TestMaskOperationsIntegration:
    """Integration tests for mask operations"""

    def test_dilate_then_apply_mask(self, circles_image, epidermis_mask):
        """Test dilation followed by mask application"""
        # Dilate mask
        dilated_mask = dilate_epidermis_vertically(epidermis_mask, offset_px=20)

        # Apply dilated mask
        result = apply_mask(circles_image, dilated_mask)

        assert result.shape == circles_image.shape
        # Masked area should be black
        assert np.all(result[dilated_mask == 0] == 0)

    def test_invert_then_apply_mask(self, fiber_like_image, simple_binary_mask):
        """Test mask inversion followed by application"""
        # Invert mask
        inverted_mask = invert_mask(simple_binary_mask)

        # Apply inverted mask
        result = apply_mask(fiber_like_image, inverted_mask)

        # Regions that were white in original mask should now be masked
        assert np.all(result[simple_binary_mask == 255] == 0)

    def test_combine_then_apply(self, gradient_image, epidermis_mask, irregular_mask):
        """Test mask combination followed by application"""
        # Combine masks
        combined_mask = combine_masks_or(epidermis_mask, irregular_mask)

        # Apply combined mask
        result = apply_mask(gradient_image, combined_mask)

        assert result.shape == gradient_image.shape
        # Combined masked area should be black
        assert np.all(result[combined_mask == 0] == 0)

    def test_dilate_combine_invert_apply(self, circles_image, epidermis_mask, irregular_mask):
        """Test full pipeline: dilate, combine, invert, apply"""
        # Dilate epidermis mask
        dilated = dilate_epidermis_vertically(epidermis_mask, offset_px=15)

        # Combine with irregular mask
        combined = combine_masks_or(dilated, irregular_mask)

        # Invert to mask the opposite region
        inverted = invert_mask(combined)

        # Apply to image
        result = apply_mask(circles_image, inverted)

        # Validate final result
        assert result.shape == circles_image.shape
        assert result.dtype == circles_image.dtype
        assert np.all(result[inverted == 0] == 0)

    def test_multiple_mask_combinations(self, simple_binary_mask, epidermis_mask, irregular_mask):
        """Test combining multiple masks"""
        # Combine first two masks
        combined1 = combine_masks_or(simple_binary_mask, epidermis_mask)

        # Combine with third mask
        combined2 = combine_masks_or(combined1, irregular_mask)

        # Result should contain all regions
        assert np.any(combined2[simple_binary_mask == 255] == 255)
        assert np.any(combined2[epidermis_mask == 255] == 255)
        assert np.any(combined2[irregular_mask == 255] == 255)

    def test_mask_chain_preserves_dtype(self, circles_image, epidermis_mask, irregular_mask):
        """Test that mask operation chain preserves dtype"""
        dilated = dilate_epidermis_vertically(epidermis_mask, offset_px=10)
        combined = combine_masks_or(dilated, irregular_mask)
        inverted = invert_mask(combined)
        result = apply_mask(circles_image, inverted)

        # All intermediate results should be uint8
        assert dilated.dtype == np.uint8
        assert combined.dtype == np.uint8
        assert inverted.dtype == np.uint8
        assert result.dtype == np.uint8
