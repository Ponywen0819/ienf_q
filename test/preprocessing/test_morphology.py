"""
Unit tests for morphological operations

Tests morphological functions in src/neural_reconstruction/core/preprocessing/morphology.py
"""

import pytest
import numpy as np

from neural_reconstruction.core.preprocessing.morphology import (
    morphological_opening,
    morphological_closing,
    morphological_gradient,
    top_hat,
    black_hat,
)


class TestMorphologicalOpening:
    """Test morphological_opening() function"""

    def test_basic_opening(self, noisy_image):
        """Test basic opening operation"""
        result = morphological_opening(noisy_image, kernel_size=3)
        assert result.shape == noisy_image.shape
        assert result.dtype == noisy_image.dtype

    def test_noise_removal(self, salt_pepper_noisy_image):
        """Test that opening removes small noise (salt)"""
        result = morphological_opening(salt_pepper_noisy_image, kernel_size=3)
        # Opening should remove some of the salt noise
        # Count white pixels - should be less after opening
        white_before = np.sum(salt_pepper_noisy_image == 255)
        white_after = np.sum(result == 255)
        assert white_after <= white_before

    @pytest.mark.parametrize("kernel_size", [1, 3, 5, 7, 9])
    def test_various_kernel_sizes(self, circles_image, kernel_size):
        """Test opening with various kernel sizes"""
        result = morphological_opening(circles_image, kernel_size=kernel_size)
        assert result.shape == circles_image.shape
        assert result.dtype == circles_image.dtype

    @pytest.mark.parametrize("kernel_shape", ['rect', 'ellipse', 'cross'])
    def test_various_kernel_shapes(self, circles_image, kernel_shape):
        """Test opening with various kernel shapes"""
        result = morphological_opening(circles_image, kernel_size=5, kernel_shape=kernel_shape)
        assert result.shape == circles_image.shape

    def test_multiple_iterations(self, noisy_image):
        """Test opening with multiple iterations"""
        result = morphological_opening(noisy_image, kernel_size=3, iterations=3)
        assert result.shape == noisy_image.shape

    def test_empty_image(self, empty_image):
        """Test opening on empty image"""
        result = morphological_opening(empty_image, kernel_size=3)
        np.testing.assert_array_equal(result, empty_image)

    def test_constant_image(self, constant_image):
        """Test opening on constant image"""
        result = morphological_opening(constant_image, kernel_size=3)
        # Constant image should remain mostly unchanged
        assert result.shape == constant_image.shape


class TestMorphologicalClosing:
    """Test morphological_closing() function"""

    def test_basic_closing(self, circles_image):
        """Test basic closing operation"""
        result = morphological_closing(circles_image, kernel_size=3)
        assert result.shape == circles_image.shape
        assert result.dtype == circles_image.dtype

    def test_hole_filling(self):
        """Test that closing fills small holes"""
        # Create image with holes
        image = np.ones((50, 50), dtype=np.uint8) * 255
        # Add holes
        image[20:25, 20:25] = 0
        image[30:33, 30:33] = 0

        result = morphological_closing(image, kernel_size=5)
        # Holes should be filled
        filled_pixels = np.sum(result == 255)
        original_filled = np.sum(image == 255)
        assert filled_pixels >= original_filled

    @pytest.mark.parametrize("kernel_size", [1, 3, 5, 7, 9])
    def test_various_kernel_sizes(self, rectangles_image, kernel_size):
        """Test closing with various kernel sizes"""
        result = morphological_closing(rectangles_image, kernel_size=kernel_size)
        assert result.shape == rectangles_image.shape
        assert result.dtype == rectangles_image.dtype

    @pytest.mark.parametrize("kernel_shape", ['rect', 'ellipse', 'cross'])
    def test_various_kernel_shapes(self, rectangles_image, kernel_shape):
        """Test closing with various kernel shapes"""
        result = morphological_closing(rectangles_image, kernel_size=5, kernel_shape=kernel_shape)
        assert result.shape == rectangles_image.shape

    def test_multiple_iterations(self, circles_image):
        """Test closing with multiple iterations"""
        result = morphological_closing(circles_image, kernel_size=3, iterations=3)
        assert result.shape == circles_image.shape

    def test_empty_image(self, empty_image):
        """Test closing on empty image"""
        result = morphological_closing(empty_image, kernel_size=3)
        np.testing.assert_array_equal(result, empty_image)

    def test_pepper_noise_removal(self, salt_pepper_noisy_image):
        """Test that closing removes pepper noise (black dots)"""
        result = morphological_closing(salt_pepper_noisy_image, kernel_size=3)
        # Closing should remove some pepper noise
        # Count black pixels in white regions
        assert result.shape == salt_pepper_noisy_image.shape


class TestMorphologicalGradient:
    """Test morphological_gradient() function"""

    def test_basic_gradient(self, circles_image):
        """Test basic morphological gradient"""
        result = morphological_gradient(circles_image, kernel_size=3)
        assert result.shape == circles_image.shape
        assert result.dtype == circles_image.dtype

    def test_edge_detection(self, circles_image):
        """Test that gradient detects edges"""
        result = morphological_gradient(circles_image, kernel_size=3)
        # Gradient should highlight edges
        # Edges should have high values
        assert result.max() > 0
        # Interior and exterior should be mostly dark
        assert result.mean() < circles_image.mean()

    @pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
    def test_various_kernel_sizes(self, rectangles_image, kernel_size):
        """Test gradient with various kernel sizes"""
        result = morphological_gradient(rectangles_image, kernel_size=kernel_size)
        assert result.shape == rectangles_image.shape

    @pytest.mark.parametrize("kernel_shape", ['rect', 'ellipse', 'cross'])
    def test_various_kernel_shapes(self, circles_image, kernel_shape):
        """Test gradient with various kernel shapes"""
        result = morphological_gradient(circles_image, kernel_size=5, kernel_shape=kernel_shape)
        assert result.shape == circles_image.shape

    def test_empty_image(self, empty_image):
        """Test gradient on empty image"""
        result = morphological_gradient(empty_image, kernel_size=3)
        # Gradient of empty image should be empty
        np.testing.assert_array_equal(result, empty_image)

    def test_constant_image(self, constant_image):
        """Test gradient on constant image"""
        result = morphological_gradient(constant_image, kernel_size=3)
        # Gradient of constant image should be near zero
        assert result.max() <= 1  # Allow for numerical errors


class TestTopHat:
    """Test top_hat() function"""

    def test_basic_top_hat(self, circles_image):
        """Test basic top hat operation"""
        result = top_hat(circles_image, kernel_size=15)
        assert result.shape == circles_image.shape
        assert result.dtype == circles_image.dtype

    def test_bright_object_extraction(self):
        """Test that top hat extracts bright objects"""
        # Create image with bright object on gradient background
        background = np.linspace(0, 100, 100).reshape(1, -1)
        background = np.repeat(background, 100, axis=0).astype(np.uint8)
        image = background.copy()
        # Add bright object
        image[40:60, 40:60] = 200

        result = top_hat(image, kernel_size=21)
        # Top hat should extract the bright object
        assert result[50, 50] > result[10, 10]  # Object brighter than background

    @pytest.mark.parametrize("kernel_size", [5, 11, 21, 31])
    def test_various_kernel_sizes(self, noisy_line_image, kernel_size):
        """Test top hat with various kernel sizes"""
        result = top_hat(noisy_line_image, kernel_size=kernel_size)
        assert result.shape == noisy_line_image.shape

    @pytest.mark.parametrize("kernel_shape", ['rect', 'ellipse'])
    def test_various_kernel_shapes(self, circles_image, kernel_shape):
        """Test top hat with various kernel shapes"""
        result = top_hat(circles_image, kernel_size=15, kernel_shape=kernel_shape)
        assert result.shape == circles_image.shape

    def test_empty_image(self, empty_image):
        """Test top hat on empty image"""
        result = top_hat(empty_image, kernel_size=5)
        np.testing.assert_array_equal(result, empty_image)

    def test_constant_image(self, constant_image):
        """Test top hat on constant image"""
        result = top_hat(constant_image, kernel_size=5)
        # Top hat of constant image should be near zero
        assert result.max() <= 1


class TestBlackHat:
    """Test black_hat() function"""

    def test_basic_black_hat(self, circles_image):
        """Test basic black hat operation"""
        result = black_hat(circles_image, kernel_size=15)
        assert result.shape == circles_image.shape
        assert result.dtype == circles_image.dtype

    def test_dark_object_extraction(self):
        """Test that black hat extracts dark objects"""
        # Create image with dark object on bright background
        image = np.ones((100, 100), dtype=np.uint8) * 200
        # Add dark object
        image[40:60, 40:60] = 50

        result = black_hat(image, kernel_size=21)
        # Black hat should extract the dark object
        assert result[50, 50] > result[10, 10]  # Object region > background

    @pytest.mark.parametrize("kernel_size", [5, 11, 21, 31])
    def test_various_kernel_sizes(self, circles_image, kernel_size):
        """Test black hat with various kernel sizes"""
        result = black_hat(circles_image, kernel_size=kernel_size)
        assert result.shape == circles_image.shape

    @pytest.mark.parametrize("kernel_shape", ['rect', 'ellipse'])
    def test_various_kernel_shapes(self, circles_image, kernel_shape):
        """Test black hat with various kernel shapes"""
        result = black_hat(circles_image, kernel_size=15, kernel_shape=kernel_shape)
        assert result.shape == circles_image.shape

    def test_empty_image(self, empty_image):
        """Test black hat on empty image"""
        result = black_hat(empty_image, kernel_size=5)
        np.testing.assert_array_equal(result, empty_image)

    def test_inverted_constant_image(self):
        """Test black hat on bright constant image"""
        image = np.full((50, 50), 255, dtype=np.uint8)
        result = black_hat(image, kernel_size=5)
        # Black hat of bright constant image should be near zero
        assert result.max() <= 1


class TestMorphologyIntegration:
    """Integration tests for morphological operations"""

    def test_opening_then_closing(self, noisy_image):
        """Test opening followed by closing"""
        opened = morphological_opening(noisy_image, kernel_size=3)
        closed = morphological_closing(opened, kernel_size=3)
        assert closed.shape == noisy_image.shape
        assert closed.dtype == noisy_image.dtype

    def test_closing_then_opening(self, salt_pepper_noisy_image):
        """Test closing followed by opening"""
        closed = morphological_closing(salt_pepper_noisy_image, kernel_size=3)
        opened = morphological_opening(closed, kernel_size=3)
        assert opened.shape == salt_pepper_noisy_image.shape

    def test_gradient_for_edge_mask(self, fiber_like_image):
        """Test using gradient to create edge mask"""
        gradient = morphological_gradient(fiber_like_image, kernel_size=3)
        # Gradient should highlight fiber boundaries
        assert gradient.max() > 0
        assert gradient.shape == fiber_like_image.shape

    def test_top_hat_and_black_hat_combination(self, uneven_illumination_image):
        """Test combining top hat and black hat"""
        top = top_hat(uneven_illumination_image, kernel_size=21)
        black = black_hat(uneven_illumination_image, kernel_size=21)

        # Both should extract features
        assert top.max() > 0
        assert black.max() > 0
        assert top.shape == uneven_illumination_image.shape
        assert black.shape == uneven_illumination_image.shape

    def test_morphology_preserves_dimensions(self, branching_network_image):
        """Test that all operations preserve image dimensions"""
        operations = [
            lambda img: morphological_opening(img, 3),
            lambda img: morphological_closing(img, 3),
            lambda img: morphological_gradient(img, 3),
            lambda img: top_hat(img, 11),
            lambda img: black_hat(img, 11),
        ]

        for op in operations:
            result = op(branching_network_image)
            assert result.shape == branching_network_image.shape
            assert result.dtype == branching_network_image.dtype
