"""
Unit tests for SkinAnalysisPipeline

Tests the main pipeline class in src/neural_reconstruction/core/preprocessing/pipeline.py
"""

import pytest
import numpy as np

from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline


class TestPipelineInitialization:
    """Test SkinAnalysisPipeline initialization"""

    def test_default_initialization(self, default_config):
        """Test pipeline with default configuration"""
        pipeline = SkinAnalysisPipeline(default_config)
        assert pipeline.config is not None
        assert pipeline.config == default_config

    def test_rolling_ball_initialization(self, rolling_ball_config):
        """Test pipeline with rolling ball configuration"""
        pipeline = SkinAnalysisPipeline(rolling_ball_config)
        assert pipeline.config['background']['method'] == 'rolling_ball'

    def test_gaussian_initialization(self, gaussian_config):
        """Test pipeline with Gaussian configuration"""
        pipeline = SkinAnalysisPipeline(gaussian_config)
        assert pipeline.config['background']['method'] == 'gaussian'

    def test_invalid_config(self, invalid_config):
        """Test pipeline initialization with invalid config"""
        with pytest.raises((ValueError, KeyError, AssertionError)):
            SkinAnalysisPipeline(invalid_config)

    def test_missing_morphology_config(self):
        """Test initialization with missing morphology config"""
        config = {
            'mask': {'dilate_offset': 50},
            'background': {'method': 'morphology', 'radius': 12},
        }
        with pytest.raises((ValueError, KeyError, AssertionError)):
            SkinAnalysisPipeline(config)

    def test_missing_background_config(self):
        """Test initialization with missing background config"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
        }
        with pytest.raises((ValueError, KeyError, AssertionError)):
            SkinAnalysisPipeline(config)


class TestPipelineRun:
    """Test SkinAnalysisPipeline.run() method"""

    def test_basic_pipeline_run(self, default_pipeline, circles_image,
                                epidermis_mask, gradient_image,
                                pipeline_output_validator):
        """Test basic pipeline execution"""
        final_label, roi_image = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        assert pipeline_output_validator((final_label, roi_image))

    def test_pipeline_with_debug(self, default_pipeline, circles_image,
                                 epidermis_mask, gradient_image,
                                 pipeline_output_validator, debug_output_validator):
        """Test pipeline execution with debug mode"""
        final_label, roi_image, debug_output = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image,
            debug=True
        )

        assert pipeline_output_validator((final_label, roi_image, debug_output), expect_debug=True)
        assert debug_output_validator(debug_output)

    def test_pipeline_rolling_ball(self, rolling_ball_pipeline, fiber_like_image,
                                   epidermis_mask, uneven_illumination_image):
        """Test pipeline with rolling ball background correction"""
        final_label, roi_image = rolling_ball_pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        assert final_label.shape == fiber_like_image.shape
        assert roi_image.shape == fiber_like_image.shape

    def test_pipeline_gaussian(self, gaussian_pipeline, branching_network_image,
                               irregular_mask, uneven_illumination_image):
        """Test pipeline with Gaussian background correction"""
        final_label, roi_image = gaussian_pipeline.run(
            label_image=branching_network_image,
            epidermis_mask=irregular_mask,
            original_image=uneven_illumination_image
        )

        assert final_label.shape == branching_network_image.shape
        assert roi_image.shape == branching_network_image.shape

    def test_pipeline_preserves_shapes(self, default_pipeline, rectangles_image,
                                       simple_binary_mask, gradient_image):
        """Test that pipeline preserves image shapes"""
        final_label, roi_image = default_pipeline.run(
            label_image=rectangles_image,
            epidermis_mask=simple_binary_mask,
            original_image=gradient_image
        )

        assert final_label.shape == rectangles_image.shape
        assert roi_image.shape == rectangles_image.shape

    def test_pipeline_with_rgb_image(self, default_pipeline, circles_image,
                                     epidermis_mask, rgb_fiber_image):
        """Test pipeline with RGB original image"""
        final_label, roi_image = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=rgb_fiber_image
        )

        # Should convert RGB to grayscale and process
        assert final_label.shape == circles_image.shape
        assert roi_image.ndim == 2  # Should be grayscale output

    def test_pipeline_empty_label(self, default_pipeline, empty_image,
                                  epidermis_mask, gradient_image):
        """Test pipeline with empty label image"""
        final_label, roi_image = default_pipeline.run(
            label_image=empty_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        # Empty label should remain mostly empty
        assert final_label.shape == empty_image.shape
        assert np.sum(final_label) < empty_image.size * 0.1  # Allow some processing artifacts

    def test_pipeline_empty_mask(self, default_pipeline, circles_image,
                                 empty_image, gradient_image):
        """Test pipeline with empty epidermis mask"""
        final_label, roi_image = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=empty_image,
            original_image=gradient_image
        )

        # Should still process, mask just won't affect anything
        assert final_label.shape == circles_image.shape

    def test_shape_mismatch(self, default_pipeline):
        """Test that shape mismatch raises error"""
        label = np.zeros((100, 100), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        image = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises((ValueError, AssertionError)):
            default_pipeline.run(label_image=label,
                               epidermis_mask=mask,
                               original_image=image)


class TestPipelineComponents:
    """Test individual pipeline components"""

    def test_label_processing(self, default_pipeline, noisy_image):
        """Test label processing path (morphological operations)"""
        # Access internal method (if available) or test through full pipeline
        # For now, test through full pipeline with debug mode
        mask = np.full(noisy_image.shape, 255, dtype=np.uint8)
        original = noisy_image.copy()

        final_label, roi_image, debug_output = default_pipeline.run(
            label_image=noisy_image,
            epidermis_mask=mask,
            original_image=original,
            debug=True
        )

        # Check that morphological operations were applied
        assert debug_output.label_after_closing is not None
        assert debug_output.label_after_opening is not None

    def test_mask_dilation(self, default_pipeline, circles_image, epidermis_mask, gradient_image):
        """Test mask dilation path"""
        final_label, roi_image, debug_output = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image,
            debug=True
        )

        # Check that mask was dilated
        assert debug_output.dilated_mask is not None
        dilated_white = np.sum(debug_output.dilated_mask == 255)
        original_white = np.sum(epidermis_mask == 255)
        # Dilated should have more white pixels
        assert dilated_white >= original_white

    def test_background_correction(self, default_pipeline, circles_image,
                                   epidermis_mask, uneven_illumination_image):
        """Test background correction path"""
        final_label, roi_image, debug_output = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image,
            debug=True
        )

        # Check that background correction was applied
        assert debug_output.background_corrected is not None
        assert debug_output.background_corrected.shape == uneven_illumination_image.shape

    def test_pseudo_label_generation(self, default_pipeline, circles_image,
                                     epidermis_mask, gradient_image):
        """Test pseudo-label generation"""
        final_label, roi_image, debug_output = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image,
            debug=True
        )

        # Check that pseudo label was generated
        assert debug_output.pseudo_label is not None
        assert debug_output.pseudo_label.dtype == np.uint8

    def test_label_merging(self, default_pipeline, circles_image,
                          epidermis_mask, gradient_image):
        """Test label merging (OR combination)"""
        final_label, roi_image, debug_output = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image,
            debug=True
        )

        # Final label should be combination of label path and pseudo label
        # Should have at least as many white pixels as the label
        final_white = np.sum(final_label == 255)
        label_white = np.sum(debug_output.label_after_opening == 255)
        assert final_white >= label_white


class TestPipelineConfiguration:
    """Test pipeline with various configurations"""

    def test_large_morphology_kernels(self, circles_image, epidermis_mask, gradient_image):
        """Test pipeline with large morphology kernels"""
        config = {
            'morphology': {'closing_kernel': 9, 'opening_kernel': 7},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'morphology', 'radius': 12, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }
        pipeline = SkinAnalysisPipeline(config)

        final_label, roi_image = pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        assert final_label.shape == circles_image.shape

    def test_small_morphology_kernels(self, circles_image, epidermis_mask, gradient_image):
        """Test pipeline with small morphology kernels"""
        config = {
            'morphology': {'closing_kernel': 1, 'opening_kernel': 1},
            'mask': {'dilate_offset': 10},
            'background': {'method': 'morphology', 'radius': 5, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }
        pipeline = SkinAnalysisPipeline(config)

        final_label, roi_image = pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        assert final_label.shape == circles_image.shape

    def test_large_dilation_offset(self, fiber_like_image, epidermis_mask, gradient_image):
        """Test pipeline with large dilation offset"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 200},
            'background': {'method': 'morphology', 'radius': 12, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }
        pipeline = SkinAnalysisPipeline(config)

        final_label, roi_image = pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        assert final_label.shape == fiber_like_image.shape

    def test_zero_dilation_offset(self, circles_image, epidermis_mask, gradient_image):
        """Test pipeline with zero dilation offset"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 0},
            'background': {'method': 'morphology', 'radius': 12, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }
        pipeline = SkinAnalysisPipeline(config)

        final_label, roi_image = pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        assert final_label.shape == circles_image.shape

    def test_normalization_enabled(self, fiber_like_image, epidermis_mask,
                                   uneven_illumination_image):
        """Test pipeline with normalization enabled"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'gaussian', 'sigma': 10.0, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': True}
        }
        pipeline = SkinAnalysisPipeline(config)

        final_label, roi_image = pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        assert final_label.shape == fiber_like_image.shape


class TestPipelineEdgeCases:
    """Test pipeline edge cases"""

    def test_tiny_images(self, default_pipeline, tiny_image):
        """Test pipeline with tiny images"""
        mask = np.full(tiny_image.shape, 255, dtype=np.uint8)
        original = tiny_image.copy()

        final_label, roi_image = default_pipeline.run(
            label_image=tiny_image,
            epidermis_mask=mask,
            original_image=original
        )

        assert final_label.shape == tiny_image.shape

    def test_constant_images(self, default_pipeline, constant_image):
        """Test pipeline with constant value images"""
        mask = np.full(constant_image.shape, 255, dtype=np.uint8)
        original = constant_image.copy()

        final_label, roi_image = default_pipeline.run(
            label_image=constant_image,
            epidermis_mask=mask,
            original_image=original
        )

        assert final_label.shape == constant_image.shape

    def test_all_white_label(self, default_pipeline, epidermis_mask, gradient_image):
        """Test pipeline with all-white label"""
        label = np.full((100, 100), 255, dtype=np.uint8)

        final_label, roi_image = default_pipeline.run(
            label_image=label,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        # Should have significant white regions
        assert np.sum(final_label == 255) > label.size * 0.5

    def test_complex_irregular_mask(self, default_pipeline, branching_network_image,
                                   irregular_mask, uneven_illumination_image):
        """Test pipeline with complex irregular mask"""
        final_label, roi_image = default_pipeline.run(
            label_image=branching_network_image,
            epidermis_mask=irregular_mask,
            original_image=uneven_illumination_image
        )

        assert final_label.shape == branching_network_image.shape


class TestPipelineRobustness:
    """Test pipeline robustness and error handling"""

    def test_multiple_runs_same_pipeline(self, default_pipeline, circles_image,
                                        epidermis_mask, gradient_image):
        """Test running pipeline multiple times"""
        result1 = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        result2 = default_pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        # Results should be identical for same inputs
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])

    def test_different_images_same_pipeline(self, default_pipeline, epidermis_mask, gradient_image):
        """Test same pipeline with different images"""
        images = [
            np.zeros((100, 100), dtype=np.uint8),
            np.full((100, 100), 128, dtype=np.uint8),
            np.random.randint(0, 256, (100, 100), dtype=np.uint8),
        ]

        for img in images:
            final_label, roi_image = default_pipeline.run(
                label_image=img,
                epidermis_mask=epidermis_mask,
                original_image=gradient_image
            )
            assert final_label.shape == img.shape
