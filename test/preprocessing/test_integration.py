"""
Integration tests for preprocessing module

End-to-end tests combining preprocessing with reconstruction and testing
realistic scenarios.
"""

import pytest
import numpy as np

from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline


class TestEndToEndPipeline:
    """Test complete preprocessing pipeline end-to-end"""

    def test_full_pipeline_with_realistic_data(self, fiber_like_image,
                                               epidermis_mask, uneven_illumination_image):
        """Test full pipeline with realistic fiber-like data"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'rolling_ball', 'radius': 20, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        # Validate output
        assert final_label.shape == fiber_like_image.shape
        assert roi_image.shape == fiber_like_image.shape
        assert final_label.dtype == np.uint8
        assert roi_image.dtype == np.uint8

        # Should have some foreground (fibers)
        assert np.sum(final_label == 255) > 0
        assert np.sum(roi_image > 0) > 0

    def test_full_pipeline_with_branching_network(self, branching_network_image,
                                                  irregular_mask, uneven_illumination_image):
        """Test full pipeline with branching network structure"""
        config = {
            'morphology': {'closing_kernel': 5, 'opening_kernel': 3},
            'mask': {'dilate_offset': 100},
            'background': {'method': 'gaussian', 'sigma': 15.0, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': True}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=branching_network_image,
            epidermis_mask=irregular_mask,
            original_image=uneven_illumination_image
        )

        # Validate network structure is preserved
        assert final_label.shape == branching_network_image.shape
        assert np.sum(final_label == 255) > 0  # Network should be present

    def test_pipeline_with_rgb_input(self, fiber_like_image, epidermis_mask, rgb_fiber_image):
        """Test pipeline with RGB original image input"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'morphology', 'radius': 20, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=rgb_fiber_image
        )

        # Should handle RGB input and convert to grayscale
        assert final_label.shape == fiber_like_image.shape
        assert roi_image.ndim == 2  # Output should be grayscale


class TestConfigurationVariations:
    """Test pipeline with various configuration combinations"""

    @pytest.mark.parametrize("closing_kernel,opening_kernel", [
        (1, 1), (3, 3), (5, 3), (3, 5), (7, 5)
    ])
    def test_morphology_kernel_combinations(self, circles_image, epidermis_mask,
                                           gradient_image, closing_kernel, opening_kernel):
        """Test various morphology kernel combinations"""
        config = {
            'morphology': {'closing_kernel': closing_kernel, 'opening_kernel': opening_kernel},
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

    @pytest.mark.parametrize("method,radius", [
        ('morphology', 10),
        ('morphology', 30),
        ('rolling_ball', 10),
        ('rolling_ball', 30),
    ])
    def test_background_method_variations(self, fiber_like_image, epidermis_mask,
                                         uneven_illumination_image, method, radius):
        """Test various background correction methods and radii"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': method, 'radius': radius, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        assert final_label.shape == fiber_like_image.shape

    @pytest.mark.parametrize("sigma", [5.0, 10.0, 20.0, 50.0])
    def test_gaussian_sigma_variations(self, circles_image, epidermis_mask,
                                      uneven_illumination_image, sigma):
        """Test various Gaussian sigma values"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'gaussian', 'sigma': sigma, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=circles_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        assert final_label.shape == circles_image.shape

    @pytest.mark.parametrize("dilate_offset", [0, 10, 50, 100, 200])
    def test_dilation_offset_variations(self, fiber_like_image, epidermis_mask,
                                       gradient_image, dilate_offset):
        """Test various dilation offset values"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': dilate_offset},
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


class TestPreprocessingToReconstruction:
    """Test preprocessing as preparation for reconstruction"""

    def test_preprocessing_output_suitable_for_reconstruction(self, fiber_like_image,
                                                              epidermis_mask,
                                                              uneven_illumination_image):
        """Test that preprocessing output is suitable for reconstruction"""
        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'rolling_ball', 'radius': 20, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        # Check outputs are suitable for reconstruction
        # 1. Binary label (0 or 255)
        unique_labels = np.unique(final_label)
        assert all(v in [0, 255] for v in unique_labels)

        # 2. ROI image is grayscale uint8
        assert roi_image.dtype == np.uint8
        assert roi_image.ndim == 2

        # 3. Both images same shape
        assert final_label.shape == roi_image.shape

        # 4. ROI has reasonable intensity range
        assert roi_image.min() >= 0
        assert roi_image.max() <= 255

    def test_preprocessing_preserves_fiber_connectivity(self, branching_network_image,
                                                       epidermis_mask, gradient_image):
        """Test that preprocessing preserves fiber connectivity"""
        config = {
            'morphology': {'closing_kernel': 5, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'morphology', 'radius': 15, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=branching_network_image,
            epidermis_mask=epidermis_mask,
            original_image=gradient_image
        )

        # Count connected components before and after
        from scipy.ndimage import label as scipy_label

        original_labeled, original_count = scipy_label(branching_network_image > 0)
        final_labeled, final_count = scipy_label(final_label > 0)

        # After preprocessing, should have similar or fewer components
        # (closing should merge nearby components)
        assert final_count <= original_count + 5  # Allow some tolerance


class TestRealWorldScenarios:
    """Test realistic real-world scenarios"""

    def test_low_contrast_image(self, fiber_like_image, epidermis_mask):
        """Test pipeline with low contrast image"""
        # Create low contrast image
        low_contrast = (fiber_like_image.astype(np.float32) * 0.3 + 50).astype(np.uint8)

        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'rolling_ball', 'radius': 20, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': True}  # Enable normalization for low contrast
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=low_contrast
        )

        # Should still produce reasonable output
        assert final_label.shape == fiber_like_image.shape
        assert np.sum(final_label == 255) > 0

    def test_noisy_input_images(self, noisy_image, epidermis_mask, salt_pepper_noisy_image):
        """Test pipeline with noisy input images"""
        config = {
            'morphology': {'closing_kernel': 5, 'opening_kernel': 5},  # Larger kernels for noise
            'mask': {'dilate_offset': 50},
            'background': {'method': 'morphology', 'radius': 20, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=noisy_image,
            epidermis_mask=epidermis_mask,
            original_image=salt_pepper_noisy_image
        )

        # Morphological operations should clean up noise
        assert final_label.shape == noisy_image.shape

    def test_extreme_uneven_illumination(self, fiber_like_image, epidermis_mask):
        """Test pipeline with extreme uneven illumination"""
        # Create extreme illumination gradient
        h, w = fiber_like_image.shape
        illumination = np.linspace(50, 250, w, dtype=np.uint8)
        illumination = np.tile(illumination, (h, 1))

        config = {
            'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
            'mask': {'dilate_offset': 50},
            'background': {'method': 'rolling_ball', 'radius': 50, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': True}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=illumination
        )

        # Should handle extreme illumination
        assert final_label.shape == fiber_like_image.shape

    def test_sparse_annotations(self, epidermis_mask, uneven_illumination_image):
        """Test pipeline with very sparse label annotations"""
        # Create sparse labels (only a few pixels)
        sparse_label = np.zeros((200, 200), dtype=np.uint8)
        sparse_label[50, 50] = 255
        sparse_label[100, 100] = 255
        sparse_label[150, 150] = 255

        config = {
            'morphology': {'closing_kernel': 7, 'opening_kernel': 3},  # Large closing to connect
            'mask': {'dilate_offset': 50},
            'background': {'method': 'rolling_ball', 'radius': 20, 'light_background': True},
            'threshold': {'method': 'binary'},
            'normalization': {'enabled': False}
        }

        pipeline = SkinAnalysisPipeline(config)
        final_label, roi_image = pipeline.run(
            label_image=sparse_label,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        # Closing should fill in gaps
        final_white = np.sum(final_label == 255)
        original_white = np.sum(sparse_label == 255)
        assert final_white >= original_white


class TestDebugOutputCompleteness:
    """Test debug output provides complete information"""

    def test_debug_output_has_all_stages(self, default_pipeline, fiber_like_image,
                                        epidermis_mask, uneven_illumination_image,
                                        debug_output_validator):
        """Test that debug output contains all processing stages"""
        final_label, roi_image, debug_output = default_pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image,
            debug=True
        )

        # Validate debug output has all required fields
        assert debug_output_validator(debug_output)

        # Check individual stages
        assert debug_output.label_after_closing.shape == fiber_like_image.shape
        assert debug_output.label_after_opening.shape == fiber_like_image.shape
        assert debug_output.dilated_mask.shape == epidermis_mask.shape
        assert debug_output.roi_from_original.shape == uneven_illumination_image.shape
        assert debug_output.background_corrected.shape == uneven_illumination_image.shape
        assert debug_output.pseudo_label.shape == fiber_like_image.shape

    def test_debug_output_stages_are_different(self, default_pipeline, noisy_image,
                                               epidermis_mask, uneven_illumination_image):
        """Test that debug output stages show progression"""
        final_label, roi_image, debug_output = default_pipeline.run(
            label_image=noisy_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image,
            debug=True
        )

        # Stages should be different (processing is happening)
        assert not np.array_equal(debug_output.label_after_closing, noisy_image)
        # Opening after closing should also be different
        assert not np.array_equal(debug_output.label_after_closing,
                                 debug_output.label_after_opening)


class TestPerformance:
    """Test pipeline performance characteristics"""

    def test_pipeline_handles_large_images(self, default_pipeline, epidermis_mask):
        """Test pipeline can handle large images"""
        # Create large image (500x500)
        large_label = np.random.randint(0, 2, (500, 500), dtype=np.uint8) * 255
        large_mask = np.full((500, 500), 255, dtype=np.uint8)
        large_original = np.random.randint(0, 256, (500, 500), dtype=np.uint8)

        # Should complete without crashing or excessive time
        final_label, roi_image = default_pipeline.run(
            label_image=large_label,
            epidermis_mask=large_mask,
            original_image=large_original
        )

        assert final_label.shape == (500, 500)

    def test_pipeline_deterministic(self, default_pipeline, fiber_like_image,
                                   epidermis_mask, uneven_illumination_image):
        """Test that pipeline produces deterministic results"""
        # Run twice with same inputs
        result1 = default_pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        result2 = default_pipeline.run(
            label_image=fiber_like_image,
            epidermis_mask=epidermis_mask,
            original_image=uneven_illumination_image
        )

        # Results should be identical
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])


class TestErrorRecovery:
    """Test pipeline error recovery and robustness"""

    def test_pipeline_with_all_zero_inputs(self, default_pipeline):
        """Test pipeline with all-zero inputs"""
        zero_image = np.zeros((100, 100), dtype=np.uint8)

        final_label, roi_image = default_pipeline.run(
            label_image=zero_image,
            epidermis_mask=zero_image,
            original_image=zero_image
        )

        # Should not crash
        assert final_label.shape == (100, 100)
        assert roi_image.shape == (100, 100)

    def test_pipeline_with_all_white_inputs(self, default_pipeline):
        """Test pipeline with all-white inputs"""
        white_image = np.full((100, 100), 255, dtype=np.uint8)

        final_label, roi_image = default_pipeline.run(
            label_image=white_image,
            epidermis_mask=white_image,
            original_image=white_image
        )

        # Should not crash
        assert final_label.shape == (100, 100)
        assert roi_image.shape == (100, 100)
