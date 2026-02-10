"""
Pytest fixtures for preprocessing tests

Provides reusable fixtures for:
- Synthetic test images
- Masks
- Configuration dictionaries
- Pipeline instances
- Validation helpers
"""

import pytest
import numpy as np


# =============================================================================
# Image Fixtures - Simple Patterns
# =============================================================================

@pytest.fixture
def noisy_line_image():
    """Horizontal line with Gaussian noise"""
    from test.preprocessing.fixtures.synthetic_images import create_noisy_line
    return create_noisy_line()


@pytest.fixture
def circles_image():
    """Image with multiple circles"""
    from test.preprocessing.fixtures.synthetic_images import create_circles
    return create_circles()


@pytest.fixture
def rectangles_image():
    """Image with multiple rectangles"""
    from test.preprocessing.fixtures.synthetic_images import create_rectangles
    return create_rectangles()


@pytest.fixture
def gradient_image():
    """Horizontal gradient image"""
    from test.preprocessing.fixtures.synthetic_images import create_gradient
    return create_gradient(direction='horizontal')


@pytest.fixture
def radial_gradient_image():
    """Radial gradient image"""
    from test.preprocessing.fixtures.synthetic_images import create_gradient
    return create_gradient(direction='radial')


# =============================================================================
# Image Fixtures - Complex Patterns
# =============================================================================

@pytest.fixture
def uneven_illumination_image():
    """Image with uneven illumination"""
    from test.preprocessing.fixtures.synthetic_images import create_uneven_illumination
    return create_uneven_illumination()


@pytest.fixture
def noisy_image():
    """Noisy image with Gaussian noise"""
    from test.preprocessing.fixtures.synthetic_images import create_noisy_image
    return create_noisy_image(noise_type='gaussian')


@pytest.fixture
def salt_pepper_noisy_image():
    """Noisy image with salt and pepper noise"""
    from test.preprocessing.fixtures.synthetic_images import create_noisy_image
    return create_noisy_image(noise_type='salt_pepper')


@pytest.fixture
def fiber_like_image():
    """Realistic fiber-like structure"""
    from test.preprocessing.fixtures.synthetic_images import create_fiber_like_structure
    return create_fiber_like_structure()


@pytest.fixture
def branching_network_image():
    """Complex branching network"""
    from test.preprocessing.fixtures.synthetic_images import create_branching_network
    return create_branching_network()


# =============================================================================
# Image Fixtures - Edge Cases
# =============================================================================

@pytest.fixture
def empty_image():
    """Empty image (all zeros)"""
    from test.preprocessing.fixtures.synthetic_images import create_empty_image
    return create_empty_image()


@pytest.fixture
def constant_image():
    """Image with constant value (128)"""
    from test.preprocessing.fixtures.synthetic_images import create_constant_image
    return create_constant_image(value=128)


@pytest.fixture
def tiny_image():
    """Very small image (5x5)"""
    from test.preprocessing.fixtures.synthetic_images import create_tiny_image
    return create_tiny_image()


@pytest.fixture
def large_image():
    """Large image (1000x1000)"""
    from test.preprocessing.fixtures.synthetic_images import create_large_image
    return create_large_image()


# =============================================================================
# RGB Image Fixtures
# =============================================================================

@pytest.fixture
def rgb_gradient_image():
    """RGB image with gradient pattern"""
    from test.preprocessing.fixtures.synthetic_images import create_rgb_image
    return create_rgb_image(pattern='gradient')


@pytest.fixture
def rgb_circles_image():
    """RGB image with circles pattern"""
    from test.preprocessing.fixtures.synthetic_images import create_rgb_image
    return create_rgb_image(pattern='circles')


@pytest.fixture
def rgb_fiber_image():
    """RGB image with fiber pattern (strong green channel)"""
    from test.preprocessing.fixtures.synthetic_images import create_rgb_image
    return create_rgb_image(pattern='fiber')


# =============================================================================
# Mask Fixtures
# =============================================================================

@pytest.fixture
def epidermis_mask():
    """Horizontal epidermis boundary mask"""
    from test.preprocessing.fixtures.synthetic_images import create_epidermis_mask
    return create_epidermis_mask()


@pytest.fixture
def irregular_mask():
    """Irregular ROI mask"""
    from test.preprocessing.fixtures.synthetic_images import create_irregular_mask
    return create_irregular_mask()


@pytest.fixture
def multi_region_mask():
    """Mask with multiple disconnected regions"""
    from test.preprocessing.fixtures.synthetic_images import create_multi_region_mask
    return create_multi_region_mask()


@pytest.fixture
def simple_binary_mask():
    """Simple binary mask (top half white, bottom half black)"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 255
    return mask


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default preprocessing configuration"""
    return {
        'morphology': {
            'closing_kernel': 3,
            'opening_kernel': 3,
        },
        'mask': {
            'dilate_offset': 50,
        },
        'background': {
            'method': 'morphology',
            'radius': 12,
        },
        'threshold': {
            'use_full_roi': False,
        },
        'normalization': {
            'enabled': False,
        }
    }


@pytest.fixture
def rolling_ball_config():
    """Configuration with rolling ball background correction"""
    return {
        'morphology': {
            'closing_kernel': 3,
            'opening_kernel': 3,
        },
        'mask': {
            'dilate_offset': 50,
        },
        'background': {
            'method': 'rolling_ball',
            'radius': 20,
        },
        'threshold': {
            'use_full_roi': False,
        },
        'normalization': {
            'enabled': False,
        }
    }


@pytest.fixture
def gaussian_config():
    """Configuration with Gaussian background correction"""
    return {
        'morphology': {
            'closing_kernel': 5,
            'opening_kernel': 3,
        },
        'mask': {
            'dilate_offset': 100,
        },
        'background': {
            'method': 'gaussian',
        },
        'threshold': {
            'use_full_roi': False,
        },
        'normalization': {
            'enabled': True,
        }
    }


@pytest.fixture
def invalid_config():
    """Invalid configuration (missing required keys)"""
    return {
        'morphology': {
            'closing_kernel': 3,
        },
        # Missing other required sections
    }


# =============================================================================
# Pipeline Instance Fixtures
# =============================================================================

@pytest.fixture
def default_pipeline(default_config):
    """SkinAnalysisPipeline with default configuration"""
    from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline
    return SkinAnalysisPipeline(default_config)


@pytest.fixture
def rolling_ball_pipeline(rolling_ball_config):
    """SkinAnalysisPipeline with rolling ball background correction"""
    from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline
    return SkinAnalysisPipeline(rolling_ball_config)


@pytest.fixture
def gaussian_pipeline(gaussian_config):
    """SkinAnalysisPipeline with Gaussian background correction"""
    from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline
    return SkinAnalysisPipeline(gaussian_config)


# =============================================================================
# Validation Helper Fixtures
# =============================================================================

@pytest.fixture
def image_property_validator():
    """
    Helper to validate image properties

    Usage:
        assert image_property_validator(image, expected_shape=(100, 100), expected_dtype=np.uint8)
    """
    def _validate(image, expected_shape=None, expected_dtype=None,
                  min_value=None, max_value=None):
        assert image is not None, "Image should not be None"
        assert isinstance(image, np.ndarray), f"Image should be numpy array, got {type(image)}"

        if expected_shape is not None:
            assert image.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {image.shape}"

        if expected_dtype is not None:
            assert image.dtype == expected_dtype, \
                f"Expected dtype {expected_dtype}, got {image.dtype}"

        if min_value is not None:
            assert image.min() >= min_value, \
                f"Minimum value {image.min()} is less than {min_value}"

        if max_value is not None:
            assert image.max() <= max_value, \
                f"Maximum value {image.max()} is greater than {max_value}"

        return True

    return _validate


@pytest.fixture
def mask_property_validator():
    """
    Helper to validate binary mask properties

    Usage:
        assert mask_property_validator(mask)
    """
    def _validate(mask, binary=True):
        assert mask is not None, "Mask should not be None"
        assert isinstance(mask, np.ndarray), f"Mask should be numpy array, got {type(mask)}"
        assert mask.dtype == np.uint8, f"Mask should be uint8, got {mask.dtype}"
        assert mask.ndim == 2, f"Mask should be 2D, got {mask.ndim}D"

        if binary:
            unique_values = np.unique(mask)
            assert len(unique_values) <= 2, \
                f"Binary mask should have at most 2 unique values, got {len(unique_values)}"
            assert all(v in [0, 255] for v in unique_values), \
                f"Binary mask values should be 0 or 255, got {unique_values}"

        return True

    return _validate


@pytest.fixture
def pipeline_output_validator():
    """
    Helper to validate pipeline output structure

    Usage:
        assert pipeline_output_validator(result)
    """
    def _validate(result, expect_debug=False):
        assert result is not None, "Result should not be None"

        if expect_debug:
            # Expecting (final_label, roi_image, debug_output)
            assert isinstance(result, tuple), "Result should be tuple"
            assert len(result) == 3, f"Result should have 3 elements, got {len(result)}"
            final_label, roi_image, debug_output = result

            # Validate debug output
            assert debug_output is not None, "Debug output should not be None"
            assert hasattr(debug_output, 'label_after_closing'), \
                "Debug output should have label_after_closing"
        else:
            # Expecting (final_label, roi_image)
            assert isinstance(result, tuple), "Result should be tuple"
            assert len(result) == 2, f"Result should have 2 elements, got {len(result)}"
            final_label, roi_image = result

        # Validate images
        assert isinstance(final_label, np.ndarray), \
            f"final_label should be numpy array, got {type(final_label)}"
        assert isinstance(roi_image, np.ndarray), \
            f"roi_image should be numpy array, got {type(roi_image)}"

        assert final_label.dtype == np.uint8, \
            f"final_label should be uint8, got {final_label.dtype}"
        assert roi_image.dtype == np.uint8, \
            f"roi_image should be uint8, got {roi_image.dtype}"

        assert final_label.ndim == 2, \
            f"final_label should be 2D, got {final_label.ndim}D"
        assert roi_image.ndim == 2, \
            f"roi_image should be 2D, got {roi_image.ndim}D"

        # Shapes should match
        assert final_label.shape == roi_image.shape, \
            f"Shapes should match: final_label {final_label.shape} vs roi_image {roi_image.shape}"

        return True

    return _validate


@pytest.fixture
def debug_output_validator():
    """
    Helper to validate debug output completeness

    Usage:
        assert debug_output_validator(debug_output)
    """
    def _validate(debug_output):
        assert debug_output is not None, "Debug output should not be None"

        # Check required attributes
        required_attrs = [
            'label_after_closing',
            'label_after_opening',
            'dilated_mask',
            'roi_from_original',
            'background_corrected',
            'pseudo_label',
        ]

        for attr in required_attrs:
            assert hasattr(debug_output, attr), \
                f"Debug output should have attribute '{attr}'"
            value = getattr(debug_output, attr)
            assert isinstance(value, np.ndarray), \
                f"Debug output.{attr} should be numpy array, got {type(value)}"

        return True

    return _validate


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def image_generator():
    """
    Factory fixture to generate images with custom parameters

    Usage:
        image = image_generator('noisy_line', height=50, width=100)
    """
    def _generate(image_type, **kwargs):
        if image_type == 'noisy_line':
            from test.preprocessing.fixtures.synthetic_images import create_noisy_line
            return create_noisy_line(**kwargs)
        elif image_type == 'circles':
            from test.preprocessing.fixtures.synthetic_images import create_circles
            return create_circles(**kwargs)
        elif image_type == 'gradient':
            from test.preprocessing.fixtures.synthetic_images import create_gradient
            return create_gradient(**kwargs)
        elif image_type == 'uneven_illumination':
            from test.preprocessing.fixtures.synthetic_images import create_uneven_illumination
            return create_uneven_illumination(**kwargs)
        elif image_type == 'fiber':
            from test.preprocessing.fixtures.synthetic_images import create_fiber_like_structure
            return create_fiber_like_structure(**kwargs)
        else:
            raise ValueError(f"Unknown image type: {image_type}")

    return _generate


@pytest.fixture
def mask_generator():
    """
    Factory fixture to generate masks with custom parameters

    Usage:
        mask = mask_generator('epidermis', height=200, boundary_y=100)
    """
    def _generate(mask_type, **kwargs):
        if mask_type == 'epidermis':
            from test.preprocessing.fixtures.synthetic_images import create_epidermis_mask
            return create_epidermis_mask(**kwargs)
        elif mask_type == 'irregular':
            from test.preprocessing.fixtures.synthetic_images import create_irregular_mask
            return create_irregular_mask(**kwargs)
        elif mask_type == 'multi_region':
            from test.preprocessing.fixtures.synthetic_images import create_multi_region_mask
            return create_multi_region_mask(**kwargs)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    return _generate
