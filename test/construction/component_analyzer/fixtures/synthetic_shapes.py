"""
Synthetic Shape Generators for Testing

Provides functions to generate synthetic component shapes for testing
the component analyzer module.
"""

import numpy as np
from skimage.draw import line
from scipy.ndimage import binary_dilation


def create_simple_line(height=10, width=50, thickness=2):
    """
    Create horizontal line

    Args:
        height: Image height in pixels
        width: Image width in pixels
        thickness: Line thickness in pixels

    Returns:
        np.ndarray: Binary mask with horizontal line
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    y_center = height // 2
    mask[y_center - thickness // 2 : y_center + thickness // 2, :] = 255
    return mask


def create_l_shape(size=50, thickness=5):
    """
    Create L-shaped component

    Args:
        size: Image size (size x size)
        thickness: Line thickness in pixels

    Returns:
        np.ndarray: Binary mask with L-shape
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    # Vertical part
    mask[10:40, 10 : 10 + thickness] = 255
    # Horizontal part
    mask[35:40, 10:40] = 255
    return mask


def create_y_junction(size=50, thickness=2):
    """
    Create Y-junction component

    Args:
        size: Image size (size x size)
        thickness: Line thickness in pixels

    Returns:
        np.ndarray: Binary mask with Y-junction
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)

    # Three branches from center
    # Top branch
    rr, cc = line(center[0], center[1], 5, center[1])
    mask[rr, cc] = 255

    # Bottom branch
    rr, cc = line(center[0], center[1], size - 5, center[1])
    mask[rr, cc] = 255

    # Left diagonal branch
    rr, cc = line(center[0], center[1], 15, 10)
    mask[rr, cc] = 255

    # Thicken lines
    if thickness > 1:
        mask = binary_dilation(mask, iterations=thickness // 2).astype(np.uint8) * 255

    return mask


def create_complex_branch(size=100):
    """
    Create complex multi-branch structure

    Args:
        size: Image size (size x size)

    Returns:
        np.ndarray: Binary mask with complex branching structure
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    # Main trunk (vertical)
    mask[10:90, 48:52] = 255
    # Left branches
    mask[20:30, 20:50] = 255
    mask[40:50, 20:50] = 255
    # Right branch
    mask[60:70, 50:80] = 255
    return mask


def create_tiny_component(size=10):
    """
    Create tiny filled square component

    Args:
        size: Image size (size x size)

    Returns:
        np.ndarray: Binary mask with small filled square
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    # Small filled square in center
    mask[3:7, 3:7] = 255
    return mask


def save_fixtures():
    """Save all fixture images as .npy files"""
    import os

    fixture_dir = os.path.dirname(__file__)

    fixtures = {
        "simple_line.npy": create_simple_line(),
        "l_shape.npy": create_l_shape(),
        "y_junction.npy": create_y_junction(),
        "complex_branch.npy": create_complex_branch(),
        "tiny_component.npy": create_tiny_component(),
    }

    for filename, data in fixtures.items():
        filepath = os.path.join(fixture_dir, filename)
        np.save(filepath, data)
        print(f"Saved: {filepath}")

    print(f"\nTotal {len(fixtures)} fixture images saved!")


if __name__ == "__main__":
    save_fixtures()
