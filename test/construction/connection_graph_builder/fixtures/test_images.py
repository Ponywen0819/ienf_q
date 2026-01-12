"""
Test Image Generators for Connection Graph Builder Tests

Provides functions to generate synthetic test images for pathfinding and
connection graph builder testing.
"""

import numpy as np
from skimage.draw import line as draw_line


def create_uniform_image(size=(100, 100), intensity=200):
    """
    Create uniform bright image

    Args:
        size: Image dimensions (height, width)
        intensity: Pixel intensity value (0-255)

    Returns:
        np.ndarray: Uniform image
    """
    return np.full(size, intensity, dtype=np.uint8)


def create_gradient_image(size=(100, 100), horizontal=True):
    """
    Create gradient image from dark to bright

    Args:
        size: Image dimensions (height, width)
        horizontal: If True, gradient is horizontal; if False, vertical

    Returns:
        np.ndarray: Gradient image
    """
    if horizontal:
        gradient = np.linspace(0, 255, size[1], dtype=np.uint8)
        return np.tile(gradient, (size[0], 1))
    else:
        gradient = np.linspace(0, 255, size[0], dtype=np.uint8)
        return np.tile(gradient[:, np.newaxis], (1, size[1]))


def create_nerve_like_image(size=(100, 100), num_lines=3, intensity=220):
    """
    Create image with bright line structures (simulating nerve fibers)

    Args:
        size: Image dimensions (height, width)
        num_lines: Number of bright lines to add
        intensity: Line intensity value

    Returns:
        np.ndarray: Image with bright lines
    """
    image = np.zeros(size, dtype=np.uint8)
    height, width = size

    # Add bright lines at various angles
    for i in range(num_lines):
        # Distribute lines evenly across image
        y_start = int((i + 1) * height / (num_lines + 1))
        y_end = y_start + int(height * 0.3)  # Lines extend about 30% of height

        # Slight angle variation
        x_start = 10 + i * 10
        x_end = width - 10 - i * 10

        # Draw line
        rr, cc = draw_line(y_start, x_start, y_end, x_end)

        # Clip to image bounds
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid], cc[valid]

        image[rr, cc] = intensity

        # Thicken the line slightly
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                rr_thick = np.clip(rr + dy, 0, height - 1)
                cc_thick = np.clip(cc + dx, 0, width - 1)
                image[rr_thick, cc_thick] = intensity

    return image


def create_obstacle_image(size=(100, 100), background_intensity=150):
    """
    Create image with bright paths and dark obstacles

    Args:
        size: Image dimensions (height, width)
        background_intensity: Background pixel intensity

    Returns:
        np.ndarray: Image with obstacles
    """
    image = create_uniform_image(size, intensity=background_intensity)

    # Add dark rectangular obstacles
    height, width = size

    # Obstacle 1: top-left corner
    image[10:30, 10:30] = 50

    # Obstacle 2: center
    image[40:60, 40:60] = 30

    # Obstacle 3: bottom-right
    image[70:85, 70:85] = 40

    return image


def create_two_regions_image(size=(100, 100)):
    """
    Create image with two bright regions separated by dark area

    Args:
        size: Image dimensions (height, width)

    Returns:
        np.ndarray: Image with two bright regions
    """
    image = np.zeros(size, dtype=np.uint8)

    # Bright region 1 (left side)
    image[10:40, 10:40] = 200

    # Bright region 2 (right side)
    image[60:90, 60:90] = 200

    # Add a narrow bright path connecting them
    image[45:55, 30:70] = 180

    return image


def create_bright_path_image(size=(100, 100)):
    """
    Create image with obvious bright path for pathfinding testing

    Args:
        size: Image dimensions (height, width)

    Returns:
        np.ndarray: Image with bright S-shaped path
    """
    image = np.full(size, 100, dtype=np.uint8)  # Medium gray background

    # Create S-shaped bright path
    height, width = size

    # Top segment (horizontal)
    image[20:25, 10:50] = 240

    # Middle segment (diagonal)
    rr, cc = draw_line(22, 50, 50, 50)
    valid = (rr < height) & (cc < width)
    image[rr[valid], cc[valid]] = 240

    # Bottom segment (horizontal)
    image[48:53, 50:90] = 240

    # Thicken the path
    from scipy.ndimage import binary_dilation
    bright_mask = image > 200
    dilated_mask = binary_dilation(bright_mask, iterations=2)
    image[dilated_mask] = 240

    return image


def create_complex_network_image(size=(200, 200)):
    """
    Create larger image with multiple bright paths forming a network

    Args:
        size: Image dimensions (height, width)

    Returns:
        np.ndarray: Complex network image
    """
    image = np.full(size, 80, dtype=np.uint8)  # Dark background
    height, width = size

    # Create a grid-like network of bright paths
    # Horizontal lines
    for y in [30, 70, 110, 150]:
        image[y:y+5, 20:width-20] = 200

    # Vertical lines
    for x in [40, 80, 120, 160]:
        image[20:height-20, x:x+5] = 200

    return image


def create_unreachable_target_image(size=(100, 100)):
    """
    Create image with isolated regions (no path between them)

    Args:
        size: Image dimensions (height, width)

    Returns:
        np.ndarray: Image with isolated bright regions
    """
    image = np.zeros(size, dtype=np.uint8)  # Black background

    # Bright region 1 (top-left, isolated)
    image[10:30, 10:30] = 200

    # Bright region 2 (bottom-right, isolated)
    image[70:90, 70:90] = 200

    # No connecting path - both regions are isolated by black pixels

    return image
