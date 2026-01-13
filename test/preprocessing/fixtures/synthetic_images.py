"""
Synthetic Image Generators for Preprocessing Tests

Provides functions to generate synthetic images for testing the preprocessing module.
"""

import numpy as np
from skimage.draw import line, rectangle
try:
    from skimage.draw import disk as circle  # Newer scikit-image versions
except ImportError:
    from skimage.draw import circle  # Older scikit-image versions
from scipy.ndimage import gaussian_filter


def create_noisy_line(height=100, width=200, thickness=3, noise_level=0.1):
    """
    Create horizontal line with Gaussian noise

    Args:
        height: Image height in pixels
        width: Image width in pixels
        thickness: Line thickness in pixels
        noise_level: Standard deviation of Gaussian noise (0-1)

    Returns:
        np.ndarray: Grayscale image with noisy line (uint8)
    """
    image = np.zeros((height, width), dtype=np.float32)
    y_center = height // 2
    image[y_center - thickness // 2 : y_center + thickness // 2, :] = 1.0

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, image.shape)
    image = np.clip(image + noise, 0, 1)

    return (image * 255).astype(np.uint8)


def create_circles(size=100, num_circles=5, min_radius=5, max_radius=15):
    """
    Create image with multiple circles

    Args:
        size: Image size (size x size)
        num_circles: Number of circles to create
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius

    Returns:
        np.ndarray: Binary image with circles (uint8)
    """
    image = np.zeros((size, size), dtype=np.uint8)

    np.random.seed(42)  # For reproducibility
    for _ in range(num_circles):
        center_y = np.random.randint(max_radius, size - max_radius)
        center_x = np.random.randint(max_radius, size - max_radius)
        radius = np.random.randint(min_radius, max_radius)

        rr, cc = circle(center_y, center_x, radius, shape=image.shape)
        image[rr, cc] = 255

    return image


def create_rectangles(size=100, num_rects=3):
    """
    Create image with multiple rectangles

    Args:
        size: Image size (size x size)
        num_rects: Number of rectangles to create

    Returns:
        np.ndarray: Binary image with rectangles (uint8)
    """
    image = np.zeros((size, size), dtype=np.uint8)

    np.random.seed(42)
    for _ in range(num_rects):
        y1 = np.random.randint(0, size // 2)
        x1 = np.random.randint(0, size // 2)
        y2 = np.random.randint(y1 + 10, min(y1 + 40, size))
        x2 = np.random.randint(x1 + 10, min(x1 + 40, size))

        rr, cc = rectangle((y1, x1), (y2, x2), shape=image.shape)
        image[rr, cc] = 255

    return image


def create_gradient(size=100, direction='horizontal'):
    """
    Create gradient image

    Args:
        size: Image size (size x size)
        direction: 'horizontal', 'vertical', or 'radial'

    Returns:
        np.ndarray: Gradient image (uint8)
    """
    if direction == 'horizontal':
        gradient = np.linspace(0, 255, size, dtype=np.uint8)
        image = np.tile(gradient, (size, 1))
    elif direction == 'vertical':
        gradient = np.linspace(0, 255, size, dtype=np.uint8)
        image = np.tile(gradient.reshape(-1, 1), (1, size))
    elif direction == 'radial':
        y, x = np.ogrid[:size, :size]
        center_y, center_x = size // 2, size // 2
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_dist = np.sqrt(2 * (size // 2)**2)
        image = (255 * distance / max_dist).astype(np.uint8)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return image


def create_uneven_illumination(size=200, object_intensity=200, bg_intensity=50):
    """
    Create image with uneven illumination (bright center, dark edges)

    Args:
        size: Image size (size x size)
        object_intensity: Intensity of central object
        bg_intensity: Background intensity

    Returns:
        np.ndarray: Image with uneven illumination (uint8)
    """
    # Create object in center
    image = np.ones((size, size), dtype=np.float32) * bg_intensity

    # Add bright object in center
    center = size // 2
    object_size = size // 4
    image[center - object_size : center + object_size,
          center - object_size : center + object_size] = object_intensity

    # Create illumination gradient (bright center, dark edges)
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((y - center)**2 + (x - center)**2)
    max_dist = np.sqrt(2 * center**2)
    illumination = 1.5 - 0.5 * (distance / max_dist)  # 1.5 at center, 1.0 at edges

    image = image * illumination
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def create_noisy_image(size=100, noise_type='gaussian', noise_level=0.2):
    """
    Create noisy image for testing denoising

    Args:
        size: Image size (size x size)
        noise_type: 'gaussian', 'salt_pepper', or 'speckle'
        noise_level: Noise intensity (0-1)

    Returns:
        np.ndarray: Noisy image (uint8)
    """
    # Create base pattern
    image = create_circles(size, num_circles=3)
    image = image.astype(np.float32) / 255.0

    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, image.shape)
        image = image + noise
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        salt = np.random.random(image.shape) > (1 - noise_level / 2)
        pepper = np.random.random(image.shape) < (noise_level / 2)
        image[salt] = 1.0
        image[pepper] = 0.0
    elif noise_type == 'speckle':
        # Multiplicative noise
        noise = np.random.normal(1, noise_level, image.shape)
        image = image * noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def create_fiber_like_structure(size=100, num_fibers=5, thickness=2):
    """
    Create realistic fiber-like branching structure

    Args:
        size: Image size (size x size)
        num_fibers: Number of fiber strands
        thickness: Fiber thickness

    Returns:
        np.ndarray: Binary image with fiber structures (uint8)
    """
    image = np.zeros((size, size), dtype=np.uint8)

    np.random.seed(42)
    for i in range(num_fibers):
        # Random starting point
        y_start = np.random.randint(0, size)
        x_start = np.random.randint(0, size)

        # Create curved path
        num_points = 20
        y_points = [y_start]
        x_points = [x_start]

        angle = np.random.uniform(0, 2 * np.pi)
        for _ in range(num_points):
            angle += np.random.uniform(-0.5, 0.5)  # Random turning
            step_size = np.random.uniform(2, 5)

            y_new = y_points[-1] + step_size * np.sin(angle)
            x_new = x_points[-1] + step_size * np.cos(angle)

            # Keep within bounds
            y_new = np.clip(y_new, 0, size - 1)
            x_new = np.clip(x_new, 0, size - 1)

            y_points.append(y_new)
            x_points.append(x_new)

        # Draw the fiber
        for j in range(len(y_points) - 1):
            rr, cc = line(int(y_points[j]), int(x_points[j]),
                         int(y_points[j+1]), int(x_points[j+1]))
            # Filter out-of-bounds
            valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
            image[rr[valid], cc[valid]] = 255

    # Thicken fibers
    if thickness > 1:
        from scipy.ndimage import binary_dilation
        image = binary_dilation(image, iterations=thickness // 2).astype(np.uint8) * 255

    return image


def create_branching_network(size=150):
    """
    Create complex branching network structure

    Args:
        size: Image size (size x size)

    Returns:
        np.ndarray: Binary image with branching network (uint8)
    """
    image = np.zeros((size, size), dtype=np.uint8)

    # Main trunk (vertical)
    trunk_x = size // 2
    trunk_width = 3
    image[10:-10, trunk_x - trunk_width : trunk_x + trunk_width] = 255

    # Add branches at regular intervals
    for y in range(20, size - 20, 30):
        # Left branch
        for i in range(40):
            x_left = trunk_x - i
            y_branch = y + i // 2
            if 0 <= x_left < size and 0 <= y_branch < size:
                image[y_branch - 1 : y_branch + 2, x_left - 1 : x_left + 2] = 255

        # Right branch
        for i in range(40):
            x_right = trunk_x + i
            y_branch = y + i // 3
            if 0 <= x_right < size and 0 <= y_branch < size:
                image[y_branch - 1 : y_branch + 2, x_right - 1 : x_right + 2] = 255

    return image


def create_empty_image(size=50):
    """
    Create empty image (all zeros)

    Args:
        size: Image size (size x size)

    Returns:
        np.ndarray: Empty image (uint8)
    """
    return np.zeros((size, size), dtype=np.uint8)


def create_constant_image(size=50, value=128):
    """
    Create image with constant value

    Args:
        size: Image size (size x size)
        value: Constant pixel value (0-255)

    Returns:
        np.ndarray: Constant-value image (uint8)
    """
    return np.full((size, size), value, dtype=np.uint8)


def create_tiny_image(size=5):
    """
    Create very small image with simple pattern

    Args:
        size: Image size (size x size)

    Returns:
        np.ndarray: Tiny image (uint8)
    """
    image = np.zeros((size, size), dtype=np.uint8)
    # Small cross pattern
    center = size // 2
    image[center, :] = 255
    image[:, center] = 255
    return image


def create_large_image(size=1000):
    """
    Create large image with pattern

    Args:
        size: Image size (size x size)

    Returns:
        np.ndarray: Large image (uint8)
    """
    # Use a simple pattern to avoid memory issues
    image = create_gradient(size, direction='radial')
    return image


def create_epidermis_mask(height=200, width=200, boundary_y=100, boundary_width=5):
    """
    Create horizontal epidermis boundary mask

    Args:
        height: Image height
        width: Image width
        boundary_y: Y-coordinate of boundary
        boundary_width: Width/thickness of boundary

    Returns:
        np.ndarray: Binary mask with horizontal boundary (uint8)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    y_start = boundary_y - boundary_width // 2
    y_end = boundary_y + boundary_width // 2
    mask[y_start:y_end, :] = 255
    return mask


def create_irregular_mask(size=200):
    """
    Create irregular ROI mask

    Args:
        size: Image size (size x size)

    Returns:
        np.ndarray: Binary irregular mask (uint8)
    """
    mask = np.zeros((size, size), dtype=np.uint8)

    # Create irregular shape using circles
    centers = [(50, 50), (50, 150), (150, 50), (150, 150), (100, 100)]
    for cy, cx in centers:
        rr, cc = circle(cy, cx, 40, shape=mask.shape)
        mask[rr, cc] = 255

    # Smooth the boundary
    mask = gaussian_filter(mask.astype(np.float32), sigma=5)
    mask = (mask > 127).astype(np.uint8) * 255

    return mask


def create_multi_region_mask(size=200, num_regions=3):
    """
    Create mask with multiple disconnected regions

    Args:
        size: Image size (size x size)
        num_regions: Number of disconnected regions

    Returns:
        np.ndarray: Binary mask with multiple regions (uint8)
    """
    mask = np.zeros((size, size), dtype=np.uint8)

    region_size = size // (num_regions + 1)
    for i in range(num_regions):
        y_center = (i + 1) * region_size
        x_center = size // 2
        rr, cc = circle(y_center, x_center, region_size // 2, shape=mask.shape)
        mask[rr, cc] = 255

    return mask


def create_rgb_image(size=100, pattern='gradient'):
    """
    Create RGB image with specified pattern

    Args:
        size: Image size (size x size)
        pattern: Pattern type ('gradient', 'circles', 'fiber')

    Returns:
        np.ndarray: RGB image (H, W, 3) uint8
    """
    if pattern == 'gradient':
        # Gradient in different channels
        r = create_gradient(size, 'horizontal')
        g = create_gradient(size, 'vertical')
        b = create_gradient(size, 'radial')
    elif pattern == 'circles':
        # Circles with different colors
        base = create_circles(size, num_circles=5)
        r = base
        g = (base * 0.7).astype(np.uint8)
        b = (base * 0.5).astype(np.uint8)
    elif pattern == 'fiber':
        # Fiber structure - strong green channel
        base = create_fiber_like_structure(size)
        r = (base * 0.5).astype(np.uint8)
        g = base  # Strong green channel
        b = (base * 0.3).astype(np.uint8)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return np.stack([r, g, b], axis=2)
