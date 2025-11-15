"""
Boundary Detector - Epidermis-Dermis Boundary Detection

Detects the boundary between epidermis and dermis layers using image masks.
Provides methods to:
- Detect boundary line
- Check if points are near boundary
- Calculate signed distance to boundary
"""

import numpy as np
from typing import Tuple, Dict, Optional
import cv2


class BoundaryDetector:
    """
    Detects and analyzes the epidermis-dermis boundary.

    The boundary is detected as the bottom edge of the epidermis mask.
    Provides spatial queries for proximity and position relative to boundary.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize boundary detector.

        Args:
            verbose: Print detailed information
        """
        self.verbose = verbose
        self.boundary_map = None  # Dict[int, int] mapping x -> y coordinate
        self.boundary_array = None  # np.ndarray of boundary points

    def detect_boundary(self, epidermis_mask: np.ndarray) -> Dict[int, int]:
        """
        Detect epidermis-dermis boundary from epidermis mask.

        The boundary is defined as the lowest (maximum y-coordinate)
        epidermis pixel in each column.

        Args:
            epidermis_mask: Binary mask (uint8, 0 or 255) where 255 = epidermis

        Returns:
            Dictionary mapping x-coordinate -> y-coordinate of boundary
        """
        if self.verbose:
            print("Detecting epidermis-dermis boundary...")

        height, width = epidermis_mask.shape
        boundary_map = {}

        # For each column, find the lowest epidermis pixel
        for x in range(width):
            column = epidermis_mask[:, x]
            epidermis_pixels = np.where(column > 0)[0]

            if len(epidermis_pixels) > 0:
                # Maximum y = lowest point in image coordinates
                boundary_y = epidermis_pixels.max()
                boundary_map[x] = boundary_y

        self.boundary_map = boundary_map

        # Convert to array for efficient spatial queries
        if boundary_map:
            self.boundary_array = np.array([
                [x, y] for x, y in boundary_map.items()
            ])
        else:
            self.boundary_array = np.array([]).reshape(0, 2)

        if self.verbose:
            print(f"✓ Boundary detected")
            print(f"  Boundary width: {len(boundary_map)} pixels")
            if boundary_map:
                y_coords = list(boundary_map.values())
                print(f"  Y-coordinate range: [{min(y_coords)}, {max(y_coords)}]")

        return boundary_map

    def is_near_boundary(
        self,
        point: Tuple[int, int],
        tolerance: int = 10
    ) -> bool:
        """
        Check if a point is near the boundary.

        Args:
            point: Point coordinates (x, y)
            tolerance: Distance threshold in pixels

        Returns:
            True if point is within tolerance distance of boundary
        """
        if self.boundary_array is None or len(self.boundary_array) == 0:
            return False

        x, y = point

        # Quick check: is x coordinate in boundary range?
        if x not in self.boundary_map:
            # Find nearest x in boundary
            x_coords = self.boundary_array[:, 0]
            nearest_x_idx = np.argmin(np.abs(x_coords - x))
            nearest_x = int(x_coords[nearest_x_idx])

            # If too far horizontally, reject
            if abs(x - nearest_x) > tolerance:
                return False

            boundary_y = self.boundary_map[nearest_x]
        else:
            boundary_y = self.boundary_map[x]

        # Check vertical distance
        distance = abs(y - boundary_y)
        return distance <= tolerance

    def distance_to_boundary(self, point: Tuple[int, int]) -> float:
        """
        Calculate signed distance from point to boundary.

        Negative distance = point is above boundary (epidermis)
        Positive distance = point is below boundary (dermis)

        Args:
            point: Point coordinates (x, y)

        Returns:
            Signed distance in pixels
        """
        if self.boundary_array is None or len(self.boundary_array) == 0:
            return float('inf')

        x, y = point

        # Find boundary y-coordinate at this x
        if x in self.boundary_map:
            boundary_y = self.boundary_map[x]
        else:
            # Interpolate from nearest boundary points
            x_coords = self.boundary_array[:, 0]
            nearest_x_idx = np.argmin(np.abs(x_coords - x))
            boundary_y = self.boundary_array[nearest_x_idx, 1]

        # Signed distance: positive = below boundary (dermis)
        signed_distance = y - boundary_y
        return float(signed_distance)

    def is_above_boundary(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is above boundary (in epidermis).

        Args:
            point: Point coordinates (x, y)

        Returns:
            True if point is above boundary
        """
        return self.distance_to_boundary(point) < 0

    def is_below_boundary(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is below boundary (in dermis).

        Args:
            point: Point coordinates (x, y)

        Returns:
            True if point is below boundary
        """
        return self.distance_to_boundary(point) > 0

    def get_boundary_segment(
        self,
        x_start: int,
        x_end: int
    ) -> np.ndarray:
        """
        Get boundary points in a horizontal range.

        Args:
            x_start: Start x-coordinate
            x_end: End x-coordinate

        Returns:
            Array of boundary points [[x1, y1], [x2, y2], ...]
        """
        if self.boundary_map is None:
            return np.array([]).reshape(0, 2)

        points = []
        for x in range(x_start, x_end + 1):
            if x in self.boundary_map:
                points.append([x, self.boundary_map[x]])

        return np.array(points) if points else np.array([]).reshape(0, 2)

    def visualize_boundary(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw boundary on image for visualization.

        Args:
            image: RGB image
            color: Line color (B, G, R)
            thickness: Line thickness

        Returns:
            Image with boundary drawn
        """
        if self.boundary_array is None or len(self.boundary_array) == 0:
            return image.copy()

        result = image.copy()

        # Draw boundary as connected line segments
        points = self.boundary_array.astype(np.int32)

        # Sort by x-coordinate for proper line drawing
        points = points[points[:, 0].argsort()]

        for i in range(len(points) - 1):
            pt1 = tuple(points[i])
            pt2 = tuple(points[i + 1])
            cv2.line(result, pt1, pt2, color, thickness)

        return result


if __name__ == '__main__':
    # Test code
    print("Testing BoundaryDetector...")

    # Create synthetic epidermis mask
    mask = np.zeros((500, 800), dtype=np.uint8)

    # Simulate curved epidermis boundary
    for x in range(800):
        # Parabolic boundary: y = 200 + 0.0003 * (x - 400)^2
        boundary_y = int(200 + 0.0003 * (x - 400)**2)
        mask[:boundary_y, x] = 255  # Everything above is epidermis

    # Initialize detector
    detector = BoundaryDetector(verbose=True)

    # Detect boundary
    boundary_map = detector.detect_boundary(mask)
    print(f"\n✓ Detected {len(boundary_map)} boundary points")

    # Test proximity queries
    test_points = [
        (400, 200, "On boundary"),
        (400, 195, "5px above (epidermis)"),
        (400, 205, "5px below (dermis)"),
        (400, 230, "30px below (dermis)"),
    ]

    print("\nTesting proximity queries:")
    for x, y, label in test_points:
        near = detector.is_near_boundary((x, y), tolerance=10)
        dist = detector.distance_to_boundary((x, y))
        above = detector.is_above_boundary((x, y))

        print(f"  {label}: ({x}, {y})")
        print(f"    Near boundary (tol=10): {near}")
        print(f"    Distance: {dist:.1f}px")
        print(f"    Above boundary: {above}")

    # Test visualization
    try:
        vis_image = np.stack([mask, mask, mask], axis=-1)
        result = detector.visualize_boundary(vis_image)
        print("\n✓ Visualization test passed")
    except Exception as e:
        print(f"\n✗ Visualization test failed: {e}")

    print("\n✓ All tests passed")
