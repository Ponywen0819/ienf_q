"""
Boundary Detector

Detects the epidermis-dermis boundary line from the epidermis mask.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from scipy.ndimage import uniform_filter1d


class BoundaryDetector:
    """Detect epidermis-dermis boundary from mask"""

    def __init__(self, config: dict):
        self.config = config
        self.boundary = {}  # Maps x-coordinate to y-coordinate

    def detect_boundary(self, epidermis_mask: np.ndarray) -> Dict[int, int]:
        """
        Detect the bottom boundary of the epidermis mask

        Args:
            epidermis_mask: Binary mask where epidermis region is marked

        Returns:
            Dictionary mapping x-coordinate to y-coordinate of boundary
        """
        print("Detecting epidermis-dermis boundary...")

        self.boundary = {}
        height, width = epidermis_mask.shape

        # For each column (x), find the lowest epidermis pixel
        for x in range(width):
            column = epidermis_mask[:, x]
            epidermis_pixels = np.where(column > 0)[0]

            if len(epidermis_pixels) > 0:
                # The boundary is at the bottom-most epidermis pixel
                self.boundary[x] = int(np.max(epidermis_pixels))

        print(f"  Boundary detected at {len(self.boundary)} x-coordinates")

        # Optional: smooth the boundary
        if self.config.get('boundary_smoothing', True):
            self._smooth_boundary()

        return self.boundary

    def _smooth_boundary(self):
        """Smooth the boundary line using moving average"""
        if len(self.boundary) < 3:
            return

        window = self.config.get('smoothing_window', 5)

        # Convert to arrays for smoothing
        x_coords = np.array(sorted(self.boundary.keys()))
        y_coords = np.array([self.boundary[x] for x in x_coords])

        # Apply moving average
        smoothed_y = uniform_filter1d(y_coords, size=window, mode='nearest')

        # Update boundary
        for x, y in zip(x_coords, smoothed_y):
            self.boundary[int(x)] = int(np.round(y))

        print(f"  Boundary smoothed with window size {window}")

    def get_boundary_y(self, x: int) -> Optional[int]:
        """Get boundary y-coordinate at given x, or None if not available"""
        return self.boundary.get(x)

    def is_near_boundary(
        self,
        point: Tuple[int, int],
        tolerance: int = 5
    ) -> bool:
        """
        Check if a point is near the boundary

        Args:
            point: (x, y) coordinates
            tolerance: Maximum distance to consider as "near"

        Returns:
            True if point is within tolerance of boundary
        """
        x, y = point
        if x not in self.boundary:
            return False

        boundary_y = self.boundary[x]
        distance = abs(y - boundary_y)
        return distance <= tolerance

    def is_above_boundary(self, point: Tuple[int, int]) -> bool:
        """Check if point is above (inside epidermis) the boundary"""
        x, y = point
        if x not in self.boundary:
            return True  # If no boundary defined, assume inside

        return y <= self.boundary[x]

    def is_below_boundary(self, point: Tuple[int, int]) -> bool:
        """Check if point is below (in dermis) the boundary"""
        x, y = point
        if x not in self.boundary:
            return False

        return y > self.boundary[x]

    def get_boundary_points(self) -> list:
        """Get all boundary points as a list of (x, y) tuples"""
        return [(x, y) for x, y in sorted(self.boundary.items())]

    def distance_to_boundary(self, point: Tuple[int, int]) -> float:
        """
        Calculate the signed distance from point to boundary
        Positive: below boundary (in dermis)
        Negative: above boundary (in epidermis)
        """
        x, y = point
        if x not in self.boundary:
            # Find nearest x with boundary
            x_coords = list(self.boundary.keys())
            if not x_coords:
                return 0.0
            nearest_x = min(x_coords, key=lambda bx: abs(bx - x))
            boundary_y = self.boundary[nearest_x]
        else:
            boundary_y = self.boundary[x]

        return float(y - boundary_y)

    def create_boundary_mask(self, shape: Tuple[int, int], thickness: int = 1) -> np.ndarray:
        """
        Create a binary mask marking the boundary line

        Args:
            shape: (height, width) of the output mask
            thickness: Thickness of the boundary line

        Returns:
            Binary mask with boundary marked
        """
        mask = np.zeros(shape, dtype=np.uint8)

        for x, y in self.boundary.items():
            if 0 <= x < shape[1]:
                # Draw vertical line around boundary point
                y_start = max(0, y - thickness // 2)
                y_end = min(shape[0], y + thickness // 2 + 1)
                mask[y_start:y_end, x] = 255

        return mask

    def visualize_boundary(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw boundary line on image

        Args:
            image: Input image (will be copied)
            color: BGR color for boundary line
            thickness: Line thickness

        Returns:
            Image with boundary drawn
        """
        vis_image = image.copy()

        # Convert to BGR if grayscale
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        # Draw boundary as connected line
        boundary_points = self.get_boundary_points()
        if len(boundary_points) > 1:
            pts = np.array(boundary_points, dtype=np.int32)
            cv2.polylines(vis_image, [pts], False, color, thickness)

        return vis_image

    def get_statistics(self) -> Dict:
        """Get statistics about the detected boundary"""
        if not self.boundary:
            return {}

        y_values = list(self.boundary.values())

        return {
            'num_points': len(self.boundary),
            'min_y': int(np.min(y_values)),
            'max_y': int(np.max(y_values)),
            'mean_y': float(np.mean(y_values)),
            'std_y': float(np.std(y_values)),
            'x_range': (min(self.boundary.keys()), max(self.boundary.keys())),
        }
