"""
Epidermis Statistics Builder

Extracts statistical features from successfully reconstructed epidermis nerves
to guide the boundary crossing detection.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx


class EpidermisStatisticsBuilder:
    """Build statistical model from epidermis nerve network"""

    def __init__(self, config: dict):
        self.config = config
        self.statistics = {}

    def build_statistics(
        self,
        network: nx.Graph,
        image: np.ndarray,
        epidermis_mask: np.ndarray = None
    ) -> Dict:
        """
        Extract statistical features from epidermis nerve network

        Args:
            network: NetworkX graph representing the nerve network
            image: Original RGB/grayscale image
            epidermis_mask: Optional mask to restrict to epidermis region

        Returns:
            Dictionary containing statistical features
        """
        print("Building epidermis statistics...")

        # Extract green channel
        if len(image.shape) == 3:
            green_channel = image[:, :, 1]
        else:
            green_channel = image

        # Initialize feature collectors
        green_intensities = []
        widths = []
        curvatures = []
        segment_lengths = []

        # Iterate through all paths in the network
        paths = self._extract_all_paths(network)
        print(f"  Found {len(paths)} paths in the network")

        for path_idx, path in enumerate(paths):
            if len(path) < 3:
                continue  # Skip very short paths

            # Filter by epidermis mask if provided
            if epidermis_mask is not None:
                path = self._filter_path_by_mask(path, epidermis_mask)
                if len(path) < 3:
                    continue

            # Extract features along this path
            path_intensities = self._extract_intensities(path, green_channel)
            green_intensities.extend(path_intensities)

            if self.config.get('extract_width', True):
                path_widths = self._extract_widths(path, green_channel)
                widths.extend(path_widths)

            if self.config.get('extract_curvature', True):
                path_curvatures = self._extract_curvatures(path)
                curvatures.extend(path_curvatures)

            path_segment_lengths = self._extract_segment_lengths(path)
            segment_lengths.extend(path_segment_lengths)

        # Compute statistics
        self.statistics = self._compute_statistics(
            green_intensities, widths, curvatures, segment_lengths
        )

        print(f"  Statistics built from {len(green_intensities)} sample points")

        # Debug info
        if len(green_intensities) == 0:
            print("  WARNING: No intensity values extracted! Statistics will be incomplete.")

        return self.statistics

    def _extract_all_paths(self, network: nx.Graph) -> List[List[Tuple[int, int]]]:
        """Extract all paths from the network"""
        import ast
        paths = []

        # Find all edges in the network
        for edge in network.edges():
            node1, node2 = edge
            # Get the path stored in the edge (if available)
            edge_data = network.get_edge_data(node1, node2)

            if 'path' in edge_data and edge_data['path']:
                path = edge_data['path']

                # Handle string representation
                if isinstance(path, str):
                    try:
                        path = ast.literal_eval(path)
                    except:
                        path = None

                if path:
                    paths.append(path)
                else:
                    # Fallback to endpoints
                    pos1 = (network.nodes[node1]['x'], network.nodes[node1]['y'])
                    pos2 = (network.nodes[node2]['x'], network.nodes[node2]['y'])
                    paths.append([pos1, pos2])
            else:
                # If no path stored, just use the two endpoints
                pos1 = (network.nodes[node1]['x'], network.nodes[node1]['y'])
                pos2 = (network.nodes[node2]['x'], network.nodes[node2]['y'])
                paths.append([pos1, pos2])

        return paths

    def _filter_path_by_mask(
        self,
        path: List[Tuple[int, int]],
        mask: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Keep only points inside the mask"""
        filtered = []
        for x, y in path:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] > 0:
                    filtered.append((x, y))
        return filtered

    def _extract_intensities(
        self,
        path: List[Tuple[int, int]],
        green_channel: np.ndarray
    ) -> List[float]:
        """Extract green intensities along the path"""
        intensities = []
        for x, y in path:
            if 0 <= y < green_channel.shape[0] and 0 <= x < green_channel.shape[1]:
                intensities.append(float(green_channel[y, x]))
        return intensities

    def _extract_widths(
        self,
        path: List[Tuple[int, int]],
        green_channel: np.ndarray
    ) -> List[float]:
        """Estimate nerve width along the path"""
        widths = []
        window = self.config.get('width_estimation_window', 5)

        for i in range(0, len(path) - window + 1, window):
            segment = path[i:i + window]
            width = self._estimate_width_at_segment(segment, green_channel)
            if width > 0:
                widths.append(width)

        return widths

    def _estimate_width_at_segment(
        self,
        segment: List[Tuple[int, int]],
        green_channel: np.ndarray
    ) -> float:
        """
        Estimate width at a segment by sampling perpendicular to the path

        Simple approach: measure the extent of high-intensity pixels
        perpendicular to the path direction
        """
        if len(segment) < 2:
            return 0.0

        # Get segment direction
        start, end = segment[0], segment[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)

        if length < 1:
            return 0.0

        # Perpendicular direction
        perp_dx = -dy / length
        perp_dy = dx / length

        # Sample along perpendicular at the midpoint
        mid_x, mid_y = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2

        # Sample in both directions
        max_width = 10  # Maximum width to search
        left_extent = 0
        right_extent = 0

        # Get reference intensity
        if not (0 <= mid_y < green_channel.shape[0] and 0 <= mid_x < green_channel.shape[1]):
            return 0.0
        ref_intensity = green_channel[mid_y, mid_x]
        threshold = ref_intensity * 0.7  # 70% of reference

        # Search left
        for dist in range(1, max_width):
            x = int(mid_x - dist * perp_dx)
            y = int(mid_y - dist * perp_dy)
            if 0 <= y < green_channel.shape[0] and 0 <= x < green_channel.shape[1]:
                if green_channel[y, x] >= threshold:
                    left_extent = dist
                else:
                    break
            else:
                break

        # Search right
        for dist in range(1, max_width):
            x = int(mid_x + dist * perp_dx)
            y = int(mid_y + dist * perp_dy)
            if 0 <= y < green_channel.shape[0] and 0 <= x < green_channel.shape[1]:
                if green_channel[y, x] >= threshold:
                    right_extent = dist
                else:
                    break
            else:
                break

        return float(left_extent + right_extent)

    def _extract_curvatures(self, path: List[Tuple[int, int]]) -> List[float]:
        """Calculate curvature along the path"""
        curvatures = []
        window = 5

        for i in range(len(path) - window + 1):
            segment = path[i:i + window]
            curvature = self._calculate_curvature_at_segment(segment)
            curvatures.append(curvature)

        return curvatures

    def _calculate_curvature_at_segment(self, segment: List[Tuple[int, int]]) -> float:
        """
        Calculate curvature as the angle between two half-segments
        Returns curvature in degrees
        """
        if len(segment) < 5:
            return 0.0

        mid = len(segment) // 2
        first_half = segment[:mid + 1]
        second_half = segment[mid:]

        # Direction of first half
        dx1 = first_half[-1][0] - first_half[0][0]
        dy1 = first_half[-1][1] - first_half[0][1]

        # Direction of second half
        dx2 = second_half[-1][0] - second_half[0][0]
        dy2 = second_half[-1][1] - second_half[0][1]

        # Calculate angle
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)

        angle_diff = np.abs(angle2 - angle1)
        # Normalize to [0, pi]
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        return np.degrees(angle_diff)

    def _extract_segment_lengths(self, path: List[Tuple[int, int]]) -> List[float]:
        """Calculate distances between consecutive points"""
        lengths = []
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            length = np.sqrt(dx**2 + dy**2)
            lengths.append(length)
        return lengths

    def _compute_statistics(
        self,
        green_intensities: List[float],
        widths: List[float],
        curvatures: List[float],
        segment_lengths: List[float]
    ) -> Dict:
        """Compute statistical summaries"""
        stats = {}

        # Green intensity statistics
        if green_intensities:
            stats['green_intensity_mean'] = float(np.mean(green_intensities))
            stats['green_intensity_std'] = float(np.std(green_intensities))
            stats['green_intensity_median'] = float(np.median(green_intensities))
            stats['green_intensity_25p'] = float(np.percentile(green_intensities, 25))
            stats['green_intensity_75p'] = float(np.percentile(green_intensities, 75))
            stats['green_intensity_min'] = float(np.min(green_intensities))
            stats['green_intensity_max'] = float(np.max(green_intensities))

        # Width statistics
        if widths:
            stats['width_mean'] = float(np.mean(widths))
            stats['width_std'] = float(np.std(widths))
            stats['width_median'] = float(np.median(widths))

        # Curvature statistics
        if curvatures:
            stats['curvature_mean'] = float(np.mean(curvatures))
            stats['curvature_std'] = float(np.std(curvatures))
            stats['curvature_90p'] = float(np.percentile(curvatures, 90))

        # Segment length statistics
        if segment_lengths:
            stats['segment_length_mean'] = float(np.mean(segment_lengths))
            stats['segment_length_std'] = float(np.std(segment_lengths))

        return stats

    def save_statistics(self, output_path: Path):
        """Save statistics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)

        print(f"Statistics saved to: {output_path}")

    def load_statistics(self, input_path: Path):
        """Load statistics from JSON file"""
        with open(input_path, 'r') as f:
            self.statistics = json.load(f)

        print(f"Statistics loaded from: {input_path}")
        return self.statistics

    def get_intensity_threshold(self, sigma_multiplier: float = 2.0) -> float:
        """Get the minimum intensity threshold based on statistics"""
        mean = self.statistics.get('green_intensity_mean', 128)
        std = self.statistics.get('green_intensity_std', 30)
        threshold = mean - sigma_multiplier * std
        return max(0, threshold)  # Ensure non-negative
