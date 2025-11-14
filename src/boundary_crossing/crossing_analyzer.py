"""
Crossing Analyzer

Core functionality for detecting nerve fibers crossing the epidermis-dermis boundary.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import networkx as nx
from dataclasses import dataclass


@dataclass
class CrossingCandidate:
    """Represents a candidate nerve endpoint near the boundary"""
    node_id: int
    position: Tuple[int, int]
    direction: Tuple[float, float]
    boundary_y: int
    distance_to_boundary: float


@dataclass
class CrossingResult:
    """Result of crossing detection for one candidate"""
    candidate: CrossingCandidate
    success: bool
    path: List[Tuple[int, int]]
    crossing_point: Optional[Tuple[int, int]]
    confidence: float
    length: int
    mean_intensity: float


class CrossingAnalyzer:
    """Analyze nerve fibers crossing the epidermis-dermis boundary"""

    def __init__(self, config: dict, statistics: dict, boundary_detector):
        self.config = config
        self.statistics = statistics
        self.boundary_detector = boundary_detector

    def find_boundary_candidates(self, network: nx.Graph) -> List[CrossingCandidate]:
        """
        Find nerve endpoints near the boundary that might cross into dermis

        Args:
            network: NetworkX graph of the nerve network

        Returns:
            List of crossing candidates
        """
        print("Finding boundary crossing candidates...")

        candidates = []
        tolerance = self.config.get('boundary_tolerance', 5)

        # Find all endpoints (degree = 1)
        for node in network.nodes():
            if network.degree(node) != 1:
                continue  # Not an endpoint

            x = network.nodes[node]['x']
            y = network.nodes[node]['y']
            position = (x, y)

            # Check if near boundary
            if not self.boundary_detector.is_near_boundary(position, tolerance):
                continue

            # Get boundary y-coordinate
            boundary_y = self.boundary_detector.get_boundary_y(x)
            if boundary_y is None:
                continue

            # Estimate direction
            direction = self._estimate_direction(node, network)
            if direction is None:
                continue

            # Check if direction is downward (into dermis)
            if direction[1] <= 0:
                continue  # Not pointing down

            # Calculate distance to boundary
            distance = self.boundary_detector.distance_to_boundary(position)

            candidate = CrossingCandidate(
                node_id=node,
                position=position,
                direction=direction,
                boundary_y=boundary_y,
                distance_to_boundary=distance
            )
            candidates.append(candidate)

        print(f"  Found {len(candidates)} boundary crossing candidates")
        return candidates

    def _estimate_direction(
        self,
        node: int,
        network: nx.Graph
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate the outward direction at an endpoint

        Args:
            node: Node ID of the endpoint
            network: The nerve network

        Returns:
            Normalized direction vector (dx, dy) or None
        """
        # Find the neighbor (endpoint has degree 1)
        neighbors = list(network.neighbors(node))
        if len(neighbors) == 0:
            return None

        neighbor = neighbors[0]

        # Get edge path if available
        edge_data = network.get_edge_data(node, neighbor)
        if edge_data and 'path' in edge_data and edge_data['path']:
            path = edge_data['path']

            # Determine which end of the path is our node
            node_pos = (network.nodes[node]['x'], network.nodes[node]['y'])

            # Check if node is at the start or end of path
            if path[0] == node_pos:
                # Node is at start, use first few points
                points = path[:min(len(path), self.config.get('direction_window', 5))]
            else:
                # Node is at end, use last few points
                points = path[-min(len(path), self.config.get('direction_window', 5)):]
                points = points[::-1]  # Reverse to go outward

            if len(points) >= 2:
                return self._calculate_direction_from_points(points)

        # Fallback: use direct line to neighbor
        node_pos = (network.nodes[node]['x'], network.nodes[node]['y'])
        neighbor_pos = (network.nodes[neighbor]['x'], network.nodes[neighbor]['y'])

        dx = node_pos[0] - neighbor_pos[0]
        dy = node_pos[1] - neighbor_pos[1]
        length = np.sqrt(dx**2 + dy**2)

        if length < 1:
            return None

        return (dx / length, dy / length)

    def _calculate_direction_from_points(
        self,
        points: List[Tuple[int, int]]
    ) -> Tuple[float, float]:
        """
        Calculate direction from a sequence of points

        Uses the vector from the first point to the last point
        """
        if len(points) < 2:
            return (0, 1)  # Default: downward

        start = points[0]
        end = points[-1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)

        if length < 1:
            return (0, 1)

        return (dx / length, dy / length)

    def extend_and_verify(
        self,
        candidate: CrossingCandidate,
        image: np.ndarray
    ) -> CrossingResult:
        """
        Extend from candidate endpoint and verify if it crosses the boundary

        Args:
            candidate: Crossing candidate
            image: Input image (should be green channel or grayscale)

        Returns:
            CrossingResult with detection outcome
        """
        # Extract green channel if needed
        if len(image.shape) == 3:
            green_channel = image[:, :, 1]
        else:
            green_channel = image

        # Initialize path with starting point
        path = [candidate.position]
        current_point = candidate.position
        current_direction = candidate.direction

        # Get intensity threshold
        threshold = self._get_intensity_threshold()

        # Extension parameters
        max_length = self.config.get('max_extension_length', 15)
        step_size = self.config.get('step_size', 2)
        max_steps = max_length // step_size

        # Extend step by step
        for step in range(max_steps):
            # Propose next point
            next_point = self._propose_next_point(current_point, current_direction, step_size)

            # Check bounds
            if not self._is_valid_point(next_point, green_channel.shape):
                break

            # Check green intensity
            intensity = green_channel[next_point[1], next_point[0]]
            if intensity < threshold:
                # Too low intensity, stop extending
                break

            # Add point to path
            path.append(next_point)
            current_point = next_point

            # Update direction based on recent path
            if len(path) >= self.config.get('min_direction_points', 3):
                current_direction = self._calculate_direction_from_points(
                    path[-self.config.get('min_direction_points', 3):]
                )

        # Verify crossing
        success = self._verify_crossing(path, candidate.boundary_y)

        # Calculate confidence
        confidence = self._calculate_confidence(path, green_channel)

        # Determine crossing point
        crossing_point = path[-1] if success else None

        # Calculate mean intensity
        mean_intensity = self._calculate_mean_intensity(path, green_channel)

        return CrossingResult(
            candidate=candidate,
            success=success,
            path=path,
            crossing_point=crossing_point,
            confidence=confidence,
            length=len(path) * step_size,
            mean_intensity=mean_intensity
        )

    def _get_intensity_threshold(self) -> float:
        """Get the minimum intensity threshold based on statistics"""
        mean = self.statistics.get('green_intensity_mean', 128)
        std = self.statistics.get('green_intensity_std', 30)
        sigma = self.config.get('intensity_sigma_threshold', 2.0)

        threshold = mean - sigma * std
        return max(0, threshold)

    def _propose_next_point(
        self,
        current: Tuple[int, int],
        direction: Tuple[float, float],
        step_size: int
    ) -> Tuple[int, int]:
        """Propose the next point along the direction"""
        next_x = int(round(current[0] + direction[0] * step_size))
        next_y = int(round(current[1] + direction[1] * step_size))
        return (next_x, next_y)

    def _is_valid_point(self, point: Tuple[int, int], shape: Tuple[int, int]) -> bool:
        """Check if point is within image bounds"""
        x, y = point
        height, width = shape
        return 0 <= x < width and 0 <= y < height

    def _verify_crossing(self, path: List[Tuple[int, int]], boundary_y: int) -> bool:
        """
        Verify if the path successfully crossed the boundary

        Args:
            path: List of points in the extension path
            boundary_y: Y-coordinate of the boundary

        Returns:
            True if path crosses boundary with sufficient depth
        """
        if len(path) < 2:
            return False

        min_depth = self.config.get('min_crossing_depth', 3)
        final_y = path[-1][1]

        # Check if final point is below boundary by at least min_depth
        return final_y > boundary_y + min_depth

    def _calculate_confidence(
        self,
        path: List[Tuple[int, int]],
        green_channel: np.ndarray
    ) -> float:
        """
        Calculate confidence score for the crossing detection

        Based on how similar the path's intensity is to epidermis statistics
        """
        min_path_length = self.config.get('min_path_length', 3)
        if len(path) < min_path_length:
            return 0.0

        # Extract intensities along path
        intensities = []
        for x, y in path:
            if 0 <= y < green_channel.shape[0] and 0 <= x < green_channel.shape[1]:
                intensities.append(green_channel[y, x])

        if not intensities:
            return 0.0

        # Calculate mean intensity
        mean_intensity = np.mean(intensities)

        # Compare with epidermis statistics
        expected_mean = self.statistics.get('green_intensity_mean', 128)
        expected_std = self.statistics.get('green_intensity_std', 30)

        # Calculate Z-score
        z_score = abs(mean_intensity - expected_mean) / max(expected_std, 1)

        # Convert to confidence (0-1)
        # z_score=0 → confidence=1.0
        # z_score=2 → confidence≈0.14
        # z_score=3 → confidence≈0.01
        confidence = np.exp(-z_score / 2)

        # Apply length penalty for very short paths
        length_factor = min(len(path) / (min_path_length * 2), 1.0)

        return float(confidence * length_factor)

    def _calculate_mean_intensity(
        self,
        path: List[Tuple[int, int]],
        green_channel: np.ndarray
    ) -> float:
        """Calculate mean green intensity along path"""
        intensities = []
        for x, y in path:
            if 0 <= y < green_channel.shape[0] and 0 <= x < green_channel.shape[1]:
                intensities.append(green_channel[y, x])

        return float(np.mean(intensities)) if intensities else 0.0

    def analyze_all_crossings(
        self,
        candidates: List[CrossingCandidate],
        image: np.ndarray
    ) -> Tuple[List[CrossingResult], Dict]:
        """
        Analyze all candidates and compute statistics

        Args:
            candidates: List of crossing candidates
            image: Input image

        Returns:
            Tuple of (results, statistics)
        """
        print(f"Analyzing {len(candidates)} crossing candidates...")

        results = []
        for i, candidate in enumerate(candidates):
            result = self.extend_and_verify(candidate, image)
            results.append(result)

            if self.config.get('verbose', False):
                status = "✓" if result.success else "✗"
                print(f"  {status} Candidate {i+1}: "
                      f"{'Success' if result.success else 'Failed'} "
                      f"(conf={result.confidence:.2f}, len={result.length}px)")

        # Compute statistics
        statistics = self._compute_statistics(results)

        return results, statistics

    def _compute_statistics(self, results: List[CrossingResult]) -> Dict:
        """Compute summary statistics from results"""
        total = len(results)
        successful = [r for r in results if r.success]
        high_confidence = [
            r for r in successful
            if r.confidence >= self.config.get('min_confidence', 0.7)
        ]

        stats = {
            'total_candidates': total,
            'successful_crossings': len(successful),
            'high_confidence_crossings': len(high_confidence),
            'success_rate': len(successful) / total if total > 0 else 0,
        }

        if successful:
            stats['mean_confidence'] = float(np.mean([r.confidence for r in successful]))
            stats['std_confidence'] = float(np.std([r.confidence for r in successful]))
            stats['mean_length'] = float(np.mean([r.length for r in successful]))
            stats['mean_intensity'] = float(np.mean([r.mean_intensity for r in successful]))

        if high_confidence:
            stats['high_conf_mean_length'] = float(np.mean([r.length for r in high_confidence]))

        return stats

    def filter_high_confidence(
        self,
        results: List[CrossingResult]
    ) -> List[CrossingResult]:
        """Filter results to keep only high-confidence crossings"""
        min_conf = self.config.get('min_confidence', 0.7)
        return [r for r in results if r.success and r.confidence >= min_conf]
