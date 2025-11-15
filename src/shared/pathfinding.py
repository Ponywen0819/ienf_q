"""
A* Pathfinding (Shared Module)

Image-based pathfinding using A* algorithm.
Finds minimum-cost paths on green channel intensity maps.

This module is shared across multiple pipeline stages:
- Network Building (Stage 03)
- Boundary Crossing (Stage 05)
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Any


class ImagePathfinder:
    """
    A* pathfinding on image cost maps.

    Finds minimum-cost paths from start to end positions.
    Cost map = 255 - green_channel (higher green intensity = lower cost)
    """

    def __init__(self, green_channel: np.ndarray, verbose: bool = False):
        """
        Initialize pathfinder.

        Args:
            green_channel: Green channel image (uint8, 0-255)
            verbose: Print detailed information
        """
        self.green_channel = green_channel
        self.cost_map = 255 - green_channel.astype(np.float32)
        self.height, self.width = self.cost_map.shape
        self.verbose = verbose

        if verbose:
            print(f"✓ Initialized A* pathfinder")
            print(f"  Image size: {self.height} x {self.width}")
            print(f"  Cost range: [{self.cost_map.min():.1f}, {self.cost_map.max():.1f}]")

    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        max_g_cost: float = None,
        max_distance_from_start: float = 30.0
    ) -> Dict[str, Any]:
        """
        Find shortest path using A* algorithm.

        Args:
            start: Start position (y, x)
            end: End position (y, x)
            max_g_cost: Maximum path cost (g_score) limit
            max_distance_from_start: Maximum Euclidean distance from start

        Returns:
            Dictionary with status and results:
            - {'status': 'success', 'path': path, 'cost': total_cost}
            - {'status': 'cutoff', 'path': partial_path, 'distance': distance}
            - {'status': 'no_path'}
        """
        # Boundary check
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            return {'status': 'no_path', 'reason': 'Invalid start or end position'}

        # A* data structures
        open_set = []  # Priority queue: (f_score, counter, position)
        counter = 0  # Ensure unique queue items
        heapq.heappush(open_set, (0, counter, start))
        counter += 1

        came_from = {}  # Path reconstruction
        g_score = {start: 0}  # Actual cost from start to current
        f_score = {start: self._heuristic(start, end)}  # g + h

        visited = set()

        # A* main loop
        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            # Reached goal
            if current == end:
                path = self._reconstruct_path(came_from, current)
                total_cost = g_score[current]
                return {'status': 'success', 'path': path, 'cost': total_cost}

            # Skip visited nodes
            if current in visited:
                continue
            visited.add(current)

            # Early termination: too far from start
            distance_from_start = self._euclidean_distance(start, current)
            if max_distance_from_start is not None and distance_from_start > max_distance_from_start:
                partial_path = self._reconstruct_path(came_from, current)
                return {
                    'status': 'cutoff',
                    'reason': 'distance_from_start',
                    'path': partial_path,
                    'distance': distance_from_start
                }

            # Early termination: total cost too high
            if max_g_cost is not None and g_score[current] > max_g_cost:
                partial_path = self._reconstruct_path(came_from, current)
                return {
                    'status': 'cutoff',
                    'reason': 'max_g_cost',
                    'path': partial_path,
                    'cost': g_score[current]
                }

            # Explore 8-connected neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue

                # Calculate new g_score
                edge_cost = self._edge_cost(current, neighbor)
                tentative_g = g_score[current] + edge_cost

                # Update or add neighbor
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        # No path found
        return {'status': 'no_path', 'reason': 'Open set exhausted'}

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Heuristic function: Euclidean distance (admissible lower bound estimate).
        """
        return self._euclidean_distance(pos, goal)

    def _euclidean_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> float:
        """Calculate Euclidean distance."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _edge_cost(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> float:
        """
        Calculate edge cost.
        Cost = movement distance × target position cost map value
        """
        # Movement distance
        dy = abs(pos1[0] - pos2[0])
        dx = abs(pos1[1] - pos2[1])
        distance = 1.414 if dy == 1 and dx == 1 else 1.0

        # Target position cost
        pixel_cost = self.cost_map[pos2[0], pos2[1]]

        return (distance * (pixel_cost**3)) / (255**3)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get 8-connected neighbors.
        """
        y, x = pos
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if self._is_valid_position((ny, nx)):
                    neighbors.append((ny, nx))
        return neighbors

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within image bounds."""
        y, x = pos
        return 0 <= y < self.height and 0 <= x < self.width

    def _reconstruct_path(
        self,
        came_from: dict,
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Backtrack to reconstruct path.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """
        Calculate total path cost.
        """
        if len(path) < 2:
            return 0.0
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._edge_cost(path[i], path[i+1])
        return total_cost

    def get_path_intensity_profile(
        self,
        path: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Get green channel intensity distribution along path.
        """
        intensities = np.array([self.green_channel[y, x] for y, x in path])
        return intensities


if __name__ == '__main__':
    # Test code
    import cv2

    # Load test image
    try:
        green_channel = cv2.imread('data/Original/S163-2_a_corrected_normalized.tif', cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        green_channel = None

    if green_channel is not None:
        pathfinder = ImagePathfinder(green_channel, verbose=True)

        # Test pathfinding
        start = (349, 3489)
        end = (353, 3484)

        print(f"\nSearching path: {start} → {end}")
        result = pathfinder.find_path(start, end)

        if result['status'] == 'success':
            path = result['path']
            cost = result['cost']
            print(f"✓ Path found")
            print(f"  Path length: {len(path)} pixels")
            print(f"  Total cost: {cost:.2f}")
            print(f"  Average cost: {cost/len(path):.2f}")

            # Path intensity
            intensities = pathfinder.get_path_intensity_profile(path)
            print(f"  Average green intensity: {intensities.mean():.1f}")
        else:
            print(f"✗ No path found, status: {result['status']}, reason: {result.get('reason', 'N/A')}")
    else:
        print("Test image not found, skipping test")
