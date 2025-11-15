"""
Connection Optimizer

Computes optimal connections between epidermis and dermis components
using A* pathfinding on green channel image.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from shared.pathfinding import ImagePathfinder


class ConnectionOptimizer:
    """
    Optimizes connections between epidermis and dermis components
    by finding minimum-cost paths using A* pathfinding.
    """

    def __init__(
        self,
        pathfinder: ImagePathfinder,
        max_crossing_distance: int = 100,
        verbose: bool = False
    ):
        """
        Initialize ConnectionOptimizer.

        Args:
            pathfinder: ImagePathfinder instance for A* pathfinding
            max_crossing_distance: Maximum distance for boundary crossing (pixels)
            verbose: Print debug information
        """
        self.pathfinder = pathfinder
        self.max_crossing_distance = max_crossing_distance
        self.verbose = verbose

    def find_best_node_pair(
        self,
        epi_nodes: List[Dict],
        derm_nodes: List[Dict]
    ) -> Optional[Dict]:
        """
        Find the best node pair between two components.

        Tries all combinations of boundary nodes and returns the pair
        with minimum A* path cost.

        Args:
            epi_nodes: Epidermis component boundary nodes
            derm_nodes: Dermis component boundary nodes

        Returns:
            Dictionary with connection info or None if no valid path found:
            {
                'epi_node': epidermis node dict,
                'derm_node': dermis node dict,
                'path': A* path as list of (y, x) tuples,
                'cost': path cost,
                'status': 'success'
            }
        """
        if not epi_nodes or not derm_nodes:
            return None

        best_cost = float('inf')
        best_connection = None

        for epi_node in epi_nodes:
            for derm_node in derm_nodes:
                # Calculate Euclidean distance first
                euclidean_dist = np.linalg.norm([
                    epi_node['x'] - derm_node['x'],
                    epi_node['y'] - derm_node['y']
                ])

                # Skip if too far apart
                if euclidean_dist > self.max_crossing_distance:
                    continue

                # Use A* to find path
                start_pos = (epi_node['y'], epi_node['x'])
                end_pos = (derm_node['y'], derm_node['x'])

                path_result = self.pathfinder.find_path(
                    start_pos,
                    end_pos,
                    max_distance_from_start=self.max_crossing_distance
                )

                if path_result['status'] == 'success':
                    path_cost = path_result['cost']

                    if self.verbose:
                        print(f"  Path found: epi_node {epi_node['id']} -> derm_node {derm_node['id']}, cost={path_cost:.2f}")

                    if path_cost < best_cost:
                        best_cost = path_cost
                        best_connection = {
                            'epi_node': epi_node,
                            'derm_node': derm_node,
                            'path': path_result['path'],
                            'cost': path_cost,
                            'euclidean_distance': euclidean_dist,
                            'path_length': len(path_result['path']),
                            'status': 'success'
                        }

        return best_connection

    def compute_component_connection_costs(
        self,
        epi_components: Dict[int, List[Dict]],
        derm_components: Dict[int, List[Dict]]
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Compute connection costs between all epidermis-dermis component pairs.

        Args:
            epi_components: {epi_comp_id: boundary_nodes} mapping
            derm_components: {derm_comp_id: boundary_nodes} mapping

        Returns:
            Dictionary mapping (epi_id, derm_id) -> connection_info
        """
        connection_costs = {}
        total_pairs = len(epi_components) * len(derm_components)
        current = 0

        print(f"\nComputing connection costs for {len(epi_components)} epidermis x {len(derm_components)} dermis components...")

        for epi_id, epi_nodes in epi_components.items():
            for derm_id, derm_nodes in derm_components.items():
                current += 1
                if self.verbose or current % 100 == 0:
                    print(f"  Progress: {current}/{total_pairs} - Epi {epi_id} -> Derm {derm_id}")

                connection = self.find_best_node_pair(epi_nodes, derm_nodes)

                if connection and connection['status'] == 'success':
                    connection_costs[(epi_id, derm_id)] = connection
                    if self.verbose:
                        print(f"    ✓ Connection found: cost={connection['cost']:.2f}")
                else:
                    if self.verbose:
                        print(f"    ✗ No valid connection")

        print(f"  ✓ Found {len(connection_costs)} valid connections out of {total_pairs} pairs")
        return connection_costs

    def match_components(
        self,
        epi_component_ids: List[int],
        connection_costs: Dict[Tuple[int, int], Dict]
    ) -> Dict[int, Tuple[int, Dict]]:
        """
        Match each epidermis component to the best dermis component.

        Each epidermis component is matched to the dermis component
        with minimum connection cost. Multiple epidermis components
        can match to the same dermis component (one-to-many).

        Args:
            epi_component_ids: List of epidermis component IDs
            connection_costs: Pre-computed connection costs

        Returns:
            Dictionary mapping epi_id -> (derm_id, connection_info)
        """
        matches = {}
        unmatched = []

        print(f"\nMatching {len(epi_component_ids)} epidermis components to dermis components...")

        for epi_id in epi_component_ids:
            # Find all possible dermis components for this epidermis component
            possible_connections = {
                (e_id, d_id): conn
                for (e_id, d_id), conn in connection_costs.items()
                if e_id == epi_id
            }

            if not possible_connections:
                unmatched.append(epi_id)
                if self.verbose:
                    print(f"  Epi {epi_id}: No valid connections found")
                continue

            # Select the connection with minimum cost
            best_pair, best_connection = min(
                possible_connections.items(),
                key=lambda item: item[1]['cost']
            )
            _, derm_id = best_pair

            matches[epi_id] = (derm_id, best_connection)
            print(f"  Epi {epi_id} -> Derm {derm_id} (cost={best_connection['cost']:.2f})")

        print(f"\n  ✓ Matched: {len(matches)} epidermis components")
        print(f"  ✗ Unmatched: {len(unmatched)} epidermis components")

        if unmatched and self.verbose:
            print(f"  Unmatched component IDs: {unmatched}")

        return matches

    def get_matching_statistics(
        self,
        matches: Dict[int, Tuple[int, Dict]]
    ) -> Dict:
        """
        Compute statistics about the matching results.

        Args:
            matches: Matching results from match_components()

        Returns:
            Dictionary with statistics
        """
        if not matches:
            return {
                'total_matches': 0,
                'unique_dermis': 0,
                'dermis_with_multiple': 0,
                'avg_cost': 0.0,
                'min_cost': 0.0,
                'max_cost': 0.0,
                'std_cost': 0.0,
                'median_cost': 0.0
            }

        # Count dermis components
        dermis_counts = {}
        costs = []

        for epi_id, (derm_id, connection) in matches.items():
            dermis_counts[derm_id] = dermis_counts.get(derm_id, 0) + 1
            costs.append(connection['cost'])

        # Dermis with multiple epidermis connections
        dermis_with_multiple = sum(1 for count in dermis_counts.values() if count > 1)

        return {
            'total_matches': len(matches),
            'unique_dermis': len(dermis_counts),
            'dermis_with_multiple': dermis_with_multiple,
            'avg_cost': np.mean(costs),
            'min_cost': np.min(costs),
            'max_cost': np.max(costs),
            'std_cost': np.std(costs),
            'median_cost': np.median(costs)
        }


if __name__ == '__main__':
    # Test code
    import json
    import cv2

    # Load test data
    green_channel = cv2.imread('intermediates/green_channel.png', cv2.IMREAD_GRAYSCALE)

    with open('epidermis_res/mst_forest_with_paths.json', 'r') as f:
        epi_mst = json.load(f)

    with open('dermis_res/mst_forest_with_paths.json', 'r') as f:
        derm_mst = json.load(f)

    # Create pathfinder
    pathfinder = ImagePathfinder(green_channel, verbose=False)

    # Create optimizer
    optimizer = ConnectionOptimizer(pathfinder, max_crossing_distance=100, verbose=True)

    # Test with first few components
    epi_test = {1: epi_mst['nodes'][:5]}
    derm_test = {1: derm_mst['nodes'][:5]}

    costs = optimizer.compute_component_connection_costs(epi_test, derm_test)
    print(f"\nConnection costs: {costs}")
