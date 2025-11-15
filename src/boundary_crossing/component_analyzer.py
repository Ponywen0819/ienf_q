"""
Component Analyzer

Analyzes MST forest components and identifies boundary-proximal nodes.

IMPORTANT: "Component" here refers to independent trees in the MST forest,
NOT connected components from skeletonization (which are stored in node['component_id']).
"""

from typing import List, Dict, Tuple, Set
import numpy as np

from .boundary_detector import BoundaryDetector


class ComponentAnalyzer:
    """
    Analyzes independent trees in MST forest and identifies boundary-proximal nodes.

    Note: This class identifies MST trees (independent neural fibers), not skeleton
    connected components (which are stored in node['component_id']).
    """

    def __init__(self, boundary_detector: BoundaryDetector, boundary_tolerance: int = 10):
        """
        Initialize ComponentAnalyzer.

        Args:
            boundary_detector: BoundaryDetector instance
            boundary_tolerance: Distance tolerance for boundary proximity (pixels)
        """
        self.boundary_detector = boundary_detector
        self.boundary_tolerance = boundary_tolerance

    def _identify_mst_trees(self, mst_data: Dict) -> Dict[int, List[str]]:
        """
        Identify all independent trees in the MST forest by analyzing edge connectivity.

        Args:
            mst_data: MST forest data with 'nodes' and 'edges' keys

        Returns:
            Dictionary mapping tree_id -> list of node IDs in that tree
        """
        nodes = mst_data.get('nodes', [])
        edges = mst_data.get('edges', [])

        # Create node ID to index mapping
        node_ids = [str(node['id']) for node in nodes]
        node_id_set = set(node_ids)

        # Build adjacency list from edges
        adjacency = {node_id: [] for node_id in node_ids}
        for edge in edges:
            # Handle both 'source'/'target' and 'source_id'/'target_id' formats
            source = str(edge.get('source', edge.get('source_id')))
            target = str(edge.get('target', edge.get('target_id')))

            # Only add edge if both nodes exist
            if source in node_id_set and target in node_id_set:
                adjacency[source].append(target)
                adjacency[target].append(source)

        # Find connected components using DFS
        visited = set()
        trees = {}
        tree_id = 0

        for node_id in node_ids:
            if node_id not in visited:
                # Start a new tree
                tree_nodes = []
                stack = [node_id]

                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue

                    visited.add(current)
                    tree_nodes.append(current)

                    # Add unvisited neighbors to stack
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)

                trees[tree_id] = tree_nodes
                tree_id += 1

        return trees

    def get_component_nodes(self, mst_data: Dict, tree_id: int) -> List[Dict]:
        """
        Get all nodes belonging to a specific MST tree.

        Args:
            mst_data: MST forest data with 'nodes' key
            tree_id: Tree ID (0-indexed)

        Returns:
            List of nodes in the tree
        """
        trees = self._identify_mst_trees(mst_data)
        if tree_id not in trees:
            return []

        tree_node_ids = set(trees[tree_id])
        nodes = mst_data.get('nodes', [])

        tree_nodes = [
            node for node in nodes
            if str(node['id']) in tree_node_ids
        ]
        return tree_nodes

    def get_all_component_ids(self, mst_data: Dict) -> Set[int]:
        """
        Get all unique tree IDs in the MST forest.

        This identifies independent MST trees, NOT skeleton connected components.

        Args:
            mst_data: MST forest data

        Returns:
            Set of tree IDs (0-indexed integers)
        """
        trees = self._identify_mst_trees(mst_data)
        return set(trees.keys())

    def filter_boundary_nodes(self, nodes: List[Dict], tolerance: int = None) -> List[Dict]:
        """
        Filter nodes that are near the boundary.

        Args:
            nodes: List of node dictionaries
            tolerance: Distance tolerance (uses self.boundary_tolerance if None)

        Returns:
            List of boundary-proximal nodes
        """
        if tolerance is None:
            tolerance = self.boundary_tolerance

        boundary_nodes = []
        for node in nodes:
            pos = (node['x'], node['y'])
            if self.boundary_detector.is_near_boundary(pos, tolerance=tolerance):
                # Add boundary distance information
                node_with_dist = node.copy()
                node_with_dist['boundary_distance'] = self.boundary_detector.distance_to_boundary(pos)
                boundary_nodes.append(node_with_dist)

        return boundary_nodes

    def get_component_boundary_nodes(
        self,
        mst_data: Dict,
        tree_id: int,
        tolerance: int = None
    ) -> List[Dict]:
        """
        Get all boundary-proximal nodes for a specific MST tree.

        Args:
            mst_data: MST forest data
            tree_id: MST tree ID (0-indexed)
            tolerance: Distance tolerance

        Returns:
            List of boundary nodes in the tree
        """
        tree_nodes = self.get_component_nodes(mst_data, tree_id)
        boundary_nodes = self.filter_boundary_nodes(tree_nodes, tolerance)
        return boundary_nodes

    def get_component_info(self, mst_data: Dict, tree_id: int) -> Dict:
        """
        Get statistical information about an MST tree.

        Args:
            mst_data: MST forest data
            tree_id: MST tree ID (0-indexed)

        Returns:
            Dictionary with tree statistics
        """
        nodes = self.get_component_nodes(mst_data, tree_id)
        boundary_nodes = self.filter_boundary_nodes(nodes)

        if not nodes:
            return {
                'tree_id': tree_id,
                'total_nodes': 0,
                'boundary_nodes': 0,
                'has_boundary_nodes': False
            }

        # Calculate tree spatial extent
        xs = [n['x'] for n in nodes]
        ys = [n['y'] for n in nodes]

        # Count node types
        node_types = {}
        for node in nodes:
            node_type = node.get('seed_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        return {
            'tree_id': tree_id,
            'total_nodes': len(nodes),
            'boundary_nodes': len(boundary_nodes),
            'has_boundary_nodes': len(boundary_nodes) > 0,
            'spatial_extent': {
                'x_min': min(xs),
                'x_max': max(xs),
                'y_min': min(ys),
                'y_max': max(ys),
                'width': max(xs) - min(xs),
                'height': max(ys) - min(ys)
            },
            'node_types': node_types,
            'endpoints': node_types.get('endpoint', 0),
            'centroids': node_types.get('centroid', 0)
        }

    def analyze_all_components(self, mst_data: Dict) -> Dict[int, Dict]:
        """
        Analyze all MST trees in the forest.

        Args:
            mst_data: MST forest data

        Returns:
            Dictionary mapping tree_id to tree info
        """
        tree_ids = self.get_all_component_ids(mst_data)
        analysis = {}

        for tree_id in tree_ids:
            analysis[tree_id] = self.get_component_info(mst_data, tree_id)

        return analysis

    def get_epidermis_components(self, mst_data: Dict) -> List[int]:
        """
        Get all epidermis tree IDs (those with boundary_distance < 0).

        Args:
            mst_data: MST forest data

        Returns:
            List of epidermis tree IDs
        """
        analysis = self.analyze_all_components(mst_data)
        epidermis_trees = []

        for tree_id, info in analysis.items():
            if info['has_boundary_nodes']:
                # Check if any boundary node has negative distance (epidermis side)
                boundary_nodes = self.get_component_boundary_nodes(mst_data, tree_id)
                if any(n.get('boundary_distance', 0) < 0 for n in boundary_nodes):
                    epidermis_trees.append(tree_id)

        return epidermis_trees

    def get_dermis_components(self, mst_data: Dict) -> List[int]:
        """
        Get all dermis tree IDs (those with boundary_distance > 0).

        Args:
            mst_data: MST forest data

        Returns:
            List of dermis tree IDs
        """
        analysis = self.analyze_all_components(mst_data)
        dermis_trees = []

        for tree_id, info in analysis.items():
            if info['has_boundary_nodes']:
                # Check if any boundary node has positive distance (dermis side)
                boundary_nodes = self.get_component_boundary_nodes(mst_data, tree_id)
                if any(n.get('boundary_distance', 0) > 0 for n in boundary_nodes):
                    dermis_trees.append(tree_id)

        return dermis_trees


if __name__ == '__main__':
    # Test code
    import json
    import cv2

    # Load test data
    with open('epidermis_res/mst_forest_with_paths.json', 'r') as f:
        mst_data = json.load(f)

    epidermis_mask = cv2.imread('intermediates/epidermis_mask.png', cv2.IMREAD_GRAYSCALE)

    # Create analyzer
    boundary_detector = BoundaryDetector()
    boundary_detector.detect_boundary(epidermis_mask)

    analyzer = ComponentAnalyzer(boundary_detector, boundary_tolerance=10)

    # Analyze all components
    analysis = analyzer.analyze_all_components(mst_data)

    print(f"Total components: {len(analysis)}")
    for comp_id, info in list(analysis.items())[:5]:
        print(f"\nComponent {comp_id}:")
        print(f"  Total nodes: {info['total_nodes']}")
        print(f"  Boundary nodes: {info['boundary_nodes']}")
        print(f"  Has boundary: {info['has_boundary_nodes']}")
