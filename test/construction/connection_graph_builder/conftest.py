"""
Pytest fixtures for connection_graph_builder tests

Provides reusable fixtures for:
- Test images
- ComponentAnalysisResult mock objects
- Builder and Pathfinder instances
- Validation helpers
"""

import pytest
import numpy as np
import networkx as nx
from typing import Tuple, List

from neural_reconstruction.common.data_types import ComponentAnalysisResult


# =============================================================================
# Test Image Fixtures
# =============================================================================

@pytest.fixture
def simple_bright_image():
    """100x100 uniform bright image (intensity 200)"""
    from .fixtures.test_images import create_uniform_image
    return create_uniform_image(size=(100, 100), intensity=200)


@pytest.fixture
def gradient_image():
    """100x100 horizontal gradient from dark to bright"""
    from .fixtures.test_images import create_gradient_image
    return create_gradient_image(size=(100, 100), horizontal=True)


@pytest.fixture
def nerve_like_image():
    """100x100 image with bright line structures"""
    from .fixtures.test_images import create_nerve_like_image
    return create_nerve_like_image(size=(100, 100), num_lines=3)


@pytest.fixture
def obstacle_image():
    """100x100 image with bright paths and dark obstacles"""
    from .fixtures.test_images import create_obstacle_image
    return create_obstacle_image(size=(100, 100))


@pytest.fixture
def bright_path_image():
    """100x100 image with obvious bright S-shaped path"""
    from .fixtures.test_images import create_bright_path_image
    return create_bright_path_image(size=(100, 100))


@pytest.fixture
def complex_network_image():
    """200x200 image with multiple bright paths"""
    from .fixtures.test_images import create_complex_network_image
    return create_complex_network_image(size=(200, 200))


@pytest.fixture
def unreachable_target_image():
    """100x100 image with isolated regions"""
    from .fixtures.test_images import create_unreachable_target_image
    return create_unreachable_target_image(size=(100, 100))


# =============================================================================
# ComponentAnalysisResult Mock Factory
# =============================================================================

@pytest.fixture
def create_mock_component_result():
    """
    Factory fixture to create mock ComponentAnalysisResult

    Usage:
        result = create_mock_component_result(
            component_id=1,
            bbox=(10, 20, 50, 60),
            node_positions=[(5, 10), (15, 20), (25, 30)]
        )
    """
    def _create(
        component_id: int,
        bbox: Tuple[int, int, int, int],
        node_positions: List[Tuple[int, int]]
    ) -> ComponentAnalysisResult:
        """
        Create mock ComponentAnalysisResult with specified nodes

        Args:
            component_id: Component ID
            bbox: Bounding box (minr, minc, maxr, maxc) in global coords
            node_positions: List of (y, x) positions in LOCAL coords (relative to bbox)

        Returns:
            ComponentAnalysisResult with topology graph
        """
        topology = nx.MultiGraph()

        # Add nodes
        for node in node_positions:
            topology.add_node(node)

        # Add edges if more than one node (create simple chain)
        if len(node_positions) > 1:
            for i in range(len(node_positions) - 1):
                topology.add_edge(
                    node_positions[i],
                    node_positions[i + 1],
                    weight=1.0,
                    path=[node_positions[i], node_positions[i + 1]]
                )

        return ComponentAnalysisResult(
            component_id=component_id,
            bbox=bbox,
            topology=topology
        )

    return _create


# =============================================================================
# Pathfinder Instance Fixtures
# =============================================================================

@pytest.fixture
def default_pathfinder(simple_bright_image):
    """Pathfinder with default balanced weights"""
    from neural_reconstruction.algorithms.pure_mst.connection_graph_builder.path_finder import Pathfinder
    return Pathfinder(
        image=simple_bright_image,
        intensity_weight=0.6,
        shape_weight=0.4
    )


@pytest.fixture
def intensity_focused_pathfinder(simple_bright_image):
    """Pathfinder with high intensity weight"""
    from neural_reconstruction.algorithms.pure_mst.connection_graph_builder.path_finder import Pathfinder
    return Pathfinder(
        image=simple_bright_image,
        intensity_weight=0.9,
        shape_weight=0.1
    )


@pytest.fixture
def shape_focused_pathfinder(simple_bright_image):
    """Pathfinder with high shape weight"""
    from neural_reconstruction.algorithms.pure_mst.connection_graph_builder.path_finder import Pathfinder
    return Pathfinder(
        image=simple_bright_image,
        intensity_weight=0.1,
        shape_weight=0.9
    )


# =============================================================================
# NetworkBuilder Instance Fixtures
# =============================================================================

@pytest.fixture
def default_builder(simple_bright_image):
    """NetworkBuilder with default parameters"""
    from neural_reconstruction.algorithms.pure_mst.connection_graph_builder.builder import NetworkBuilder
    return NetworkBuilder(
        image=simple_bright_image,
        search_radius=50.0,
        max_cost_threshold=0.98,
        intensity_weight=0.6,
        shape_weight=0.4
    )


@pytest.fixture
def tight_threshold_builder(simple_bright_image):
    """NetworkBuilder with strict cost threshold"""
    from neural_reconstruction.algorithms.pure_mst.connection_graph_builder.builder import NetworkBuilder
    return NetworkBuilder(
        image=simple_bright_image,
        search_radius=50.0,
        max_cost_threshold=0.5,  # Lower threshold
        intensity_weight=0.6,
        shape_weight=0.4
    )


@pytest.fixture
def large_radius_builder(simple_bright_image):
    """NetworkBuilder with large search radius"""
    from neural_reconstruction.algorithms.pure_mst.connection_graph_builder.builder import NetworkBuilder
    return NetworkBuilder(
        image=simple_bright_image,
        search_radius=100.0,  # Larger radius
        max_cost_threshold=0.98,
        intensity_weight=0.6,
        shape_weight=0.4
    )


# =============================================================================
# Component Result Fixtures
# =============================================================================

@pytest.fixture
def single_component_result(create_mock_component_result):
    """Single component with 3 nodes in simple chain"""
    return create_mock_component_result(
        component_id=1,
        bbox=(10, 10, 30, 40),  # (minr, minc, maxr, maxc)
        node_positions=[(5, 5), (10, 15), (15, 25)]  # Local coordinates
    )


@pytest.fixture
def two_component_results(create_mock_component_result):
    """Two components for connection testing"""
    comp1 = create_mock_component_result(
        component_id=1,
        bbox=(10, 10, 30, 40),
        node_positions=[(10, 10), (15, 20)]  # Local coords
    )

    comp2 = create_mock_component_result(
        component_id=2,
        bbox=(60, 60, 80, 90),
        node_positions=[(10, 10), (15, 20)]  # Local coords
    )

    return [comp1, comp2]


@pytest.fixture
def multiple_component_results(create_mock_component_result):
    """Three components for complex network testing"""
    components = []

    # Component 1: top-left
    components.append(create_mock_component_result(
        component_id=1,
        bbox=(10, 10, 30, 30),
        node_positions=[(10, 10)]
    ))

    # Component 2: top-right
    components.append(create_mock_component_result(
        component_id=2,
        bbox=(10, 70, 30, 90),
        node_positions=[(10, 10)]
    ))

    # Component 3: bottom-center
    components.append(create_mock_component_result(
        component_id=3,
        bbox=(70, 40, 90, 60),
        node_positions=[(10, 10)]
    ))

    return components


# =============================================================================
# Validation Helper Fixtures
# =============================================================================

@pytest.fixture
def path_validator():
    """Helper to validate path structure"""
    def _validate(path, start, end):
        """
        Validate path from start to end

        Args:
            path: List of (y, x) tuples
            start: Starting position tuple
            end: Ending position tuple
        """
        assert isinstance(path, list), "Path should be a list"
        assert len(path) >= 2, "Path should have at least 2 points"

        # Check all points are tuples
        for point in path:
            assert isinstance(point, (tuple, list)), f"Point {point} should be tuple/list"
            assert len(point) == 2, f"Point {point} should have 2 coordinates"

        # Check start and end points
        assert path[0] == start or path[0] == tuple(start), \
            f"Path should start at {start}, got {path[0]}"
        assert path[-1] == end or path[-1] == tuple(end), \
            f"Path should end at {end}, got {path[-1]}"

        return True

    return _validate


@pytest.fixture
def graph_validator():
    """Helper to validate NetworkX graph structure"""
    def _validate(graph, expected_min_nodes=0, expected_min_edges=0):
        """
        Validate NetworkX graph structure

        Args:
            graph: NetworkX graph
            expected_min_nodes: Minimum expected number of nodes
            expected_min_edges: Minimum expected number of edges
        """
        assert graph is not None, "Graph should not be None"
        assert hasattr(graph, 'number_of_nodes'), "Graph should have number_of_nodes method"
        assert hasattr(graph, 'number_of_edges'), "Graph should have number_of_edges method"

        assert graph.number_of_nodes() >= expected_min_nodes, \
            f"Expected at least {expected_min_nodes} nodes, got {graph.number_of_nodes()}"
        assert graph.number_of_edges() >= expected_min_edges, \
            f"Expected at least {expected_min_edges} edges, got {graph.number_of_edges()}"

        # Validate node format (should be coordinate tuples)
        for node in graph.nodes():
            assert isinstance(node, tuple), f"Node should be tuple, got {type(node)}"
            assert len(node) == 2, f"Node should be (y, x), got {node}"

        return True

    return _validate


@pytest.fixture
def connection_result_validator():
    """Helper to validate ConnectionGraphBuilderResult"""
    def _validate(result):
        """
        Validate ConnectionGraphBuilderResult structure

        Args:
            result: ConnectionGraphBuilderResult instance
        """
        assert result is not None, "Result should not be None"
        assert hasattr(result, 'nodes'), "Result should have nodes attribute"
        assert hasattr(result, 'graph'), "Result should have graph attribute"

        # Validate nodes array
        if result.nodes is not None and len(result.nodes) > 0:
            assert isinstance(result.nodes, np.ndarray), "Nodes should be numpy array"
            assert result.nodes.shape[1] == 2, "Nodes should have shape (N, 2)"

        # Validate graph
        assert hasattr(result.graph, 'number_of_nodes'), "Graph should be NetworkX graph"

        return True

    return _validate
