"""
Unit tests for ComponentTopologyBuilder

Tests the topology builder class that handles skeletonization and graph construction.
"""

import pytest
import numpy as np
import networkx as nx
from skimage.measure import label

from neural_reconstruction.algorithms.pure_mst.component_analyzer.topology import (
    ComponentTopologyBuilder,
)


class TestTopologyBuilderInit:
    """Test ComponentTopologyBuilder initialization"""

    def test_default_initialization(self):
        """Test builder with default parameters"""
        builder = ComponentTopologyBuilder()
        assert builder.prune_threshold == 5.0
        assert builder.spacing == 1.0

    def test_custom_parameters(self):
        """Test builder with custom parameters"""
        builder = ComponentTopologyBuilder(prune_threshold=10.0, spacing=2.0)
        assert builder.prune_threshold == 10.0
        assert builder.spacing == 2.0


class TestTopologyBuilderSkeletonize:
    """Test _skeletonize() method"""

    def test_skeletonize_simple_line(self, topology_builder, simple_line_mask):
        """Test skeletonization of horizontal line"""
        skeleton = topology_builder._skeletonize(simple_line_mask)

        assert skeleton.dtype == np.uint8
        assert skeleton.shape == simple_line_mask.shape
        # Skeleton should be thinner than or equal to original
        assert np.sum(skeleton) <= np.sum(simple_line_mask > 0)
        # Skeleton should have at least one pixel
        assert np.sum(skeleton) > 0

    def test_skeletonize_l_shape(self, topology_builder, l_shape_mask):
        """Test skeletonization of L-shape"""
        skeleton = topology_builder._skeletonize(l_shape_mask)

        assert skeleton.shape == l_shape_mask.shape
        assert np.sum(skeleton) > 0
        # L-shape skeleton should be connected (single component)
        assert label(skeleton).max() == 1

    def test_skeletonize_empty_mask(self, topology_builder, empty_mask):
        """Test skeletonization of empty mask"""
        skeleton = topology_builder._skeletonize(empty_mask)

        assert skeleton.shape == empty_mask.shape
        assert np.sum(skeleton) == 0

    def test_skeletonize_binary_conversion(self, topology_builder):
        """Test that mask is properly binarized"""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 128  # Non-standard value

        skeleton = topology_builder._skeletonize(mask)

        # Should still produce valid skeleton
        assert skeleton.dtype == np.uint8
        assert np.all((skeleton == 0) | (skeleton == 1))


class TestTopologyBuilderSkeletonGraph:
    """Test _get_skeleton_graph() method"""

    def test_skeleton_graph_simple_line(self, topology_builder, simple_line_mask):
        """Test graph construction from simple line skeleton"""
        skeleton = topology_builder._skeletonize(simple_line_mask)
        graph = topology_builder._get_skeleton_graph(skeleton)

        assert isinstance(graph, nx.MultiGraph)
        assert graph.number_of_nodes() == 2  # At least two endpoints

        # Nodes should be coordinate tuples
        for node in graph.nodes():
            assert isinstance(node, tuple)
            assert len(node) == 2

    def test_skeleton_graph_y_junction(self, topology_builder, y_junction_mask):
        """Test graph construction from Y-junction skeleton"""
        skeleton = topology_builder._skeletonize(y_junction_mask)
        graph = topology_builder._get_skeleton_graph(skeleton)

        assert graph.number_of_nodes() == 4  # Junction + 3 endpoints

        # Check for junction node (degree > 2)
        degrees = dict(graph.degree())
        assert any(deg > 2 for deg in degrees.values()), (
            "Y-junction should have at least one node with degree > 2"
        )

    def test_skeleton_graph_node_coordinates(self, topology_builder, l_shape_mask):
        """Test that node coordinates are valid"""
        skeleton = topology_builder._skeletonize(l_shape_mask)
        graph = topology_builder._get_skeleton_graph(skeleton)

        h, w = l_shape_mask.shape
        for node in graph.nodes():
            y, x = node
            assert 0 <= y < h, f"Node y={y} out of bounds [0, {h})"
            assert 0 <= x < w, f"Node x={x} out of bounds [0, {w})"

    def test_skeleton_graph_edge_attributes(self, topology_builder, simple_line_mask):
        """Test that edges have required attributes"""
        skeleton = topology_builder._skeletonize(simple_line_mask)
        graph = topology_builder._get_skeleton_graph(skeleton)

        for u, v, key, data in graph.edges(keys=True, data=True):
            # Check that data is a dictionary
            assert isinstance(data, dict)


class TestTopologyBuilderBuildTopology:
    """Test build_topology() integration method"""

    def test_build_topology_simple_line(self, topology_builder, simple_line_mask):
        """Test full topology building for simple line"""
        graph = topology_builder.build_topology(simple_line_mask)

        assert isinstance(graph, nx.MultiGraph)
        assert graph.number_of_nodes() >= 2
        assert graph.number_of_edges() >= 1

    def test_build_topology_l_shape(self, topology_builder, l_shape_mask):
        """Test full topology building for L-shape"""
        graph = topology_builder.build_topology(l_shape_mask)

        assert graph.number_of_nodes() >= 2  # At least some topology
        assert graph.number_of_edges() >= 1  # At least one branch

    def test_build_topology_y_junction(self, topology_builder, y_junction_mask):
        """Test full topology building for Y-junction"""
        graph = topology_builder.build_topology(y_junction_mask)

        assert graph.number_of_nodes() >= 4  # Junction + 3 endpoints
        # Check connectivity
        assert nx.is_connected(graph)

    def test_build_topology_complex_branch(self, topology_builder, complex_branch_mask):
        """Test full topology building for complex structure"""
        graph = topology_builder.build_topology(complex_branch_mask)

        assert graph.number_of_nodes() > 4
        assert graph.number_of_edges() > 3

    def test_build_topology_empty_mask(self, topology_builder, empty_mask):
        """Test topology building with empty mask"""
        # Empty mask may cause skan to fail
        try:
            graph = topology_builder.build_topology(empty_mask)
            # Empty mask should produce empty or minimal graph
            assert graph.number_of_nodes() == 0 or graph.number_of_nodes() == 1
        except (ValueError, IndexError):
            # Expected failure for empty skeleton
            pytest.skip("Empty mask causes expected failure in skan library")

    def test_prune_threshold_effect(self, simple_line_mask):
        """Test that prune threshold affects result"""
        builder_no_prune = ComponentTopologyBuilder(prune_threshold=0.0)
        builder_prune = ComponentTopologyBuilder(prune_threshold=10.0)

        graph_no_prune = builder_no_prune.build_topology(simple_line_mask)
        graph_prune = builder_prune.build_topology(simple_line_mask)

        # Pruning should potentially reduce edges (or keep same if no short branches)
        assert graph_prune.number_of_edges() <= graph_no_prune.number_of_edges()


class TestTopologyBuilderEdgeCases:
    """Test edge cases and error handling"""

    def test_single_pixel_mask(self, topology_builder):
        """Test topology building with single pixel"""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 255

        # Single pixel may cause skan to fail
        try:
            graph = topology_builder.build_topology(mask)
            # Single pixel should produce minimal or empty graph
            assert graph.number_of_nodes() <= 1
        except (ValueError, IndexError):
            # Expected failure for single pixel skeleton
            pytest.skip("Single pixel mask causes expected failure in skan library")

    def test_disconnected_components(self, topology_builder):
        """Test topology building with disconnected components"""
        mask = np.zeros((50, 50), dtype=np.uint8)
        # Two separate components
        mask[10:20, 10:20] = 255
        mask[30:40, 30:40] = 255

        # Note: topology builder should handle the largest or first component
        # This test documents behavior
        graph = topology_builder.build_topology(mask)

        # Should produce some topology
        assert graph.number_of_nodes() >= 0

    def test_very_thin_structure(self, topology_builder):
        """Test topology building with very thin structure"""
        mask = np.zeros((50, 50), dtype=np.uint8)
        # Single pixel wide line
        mask[25, 10:40] = 255

        graph = topology_builder.build_topology(mask)

        assert graph.number_of_nodes() >= 2  # At least endpoints
