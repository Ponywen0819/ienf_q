"""
Unit tests for EdgeSeedGenerator

Tests the seed extraction class that generates seed points along skeleton edges.
"""

import pytest
import numpy as np
import networkx as nx

from neural_reconstruction.core.construction.component_analyzer.seed_extraction import (
    EdgeSeedGenerator,
)


class TestEdgeSeedGeneratorInit:
    """Test EdgeSeedGenerator initialization"""

    def test_default_initialization(self):
        """Test generator with default parameters"""
        generator = EdgeSeedGenerator()
        assert generator.min_edge_length == 10.0

    def test_custom_parameters(self):
        """Test generator with custom parameters"""
        generator = EdgeSeedGenerator(min_edge_length=5.0)
        assert generator.min_edge_length == 5.0


class TestEdgeSeedGeneratorComputeCumulativeDistances:
    """Test _compute_cumulative_distances() method"""

    def test_cumulative_distances_straight_line(self, seed_generator):
        """Test cumulative distances for straight horizontal line"""
        path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        distances = seed_generator._compute_cumulative_distances(path)

        assert len(distances) == len(path)
        assert distances[0] == 0.0
        # Each step is distance 1
        assert np.allclose(distances, [0, 1, 2, 3, 4])

    def test_cumulative_distances_diagonal(self, seed_generator):
        """Test cumulative distances for diagonal line"""
        path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        distances = seed_generator._compute_cumulative_distances(path)

        assert len(distances) == len(path)
        assert distances[0] == 0.0
        # Each diagonal step is sqrt(2)
        expected = [0, np.sqrt(2), 2 * np.sqrt(2), 3 * np.sqrt(2)]
        assert np.allclose(distances, expected)

    def test_cumulative_distances_single_point(self, seed_generator):
        """Test cumulative distances for single point"""
        path = [(5, 5)]
        distances = seed_generator._compute_cumulative_distances(path)

        assert len(distances) == 1
        assert distances[0] == 0.0

    def test_cumulative_distances_two_points(self, seed_generator):
        """Test cumulative distances for two points"""
        path = [(0, 0), (3, 4)]
        distances = seed_generator._compute_cumulative_distances(path)

        assert len(distances) == 2
        assert distances[0] == 0.0
        assert distances[1] == 5.0  # 3-4-5 triangle


class TestEdgeSeedGeneratorFindPointAtDistance:
    """Test _find_point_at_distance() method"""

    def test_find_point_exact_match(self, seed_generator):
        """Test finding point at exact cumulative distance"""
        path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        cumulative = [0, 1, 2, 3, 4]

        index = seed_generator._find_point_at_distance(path, cumulative, 2.0)
        assert index == 2

    def test_find_point_between_points(self, seed_generator):
        """Test finding point when target is between points"""
        path = [(0, 0), (0, 1), (0, 2), (0, 3)]
        cumulative = [0, 1, 2, 3]

        # Target 1.5 should return index 2 (first point >= 1.5)
        index = seed_generator._find_point_at_distance(path, cumulative, 1.5)
        assert index == 2

    def test_find_point_beyond_end(self, seed_generator):
        """Test finding point beyond path end"""
        path = [(0, 0), (0, 1), (0, 2)]
        cumulative = [0, 1, 2]

        # Target beyond end should return last index
        index = seed_generator._find_point_at_distance(path, cumulative, 10.0)
        assert index == len(path) - 1

    def test_find_point_at_start(self, seed_generator):
        """Test finding point at start"""
        path = [(0, 0), (0, 1), (0, 2)]
        cumulative = [0, 1, 2]

        index = seed_generator._find_point_at_distance(path, cumulative, 0.0)
        assert index == 0


class TestEdgeSeedGeneratorExtractSeedsFromEdge:
    """Test extract_seeds_from_edge() method"""

    def test_extract_seeds_simple_line(self, seed_generator):
        """Test seed extraction from simple line"""
        # 50-pixel horizontal line
        path = [(0, i) for i in range(50)]
        segment_length = 10.0
        length = 49.0

        edges = seed_generator.extract_seeds_from_edge(path, segment_length, length)

        # Should generate 4 seeds (49 // 10 = 4) plus final segment to endpoint
        # Seeds at distance 10, 20, 30, 40, plus final segment from 40 to 49
        assert len(edges) == 5

        # Each edge should be a list of path coordinates
        for edge in edges:
            assert isinstance(edge, list)
            assert len(edge) >= 2  # At least start and end

    def test_extract_seeds_short_edge(self):
        """Test seed extraction from edge shorter than min_edge_length"""
        generator = EdgeSeedGenerator(min_edge_length=20.0)
        path = [(0, i) for i in range(10)]
        segment_length = 5.0
        length = 9.0

        edges = generator.extract_seeds_from_edge(path, segment_length, length)

        # Should not extract seeds (length < min_edge_length)
        assert len(edges) == 0

    def test_extract_seeds_zero_seeds(self, seed_generator):
        """Test edge where calculated seed count is zero"""
        path = [(0, i) for i in range(15)]
        segment_length = 50.0
        length = 15.0

        edges = seed_generator.extract_seeds_from_edge(path, segment_length, length)

        # length < segment_length, so num_seeds = 0
        assert len(edges) == 0

    def test_extract_seeds_single_seed(self, seed_generator):
        """Test extraction of exactly one seed"""
        path = [(0, i) for i in range(15)]
        segment_length = 10.0
        length = 15

        edges = seed_generator.extract_seeds_from_edge(path, segment_length, length)

        # Should generate 1 seed (15 // 10 = 1) plus final segment to endpoint
        # Seed at distance 10, plus final segment from 10 to 14
        assert len(edges) == 2

    def test_extract_seeds_edge_length_filtering(self):
        """Test that min_edge_length filters correctly"""
        generator = EdgeSeedGenerator(min_edge_length=15.0)

        # Edge with length 12 (below threshold)
        path_short = [(0, i) for i in range(13)]
        edges_short = generator.extract_seeds_from_edge(path_short, 5.0, 12.0)
        assert len(edges_short) == 0

        # Edge with length 20 (above threshold)
        path_long = [(0, i) for i in range(21)]
        edges_long = generator.extract_seeds_from_edge(path_long, 5.0, 20.0)
        assert len(edges_long) == 4


class TestEdgeSeedGeneratorExtractSeedsFromTopology:
    """Test extract_seeds_from_topology() method"""

    def test_extract_from_simple_topology(self, seed_generator):
        """Test seed extraction from simple topology graph"""
        # Create simple graph: two nodes connected by an edge
        graph = nx.MultiGraph()
        node1 = (0, 0)
        node2 = (0, 49)
        path = [(0, i) for i in range(1, 50)]  # Intermediate points

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(node1, node2, path=path, **{"branch-distance": 49.0})

        segment_length = 10.0
        result_graph = seed_generator.extract_seeds_from_topology(graph, segment_length)

        assert isinstance(result_graph, nx.MultiGraph)
        # 4 intermediate seed nodes + 2 original endpoints = 6 nodes
        assert result_graph.number_of_nodes() == 6
        # Should have created seed edges: 4 seeds + 1 final segment = 5 edges
        assert result_graph.number_of_edges() == 5

    def test_extract_from_empty_topology(self, seed_generator):
        """Test seed extraction from empty topology"""
        graph = nx.MultiGraph()

        result_graph = seed_generator.extract_seeds_from_topology(graph, 10.0)

        assert result_graph.number_of_nodes() == 0
        assert result_graph.number_of_edges() == 0

    def test_extract_preserves_nodes(self, seed_generator):
        """Test that original nodes are preserved"""
        graph = nx.MultiGraph()
        node1 = (10, 20)
        node2 = (30, 40)
        graph.add_node(node1, degree=1)
        graph.add_node(node2, degree=1)
        graph.add_edge(
            node1,
            node2,
            path=[(15, 25), (20, 30), (25, 35), (30, 40)],
            **{"branch-distance": 30.0},
        )

        result_graph = seed_generator.extract_seeds_from_topology(graph, 10.0)

        assert result_graph.number_of_nodes() == 4

        # Original nodes should be present
        assert node1 in result_graph.nodes()
        assert node2 in result_graph.nodes()

        assert (20, 30) in result_graph.nodes()
        assert (25, 35) in result_graph.nodes()

    def test_extract_from_multi_edge_topology(self, seed_generator):
        """Test seed extraction from topology with multiple edges"""
        graph = nx.MultiGraph()

        # Create Y-junction: node1 connects to node2, node3, node4
        node1 = (25, 25)  # Center junction
        node2 = (5, 25)  # Left
        node3 = (45, 25)  # Right
        node4 = (25, 5)  # Top

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        graph.add_node(node4)

        # Add edges with paths
        graph.add_edge(
            node1,
            node2,
            path=[(20, 25), (15, 25), (10, 25), (5, 25)],
            **{"branch-distance": 20.0},
        )
        graph.add_edge(
            node1,
            node3,
            path=[(30, 25), (35, 25), (40, 25), (45, 25)],
            **{"branch-distance": 20.0},
        )
        graph.add_edge(
            node1,
            node4,
            path=[(25, 20), (25, 15), (25, 10), (25, 5)],
            **{"branch-distance": 20.0},
        )

        result_graph = seed_generator.extract_seeds_from_topology(graph, 10.0)

        # Should have all original nodes
        assert node1 in result_graph.nodes()
        assert node2 in result_graph.nodes()
        assert node3 in result_graph.nodes()
        assert node4 in result_graph.nodes()

        assert (15, 25) in result_graph.nodes()
        assert (35, 25) in result_graph.nodes()
        assert (25, 15) in result_graph.nodes()

        assert result_graph.number_of_nodes() == 7
        # Should have generated seed edges from each branch

        assert result_graph.number_of_edges() == 6

    def test_extract_with_varying_segment_lengths(self, seed_generator):
        """Test that different segment lengths produce different seed counts"""
        graph = nx.MultiGraph()
        node1 = (0, 0)
        node2 = (0, 50)
        path = [(0, i) for i in range(1, 51)]

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(node1, node2, path=path, **{"branch-distance": 50.0})

        # Extract with small segments
        result_small = seed_generator.extract_seeds_from_topology(graph, 5.0)
        # Extract with large segments
        result_large = seed_generator.extract_seeds_from_topology(graph, 20.0)

        # Smaller segments should produce more edges
        assert result_small.number_of_edges() > result_large.number_of_edges()


class TestEdgeSeedGeneratorEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_path(self, seed_generator):
        """Test handling of empty path"""
        path = []
        segment_length = 10.0
        length = 0.0

        edges = seed_generator.extract_seeds_from_edge(path, segment_length, length)

        # Empty path should return empty list
        assert len(edges) == 0

    def test_very_short_path(self, seed_generator):
        """Test handling of very short path"""
        path = [(0, 0), (0, 1)]
        segment_length = 10.0
        length = 1.0

        # Path is shorter than min_edge_length (default 10.0)
        edges = seed_generator.extract_seeds_from_edge(path, segment_length, length)
        assert len(edges) == 0

    def test_topology_without_edge_attributes(self, seed_generator):
        """Test handling of topology graph without required edge attributes"""
        graph = nx.MultiGraph()
        node1 = (0, 0)
        node2 = (10, 10)
        graph.add_node(node1)
        graph.add_node(node2)
        # Add edge without 'path' or 'branch-distance' attributes
        graph.add_edge(node1, node2)

        # Should handle gracefully
        result_graph = seed_generator.extract_seeds_from_topology(graph, 5.0)

        # Should return graph with nodes preserved
        assert result_graph.number_of_nodes() >= 2
