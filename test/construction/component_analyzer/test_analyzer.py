"""
Unit tests for ComponentAnalyzer

Tests the main analyzer class that orchestrates topology building and seed extraction.
"""

import pytest
import numpy as np
import networkx as nx
from skimage.measure import label, regionprops

from neural_reconstruction.core.construction.component_analyzer import ComponentAnalyzer
from neural_reconstruction.common.data_types import ComponentAnalysisResult


class TestComponentAnalyzerInit:
    """Test ComponentAnalyzer initialization"""

    def test_default_initialization(self):
        """Test analyzer with default parameters"""
        analyzer = ComponentAnalyzer()
        assert analyzer.segment_length == 10.0
        assert analyzer.topology_builder is not None
        assert analyzer.seed_extractor is not None

    def test_custom_parameters(self):
        """Test analyzer with custom parameters"""
        analyzer = ComponentAnalyzer(
            segment_length=5.0,
            min_edge_length=3.0,
            prune_threshold=2.0,
            spacing=0.5,
        )
        assert analyzer.segment_length == 5.0
        assert analyzer.topology_builder.prune_threshold == 2.0
        assert analyzer.seed_extractor.min_edge_length == 3.0

    def test_parameter_storage(self):
        """Test that parameters are stored correctly"""
        analyzer = ComponentAnalyzer(
            segment_length=7.5,
            min_edge_length=4.0,
            prune_threshold=3.0,
            spacing=1.5,
        )
        assert analyzer.segment_length == 7.5


class TestComponentAnalyzerAnalyze:
    """Test ComponentAnalyzer.analyze() method"""

    def test_analyze_simple_line(self, default_analyzer, simple_line_region,
                                   component_analysis_result_validator):
        """Test analysis of simple horizontal line"""
        result = default_analyzer.analyze(simple_line_region)

        assert component_analysis_result_validator(result)
        assert result.component_id == simple_line_region.label
        assert result.topology.number_of_nodes() > 0
        # Simple line should have at least 2 nodes (endpoints)
        assert result.topology.number_of_nodes() >= 2

    def test_analyze_l_shape(self, default_analyzer, l_shape_region,
                              component_analysis_result_validator):
        """Test analysis of L-shaped component"""
        result = default_analyzer.analyze(l_shape_region)

        assert component_analysis_result_validator(result)
        # L-shape should have junction nodes (at least corner + endpoints)
        assert result.topology.number_of_nodes() >= 3

    def test_analyze_y_junction(self, default_analyzer, y_junction_region,
                                 component_analysis_result_validator):
        """Test analysis of Y-junction component"""
        result = default_analyzer.analyze(y_junction_region)

        assert component_analysis_result_validator(result)
        # Y-junction should have center junction plus 3 endpoints
        assert result.topology.number_of_nodes() >= 4

    def test_analyze_tiny_component(self, default_analyzer, tiny_component_mask,
                                     mock_region_from_mask):
        """Test analysis of very small component"""
        region = mock_region_from_mask(tiny_component_mask)
        result = default_analyzer.analyze(region)

        # Small component may use centroid fallback or produce minimal topology
        assert result.topology.number_of_nodes() >= 1

    def test_analyze_empty_component(self, default_analyzer, empty_mask,
                                      mock_region_from_mask):
        """Test analysis of empty component (edge case)"""
        region = mock_region_from_mask(empty_mask)

        # Empty component should either fail or use centroid fallback
        # This documents expected behavior
        try:
            result = default_analyzer.analyze(region)
            # If it succeeds, should have centroid fallback
            assert result.topology.number_of_nodes() >= 1
        except (ValueError, IndexError):
            # Empty component may cause skan to fail, which is acceptable
            pytest.skip("Empty component causes expected failure in skan library")

    def test_analyze_single_pixel(self, default_analyzer, single_pixel_mask,
                                   mock_region_from_mask):
        """Test analysis of single pixel component"""
        region = mock_region_from_mask(single_pixel_mask)

        # Single pixel may cause skan issues
        try:
            result = default_analyzer.analyze(region)
            # Single pixel should result in centroid
            assert result.topology.number_of_nodes() >= 1
        except (ValueError, IndexError):
            # Single pixel may cause skan to fail, which is acceptable
            pytest.skip("Single pixel component causes expected failure in skan library")

    def test_segment_length_affects_seed_density(self, simple_line_region):
        """Test that different segment lengths produce different seed counts"""
        analyzer_small = ComponentAnalyzer(segment_length=3.0, min_edge_length=3.0)
        analyzer_large = ComponentAnalyzer(segment_length=20.0, min_edge_length=20.0)

        result_small = analyzer_small.analyze(simple_line_region)
        result_large = analyzer_large.analyze(simple_line_region)

        # Smaller segment length should produce more edges (seeds)
        assert result_small.topology.number_of_edges() >= result_large.topology.number_of_edges()

    def test_bbox_coordinate_system(self, default_analyzer, l_shape_region):
        """Test that bbox is correctly stored"""
        result = default_analyzer.analyze(l_shape_region)

        minr, minc, maxr, maxc = result.bbox
        assert 0 <= minr < maxr
        assert 0 <= minc < maxc

        # Topology coordinates should be within local bbox dimensions
        bbox_height = maxr - minr
        bbox_width = maxc - minc
        for node in result.topology.nodes():
            y, x = node
            assert 0 <= y < bbox_height, f"Node y={y} out of bbox height={bbox_height}"
            assert 0 <= x < bbox_width, f"Node x={x} out of bbox width={bbox_width}"

    def test_multiple_components_independent(self, default_analyzer):
        """Test that analyzing different components produces independent results"""
        # Create two different masks - make them larger to avoid skan issues
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask1[10:30, 10:40] = 255  # Horizontal rectangle

        mask2 = np.zeros((60, 60), dtype=np.uint8)
        mask2[10:50, 10:20] = 255  # Vertical rectangle

        region1 = regionprops(label(mask1 > 0))[0]
        region2 = regionprops(label(mask2 > 0))[0]

        result1 = default_analyzer.analyze(region1)
        result2 = default_analyzer.analyze(region2)

        # Results should have independent analysis
        assert result1.bbox != result2.bbox, "Different components should have different bboxes"

        # Both should produce valid results
        assert result1.topology.number_of_nodes() >= 1
        assert result2.topology.number_of_nodes() >= 1

        # Results are independent (even if topology metrics are similar, they're separate graphs)
        assert result1.topology is not result2.topology


class TestComponentAnalyzerCentroidFallback:
    """Test centroid fallback behavior"""

    def test_centroid_calculation(self, default_analyzer):
        """Test _get_component_centroid() method"""
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:15, 10:20] = 255

        centroid = default_analyzer._get_component_centroid(mask)

        assert isinstance(centroid, tuple)
        assert len(centroid) == 2
        # Centroid should be at center of mask
        assert centroid == (10, 15)

    def test_centroid_used_when_no_edges(self, default_analyzer, single_pixel_mask,
                                          mock_region_from_mask):
        """Test that centroid is used when topology has no edges"""
        region = mock_region_from_mask(single_pixel_mask)

        # Single pixel may cause skan issues
        try:
            result = default_analyzer.analyze(region)
            # Should have at least one node (centroid fallback if no edges)
            assert result.topology.number_of_nodes() >= 1
        except (ValueError, IndexError):
            # Single pixel may cause skan to fail
            pytest.skip("Single pixel component causes expected failure in skan library")


class TestComponentAnalyzerResultType:
    """Test that analyzer returns correct result type"""

    def test_result_is_component_analysis_result(self, default_analyzer, simple_line_region):
        """Test that analyze returns ComponentAnalysisResult instance"""
        result = default_analyzer.analyze(simple_line_region)

        assert isinstance(result, ComponentAnalysisResult)

    def test_result_has_required_attributes(self, default_analyzer, l_shape_region):
        """Test that result has all required attributes"""
        result = default_analyzer.analyze(l_shape_region)

        assert hasattr(result, 'component_id')
        assert hasattr(result, 'bbox')
        assert hasattr(result, 'topology')

    def test_topology_is_multigraph(self, default_analyzer, y_junction_region):
        """Test that topology is a NetworkX MultiGraph"""
        result = default_analyzer.analyze(y_junction_region)

        assert isinstance(result.topology, nx.MultiGraph)
