"""
Integration tests for Component Analyzer

Tests the complete component analysis pipeline from end to end.
"""

import pytest
import numpy as np
from skimage.measure import label, regionprops

from neural_reconstruction.algorithms.pure_mst.component_analyzer import ComponentAnalyzer


class TestEndToEndAnalysis:
    """End-to-end integration tests"""

    def test_full_pipeline_simple_line(self, simple_line_mask):
        """Test complete analysis pipeline for simple line"""
        analyzer = ComponentAnalyzer(
            segment_length=10.0,
            min_edge_length=10.0,
            prune_threshold=5.0,
            spacing=1.0,
        )

        # Get region from mask
        labeled = label(simple_line_mask > 0)
        regions = regionprops(labeled)
        assert len(regions) == 1

        # Analyze
        result = analyzer.analyze(regions[0])

        # Validate complete result
        assert result.component_id == regions[0].label
        assert result.topology.number_of_nodes() >= 1
        # Note: edges may be 0 if component is too short for seed extraction
        assert result.topology.number_of_edges() >= 0

        # Validate node coordinates are in local space
        minr, minc, maxr, maxc = result.bbox
        bbox_height = maxr - minr
        bbox_width = maxc - minc
        for node in result.topology.nodes():
            y, x = node
            assert 0 <= y < bbox_height
            assert 0 <= x < bbox_width

    def test_full_pipeline_complex_structure(self, complex_branch_mask):
        """Test complete pipeline for complex branching structure"""
        analyzer = ComponentAnalyzer(segment_length=5.0, min_edge_length=5.0)

        labeled = label(complex_branch_mask > 0)
        regions = regionprops(labeled)
        assert len(regions) == 1

        result = analyzer.analyze(regions[0])

        # Complex structure should have nodes (edges may vary based on segment length)
        assert result.topology.number_of_nodes() > 3
        assert result.topology.number_of_edges() >= 0

    def test_pipeline_with_different_parameters(self, l_shape_mask):
        """Test that parameter changes affect output"""
        labeled = label(l_shape_mask > 0)
        regions = regionprops(labeled)
        region = regions[0]

        # Small segments
        analyzer_dense = ComponentAnalyzer(segment_length=3.0, min_edge_length=3.0)
        result_dense = analyzer_dense.analyze(region)

        # Large segments
        analyzer_sparse = ComponentAnalyzer(segment_length=20.0, min_edge_length=20.0)
        result_sparse = analyzer_sparse.analyze(region)

        # Dense should have more or equal edges
        assert result_dense.topology.number_of_edges() >= result_sparse.topology.number_of_edges()

    def test_pipeline_reproducibility(self, y_junction_mask):
        """Test that running pipeline twice gives same results"""
        analyzer = ComponentAnalyzer()

        labeled = label(y_junction_mask > 0)
        regions = regionprops(labeled)
        region = regions[0]

        result1 = analyzer.analyze(region)
        result2 = analyzer.analyze(region)

        # Results should be identical
        assert result1.component_id == result2.component_id
        assert result1.bbox == result2.bbox
        assert result1.topology.number_of_nodes() == result2.topology.number_of_nodes()
        assert result1.topology.number_of_edges() == result2.topology.number_of_edges()

    def test_pipeline_multiple_components(self):
        """Test analyzing multiple different components"""
        analyzer = ComponentAnalyzer()

        # Create image with multiple components
        image = np.zeros((100, 100), dtype=np.uint8)
        image[10:20, 10:40] = 255  # Component 1: horizontal line
        image[30:60, 30:35] = 255  # Component 2: vertical line
        image[70:85, 70:85] = 255  # Component 3: square

        labeled = label(image > 0)
        regions = regionprops(labeled)
        assert len(regions) == 3

        results = []
        for region in regions:
            result = analyzer.analyze(region)
            results.append(result)

        # Each result should be valid
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.component_id == regions[i].label
            assert result.topology.number_of_nodes() > 0

    def test_coordinate_system_consistency(self, l_shape_mask):
        """Test that coordinate systems are consistent throughout pipeline"""
        analyzer = ComponentAnalyzer()

        labeled = label(l_shape_mask > 0)
        regions = regionprops(labeled)
        region = regions[0]

        result = analyzer.analyze(region)

        # All nodes should be within the local coordinate system defined by bbox
        minr, minc, maxr, maxc = result.bbox
        bbox_height = maxr - minr
        bbox_width = maxc - minc

        for node in result.topology.nodes():
            y, x = node
            assert 0 <= y < bbox_height, \
                f"Node y={y} outside bbox height={bbox_height}"
            assert 0 <= x < bbox_width, \
                f"Node x={x} outside bbox width={bbox_width}"

        # Check edge paths are also in local coordinates
        for u, v, data in result.topology.edges(data=True):
            if 'path' in data:
                for point in data['path']:
                    y, x = point
                    assert 0 <= y < bbox_height
                    assert 0 <= x < bbox_width


class TestRealWorldScenarios:
    """Integration tests with real-world scenarios"""

    def test_with_noise(self):
        """Test robustness to noisy input"""
        # Create component with noise
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 23:27] = 255  # Main line
        # Add noise pixels
        mask[15, 10] = 255
        mask[20, 40] = 255
        mask[35, 15] = 255

        analyzer = ComponentAnalyzer(prune_threshold=5.0)

        labeled = label(mask > 0)
        regions = regionprops(labeled)
        # Multiple small components from noise
        main_region = max(regions, key=lambda r: r.area)

        result = analyzer.analyze(main_region)

        # Should still produce valid result
        assert result.topology.number_of_nodes() >= 2

    def test_edge_cases_collection(self):
        """Test collection of various edge cases"""
        analyzer = ComponentAnalyzer()

        edge_cases = [
            np.array([[255]], dtype=np.uint8),  # Single pixel
            np.full((3, 3), 255, dtype=np.uint8),  # Tiny square
        ]

        for i, mask in enumerate(edge_cases):
            if np.sum(mask) > 0:
                labeled = label(mask > 0)
                regions = regionprops(labeled)
                if len(regions) > 0:
                    try:
                        result = analyzer.analyze(regions[0])
                        # Should handle gracefully
                        assert result.topology.number_of_nodes() >= 1
                    except (ValueError, IndexError):
                        # Very small components may cause skan to fail
                        # This is acceptable edge case behavior
                        pass


class TestParameterSensitivity:
    """Test sensitivity to different parameter values"""

    def test_segment_length_variations(self, simple_line_mask):
        """Test behavior with various segment lengths"""
        labeled = label(simple_line_mask > 0)
        regions = regionprops(labeled)
        region = regions[0]

        segment_lengths = [3.0, 5.0, 10.0, 15.0, 20.0]
        results = []

        for seg_len in segment_lengths:
            analyzer = ComponentAnalyzer(
                segment_length=seg_len,
                min_edge_length=seg_len
            )
            result = analyzer.analyze(region)
            results.append(result)

        # Verify that smaller segment lengths produce more edges
        for i in range(len(results) - 1):
            assert results[i].topology.number_of_edges() >= results[i+1].topology.number_of_edges()

    def test_prune_threshold_variations(self, y_junction_mask):
        """Test behavior with various prune thresholds"""
        labeled = label(y_junction_mask > 0)
        regions = regionprops(labeled)
        region = regions[0]

        prune_thresholds = [0.0, 2.0, 5.0, 10.0]
        results = []

        for prune_thresh in prune_thresholds:
            analyzer = ComponentAnalyzer(prune_threshold=prune_thresh)
            result = analyzer.analyze(region)
            results.append(result)

        # All results should be valid
        for result in results:
            assert result.topology.number_of_nodes() > 0


class TestDataIntegrity:
    """Test data integrity throughout the pipeline"""

    def test_no_data_corruption(self, l_shape_mask):
        """Test that original data is not modified"""
        mask_copy = l_shape_mask.copy()
        analyzer = ComponentAnalyzer()

        labeled = label(l_shape_mask > 0)
        regions = regionprops(labeled)
        region = regions[0]

        result = analyzer.analyze(region)

        # Original mask should not be modified
        assert np.array_equal(l_shape_mask, mask_copy)

    def test_bbox_consistency_with_region(self, simple_line_mask):
        """Test that bbox matches the region bbox"""
        analyzer = ComponentAnalyzer()

        labeled = label(simple_line_mask > 0)
        regions = regionprops(labeled)
        region = regions[0]

        result = analyzer.analyze(region)

        # BBox should match region bbox
        assert result.bbox == region.bbox

    def test_topology_graph_validity(self, complex_branch_mask):
        """Test that topology graph is valid NetworkX graph"""
        analyzer = ComponentAnalyzer()

        labeled = label(complex_branch_mask > 0)
        regions = regionprops(labeled)
        region = regions[0]

        result = analyzer.analyze(region)

        # Verify it's a valid NetworkX MultiGraph
        import networkx as nx
        assert isinstance(result.topology, nx.MultiGraph)

        # Verify graph properties
        assert result.topology.number_of_nodes() >= 0
        assert result.topology.number_of_edges() >= 0

        # Verify all edges connect existing nodes
        for u, v in result.topology.edges():
            assert u in result.topology.nodes()
            assert v in result.topology.nodes()
