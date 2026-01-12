"""
Pytest fixtures for component analyzer tests

Provides reusable fixtures for:
- Synthetic component masks
- RegionProperties objects
- Analyzer instances
- Validation helpers
"""

import pytest
import numpy as np
from skimage.measure import label, regionprops
from pathlib import Path


# =============================================================================
# Mask Fixtures
# =============================================================================

@pytest.fixture
def simple_line_mask():
    """Horizontal line: 10x50 pixels, 2px thick"""
    from .fixtures.synthetic_shapes import create_simple_line
    return create_simple_line()


@pytest.fixture
def l_shape_mask():
    """L-shaped component: 50x50 pixels"""
    from .fixtures.synthetic_shapes import create_l_shape
    return create_l_shape()


@pytest.fixture
def y_junction_mask():
    """Y-junction component: 50x50 pixels"""
    from .fixtures.synthetic_shapes import create_y_junction
    return create_y_junction()


@pytest.fixture
def complex_branch_mask():
    """Complex multi-branch structure: 100x100 pixels"""
    from .fixtures.synthetic_shapes import create_complex_branch
    return create_complex_branch()


@pytest.fixture
def tiny_component_mask():
    """Very small component: 10x10 pixels"""
    from .fixtures.synthetic_shapes import create_tiny_component
    return create_tiny_component()


@pytest.fixture
def empty_mask():
    """Empty mask for edge case testing"""
    return np.zeros((50, 50), dtype=np.uint8)


@pytest.fixture
def single_pixel_mask():
    """Single pixel component"""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255
    return mask


# =============================================================================
# Region Fixtures
# =============================================================================

@pytest.fixture
def mock_region_from_mask():
    """
    Factory fixture to create RegionProperties from a mask

    Usage:
        region = mock_region_from_mask(mask, component_id=1)
    """
    def _create_region(mask, component_id=1):
        binary = (mask > 0).astype(np.uint8)
        labeled = label(binary)
        regions = regionprops(labeled)

        if len(regions) == 0:
            # For empty masks, create a mock region
            # Note: Real analyzer may fail on empty regions, which is expected behavior
            from unittest.mock import Mock
            mock_region = Mock()
            mock_region.label = component_id
            mock_region.bbox = (0, 0, mask.shape[0], mask.shape[1])
            # Create an empty binary image property
            mock_region.image = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
            mock_region.area = 0
            return mock_region

        # Return actual region
        # Note: region.label is set by label() function, can't override
        return regions[0]

    return _create_region


@pytest.fixture
def simple_line_region(simple_line_mask, mock_region_from_mask):
    """RegionProperties for simple line"""
    return mock_region_from_mask(simple_line_mask, component_id=1)


@pytest.fixture
def l_shape_region(l_shape_mask, mock_region_from_mask):
    """RegionProperties for L-shape"""
    return mock_region_from_mask(l_shape_mask, component_id=2)


@pytest.fixture
def y_junction_region(y_junction_mask, mock_region_from_mask):
    """RegionProperties for Y-junction"""
    return mock_region_from_mask(y_junction_mask, component_id=3)


# =============================================================================
# Analyzer Instance Fixtures
# =============================================================================

@pytest.fixture
def default_analyzer():
    """ComponentAnalyzer with default parameters"""
    from neural_reconstruction.core.construction.component_analyzer import ComponentAnalyzer
    return ComponentAnalyzer(
        segment_length=10.0,
        min_edge_length=10.0,
        prune_threshold=5.0,
        spacing=1.0,
    )


@pytest.fixture
def small_segment_analyzer():
    """ComponentAnalyzer with small segment length for dense seeds"""
    from neural_reconstruction.core.construction.component_analyzer import ComponentAnalyzer
    return ComponentAnalyzer(
        segment_length=3.0,
        min_edge_length=3.0,
        prune_threshold=2.0,
        spacing=1.0,
    )


@pytest.fixture
def large_segment_analyzer():
    """ComponentAnalyzer with large segment length for sparse seeds"""
    from neural_reconstruction.core.construction.component_analyzer import ComponentAnalyzer
    return ComponentAnalyzer(
        segment_length=20.0,
        min_edge_length=20.0,
        prune_threshold=10.0,
        spacing=1.0,
    )


@pytest.fixture
def topology_builder():
    """ComponentTopologyBuilder with default parameters"""
    from neural_reconstruction.core.construction.component_analyzer.topology import ComponentTopologyBuilder
    return ComponentTopologyBuilder(prune_threshold=5.0, spacing=1.0)


@pytest.fixture
def seed_generator():
    """EdgeSeedGenerator with default parameters"""
    from neural_reconstruction.core.construction.component_analyzer.seed_extraction import EdgeSeedGenerator
    return EdgeSeedGenerator(min_edge_length=10.0)


# =============================================================================
# Validation Helper Fixtures
# =============================================================================

@pytest.fixture
def topology_validator():
    """Helper to validate topology graph properties"""
    def _validate(topology, expected_min_nodes=0, expected_min_edges=0):
        assert topology is not None, "Topology should not be None"
        assert hasattr(topology, 'number_of_nodes'), "Topology should have number_of_nodes method"
        assert hasattr(topology, 'number_of_edges'), "Topology should have number_of_edges method"
        assert topology.number_of_nodes() >= expected_min_nodes, \
            f"Expected at least {expected_min_nodes} nodes, got {topology.number_of_nodes()}"
        assert topology.number_of_edges() >= expected_min_edges, \
            f"Expected at least {expected_min_edges} edges, got {topology.number_of_edges()}"

        # Validate node attributes
        for node in topology.nodes():
            assert isinstance(node, tuple), f"Node should be tuple, got {type(node)}"
            assert len(node) == 2, f"Node should be (y, x) coordinates, got {node}"
            assert all(isinstance(coord, (int, np.integer)) for coord in node), \
                f"Node coordinates should be integers, got {node}"

        # Validate edge attributes
        for u, v, data in topology.edges(data=True):
            if 'path' in data:
                assert isinstance(data['path'], list), "Edge path should be a list"
                assert all(isinstance(p, tuple) for p in data['path']), \
                    "Edge path points should be tuples"

        return True

    return _validate


@pytest.fixture
def component_analysis_result_validator():
    """Helper to validate ComponentAnalysisResult"""
    def _validate(result):
        assert result is not None, "Result should not be None"
        assert hasattr(result, 'component_id'), "Result should have component_id"
        assert hasattr(result, 'bbox'), "Result should have bbox"
        assert hasattr(result, 'topology'), "Result should have topology"

        # Validate bbox format
        assert len(result.bbox) == 4, f"BBox should have 4 elements, got {len(result.bbox)}"
        minr, minc, maxr, maxc = result.bbox
        assert minr < maxr, f"BBox minr ({minr}) should be less than maxr ({maxr})"
        assert minc < maxc, f"BBox minc ({minc}) should be less than maxc ({maxc})"

        # Validate topology is a graph
        assert hasattr(result.topology, 'number_of_nodes'), "Topology should be a NetworkX graph"
        assert hasattr(result.topology, 'number_of_edges'), "Topology should be a NetworkX graph"

        return True

    return _validate


# =============================================================================
# Fixture File Loaders
# =============================================================================

@pytest.fixture
def fixture_dir():
    """Path to fixtures directory"""
    return Path(__file__).parent / 'fixtures'


@pytest.fixture
def load_fixture_from_file(fixture_dir):
    """Factory to load .npy fixture files"""
    def _load(filename):
        filepath = fixture_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Fixture file not found: {filepath}")
        return np.load(filepath)

    return _load
