"""
Boundary Connector - Main Orchestrator

Coordinates the entire epidermis-dermis connection process.
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .boundary_detector import BoundaryDetector
from shared.pathfinding import ImagePathfinder
from .component_analyzer import ComponentAnalyzer
from .connection_optimizer import ConnectionOptimizer
from .forest_merger import ForestMerger


class BoundaryConnector:
    """
    Main orchestrator for epidermis-dermis neural connection.

    Coordinates all stages of the connection process:
    1. Load MST forests and boundary information
    2. Identify boundary nodes for each component
    3. Compute connection costs between components
    4. Match epidermis components to dermis components
    5. Create crossing edges and merge forests
    """

    def __init__(
        self,
        boundary_tolerance: int = 10,
        max_crossing_distance: int = 100,
        verbose: bool = False
    ):
        """
        Initialize BoundaryConnector.

        Args:
            boundary_tolerance: Distance tolerance for boundary proximity (pixels)
            max_crossing_distance: Maximum distance for boundary crossing (pixels)
            verbose: Print debug information
        """
        self.boundary_tolerance = boundary_tolerance
        self.max_crossing_distance = max_crossing_distance
        self.verbose = verbose

    def connect_layers(
        self,
        epidermis_mst: str,
        dermis_mst: str,
        epidermis_mask: str,
        green_channel: str,
        output_dir: str
    ) -> Dict:
        """
        Execute the complete epidermis-dermis connection process.

        Args:
            epidermis_mst: Path to epidermis MST forest JSON
            dermis_mst: Path to dermis MST forest JSON
            epidermis_mask: Path to epidermis mask image
            green_channel: Path to green channel image
            output_dir: Output directory for results

        Returns:
            Merged MST forest data
        """
        print("=" * 60)
        print("EPIDERMIS-DERMIS NEURAL CONNECTION")
        print("=" * 60)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ========== Stage 1: Load Data ==========
        print("\n[1/7] Loading MST forests and images...")
        epi_mst_data = self._load_json(epidermis_mst)
        derm_mst_data = self._load_json(dermis_mst)
        epi_mask = cv2.imread(str(epidermis_mask), cv2.IMREAD_GRAYSCALE)
        green_img = cv2.imread(str(green_channel), cv2.IMREAD_GRAYSCALE)

        print(f"  ✓ Epidermis MST: {len(epi_mst_data['nodes'])} nodes, {len(epi_mst_data['edges'])} edges")
        print(f"  ✓ Dermis MST: {len(derm_mst_data['nodes'])} nodes, {len(derm_mst_data['edges'])} edges")
        print(f"  ✓ Images loaded: {green_img.shape}")

        # ========== Stage 2: Detect Boundary ==========
        print("\n[2/7] Detecting epidermis-dermis boundary...")
        boundary_detector = BoundaryDetector()
        boundary_detector.detect_boundary(epi_mask)
        print(f"  ✓ Boundary detected")

        # ========== Stage 3: Identify Boundary Nodes ==========
        print("\n[3/7] Identifying boundary nodes for each MST tree...")
        analyzer = ComponentAnalyzer(boundary_detector, self.boundary_tolerance)

        # Get all MST tree IDs (independent neural fibers)
        epi_tree_ids = analyzer.get_all_component_ids(epi_mst_data)
        derm_tree_ids = analyzer.get_all_component_ids(derm_mst_data)

        print(f"  ✓ Epidermis MST trees: {len(epi_tree_ids)}")
        print(f"  ✓ Dermis MST trees: {len(derm_tree_ids)}")

        # Get boundary nodes for each MST tree
        epi_boundary_nodes = {}
        for tree_id in epi_tree_ids:
            nodes = analyzer.get_component_boundary_nodes(epi_mst_data, tree_id)
            if nodes:  # Only include trees with boundary nodes
                epi_boundary_nodes[tree_id] = nodes

        derm_boundary_nodes = {}
        for tree_id in derm_tree_ids:
            nodes = analyzer.get_component_boundary_nodes(derm_mst_data, tree_id)
            if nodes:  # Only include trees with boundary nodes
                derm_boundary_nodes[tree_id] = nodes

        print(f"  ✓ Epidermis trees with boundary nodes: {len(epi_boundary_nodes)}")
        print(f"  ✓ Dermis trees with boundary nodes: {len(derm_boundary_nodes)}")

        # ========== Stage 4: Compute Connection Costs ==========
        print("\n[4/7] Computing connection costs between components...")
        pathfinder = ImagePathfinder(green_img, verbose=self.verbose)
        optimizer = ConnectionOptimizer(
            pathfinder,
            max_crossing_distance=self.max_crossing_distance,
            verbose=self.verbose
        )

        connection_costs = optimizer.compute_component_connection_costs(
            epi_boundary_nodes,
            derm_boundary_nodes
        )

        print(f"  ✓ Computed {len(connection_costs)} connection costs")

        # ========== Stage 5: Match Components ==========
        print("\n[5/7] Matching epidermis MST trees to dermis MST trees...")
        epi_tree_list = list(epi_boundary_nodes.keys())
        matches = optimizer.match_components(epi_tree_list, connection_costs)

        print(f"  ✓ Matched {len(matches)} epidermis MST trees")

        # Compute statistics
        statistics = optimizer.get_matching_statistics(matches)
        print(f"\n  Statistics:")
        print(f"    Total matches: {statistics['total_matches']}")
        print(f"    Unique dermis: {statistics['unique_dermis']}")
        print(f"    Dermis with multiple: {statistics['dermis_with_multiple']}")
        print(f"    Avg cost: {statistics['avg_cost']:.2f}")

        # ========== Stage 6: Create Crossing Edges ==========
        print("\n[6/7] Creating boundary crossing edges...")
        merger = ForestMerger(verbose=self.verbose)
        crossing_edges = merger.create_crossing_edges(matches)
        print(f"  ✓ Created {len(crossing_edges)} crossing edges")

        # ========== Stage 7: Merge Forests and Export ==========
        print("\n[7/7] Merging MST forests and exporting results...")
        merged_mst = merger.merge_mst_forests(epi_mst_data, derm_mst_data, crossing_edges)

        # Validate the merged result
        self._validate_merged_forest(merged_mst)

        # Export all outputs
        merger.export_merged_forest(
            merged_mst,
            output_dir / 'merged_mst_forest.json'
        )

        merger.export_crossing_details(
            matches,
            statistics,
            output_dir / 'crossing_connections.json'
        )

        merger.export_statistics_report(
            statistics,
            matches,
            output_dir / 'connection_statistics.txt'
        )

        print("\n" + "=" * 60)
        print("✓ EPIDERMIS-DERMIS CONNECTION COMPLETED!")
        print("=" * 60)
        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - merged_mst_forest.json")
        print(f"  - crossing_connections.json")
        print(f"  - connection_statistics.txt")

        return merged_mst

    def _validate_merged_forest(self, merged_mst: Dict):
        """
        Validate the merged MST forest to ensure each epidermis component
        has at most one boundary crossing edge.

        Args:
            merged_mst: Merged MST forest data

        Raises:
            ValueError: If validation fails
        """
        print("\nValidating merged forest...")

        # Count boundary crossing edges per epidermis component
        epi_comp_crossings = {}
        for edge in merged_mst['edges']:
            if edge.get('edge_type') == 'boundary_crossing':
                epi_comp = edge.get('epidermis_component')
                if epi_comp is not None:
                    epi_comp_crossings[epi_comp] = epi_comp_crossings.get(epi_comp, 0) + 1

        # Check for violations
        violations = []
        for epi_comp, count in epi_comp_crossings.items():
            if count > 1:
                violations.append((epi_comp, count))

        if violations:
            print(f"  ✗ VALIDATION FAILED!")
            print(f"     Found {len(violations)} epidermis components with multiple crossing edges:")
            for epi_comp, count in violations[:10]:  # Show first 10
                print(f"       - Epidermis component {epi_comp}: {count} edges")
            if len(violations) > 10:
                print(f"       ... and {len(violations) - 10} more")

            raise ValueError(
                f"Validation failed: {len(violations)} epidermis components have multiple crossing edges. "
                f"Each epidermis component should only have one connection."
            )
        else:
            print(f"  ✓ Validation passed:")
            print(f"     All {len(epi_comp_crossings)} epidermis components have exactly one crossing edge")

    def _load_json(self, file_path: str) -> Dict:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Connect epidermis and dermis neural networks across boundary',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--epidermis-mst',
        type=str,
        help='Path to epidermis MST forest JSON file',
        default='together/epidermis/mst_forest_with_paths.json'
    )
    parser.add_argument(
        '--dermis-mst',
        type=str,
        help='Path to dermis MST forest JSON file',
        default='together/dermis/mst_forest_with_paths.json'
    )
    parser.add_argument(
        '--epidermis-mask',
        type=str,
        help='Path to epidermis mask image',
        default="data/Mask/S163-2_a.tif"
    )
    parser.add_argument(
        '--green-channel',
        type=str,
        
        help='Path to green channel image',
        default="together/input_r_12.png"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        
        help='Output directory for results',
        default='together/boundary_connection_output'
    )

    # Optional arguments
    parser.add_argument(
        '--boundary-tolerance',
        type=int,
        default=10,
        help='Distance tolerance for boundary proximity (pixels)'
    )
    parser.add_argument(
        '--max-crossing-distance',
        type=int,
        default=100,
        help='Maximum distance for boundary crossing (pixels)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Initialize connector with parsed arguments
    connector = BoundaryConnector(
        boundary_tolerance=args.boundary_tolerance,
        max_crossing_distance=args.max_crossing_distance,
        verbose=args.verbose
    )

    # Execute connection process
    merged_mst = connector.connect_layers(
        epidermis_mst=args.epidermis_mst,
        dermis_mst=args.dermis_mst,
        epidermis_mask=args.epidermis_mask,
        green_channel=args.green_channel,
        output_dir=args.output_dir
    )

    print(f"\nMerged MST metadata: {merged_mst['metadata']}")
