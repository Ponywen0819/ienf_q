"""
Forest Merger

Merges epidermis and dermis MST forests with boundary crossing edges.
"""

from typing import List, Dict, Tuple
import json
import numpy as np
from pathlib import Path
from datetime import datetime


class ForestMerger:
    """
    Merges two MST forests and adds boundary crossing edges.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize ForestMerger.

        Args:
            verbose: Print debug information
        """
        self.verbose = verbose

    @staticmethod
    def _convert_to_json_serializable(obj):
        """
        Convert numpy types to native Python types for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [ForestMerger._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: ForestMerger._convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj

    def create_crossing_edges(
        self,
        matches: Dict[int, Tuple[int, Dict]]
    ) -> List[Dict]:
        """
        Create boundary crossing edge data from component matches.

        Args:
            matches: {epi_id: (derm_id, connection_info)} from ConnectionOptimizer

        Returns:
            List of edge dictionaries
        """
        crossing_edges = []

        for epi_id, (derm_id, connection) in matches.items():
            epi_node = connection['epi_node']
            derm_node = connection['derm_node']
            path = connection['path']
            cost = connection['cost']

            # Create edge in MST format - convert all values to JSON-serializable types
            # Use prefixed node IDs to avoid conflicts between epidermis and dermis
            edge = {
                'source_id': f"epi_{epi_node['id']}",
                'target_id': f"derm_{derm_node['id']}",
                'weight': float(cost),
                'edge_type': 'boundary_crossing',
                'image_cost': float(cost),
                'geometric_cost': float(connection['euclidean_distance']),
                'curvature_cost': 0.0,
                'path': self._convert_to_json_serializable(path),
                'path_cost': float(cost),
                'path_length': int(connection['path_length']),
                'tortuosity': float(connection['path_length'] / max(connection['euclidean_distance'], 1.0)),
                'epidermis_component': int(epi_id),
                'dermis_component': int(derm_id)
            }

            crossing_edges.append(edge)

        return crossing_edges

    def _deduplicate_crossing_edges(
        self,
        crossing_edges: List[Dict]
    ) -> List[Dict]:
        """
        Remove duplicate crossing edges for the same epidermis component.

        Each epidermis component should only have one crossing edge.
        If multiple edges exist for the same epidermis component,
        keep only the one with the lowest cost.

        Args:
            crossing_edges: List of crossing edge dictionaries

        Returns:
            Deduplicated list of crossing edges
        """
        # Group edges by epidermis component
        epi_comp_edges = {}
        for edge in crossing_edges:
            epi_comp = edge['epidermis_component']

            if epi_comp not in epi_comp_edges:
                epi_comp_edges[epi_comp] = []

            epi_comp_edges[epi_comp].append(edge)

        # For each epidermis component, keep only the edge with lowest cost
        deduplicated = []
        for epi_comp, edges in epi_comp_edges.items():
            if len(edges) > 1:
                # Sort by cost and keep the best one
                best_edge = min(edges, key=lambda e: e['weight'])
                deduplicated.append(best_edge)

                if self.verbose:
                    print(f"  ⚠ Epidermis component {epi_comp} had {len(edges)} edges, kept best (cost={best_edge['weight']:.2f})")
            else:
                # Only one edge, keep it
                deduplicated.append(edges[0])

        return deduplicated

    def merge_mst_forests(
        self,
        epi_mst: Dict,
        derm_mst: Dict,
        crossing_edges: List[Dict]
    ) -> Dict:
        """
        Merge epidermis and dermis MST forests with crossing edges.

        Args:
            epi_mst: Epidermis MST forest data
            derm_mst: Dermis MST forest data
            crossing_edges: List of crossing edge dictionaries

        Returns:
            Merged MST forest data
        """
        # Count unique components
        epi_components = len(set(n['component_id'] for n in epi_mst['nodes'] if 'component_id' in n))
        derm_components = len(set(n['component_id'] for n in derm_mst['nodes'] if 'component_id' in n))

        # Add namespace prefixes to node IDs to avoid conflicts
        # Epidermis nodes get "epi_" prefix, dermis nodes get "derm_" prefix
        epi_nodes = [
            {**node, 'id': f"epi_{node['id']}"}
            for node in epi_mst['nodes']
        ]
        derm_nodes = [
            {**node, 'id': f"derm_{node['id']}"}
            for node in derm_mst['nodes']
        ]

        # Update edge source/target IDs to match the new node IDs
        epi_edges = []
        for edge in epi_mst['edges']:
            updated_edge = edge.copy()
            # Handle both 'source'/'target' and 'source_id'/'target_id' formats
            if 'source' in edge:
                updated_edge['source'] = f"epi_{edge['source']}"
            if 'source_id' in edge:
                updated_edge['source_id'] = f"epi_{edge['source_id']}"
            if 'target' in edge:
                updated_edge['target'] = f"epi_{edge['target']}"
            if 'target_id' in edge:
                updated_edge['target_id'] = f"epi_{edge['target_id']}"
            epi_edges.append(updated_edge)

        derm_edges = []
        for edge in derm_mst['edges']:
            updated_edge = edge.copy()
            if 'source' in edge:
                updated_edge['source'] = f"derm_{edge['source']}"
            if 'source_id' in edge:
                updated_edge['source_id'] = f"derm_{edge['source_id']}"
            if 'target' in edge:
                updated_edge['target'] = f"derm_{edge['target']}"
            if 'target_id' in edge:
                updated_edge['target_id'] = f"derm_{edge['target_id']}"
            derm_edges.append(updated_edge)

        # Merge nodes (with prefixed IDs)
        merged_nodes = epi_nodes + derm_nodes

        # Deduplicate crossing edges by epidermis component
        # Ensure each epidermis component has at most one crossing edge
        deduplicated_crossing_edges = self._deduplicate_crossing_edges(crossing_edges)

        if len(deduplicated_crossing_edges) < len(crossing_edges):
            duplicates_removed = len(crossing_edges) - len(deduplicated_crossing_edges)
            print(f"  ⚠ Warning: Removed {duplicates_removed} duplicate crossing edges")
            print(f"     (Each epidermis component should only have one connection)")

        # Merge edges (with updated IDs)
        merged_edges = epi_edges + derm_edges + deduplicated_crossing_edges

        # Create merged forest structure
        merged_mst = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'epidermis_components': epi_components,
                'dermis_components': derm_components,
                'boundary_crossings': len(deduplicated_crossing_edges),
                'total_nodes': len(merged_nodes),
                'total_edges': len(merged_edges),
                'epidermis_nodes': len(epi_mst['nodes']),
                'dermis_nodes': len(derm_mst['nodes']),
                'epidermis_edges': len(epi_mst['edges']),
                'dermis_edges': len(derm_mst['edges'])
            },
            'nodes': merged_nodes,
            'edges': merged_edges
        }

        if self.verbose:
            print("\nMerged MST Forest Summary:")
            print(f"  Total nodes: {len(merged_nodes)}")
            print(f"  Total edges: {len(merged_edges)}")
            print(f"  Epidermis components: {epi_components}")
            print(f"  Dermis components: {derm_components}")
            print(f"  Boundary crossings: {len(deduplicated_crossing_edges)}")

        return merged_mst

    def export_merged_forest(
        self,
        merged_mst: Dict,
        output_path: str
    ):
        """
        Export merged MST forest to JSON file.

        Args:
            merged_mst: Merged MST forest data
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_mst, f, indent=2)

        print(f"\n  ✓ Merged MST forest saved to: {output_path}")

    def export_crossing_details(
        self,
        matches: Dict[int, Tuple[int, Dict]],
        statistics: Dict,
        output_path: str
    ):
        """
        Export detailed crossing connection information.

        Args:
            matches: Component matching results
            statistics: Matching statistics
            output_path: Output file path
        """
        # Convert matches to serializable format
        connections = []
        for epi_id, (derm_id, connection) in matches.items():
            connections.append({
                'epidermis_component': int(epi_id),
                'dermis_component': int(derm_id),
                'connection': {
                    'epi_node_id': int(connection['epi_node']['id']),
                    'derm_node_id': int(connection['derm_node']['id']),
                    'path_cost': float(connection['cost']),
                    'path_length': int(connection['path_length']),
                    'euclidean_distance': float(connection['euclidean_distance']),
                    'tortuosity': float(connection['path_length'] / max(connection['euclidean_distance'], 1.0))
                }
            })

        crossing_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_connections': len(connections)
            },
            'statistics': self._convert_to_json_serializable(statistics),
            'connections': connections
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(crossing_data, f, indent=2)

        print(f"  ✓ Crossing details saved to: {output_path}")

    def export_statistics_report(
        self,
        statistics: Dict,
        matches: Dict[int, Tuple[int, Dict]],
        output_path: str
    ):
        """
        Export human-readable statistics report.

        Args:
            statistics: Matching statistics
            matches: Component matching results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count dermis with multiple epidermis
        dermis_counts = {}
        for epi_id, (derm_id, _) in matches.items():
            dermis_counts[derm_id] = dermis_counts.get(derm_id, 0) + 1

        multi_connected_dermis = {
            derm_id: count
            for derm_id, count in dermis_counts.items()
            if count > 1
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EPIDERMIS-DERMIS CONNECTION STATISTICS\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("MATCHING RESULTS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total connections:        {statistics['total_matches']}\n")
            f.write(f"Unique dermis components: {statistics['unique_dermis']}\n")
            f.write(f"Dermis with multiple:     {statistics['dermis_with_multiple']}\n\n")

            f.write("COST STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average cost:  {statistics['avg_cost']:.2f}\n")
            f.write(f"Median cost:   {statistics['median_cost']:.2f}\n")
            f.write(f"Std deviation: {statistics['std_cost']:.2f}\n")
            f.write(f"Min cost:      {statistics['min_cost']:.2f}\n")
            f.write(f"Max cost:      {statistics['max_cost']:.2f}\n\n")

            if multi_connected_dermis:
                f.write("DERMIS COMPONENTS WITH MULTIPLE EPIDERMIS CONNECTIONS\n")
                f.write("-" * 60 + "\n")
                for derm_id, count in sorted(multi_connected_dermis.items(), key=lambda x: x[1], reverse=True):
                    epi_ids = [e_id for e_id, (d_id, _) in matches.items() if d_id == derm_id]
                    f.write(f"Dermis {derm_id}: {count} epidermis components {epi_ids}\n")

        print(f"  ✓ Statistics report saved to: {output_path}")


if __name__ == '__main__':
    # Test code
    import json

    # Load test data
    with open('epidermis_res/mst_forest_with_paths.json', 'r') as f:
        epi_mst = json.load(f)

    with open('dermis_res/mst_forest_with_paths.json', 'r') as f:
        derm_mst = json.load(f)

    # Mock matches
    mock_matches = {
        1: (5, {
            'epi_node': {'id': 10, 'x': 100, 'y': 200},
            'derm_node': {'id': 50, 'x': 105, 'y': 210},
            'path': [[200, 100], [205, 102], [210, 105]],
            'cost': 12.5,
            'euclidean_distance': 10.0,
            'path_length': 15
        })
    }

    # Create merger
    merger = ForestMerger(verbose=True)

    # Create crossing edges
    crossing_edges = merger.create_crossing_edges(mock_matches)
    print(f"\nCrossing edges: {crossing_edges}")

    # Merge forests
    merged = merger.merge_mst_forests(epi_mst, derm_mst, crossing_edges)
    print(f"\nMerged metadata: {merged['metadata']}")
