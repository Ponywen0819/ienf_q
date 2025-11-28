"""
Minimal Component Graph Builder for MST-Based Connection Filtering

This module constructs a simple graph from component pairing results:
- Nodes: Component IDs
- Edges: Inter-component connections with cost weights

The graph is used to compute MST and determine which connections to keep or remove.
"""

import networkx as nx
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ComponentGraphBuilder:
    """
    Builds a minimal component-level graph from pairing results.

    The graph is used purely for MST computation to filter connections.
    """

    def __init__(self):
        """Initialize the component graph builder."""
        logger.info("Initialized ComponentGraphBuilder")

    def build_graph(self, pairing_results: Dict) -> nx.Graph:
        """
        Build a minimal graph from component pairing results.

        Args:
            pairing_results: Component pairing analysis results containing 'connections'

        Returns:
            NetworkX graph with component IDs as nodes and costs as edge weights
        """
        logger.info("Building component graph from pairing results...")

        G = nx.Graph()
        connections = pairing_results.get('connections', [])

        if not connections:
            logger.warning("No connections found in pairing results")
            return G

        # Add edges (nodes are created automatically)
        for conn in connections:
            component_a = conn['component_a_id']
            component_b = conn['component_b_id']
            cost = conn['cost']

            G.add_edge(component_a, component_b, weight=cost)

        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    def filter_connections_by_mst(
        self,
        pairing_results: Dict,
        mst_forest: nx.Graph
    ) -> Dict:
        """
        Filter connections based on MST forest results.

        Connections in the MST are kept, others are removed.

        Args:
            pairing_results: Original component pairing results
            mst_forest: MST forest computed from the graph

        Returns:
            Dictionary with 'connections_to_keep' and 'connections_removed'
        """
        logger.info("Filtering connections based on MST...")

        connections = pairing_results.get('connections', [])

        # Get edges in MST
        mst_edges = set()
        for u, v in mst_forest.edges():
            # Store as sorted tuple to handle undirected edges
            mst_edges.add(tuple(sorted([u, v])))

        # Filter connections
        connections_to_keep = []
        connections_removed = []

        for conn in connections:
            edge = tuple(sorted([conn['component_a_id'], conn['component_b_id']]))

            if edge in mst_edges:
                connections_to_keep.append(conn)
            else:
                connections_removed.append(conn)

        logger.info(f"Kept {len(connections_to_keep)} connections, removed {len(connections_removed)}")

        return {
            'connections_to_keep': connections_to_keep,
            'connections_removed': connections_removed,
            'num_kept': len(connections_to_keep),
            'num_removed': len(connections_removed)
        }