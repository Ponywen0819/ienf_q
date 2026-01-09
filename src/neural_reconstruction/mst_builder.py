"""
MST Forest Builder for Neural Network Reconstruction

Constructs a minimum spanning tree (MST) forest from component connections.
Allows multiple independent connected components (forest structure).
"""

import networkx as nx
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class MSTBuilder:
    """
    MST Forest Builder

    Builds a minimum spanning tree forest from a component connection graph.
    The forest may contain multiple disconnected trees if components cannot
    be connected (after cost filtering in the pairing stage).
    """

    def __init__(self):
        """Initialize MST builder."""
        logger.info("Initialized MSTBuilder")

    def build_mst_forest(self, G: nx.Graph) -> nx.Graph:
        """
        Build MST forest from component connection graph.

        Strategy:
        1. Find all connected components in the graph
        2. Build MST independently for each connected component
        3. Merge all component MSTs to form a forest

        Args:
            G: Input graph with component IDs as nodes and costs as edge weights

        Returns:
            forest: MST forest, may contain multiple connected components
        """
        if G is None or G.number_of_nodes() == 0:
            logger.warning("Input graph is empty")
            return nx.Graph()

        logger.info("Building MST forest...")
        logger.info(f"  Input graph: {G.number_of_nodes()} components, {G.number_of_edges()} connections")

        # Find all connected components
        components = list(nx.connected_components(G))
        logger.info(f"  Number of connected components: {len(components)}")

        # Build MST for each component
        forest = nx.Graph()

        for i, component_nodes in enumerate(components):
            # Extract subgraph
            subgraph = G.subgraph(component_nodes).copy()

            # If only one node, add directly
            if subgraph.number_of_nodes() == 1:
                forest.add_nodes_from(subgraph.nodes(data=True))
                continue

            # Build MST
            mst = nx.minimum_spanning_tree(subgraph, weight='weight')

            # Add to forest
            forest.add_nodes_from(mst.nodes(data=True))
            forest.add_edges_from(mst.edges(data=True))

            if i < 5 or len(components) <= 10:
                logger.info(f"    Cluster {i+1}: {mst.number_of_nodes()} components, {mst.number_of_edges()} connections")

        if len(components) > 10:
            logger.info(f"    ... ({len(components) - 5} clusters not shown)")

        logger.info(f"  MST forest complete: {forest.number_of_nodes()} components, {forest.number_of_edges()} connections")

        return forest