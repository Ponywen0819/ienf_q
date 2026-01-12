"""
MST Forest Builder for Neural Network Reconstruction

Constructs a minimum spanning tree (MST) forest from component connections.
Allows multiple independent connected components (forest structure).
"""

import networkx as nx
import logging

logger = logging.getLogger(__name__)


class BackboneExtractor:
    """
    MST Forest Builder

    Builds a minimum spanning tree forest from a component connection graph.
    The forest may contain multiple disconnected trees if components cannot
    be connected (after cost filtering in the pairing stage).
    """

    def __init__(
        self,
    ):
        """Initialize MST builder."""
        logger.info("Initialized MSTBuilder")

    def extract(self, G: nx.Graph) -> nx.Graph:
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
        logger.info(
            f"  Input graph: {G.number_of_nodes()} components, {G.number_of_edges()} connections"
        )

        # Find all connected components
        components = list(nx.connected_components(G))
        logger.info(f"  Number of connected components: {len(components)}")

        # Build MST for each component
        forest = nx.Graph()

        for i, component_nodes in enumerate(components):
            subtree_backbone = self._extract_subtree(G, component_nodes)
            forest = nx.compose(forest, subtree_backbone)

        logger.info(
            f"  MST forest complete: {forest.number_of_nodes()} components, {forest.number_of_edges()} connections"
        )

        return forest

    def _extract_subtree(self, G: nx.Graph, nodes: set) -> nx.Graph:
        """
        Extract subtree from graph given a set of nodes.

        Args:
            G: Input graph
            nodes: Set of nodes to include in the subtree
        Returns:
            subtree: Extracted subtree
        """
        subgraph = G.subgraph(nodes)

        # If only one node or no edges, create a new graph with just the nodes
        if subgraph.number_of_nodes() <= 1 or subgraph.number_of_edges() == 0:
            result = nx.Graph()
            result.add_nodes_from(subgraph.nodes(data=True))
            return result

        # Build MST
        mst = nx.minimum_spanning_tree(subgraph, weight="weight")

        return mst
