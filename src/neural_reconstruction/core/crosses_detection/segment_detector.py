"""
Segment Detector

Identifies discrete segments of neural fibers. A segment is defined as a path
from one boundary node (endpoint or branchpoint) to another boundary node.
"""

from typing import Set, Tuple, FrozenSet
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class SegmentDetector:
    """
    Identifies discrete segments of neural fibers in a topology graph.

    A segment is defined as a path between two boundary nodes:
    - Endpoint (degree == 1)
    - Branchpoint (degree >= 3)

    Each segment consists of one or more consecutive edges.
    """

    def __init__(self):
        """Initialize the SegmentDetector."""
        logger.info("Initialized SegmentDetector")

    @classmethod
    def detect_segments(cls, graph: nx.Graph) -> nx.Graph:
        """
        Identify all segments in a topology graph.

        Each edge in the returned graph is annotated with a ``segment_id``
        attribute indicating which segment it belongs to. Nodes are annotated
        with a ``node_type`` attribute (``"endpoint"`` or ``"branchpoint"``).

        Args:
            graph: NetworkX Graph representing the fiber topology.

        Returns:
            A copy of the input graph with ``segment_id`` on edges and
            ``node_type`` on boundary nodes.
        """
        res_graph = graph.copy()

        if res_graph.number_of_nodes() == 0 or res_graph.number_of_edges() == 0:
            logger.warning("Topology graph is empty")
            return res_graph

        cls._identify_boundary_nodes(res_graph)
        cls._identify_segment_nodes(res_graph)

        return res_graph

    @classmethod
    def _identify_boundary_nodes(cls, graph: nx.Graph) -> None:
        """
        Label all boundary nodes in the graph with their node type.

        A boundary node is either:
        - An endpoint (degree == 1)
        - A branchpoint (degree >= 3)

        Nodes that qualify are annotated with ``node_type`` set to
        ``"endpoint"`` or ``"branchpoint"`` respectively.

        Args:
            graph: NetworkX Graph to annotate in-place.
        """
        for node_id in graph.nodes():
            degree = graph.degree(node_id)
            if degree == 1:
                graph.nodes[node_id]["node_type"] = "endpoint"
            elif degree >= 3:
                graph.nodes[node_id]["node_type"] = "branchpoint"
            else:
                # degree 0 or 2：清除前次執行殘留的 node_type，避免重新執行時誤判邊界
                graph.nodes[node_id].pop("node_type", None)

    @classmethod
    def _identify_segment_nodes(cls, graph: nx.Graph) -> None:
        """
        Assign a ``segment_id`` to every edge in the graph.

        A segment spans all edges between two boundary nodes (endpoints or
        branchpoints). Traversal starts from each boundary node and follows
        degree-2 intermediate nodes until another boundary node is reached.

        Args:
            graph: NetworkX Graph with ``node_type`` already set on boundary
                nodes. Edges are annotated in-place with ``segment_id``.
        """
        segment_id = 0
        visited_edges: Set[FrozenSet] = set()

        boundary_nodes = [
            node
            for node in graph.nodes()
            if graph.nodes[node].get("node_type") in ["endpoint", "branchpoint"]
        ]

        logger.debug(f"Found {len(boundary_nodes)} boundary nodes")

        for boundary_node in boundary_nodes:
            for neighbor in list(graph.neighbors(boundary_node)):
                edge_id = frozenset({boundary_node, neighbor})
                if edge_id in visited_edges:
                    continue

                cls._trace_and_label_segment(
                    graph, boundary_node, neighbor, segment_id, visited_edges
                )
                logger.debug(f"Labeled segment {segment_id}")
                segment_id += 1

        logger.info(f"Segment detection complete: {segment_id} segments found")

    @classmethod
    def _trace_and_label_segment(
        cls,
        graph: nx.Graph,
        start_from: Tuple[int, int],
        start_to: Tuple[int, int],
        segment_id: int,
        visited_edges: Set[FrozenSet],
    ) -> None:
        """
        Iteratively trace and label all edges belonging to a segment.

        Starting from the edge (``start_from`` → ``start_to``), this method
        follows degree-2 intermediate nodes until a boundary node is reached,
        assigning ``segment_id`` to every traversed edge.

        Args:
            graph: NetworkX Graph to annotate in-place.
            start_from: The entry node of the first edge.
            start_to: The exit node of the first edge.
            segment_id: The segment ID to assign to each traversed edge.
            visited_edges: Set of already-visited ``frozenset({u, v})`` edge
                identifiers, updated in-place to prevent revisiting edges.
        """
        stack = [(start_from, start_to)]

        while stack:
            from_node, current_node = stack.pop()
            edge_id = frozenset({from_node, current_node})

            if edge_id in visited_edges:
                continue

            graph[from_node][current_node]["segment_id"] = segment_id
            visited_edges.add(edge_id)

            # Stop at boundary nodes
            if graph.nodes[current_node].get("node_type") in ["endpoint", "branchpoint"]:
                continue

            # degree-2 中繼節點：繼續向前走訪
            for next_node in graph.neighbors(current_node):
                if next_node == from_node:
                    continue
                if frozenset({current_node, next_node}) not in visited_edges:
                    stack.append((current_node, next_node))
