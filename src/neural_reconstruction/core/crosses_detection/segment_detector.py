"""
Segment Detector

Identifies discrete segments of neural fibers. A segment is defined as a path
from one boundary node (endpoint or branchpoint) to another boundary node.
"""

from typing import Set, Tuple
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
        nodes = res_graph.nodes(data=True)
        edges = res_graph.edges(data=True)

        if not nodes or not edges:
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
        visited_edges: Set[Tuple] = set()

        boundary_nodes = [
            node
            for node in graph.nodes()
            if graph.nodes[node].get("node_type") in ["endpoint", "branchpoint"]
        ]

        logger.debug(f"Found {len(boundary_nodes)} boundary nodes")

        for boundary_node in boundary_nodes:
            for neighbor in list(graph.neighbors(boundary_node)):
                for key in list(graph[boundary_node][neighbor].keys()):
                    edge_id = (boundary_node, neighbor, key)

                    if (
                        edge_id in visited_edges
                        or (neighbor, boundary_node, key) in visited_edges
                    ):
                        continue

                    cls._trace_and_label_segment(
                        graph, boundary_node, neighbor, key, segment_id, visited_edges
                    )
                    logger.debug(f"Labeled segment {segment_id}")
                    segment_id += 1

        logger.info(f"Segment detection complete: {segment_id} segments found")

    @classmethod
    def _trace_and_label_segment(
        cls,
        graph: nx.Graph,
        from_node: Tuple[int, int],
        current_node: Tuple[int, int],
        edge_key,
        segment_id: int,
        visited_edges: Set[Tuple],
    ) -> None:
        """
        Recursively trace and label all edges belonging to a segment.

        Starting from the edge (``from_node`` → ``current_node``), this method
        follows degree-2 intermediate nodes until a boundary node is reached,
        assigning ``segment_id`` to every traversed edge.

        Args:
            graph: NetworkX Graph to annotate in-place.
            from_node: The previously visited node (used to avoid backtracking).
            current_node: The node currently being processed.
            edge_key: The multi-edge key for the edge between ``from_node`` and
                ``current_node``.
            segment_id: The segment ID to assign to each traversed edge.
            visited_edges: Set of already-visited ``(u, v, key)`` tuples,
                updated in-place to prevent revisiting edges.
        """
        edge_id = (from_node, current_node, edge_key)

        if (
            edge_id in visited_edges
            or (current_node, from_node, edge_key) in visited_edges
        ):
            return

        graph[from_node][current_node][edge_key]["segment_id"] = segment_id
        visited_edges.add(edge_id)
        visited_edges.add((current_node, from_node, edge_key))

        # Stop at boundary nodes
        if graph.nodes[current_node].get("node_type") in ["endpoint", "branchpoint"]:
            return

        # current_node is a degree-2 intermediate node; continue traversal
        for next_node in graph.neighbors(current_node):
            if next_node == from_node:
                continue  # avoid backtracking

            for next_key in graph[current_node][next_node].keys():
                cls._trace_and_label_segment(
                    graph, current_node, next_node, next_key, segment_id, visited_edges
                )
