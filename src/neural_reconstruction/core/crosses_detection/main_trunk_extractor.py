"""
Main Trunk Extractor

Identifies the main trunk in each connected component of a fiber topology and
merges the trunk's segments into a single ``segment_id``.

Expects a graph already processed by :class:`SegmentDetector` so every edge has
a ``segment_id`` and every boundary node has ``node_type`` set to either
``"endpoint"`` or ``"branchpoint"``.

Algorithm:
1. Split the graph into connected components.
2. In each component, pick the start endpoint with the largest ``y`` (and
   smallest ``x`` to break ties).
3. Build a segment-level meta-graph (boundary nodes only; segments become
   weighted edges, weight = total ``path`` pixel length).
4. Run an iterative DFS from the start to find the longest single path.
5. Relabel every edge along the trunk with the smallest ``segment_id`` among
   them so the trunk becomes one segment.

Components without endpoints (pure cycles, isolated nodes) are skipped.
"""

from collections import defaultdict
from typing import Dict, Hashable, List, Set, Tuple
import logging

import networkx as nx

logger = logging.getLogger(__name__)


_BOUNDARY_TYPES = ("endpoint", "branchpoint")


class MainTrunkExtractor:
    """
    Merges each component's main trunk into a single segment.

    The main trunk in a component is the longest single path that starts from
    the lowest endpoint (max ``y``, ties broken by min ``x``). Length is
    measured in pixels along the edges' ``path`` attribute.
    """

    def __init__(self):
        logger.info("Initialized MainTrunkExtractor")

    @classmethod
    def extract(cls, graph: nx.Graph) -> nx.Graph:
        """
        Identify and merge the main trunk in every connected component.

        Args:
            graph: NetworkX Graph already processed by :class:`SegmentDetector`.
                Edges must carry ``segment_id``; boundary nodes must carry
                ``node_type``.

        Returns:
            A copy of the input graph with main-trunk segments merged into a
            single ``segment_id`` per component.
        """
        res_graph = graph.copy()

        if res_graph.number_of_nodes() == 0 or res_graph.number_of_edges() == 0:
            logger.warning("Topology graph is empty")
            return res_graph

        merged_components = 0
        for component in nx.connected_components(res_graph):
            if cls._process_component(res_graph, component):
                merged_components += 1

        logger.info(
            f"Main trunk extraction complete: {merged_components} component(s) merged"
        )
        return res_graph

    # ------------------------------------------------------------------
    # Per-component processing
    # ------------------------------------------------------------------

    @classmethod
    def _process_component(
        cls, graph: nx.Graph, component: Set[Hashable]
    ) -> bool:
        """
        Process a single connected component, merging its trunk segments.

        Args:
            graph: The full graph (edges in ``component`` will be relabeled
                in-place).
            component: Set of node ids forming a connected component.

        Returns:
            True if a trunk was identified and merged; False if the component
            was skipped (no endpoint, no segments, or trunk has only one
            segment).
        """
        endpoints = [n for n in component if graph.degree(n) == 1]
        if not endpoints:
            logger.debug("Skipping component with no endpoints (cycle or isolate)")
            return False

        # max y, then min x. Nodes are (y, x) tuples.
        start = sorted(endpoints, key=lambda n: (-n[0], n[1]))[0]

        seg_edges: Dict[int, List[Tuple]] = defaultdict(list)
        for u, v, data in graph.edges(component, data=True):
            seg_id = data.get("segment_id")
            if seg_id is None:
                continue
            seg_edges[seg_id].append((u, v, data))

        if not seg_edges:
            return False

        meta = nx.Graph()
        for seg_id, edges in seg_edges.items():
            boundary = cls._segment_boundary_nodes(edges, graph)
            if len(boundary) != 2:
                # Self-loops on a boundary node, or malformed segments — skip.
                continue
            a, b = boundary
            weight = cls._segment_pixel_length(edges)
            # If multiple segments share the same boundary pair (parallel
            # paths in non-tree graphs), keep the longest as the meta edge.
            if meta.has_edge(a, b) and meta[a][b]["weight"] >= weight:
                continue
            meta.add_edge(a, b, segment_id=seg_id, weight=weight)

        if start not in meta:
            return False

        trunk_segment_ids = cls._dfs_longest_path(meta, start)
        if len(trunk_segment_ids) <= 1:
            return False

        canonical = min(trunk_segment_ids)
        for seg_id in trunk_segment_ids:
            if seg_id == canonical:
                continue
            for u, v, _ in seg_edges[seg_id]:
                graph[u][v]["segment_id"] = canonical

        logger.debug(
            f"Component starting at {start}: merged {len(trunk_segment_ids)} "
            f"segments into segment_id={canonical}"
        )
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_pixel_length(edges: List[Tuple]) -> int:
        """Sum ``len(path) - 1`` over a segment's edges (fallback: 1 step)."""
        total = 0
        for u, v, data in edges:
            path = data.get("path")
            if path is None or len(path) < 2:
                total += 1
            else:
                total += len(path) - 1
        return total

    @staticmethod
    def _segment_boundary_nodes(
        edges: List[Tuple], graph: nx.Graph
    ) -> List[Hashable]:
        """Collect nodes whose ``node_type`` marks them as segment endpoints."""
        seen: Set[Hashable] = set()
        boundary: List[Hashable] = []
        for u, v, _ in edges:
            for node in (u, v):
                if node in seen:
                    continue
                seen.add(node)
                if graph.nodes[node].get("node_type") in _BOUNDARY_TYPES:
                    boundary.append(node)
        return boundary

    @staticmethod
    def _dfs_longest_path(
        meta: nx.Graph, start: Hashable
    ) -> List[int]:
        """
        Iterative DFS that returns the segment_id sequence of the longest path
        starting at ``start``.

        Uses two-pass post-order traversal: first push children, then settle
        the best (weight, segment_ids) for each node on the way back up.
        Tie-break: longer wins; same weight → smaller leading segment_id wins.
        """
        # best[node] = (best_total_weight, [segment_ids along that branch])
        best: Dict[Hashable, Tuple[int, List[int]]] = {}
        stack: List[Tuple[Hashable, Hashable, bool]] = [(start, None, False)]

        while stack:
            node, parent, processed = stack.pop()
            if not processed:
                stack.append((node, parent, True))
                for nbr in meta.neighbors(node):
                    if nbr == parent:
                        continue
                    stack.append((nbr, node, False))
                continue

            best_weight = 0
            best_segs: List[int] = []
            for nbr in meta.neighbors(node):
                if nbr == parent:
                    continue
                edge = meta[node][nbr]
                child_weight, child_segs = best.get(nbr, (0, []))
                total_weight = edge["weight"] + child_weight
                candidate_segs = [edge["segment_id"], *child_segs]
                key = (total_weight, -candidate_segs[0])
                best_key = (best_weight, -best_segs[0]) if best_segs else (0, 0)
                if key > best_key:
                    best_weight = total_weight
                    best_segs = candidate_segs

            best[node] = (best_weight, best_segs)

        return best.get(start, (0, []))[1]
