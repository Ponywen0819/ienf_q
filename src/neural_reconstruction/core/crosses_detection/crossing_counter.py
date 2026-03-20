"""
Crossing Counter

Counts the number of effective nerve fiber crossings across the
epidermis/dermis boundary. Rule: each segment counts at most once,
regardless of how many crossing edges it contains.
"""

from typing import Dict, List
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class CrossingCounter:
    """
    Counts effective nerve fiber crossings in an annotated topology graph.

    Expects a graph that has been processed by both SegmentDetector and
    RegionLabeler, so that every edge carries:
    - ``segment_id`` (int): assigned by SegmentDetector
    - ``is_crossing`` (bool): assigned by RegionLabeler

    Rule: a segment that contains one or more crossing edges counts as exactly
    one effective crossing.
    """

    def __init__(self):
        """Initialize the CrossingCounter."""
        logger.info("Initialized CrossingCounter")

    def count_effective_crossings(self, graph: nx.Graph) -> Dict:
        """
        Count effective crossings from an annotated topology graph.

        The graph must have ``segment_id`` and ``is_crossing`` attributes on
        every edge (set by SegmentDetector and RegionLabeler respectively).

        Args:
            graph: Annotated NetworkX Graph.

        Returns:
            A dict with the following keys:

            - ``effective_crossing_count`` (int): Number of segments that
              contain at least one crossing edge (deduplicated count).
            - ``total_crossing_edges`` (int): Total crossing edges across all
              segments (not deduplicated).
            - ``total_segments`` (int): Total number of distinct segments.
            - ``segments_with_crossing`` (int): Same as
              ``effective_crossing_count``; provided for clarity.
            - ``segment_details`` (list[dict]): Per-segment breakdown with
              keys ``segment_id``, ``has_crossing``, ``crossing_edge_count``.
        """
        # Group edges by segment_id
        segments: Dict[int, List[bool]] = {}
        for _, _, data in graph.edges(data=True):
            seg_id = data.get("segment_id")
            if seg_id is None:
                continue
            segments.setdefault(seg_id, []).append(data.get("is_crossing", False))

        effective_crossing_count = 0
        total_crossing_edges = 0
        segment_details: List[Dict] = []

        for seg_id, crossing_flags in segments.items():
            crossing_count = sum(crossing_flags)
            has_crossing = crossing_count > 0

            segment_details.append({
                "segment_id": seg_id,
                "has_crossing": has_crossing,
                "crossing_edge_count": crossing_count,
            })

            total_crossing_edges += crossing_count
            if has_crossing:
                effective_crossing_count += 1

        result = {
            "effective_crossing_count": effective_crossing_count,
            "total_crossing_edges": total_crossing_edges,
            "total_segments": len(segments),
            "segments_with_crossing": effective_crossing_count,
            "segment_details": segment_details,
        }

        logger.info("Crossing count complete:")
        logger.info(f"  Total segments: {len(segments)}")
        logger.info(f"  Segments with crossings: {effective_crossing_count}")
        logger.info(f"  Total crossing edges (raw): {total_crossing_edges}")
        logger.info(f"  Effective crossings (deduplicated): {effective_crossing_count}")

        return result

    def get_crossing_summary(self, result: Dict) -> str:
        """
        Return a human-readable summary string for a counting result.

        Args:
            result: Dict returned by :meth:`count_effective_crossings`.

        Returns:
            Formatted multi-line summary string.
        """
        lines = [
            "=== Nerve Fiber Crossing Summary ===",
            f"Total segments:              {result['total_segments']}",
            f"Segments with crossings:     {result['segments_with_crossing']}",
            f"Total crossing edges (raw):  {result['total_crossing_edges']}",
            f"Effective crossings (dedup): {result['effective_crossing_count']}",
            "=====================================",
        ]
        return "\n".join(lines)
