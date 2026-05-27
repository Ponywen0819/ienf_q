"""
Crossing Counter

Counts the number of effective nerve fiber crossings across the
epidermis/dermis boundary. Rule: each segment counts at most once,
regardless of how many crossing edges it contains.

Validity condition: a crossing segment is only counted if it has at least
``min_region_length`` pixels of path length in **both** the epidermis and
dermis regions.
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
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
    one effective crossing, **provided** it has sufficient path length in both
    the epidermis and dermis regions (controlled by ``min_region_length``).
    """

    def __init__(self):
        """Initialize the CrossingCounter."""
        logger.info("Initialized CrossingCounter")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_effective_crossings(
        self,
        graph: nx.Graph,
        epidermis_mask: Optional[np.ndarray] = None,
        min_region_length: float =0,
    ) -> Dict:
        """
        Count effective crossings from an annotated topology graph.

        The graph must have ``segment_id`` and ``is_crossing`` attributes on
        every edge (set by SegmentDetector and RegionLabeler respectively).
        Nodes must have a ``region`` attribute (``"epidermis"`` or
        ``"dermis"``) set by RegionLabeler.

        A crossing segment is valid only when it has **at least**
        ``min_region_length`` pixels of path length in both the epidermis and
        dermis regions.  When ``epidermis_mask`` is provided, each path step's
        midpoint is checked against the mask for an exact split; otherwise the
        node ``region`` attribute is used as a fallback.

        Args:
            graph: Annotated NetworkX Graph.
            epidermis_mask: Optional binary mask (H, W) where >127 = epidermis.
                            When supplied, region lengths are computed precisely
                            from the ``path`` pixel coordinates.
            min_region_length: Minimum path length (px) required in both
                               epidermis and dermis for a segment to be counted.
                               Default: 5.0.

        Returns:
            A dict with the following keys:

            - ``effective_crossing_count`` (int)
            - ``total_crossing_edges`` (int)
            - ``total_segments`` (int)
            - ``segments_with_crossing`` (int)
            - ``segment_details`` (list[dict]): Per-segment breakdown with
              keys ``segment_id``, ``has_crossing``, ``is_valid``,
              ``crossing_edge_count``, ``epidermis_length``, ``dermis_length``.
        """
        # Group full edge tuples by segment_id
        segments: Dict[int, List[Tuple]] = {}
        for u, v, data in graph.edges(data=True):
            seg_id = data.get("segment_id")
            if seg_id is None:
                continue
            segments.setdefault(seg_id, []).append((u, v, data))

        effective_crossing_count = 0
        total_crossing_edges = 0
        segment_details: List[Dict] = []

        for seg_id, edges in segments.items():
            crossing_count = sum(d.get("is_crossing", False) for _, _, d in edges)
            has_crossing = crossing_count > 0

            epi_len, der_len = self._compute_region_lengths(
                edges, graph, epidermis_mask
            )
            is_valid = (
                has_crossing
                and epi_len >= min_region_length
                and der_len >= min_region_length
            )

            segment_details.append({
                "segment_id": seg_id,
                "has_crossing": has_crossing,
                "is_valid": is_valid,
                "crossing_edge_count": crossing_count,
                "epidermis_length": epi_len,
                "dermis_length": der_len,
            })

            total_crossing_edges += crossing_count
            if is_valid:
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_region_lengths(
        self,
        edges: List[Tuple],
        graph: nx.Graph,
        epidermis_mask: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        """
        Compute total path length in epidermis and dermis for a segment.

        Args:
            edges: List of (u, v, data) tuples belonging to the segment.
            graph: The topology graph (used for node ``region`` fallback).
            epidermis_mask: Optional mask for exact per-pixel region lookup.

        Returns:
            (epidermis_length, dermis_length) in pixels.
        """
        epi_len = 0.0
        der_len = 0.0

        mask_h = epidermis_mask.shape[0] if epidermis_mask is not None else 0
        mask_w = epidermis_mask.shape[1] if epidermis_mask is not None else 0

        for u, v, data in edges:
            path = data.get("path", [u, v])
            if len(path) < 2:
                continue

            path_arr = np.array(path, dtype=np.float64)
            steps = np.diff(path_arr, axis=0)           # (N-1, 2)
            step_lengths = np.linalg.norm(steps, axis=1)  # (N-1,)
            midpoints = (path_arr[:-1] + path_arr[1:]) / 2  # (N-1, 2)

            for step_len, mid in zip(step_lengths, midpoints):
                region = self._region_at(
                    mid, u, v, data, graph, epidermis_mask, mask_h, mask_w
                )
                if region == "epidermis":
                    epi_len += step_len
                else:
                    der_len += step_len

        return epi_len, der_len

    @staticmethod
    def _region_at(
        mid: np.ndarray,
        u,
        v,
        edge_data: Dict,
        graph: nx.Graph,
        epidermis_mask: Optional[np.ndarray],
        mask_h: int,
        mask_w: int,
    ) -> str:
        """Return ``"epidermis"`` or ``"dermis"`` for a path midpoint."""
        if epidermis_mask is not None:
            y, x = int(round(mid[0])), int(round(mid[1]))
            if 0 <= y < mask_h and 0 <= x < mask_w:
                return "epidermis" if epidermis_mask[y, x] > 127 else "dermis"
            return "dermis"

        # Fallback: use node region attribute
        u_region = graph.nodes[u].get("region", "dermis")
        v_region = graph.nodes[v].get("region", "dermis")
        if u_region == v_region:
            return u_region
        # Crossing edge without mask — assign based on which endpoint is closer
        dist_u = float(np.linalg.norm(mid - np.array(u, dtype=np.float64)))
        dist_v = float(np.linalg.norm(mid - np.array(v, dtype=np.float64)))
        return u_region if dist_u <= dist_v else v_region

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
            f"Effective crossings (valid): {result['effective_crossing_count']}",
            "=====================================",
        ]
        return "\n".join(lines)
