"""
SkeletonLinker - no-op baseline from ROI annotation to topology.

This linker does not perform reconstruction, path finding, growing, or MST
connection. It only clips the input annotation to the ROI mask, skeletonizes it
with TopologyBuilder, and runs the standard crossing analysis.
"""

import logging
from collections import defaultdict

import cv2
import networkx as nx
import numpy as np

from neural_reconstruction.common.data_types import LinkerResult
from neural_reconstruction.core.crosses_detection import (
    CrossingCounter,
    RegionLabeler,
    SegmentDetector,
)
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.topology import TopologyBuilder


logger = logging.getLogger(__name__)


class SkeletonLinker:
    """
    Baseline linker: directly converts ``roi_annotation`` into a topology graph.

    Flow:
      1. Build ROI mask from epidermis mask using ``offset_px``
      2. Clip annotation to ROI, producing ``roi_annotation``
      3. Build skeleton topology with ``TopologyBuilder``
      4. Run the shared crossing analysis pipeline
    """

    def __init__(self, offset_px: int = 50, segment_length: float = 3.0):
        self.offset_px = offset_px
        self.segment_length = segment_length

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        annotation: np.ndarray,
    ) -> LinkerResult:
        """
        Args:
            image: RGB or grayscale original image (H, W, 3) or (H, W)
            mask: epidermis mask (H, W)
            annotation: binary annotation / weka output (H, W)
        """
        if image.ndim == 3:
            image = image[:, :, 1]
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if annotation.ndim == 3:
            annotation = annotation[:, :, 0]

        roi_mask = dilate_epidermis_vertically(mask, offset_px=self.offset_px)
        roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)

        topology_builder = TopologyBuilder(segment_length=self.segment_length)
        skeleton_graph = topology_builder.build_skeleton_graph(roi_annotation)
        logger.info(
            f"Skeleton graph: {skeleton_graph.number_of_nodes()} nodes, "
            f"{skeleton_graph.number_of_edges()} edges"
        )

        valid_count, labeled_graph = self._run_crossing_analysis(mask, skeleton_graph)

        return LinkerResult(
            annotation=roi_annotation,
            image=cv2.bitwise_and(image, image, mask=roi_mask),
            mask=roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )

    def _run_crossing_analysis(
        self,
        mask: np.ndarray,
        graph: nx.Graph,
    ) -> tuple[int, nx.Graph]:
        """
        Run the same crossing analysis used by reconstruction linkers.
        """
        region_labeler = RegionLabeler()
        segment_detector = SegmentDetector()
        crossing_counter = CrossingCounter()

        segmented_graph = segment_detector.detect_segments(graph)

        def _path_length(data, u, v):
            path = data.get("path", [u, v])
            return len(path) - 1

        seg_edges = defaultdict(list)
        for u, v, data in segmented_graph.edges(data=True):
            seg_id = data.get("segment_id")
            if seg_id is not None:
                seg_edges[seg_id].append((u, v, data))

        edges_to_remove = []
        for _seg_id, edges in seg_edges.items():
            boundary_nodes = set()
            for u, v, _ in edges:
                if segmented_graph.nodes[u].get("node_type") in (
                    "endpoint",
                    "branchpoint",
                ):
                    boundary_nodes.add(u)
                if segmented_graph.nodes[v].get("node_type") in (
                    "endpoint",
                    "branchpoint",
                ):
                    boundary_nodes.add(v)

            has_endpoint = any(
                segmented_graph.nodes[n].get("node_type") == "endpoint"
                for n in boundary_nodes
            )
            if not has_endpoint:
                continue

            total_length = sum(_path_length(data, u, v) for u, v, data in edges)
            if total_length < 5:
                edges_to_remove.extend((u, v) for u, v, _ in edges)

        segmented_graph.remove_edges_from(edges_to_remove)
        segmented_graph.remove_nodes_from(list(nx.isolates(segmented_graph)))
        logger.info(
            f"Pruned {len(edges_to_remove)} stub edges -> "
            f"{segmented_graph.number_of_nodes()} nodes, "
            f"{segmented_graph.number_of_edges()} edges"
        )

        segmented_graph = segment_detector.detect_segments(segmented_graph)
        labeled_graph, _ = region_labeler.label_topology(segmented_graph, mask)

        result = crossing_counter.count_effective_crossings(
            labeled_graph,
            epidermis_mask=mask,
        )

        return result["effective_crossing_count"], labeled_graph
