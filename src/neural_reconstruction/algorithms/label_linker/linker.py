"""
Label baseline linker (skip reconstruction).

直接對 GT label 做骨架化，不使用原始影像，也不執行任何 pathfinding 或 MST
重建。作為其他重建演算法的上界對照 baseline 使用。
"""

import logging
from collections import defaultdict

import cv2
import numpy as np
import networkx as nx

from neural_reconstruction.common.data_types import LinkerResult
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.crosses_detection import (
    RegionLabeler,
    SegmentDetector,
    CrossingCounter,
)


logger = logging.getLogger(__name__)


class LabelLinker:
    """
    Baseline：直接對 GT label 做骨架化。

    流程：
      1. 依 ``offset_px`` 將表皮遮罩向下擴張取得 ROI mask
      2. 將 label 限制在 ROI 內
      3. 用 ``TopologyBuilder`` 對 label 做骨架化（Zhang-Suen → skan）
      4. 執行既有的 crossing 分析以取得 valid_count
    """

    def __init__(self, offset_px: int = 50):
        self.offset_px = offset_px

    def run(
        self,
        mask: np.ndarray,
        label: np.ndarray,
    ) -> LinkerResult:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if label.ndim == 3:
            label = label[:, :, 0]

        roi_mask = dilate_epidermis_vertically(mask, offset_px=self.offset_px)
        roi_label = cv2.bitwise_and(label, label, mask=roi_mask)

        topology_builder = TopologyBuilder()
        skeleton_graph = topology_builder.build_skeleton_graph(roi_label)
        logger.info(
            f"Skeleton graph: {skeleton_graph.number_of_nodes()} nodes, "
            f"{skeleton_graph.number_of_edges()} edges"
        )

        valid_count, labeled_graph = self._run_crossing_analysis(mask, skeleton_graph)

        return LinkerResult(
            annotation=roi_label,
            image=roi_label,
            mask=roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )

    def _run_crossing_analysis(
        self,
        mask: np.ndarray,
        graph: nx.Graph,
    ) -> tuple[int, nx.Graph]:
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
        for seg_id, edges in seg_edges.items():
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
            f"Pruned {len(edges_to_remove)} stub edges → "
            f"{segmented_graph.number_of_nodes()} nodes, "
            f"{segmented_graph.number_of_edges()} edges"
        )

        segmented_graph = segment_detector.detect_segments(segmented_graph)
        labeled_graph, _ = region_labeler.label_topology(segmented_graph, mask)

        result = crossing_counter.count_effective_crossings(
            labeled_graph, epidermis_mask=mask
        )

        return result["effective_crossing_count"], labeled_graph
