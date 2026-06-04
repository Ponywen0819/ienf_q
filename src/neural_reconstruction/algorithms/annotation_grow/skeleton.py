"""
Build pixel-level result graph from MST backtracked paths + annotation mask.

Pipeline:
  1. Draw backtracked inter-component paths from MST edges onto a bridge mask
  2. Dilate bridges by dilate_radius px
  3. Merge with original annotation_bin
  4. Global skeletonization via TopologyBuilder
"""

import cv2
import networkx as nx
import numpy as np

from neural_reconstruction.core.topology import TopologyBuilder


def build_result_graph(
    mst: nx.Graph,
    annotation_bin: np.ndarray,
    dilate_radius: int = 3,
    stub_length_threshold: float = 5.0,
) -> nx.Graph:
    """
    Build a pixel-level graph by merging MST bridge paths with the annotation mask.

    Args:
        mst:            MST graph; edges carry 'path' = list[(y, x)] backtracked
                        from seed_A through meeting point to seed_B
        annotation_bin: uint8 (H, W) binary annotation mask (>0 = fiber)
        segment_length: TopologyBuilder seed spacing (pixels)
        dilate_radius:  dilation radius for bridge paths in pixels

    Returns:
        nx.Graph with (y, x) nodes and edge 'path' attributes
    """
    H, W = annotation_bin.shape

    # ── 1. Draw backtracked paths onto bridge mask ────────────────────────
    bridge_mask = np.zeros((H, W), dtype=np.uint8)
    for _, _, data in mst.edges(data=True):
        for py, px in data.get("path", []):
            if 0 <= py < H and 0 <= px < W:
                bridge_mask[py, px] = 1

    # ── 2. Dilate bridges ─────────────────────────────────────────────────
    ksize = dilate_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bridge_dilated = cv2.dilate(bridge_mask, kernel).astype(bool)

    # ── 3. Merge with annotation ──────────────────────────────────────────
    combined = ((annotation_bin > 0) | bridge_dilated).astype(np.uint8)

    # ── 4. Global skeletonization → pixel-level graph ────────────────────
    topology_builder = TopologyBuilder(segment_length=500)
    return topology_builder.build_seed_graph(combined, stub_length_threshold=stub_length_threshold)
