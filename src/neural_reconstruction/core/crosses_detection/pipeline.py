"""
End-to-end crossing analysis pipeline shared by all linkers.

流程:
  1. Segment detection
  2. Stub pruning (短於 stub_length_threshold 像素、且至少一端為 endpoint 的 segment)
  3. Re-segment + region labeling
  4. 排除「覆蓋 annotation 元件數 < min_tree_components」的小子樹
     (保留在圖上但 is_crossing → False, 不參與計數)
  5. Count effective crossings
"""

import logging
from collections import defaultdict

import numpy as np
import networkx as nx

from .segment_detector import SegmentDetector
from .region_labeler import RegionLabeler
from .crossing_counter import CrossingCounter


logger = logging.getLogger(__name__)


def run_crossing_analysis(
    graph: nx.Graph,
    mask: np.ndarray,
    annot_labeled: np.ndarray,
    *,
    min_tree_components: int = 5,
    stub_length_threshold: int = 5,
    neighborhood: int = 3,
) -> tuple[int, nx.Graph]:
    """
    對重建圖跑完整的交叉點分析。

    Args:
        graph:                  重建後的 pixel-level 圖 (節點為 (y, x))
        mask:                   表皮二值化遮罩 (H, W, 0/255 或 0/1)
        annot_labeled:          annotation 連通元件標籤圖 (H, W, int);
                                每個原始 annotation 連通元件有唯一 ID, 背景為 0
        min_tree_components:    子樹覆蓋的 annotation 元件下限,
                                低於此值的子樹不計入交叉數但仍保留在圖上
        stub_length_threshold:  stub segment 長度移除門檻(像素)
        neighborhood:           節點查找 annot_labeled 時的半徑容差;
                                骨架節點可能不正好落在 annotation 像素上, 故給容差

    Returns:
        (effective_crossing_count, labeled_graph)
        labeled_graph 完整保留所有 CCs (含被排除計數的小子樹), 以利視覺化
    """
    region_labeler = RegionLabeler()
    segment_detector = SegmentDetector()
    crossing_counter = CrossingCounter()

    # ── Step 1: Detect segments (segment_id 為後續 label_topology 必要屬性) ──
    segmented_graph = segment_detector.detect_segments(graph)

    # ── Step 1b: 移除短 stub segments (至少一端為 endpoint) ─────────────────
    seg_edges = defaultdict(list)
    for u, v, data in segmented_graph.edges(data=True):
        seg_id = data.get("segment_id")
        if seg_id is not None:
            seg_edges[seg_id].append((u, v, data))

    edges_to_remove = []
    for _, edges in seg_edges.items():
        boundary_nodes = set()
        for u, v, _data in edges:
            if segmented_graph.nodes[u].get("node_type") in ("endpoint", "branchpoint"):
                boundary_nodes.add(u)
            if segmented_graph.nodes[v].get("node_type") in ("endpoint", "branchpoint"):
                boundary_nodes.add(v)

        has_endpoint = any(
            segmented_graph.nodes[n].get("node_type") == "endpoint"
            for n in boundary_nodes
        )
        if not has_endpoint:
            continue

        total_length = sum(
            len(data.get("path", [u, v])) - 1 for u, v, data in edges
        )
        if total_length < stub_length_threshold:
            edges_to_remove.extend((u, v) for u, v, _ in edges)

    segmented_graph.remove_edges_from(edges_to_remove)
    segmented_graph.remove_nodes_from(list(nx.isolates(segmented_graph)))
    logger.info(
        f"Pruned {len(edges_to_remove)} stub edges → "
        f"{segmented_graph.number_of_nodes()} nodes, "
        f"{segmented_graph.number_of_edges()} edges"
    )

    # ── Step 1c: 重新做 segment detection (因 stub 修剪改變了拓樸) ───────
    segmented_graph = segment_detector.detect_segments(segmented_graph)

    # ── Step 2: Label regions and mark crossing edges ─────────────────────
    labeled_graph, _ = region_labeler.label_topology(segmented_graph, mask)

    # ── Step 2b: 排除小子樹於計數 (保留在圖上以利視覺化) ───────────────────
    excluded = _exclude_small_subtrees_from_count(
        labeled_graph,
        annot_labeled,
        min_tree_components=min_tree_components,
        neighborhood=neighborhood,
    )
    if excluded:
        logger.info(
            f"Excluded {excluded} subtree(s) from crossing count "
            f"(covering < {min_tree_components} annotation components); "
            f"they remain in the graph for visualization"
        )

    # ── Step 3: Count effective crossings ─────────────────────────────────
    result = crossing_counter.count_effective_crossings(
        labeled_graph, epidermis_mask=mask
    )

    return result["effective_crossing_count"], labeled_graph


def _exclude_small_subtrees_from_count(
    labeled_graph: nx.Graph,
    annot_labeled: np.ndarray,
    *,
    min_tree_components: int,
    neighborhood: int = 3,
) -> int:
    """
    將覆蓋 annotation 連通元件過少的子樹排除於計數之外。

    子樹仍保留在 labeled_graph 中(節點/邊都不移除),只將該 CC 內所有
    edges 的 ``is_crossing`` 標為 ``False`` -- 後續 ``CrossingCounter``
    自動跳過。

    判定方式:對每個 connected component, 蒐集其節點座標在
    ``annot_labeled`` (含 ``neighborhood`` 半徑容差)上覆蓋到的
    annotation 元件 ID。若覆蓋元件數 < ``min_tree_components``,
    則視為小子樹。

    Returns:
        被排除的 connected component 數量
    """
    H, W = annot_labeled.shape
    excluded_count = 0
    for cc_nodes in nx.connected_components(labeled_graph):
        covered: set[int] = set()
        for y, x in cc_nodes:
            y0, y1 = max(0, y - neighborhood), min(H, y + neighborhood + 1)
            x0, x1 = max(0, x - neighborhood), min(W, x + neighborhood + 1)
            covered.update(np.unique(annot_labeled[y0:y1, x0:x1]).tolist())
        covered.discard(0)

        if len(covered) < min_tree_components:
            excluded_count += 1
            for u, v in labeled_graph.subgraph(cc_nodes).edges():
                labeled_graph[u][v]["is_crossing"] = False
    return excluded_count
