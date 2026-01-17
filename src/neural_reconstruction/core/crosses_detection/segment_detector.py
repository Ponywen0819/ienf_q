"""
分段偵測器 (Segment Detector)

識別神經纖維的獨立分段。分段定義為從一個分界點（端點或分支點）
到另一個分界點的路徑。
"""

from typing import Set, Tuple
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class SegmentDetector:
    """
    分段偵測器 - 識別神經纖維的獨立分段

    分段定義：從一個分界點（端點或分支點）到另一個分界點的路徑。
    - 端點 (endpoint): degree = 1
    - 分支點 (branchpoint): degree >= 3

    每個分段包含一條或多條連續的邊。
    """

    def __init__(self):
        """初始化分段偵測器"""
        logger.info("Initialized SegmentDetector")

    @classmethod
    def detect_segments(cls, graph: nx.Graph) -> nx.Graph:
        """
        從拓樸結構中識別所有分段

        Args:
            topology: 拓樸結構字典（可能已標注區域資訊）
                {
                    'nodes': [{'id': int, 'position': (y, x), 'type': str, 'region': str?}, ...],
                    'edges': [{'source': int, 'target': int, 'path': [...], 'is_crossing': bool?}, ...]
                }

        Returns:
            segments: 分段列表
                [
                    {
                        'segment_id': int,
                        'start_node_id': int,
                        'end_node_id': int,
                        'edge_indices': [int, ...],  # 此分段包含的邊索引
                        'total_length': float
                    },
                    ...
                ]
        """
        res_graph = graph.copy()
        nodes = res_graph.nodes(data=True)
        edges = res_graph.edges(data=True)

        if not nodes or not edges:
            logger.warning("拓樸結構為空")
            return res_graph

        # 識別所有分界點
        cls._identify_boundary_nodes(res_graph)

        cls._identify_segment_nodes(res_graph)

        return res_graph

    @classmethod
    def _identify_boundary_nodes(cls, graph: nx.Graph) -> None:
        """
        識別所有分界點（端點或分支點）

        分界點條件:
        - 端點 (degree=1)
        - 分支點 (degree>=3)

        Args:
            graph: NetworkX Graph
        """

        for node_id in graph.nodes():
            degree = graph.degree(node_id)
            # endpoint (degree=1) 和 branchpoint (degree>=3) 都是分界點

            is_endpoint = degree == 1
            is_branchpoint = degree >= 3

            if is_endpoint:
                graph.nodes[node_id]["node_type"] = "endpoint"
            elif is_branchpoint:
                graph.nodes[node_id]["node_type"] = "branchpoint"

    @classmethod
    def _identify_segment_nodes(cls, graph: nx.Graph) -> None:
        """
        為每條邊分配片段 ID

        片段定義：從一個分界點到另一個分界點之間的所有邊

        Args:
            graph: NetworkX Graph，節點已標記 node_type
        """
        segment_id = 0
        visited_edges: Set[Tuple] = set()  # 記錄已訪問的邊 (u, v, key)

        # 取得所有 boundary nodes (endpoint 或 branchpoint)
        boundary_nodes = [
            node
            for node in graph.nodes()
            if graph.nodes[node].get("node_type") in ["endpoint", "branchpoint"]
        ]

        logger.debug(f"找到 {len(boundary_nodes)} 個分界點")

        # 從每個 boundary node 開始探索
        for boundary_node in boundary_nodes:
            # 檢查此節點的每條邊
            for neighbor in list(graph.neighbors(boundary_node)):
                for key in list(graph[boundary_node][neighbor].keys()):
                    edge_id = (boundary_node, neighbor, key)

                    # 如果這條邊已經訪問過，跳過
                    if (
                        edge_id in visited_edges
                        or (neighbor, boundary_node, key) in visited_edges
                    ):
                        continue

                    # 開始一個新片段：從 boundary_node 沿著這條邊遍歷
                    cls._trace_and_label_segment(
                        graph, boundary_node, neighbor, key, segment_id, visited_edges
                    )
                    logger.debug(f"完成片段 {segment_id} 的標記")
                    segment_id += 1

        logger.info(f"片段識別完成，共 {segment_id} 個片段")

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
        遞迴遍歷並標記片段中的所有邊

        Args:
            graph: NetworkX Graph
            from_node: 來源節點（前一個節點）
            current_node: 當前節點
            edge_key: Graph 的邊鍵值
            segment_id: 要分配的片段 ID
            visited_edges: 已訪問的邊集合
        """
        # 建立當前邊的標識
        edge_id = (from_node, current_node, edge_key)

        # 如果已經訪問過，停止
        if (
            edge_id in visited_edges
            or (current_node, from_node, edge_key) in visited_edges
        ):
            return

        # 標記這條邊的 segment_id
        graph[from_node][current_node][edge_key]["segment_id"] = segment_id
        visited_edges.add(edge_id)
        visited_edges.add((current_node, from_node, edge_key))

        # 如果 current_node 是 boundary node（endpoint 或 branchpoint），停止
        if graph.nodes[current_node].get("node_type") in ["endpoint", "branchpoint"]:
            return

        # current_node 是 degree-2 的中間節點，繼續遍歷
        for next_node in graph.neighbors(current_node):
            if next_node == from_node:
                continue  # 不要回頭

            # 遍歷所有連接到 next_node 的邊
            for next_key in graph[current_node][next_node].keys():
                cls._trace_and_label_segment(
                    graph, current_node, next_node, next_key, segment_id, visited_edges
                )
