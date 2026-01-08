"""
分段偵測器 (Segment Detector)

識別神經纖維的獨立分段。分段定義為從一個分界點（端點或分支點）
到另一個分界點的路徑。
"""

from typing import Dict, List, Set, Tuple
import logging

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

    def detect_segments(
        self,
        topology: Dict
    ) -> List[Dict]:
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
        nodes = topology['nodes']
        edges = topology['edges']

        if not nodes or not edges:
            logger.warning("拓樸結構為空")
            return []

        # 識別所有分界點
        boundary_nodes = self._identify_boundary_nodes(topology)
        logger.info(f"識別到 {len(boundary_nodes)} 個分界點")

        # 建立鄰接表
        adjacency = self._build_adjacency(topology)

        # 追蹤所有分段
        segments = []
        visited_edges: Set[int] = set()
        segment_id = 0

        for start_node in boundary_nodes:
            # 從每個分界點的每條出邊開始追蹤
            if start_node not in adjacency:
                continue

            for neighbor_id, edge_idx in adjacency[start_node]:
                if edge_idx in visited_edges:
                    continue

                # 追蹤此分段
                segment = self._trace_segment(
                    start_node=start_node,
                    first_edge_idx=edge_idx,
                    adjacency=adjacency,
                    boundary_nodes=boundary_nodes,
                    edges=edges,
                    visited_edges=visited_edges
                )

                if segment:
                    segment['segment_id'] = segment_id
                    segments.append(segment)
                    segment_id += 1

        logger.info(f"偵測到 {len(segments)} 個分段")

        return segments

    def _identify_boundary_nodes(
        self,
        topology: Dict
    ) -> Set[int]:
        """
        識別所有分界點（端點或分支點）

        分界點條件:
        - 端點 (type='endpoint', degree=1)
        - 分支點 (type='branchpoint', degree>=3)

        注意：由於拓樸中 type 已經標記了 endpoint 和 branchpoint，
        所有節點都應該是分界點（因為中間節點不會被加入拓樸）。

        Args:
            topology: 拓樸結構

        Returns:
            boundary_node_ids: 分界點 ID 集合
        """
        boundary_nodes: Set[int] = set()

        for node in topology['nodes']:
            node_type = node.get('type', '')
            # endpoint 和 branchpoint 都是分界點
            if node_type in ('endpoint', 'branchpoint'):
                boundary_nodes.add(node['id'])

        return boundary_nodes

    def _build_adjacency(
        self,
        topology: Dict
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        建立節點鄰接表

        Args:
            topology: 拓樸結構

        Returns:
            adjacency: {node_id: [(neighbor_id, edge_index), ...]}
        """
        adjacency: Dict[int, List[Tuple[int, int]]] = {}

        for edge_idx, edge in enumerate(topology['edges']):
            source = edge['source']
            target = edge['target']

            # 添加 source -> target
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append((target, edge_idx))

            # 添加 target -> source（無向邊）
            if target not in adjacency:
                adjacency[target] = []
            adjacency[target].append((source, edge_idx))

        return adjacency

    def _trace_segment(
        self,
        start_node: int,
        first_edge_idx: int,
        adjacency: Dict[int, List[Tuple[int, int]]],
        boundary_nodes: Set[int],
        edges: List[Dict],
        visited_edges: Set[int]
    ) -> Dict:
        """
        從起點追蹤一個完整分段

        Args:
            start_node: 起點節點 ID
            first_edge_idx: 第一條邊的索引
            adjacency: 鄰接表
            boundary_nodes: 分界點集合
            edges: 邊列表
            visited_edges: 已訪問的邊集合（會被修改）

        Returns:
            segment: 分段資訊字典，若追蹤失敗返回 None
        """
        edge_indices: List[int] = []
        current_node = start_node
        current_edge_idx = first_edge_idx

        while True:
            # 標記當前邊為已訪問
            visited_edges.add(current_edge_idx)
            edge_indices.append(current_edge_idx)

            # 取得當前邊
            edge = edges[current_edge_idx]

            # 找到下一個節點
            if edge['source'] == current_node:
                next_node = edge['target']
            else:
                next_node = edge['source']

            # 如果下一個節點是分界點，分段結束
            if next_node in boundary_nodes:
                # 計算總長度
                total_length = sum(edges[idx].get('length', 0) for idx in edge_indices)

                return {
                    'start_node_id': start_node,
                    'end_node_id': next_node,
                    'edge_indices': edge_indices,
                    'total_length': total_length
                }

            # 繼續追蹤：找到下一條未訪問的邊
            next_edge_idx = None
            if next_node in adjacency:
                for neighbor_id, edge_idx in adjacency[next_node]:
                    if edge_idx not in visited_edges:
                        next_edge_idx = edge_idx
                        break

            if next_edge_idx is None:
                # 沒有下一條邊，分段結束（可能是孤立節點或已遍歷完）
                total_length = sum(edges[idx].get('length', 0) for idx in edge_indices)
                return {
                    'start_node_id': start_node,
                    'end_node_id': next_node,
                    'edge_indices': edge_indices,
                    'total_length': total_length
                }

            # 移動到下一個節點和邊
            current_node = next_node
            current_edge_idx = next_edge_idx

    def get_segment_statistics(
        self,
        segments: List[Dict]
    ) -> Dict:
        """
        取得分段統計資訊

        Args:
            segments: 分段列表

        Returns:
            statistics: 統計資訊字典
        """
        if not segments:
            return {
                'total_segments': 0,
                'total_edges': 0,
                'total_length': 0.0,
                'avg_edges_per_segment': 0.0,
                'avg_length_per_segment': 0.0
            }

        total_edges = sum(len(s['edge_indices']) for s in segments)
        total_length = sum(s['total_length'] for s in segments)

        return {
            'total_segments': len(segments),
            'total_edges': total_edges,
            'total_length': total_length,
            'avg_edges_per_segment': total_edges / len(segments),
            'avg_length_per_segment': total_length / len(segments)
        }
