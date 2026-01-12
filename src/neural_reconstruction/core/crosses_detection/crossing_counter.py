"""
有效跨越計算器 (Crossing Counter)

計算神經纖維穿越表皮/真皮邊界的有效跨越數量。
規則：每個分段最多只能算一次有效跨越。
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossingCounter:
    """
    有效跨越計算器 - 計算有效的神經纖維跨越數量

    規則：每個分段最多只能算一次有效跨越邊。
    若一個分段包含多條 crossing edges，只計為 1 次有效跨越。
    """

    def __init__(self):
        """初始化有效跨越計算器"""
        logger.info("Initialized CrossingCounter")

    def count_effective_crossings(
        self,
        segments: List[Dict],
        labeled_topology: Dict
    ) -> Dict:
        """
        計算有效跨越數量

        Args:
            segments: 分段列表（來自 SegmentDetector）
                [
                    {
                        'segment_id': int,
                        'start_node_id': int,
                        'end_node_id': int,
                        'edge_indices': [int, ...],
                        'total_length': float
                    },
                    ...
                ]
            labeled_topology: 已標注的拓樸結構（來自 RegionLabeler）
                {
                    'nodes': [...],
                    'edges': [{'is_crossing': bool, ...}, ...]
                }

        Returns:
            result: 計算結果
                {
                    'effective_crossing_count': int,  # 有效跨越總數（去重後）
                    'total_crossing_edges': int,      # 跨越邊總數（未去重）
                    'total_segments': int,            # 分段總數
                    'segments_with_crossing': int,    # 包含跨越邊的分段數
                    'segment_details': [              # 每個分段的詳細資訊
                        {
                            'segment_id': int,
                            'has_crossing': bool,
                            'crossing_edge_count': int,
                            'crossing_edge_indices': [int, ...]
                        },
                        ...
                    ]
                }
        """
        edges = labeled_topology.get('edges', [])

        # 建立邊索引到 is_crossing 的映射
        edge_crossing_map: Dict[int, bool] = {}
        for idx, edge in enumerate(edges):
            edge_crossing_map[idx] = edge.get('is_crossing', False)

        # 統計變數
        effective_crossing_count = 0
        total_crossing_edges = 0
        segments_with_crossing = 0
        segment_details: List[Dict] = []

        # 遍歷每個分段
        for segment in segments:
            has_crossing, crossing_count, crossing_indices = self._segment_has_crossing(
                segment, edge_crossing_map
            )

            # 記錄分段詳細資訊
            segment_details.append({
                'segment_id': segment['segment_id'],
                'has_crossing': has_crossing,
                'crossing_edge_count': crossing_count,
                'crossing_edge_indices': crossing_indices
            })

            # 統計
            total_crossing_edges += crossing_count
            if has_crossing:
                effective_crossing_count += 1  # 每個分段最多算 1 次
                segments_with_crossing += 1

        result = {
            'effective_crossing_count': effective_crossing_count,
            'total_crossing_edges': total_crossing_edges,
            'total_segments': len(segments),
            'segments_with_crossing': segments_with_crossing,
            'segment_details': segment_details
        }

        # 輸出日誌
        logger.info(f"有效跨越計算完成:")
        logger.info(f"  總分段數: {len(segments)}")
        logger.info(f"  包含跨越邊的分段數: {segments_with_crossing}")
        logger.info(f"  跨越邊總數（未去重）: {total_crossing_edges}")
        logger.info(f"  有效跨越數（去重後）: {effective_crossing_count}")

        return result

    def _segment_has_crossing(
        self,
        segment: Dict,
        edge_crossing_map: Dict[int, bool]
    ) -> Tuple[bool, int, List[int]]:
        """
        判斷分段是否包含跨越邊

        Args:
            segment: 分段資訊
            edge_crossing_map: 邊索引 -> is_crossing 映射

        Returns:
            (has_crossing, crossing_count, crossing_indices):
                - has_crossing: 是否包含跨越邊
                - crossing_count: 跨越邊數量
                - crossing_indices: 跨越邊的索引列表
        """
        crossing_indices: List[int] = []

        for edge_idx in segment.get('edge_indices', []):
            if edge_crossing_map.get(edge_idx, False):
                crossing_indices.append(edge_idx)

        has_crossing = len(crossing_indices) > 0
        crossing_count = len(crossing_indices)

        return has_crossing, crossing_count, crossing_indices

    def get_crossing_summary(
        self,
        result: Dict
    ) -> str:
        """
        取得跨越計算的摘要字串

        Args:
            result: count_effective_crossings 的返回結果

        Returns:
            summary: 摘要字串
        """
        lines = [
            "=== 神經纖維跨越統計 ===",
            f"總分段數: {result['total_segments']}",
            f"包含跨越邊的分段數: {result['segments_with_crossing']}",
            f"跨越邊總數（未去重）: {result['total_crossing_edges']}",
            f"有效跨越數（去重後）: {result['effective_crossing_count']}",
            "========================"
        ]
        return "\n".join(lines)
