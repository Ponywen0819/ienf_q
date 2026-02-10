"""
拓樸比對器模組 (Topology Comparator Module)

提供高階的拓樸比對 API，整合：
- 點集提取
- 度量計算
- 結果封裝

設計原則：
- 簡潔的 API，隱藏內部複雜度
- 結構化的結果返回
- 完善的錯誤處理
"""

import logging
from typing import Optional

import networkx as nx

from .metrics import compute_average_hausdorff_distance
from .point_extractor import GraphPointExtractor
from .data_types import ComparisonResult


class TopologyComparator:
    """
    拓樸比對器

    計算兩個拓樸圖之間的相似度度量。

    Examples:
        >>> comparator = TopologyComparator()
        >>> result = comparator.compare(graph_pred, graph_gt)
        >>> print(f"Hausdorff Distance: {result.hausdorff_distance:.4f}")
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.point_extractor = GraphPointExtractor(logger=self.logger)

    def compare(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph,
        label1: str = "圖1",
        label2: str = "圖2",
        return_directional: bool = False
    ) -> ComparisonResult:
        """
        比對兩個拓樸圖

        Args:
            graph1: 第一個圖
            graph2: 第二個圖
            label1: 第一個圖的標籤
            label2: 第二個圖的標籤
            return_directional: 是否返回雙向距離分量

        Returns:
            ComparisonResult 包含完整比對結果
        """
        result = ComparisonResult(
            label1=label1,
            label2=label2,
            num_nodes1=graph1.number_of_nodes(),
            num_nodes2=graph2.number_of_nodes(),
            num_edges1=graph1.number_of_edges(),
            num_edges2=graph2.number_of_edges(),
        )

        try:
            # 提取點集
            points1 = self.point_extractor.extract_points(graph1)
            points2 = self.point_extractor.extract_points(graph2)

            result.num_points1 = len(points1)
            result.num_points2 = len(points2)

            # 檢查空點集
            if len(points1) == 0 or len(points2) == 0:
                result.status = 'failed'
                result.error = 'empty_point_set'
                self.logger.warning(
                    f"點集為空: {label1}={len(points1)}, "
                    f"{label2}={len(points2)}"
                )
                return result

            # 計算 Hausdorff 距離
            if return_directional:
                distance, d_a_to_b, d_b_to_a = compute_average_hausdorff_distance(
                    points1, points2, return_components=True
                )
                result.hausdorff_a_to_b = d_a_to_b
                result.hausdorff_b_to_a = d_b_to_a
            else:
                distance = compute_average_hausdorff_distance(
                    points1, points2
                )

            result.hausdorff_distance = distance

            self.logger.info(
                f"比對完成: {label1} vs {label2} = {distance:.4f} "
                f"(點數: {len(points1)} vs {len(points2)})"
            )

        except Exception as e:
            result.status = 'failed'
            result.error = str(e)
            self.logger.error(f"比對失敗: {e}", exc_info=True)

        return result
