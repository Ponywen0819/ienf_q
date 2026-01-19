"""
區域標注器 (Region Labeler)

為神經纖維拓樸結構中的每個節點標注其所屬區域（表皮/真皮），
並為每條邊標注是否跨越表皮/真皮邊界。
"""

import numpy as np
from typing import Tuple
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class RegionLabeler:
    """
    區域標注器 - 為拓樸節點標注區域屬性，為邊標注跨越屬性

    根據表皮遮罩判斷每個節點位於表皮(epidermis)或真皮(dermis)區域，
    並標注每條邊是否跨越表皮/真皮邊界。
    """

    def __init__(self):
        """初始化區域標注器"""
        logger.info("Initialized RegionLabeler")

    def label_topology(
        self, graph: nx.Graph, epidermis_mask: np.ndarray
    ) -> Tuple[nx.Graph, int]:
        """
        為拓樸結構標注區域資訊

        Args:
            graph: nx.Graph 拓樸結構
            epidermis_mask: 表皮遮罩

        Returns:
            nx.Graph: 標注後的拓樸結構,
            int: 跨越表皮/真皮邊界的邊數
        """
        res_graph = graph.copy()
        corossing_segments = set()

        # 標注每個節點的區域
        for node, data in res_graph.nodes(data=True):
            node_y = node[0]
            node_x = node[1]
            region = self._get_node_region((node_y, node_x), epidermis_mask)
            res_graph.nodes[node]["region"] = region

        # crossing_count = 0
        for u, v, data in res_graph.edges(data=True):
            source_region = res_graph.nodes[u]["region"]
            target_region = res_graph.nodes[v]["region"]
            is_crossing = source_region != target_region
            data["is_crossing"] = is_crossing
            if is_crossing:
                corossing_segments.add(data["segment_id"])
                # crossing_count += 1

        return res_graph, len(corossing_segments)

    def _get_node_region(
        self, position: Tuple[int, int], epidermis_mask: np.ndarray
    ) -> str:
        """
        判斷節點所屬區域

        Args:
            position: 節點座標 (y, x)
            epidermis_mask: 表皮遮罩

        Returns:
            region: 'epidermis' 或 'dermis'
        """
        y, x = position
        height, width = epidermis_mask.shape

        # 邊界檢查
        if y < 0 or y >= height or x < 0 or x >= width:
            # logger.warning(f"節點座標 ({y}, {x}) 超出遮罩範圍，預設為真皮區域")
            return "dermis"

        # 查詢遮罩值
        mask_value = epidermis_mask[y, x]

        # 255 = 表皮, 0 = 真皮
        if mask_value > 127:  # 使用閾值以處理可能的中間值
            return "epidermis"
        else:
            return "dermis"
