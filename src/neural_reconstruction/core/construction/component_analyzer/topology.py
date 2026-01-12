"""
元件拓樸建構模組 (Component Topology Module)

使用 skan 函式庫提供完整的骨架分析與拓樸建構功能：
1. ComponentTopologyBuilder - 元件拓樸建構器（骨架化 + 分析 + 拓樸建構）

輸出為 NetworkX MultiGraph，節點與邊的屬性包含：
- 節點屬性: coordinates (y, x 座標), degree (連接數)
- 邊屬性: path-coordinates (路徑座標), branch-distance (路徑長度), branch-type (分支類型)
"""

import logging

import numpy as np
import networkx as nx
from skimage import morphology
from skan import Skeleton
from skan.csr import skeleton_to_nx

logger = logging.getLogger(__name__)


class ComponentTopologyBuilder:
    """
    元件拓樸建構器

    整合骨架化、分析與拓樸建構的完整流程：
    1. 使用 skimage 進行骨架化
    2. 使用 skan 對骨架進行分析
    3. 使用 skeleton_to_nx 轉換為 NetworkX graph
    """

    def __init__(self, prune_threshold: float = 5.0, spacing: float = 1.0):
        """
        Args:
            prune_threshold: 剪枝閾值 - 移除長度小於此值的分支（像素）
            spacing: 像素間距（用於距離計算）
        """
        self.prune_threshold = prune_threshold
        self.spacing = spacing

    def build_topology(self, mask: np.ndarray) -> nx.MultiGraph:
        """
        建構元件的完整拓樸結構

        Args:
            mask: 二值 mask (0 或 255)

        Returns:
            nx.MultiGraph: NetworkX 拓樸圖
                - 節點屬性: coordinates (y, x), degree
                - 邊屬性: path-coordinates, branch-distance, branch-type
        """
        logger.debug("開始建構元件拓樸...")

        skeleton = self._skeletonize(mask)

        graph = self._get_skeleton_graph(skeleton)

        logger.debug(
            f"拓樸建構完成: {graph.number_of_nodes()} 節點, {graph.number_of_edges()} 邊"
        )
        return graph

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """
        使用 skimage 進行骨架化（Zhang-Suen 演算法）

        Args:
            mask: 二值 mask (0 或 255)

        Returns:
            骨架影像 (0 或 1)
        """
        binary = (mask > 0).astype(np.uint8)
        skeleton = morphology.skeletonize(binary)
        return skeleton.astype(np.uint8)

    def _get_skeleton_graph(self, skeleton: np.ndarray) -> nx.MultiGraph:
        """
        建立 Skeleton 物件並取得分支摘要資料

        Args:
            skeleton: 骨架影像 (0 或 255)

        Returns:
            Skeleton 物件與分支摘要資料陣列
        """
        skel_obj = Skeleton(skeleton, keep_images=False)
        graph = skeleton_to_nx(skel_obj)

        # 建立節點 ID 到座標的映射
        # skel_obj.coordinates 是一個 (N, 2) 的 numpy array
        mapping = {i: tuple(skel_obj.coordinates[i].astype(int)) for i in graph.nodes()}

        # 重標籤節點
        graph = nx.relabel_nodes(graph, mapping)

        return graph
