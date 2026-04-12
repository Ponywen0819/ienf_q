"""
點集提取模組 (Point Extractor Module)

從 NetworkX 圖中提取用於度量計算的點集：
- 節點座標
- 邊路徑點（支援多種屬性名稱）

設計原則：
- 支援 'path' 和 'path-coordinates' 兩種邊屬性
- 自動去重以提高效率
- 提供詳細的提取統計資訊
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx


class GraphPointExtractor:
    """
    圖點集提取器

    從 NetworkX 圖中提取所有點（節點 + 邊路徑點）用於度量計算。

    支援的邊路徑屬性：
    - 'path': Pipeline 生成的預測圖使用
    - 'path-coordinates': GT 拓樸使用

    Examples:
        >>> extractor = GraphPointExtractor()
        >>> points = extractor.extract_points(graph)
        >>> print(points.shape)
        (150, 2)
    """

    # 支援的邊路徑屬性名稱（按優先順序）
    EDGE_PATH_ATTRIBUTES = ["path", "path-coordinates"]

    def __init__(
        self, logger: Optional[logging.Logger] = None, remove_duplicates: bool = True
    ):
        """
        Args:
            logger: 日誌記錄器
            remove_duplicates: 是否去除重複點
        """
        self.logger = logger or logging.getLogger(__name__)
        self.remove_duplicates = remove_duplicates

    def extract_points(
        self,
        graph: nx.Graph,
        include_nodes: bool = True,
        include_edge_paths: bool = True,
    ) -> np.ndarray:
        """
        從圖中提取所有點

        Args:
            graph: NetworkX 圖，節點為 (y, x) 座標
            include_nodes: 是否包含節點座標
            include_edge_paths: 是否包含邊路徑點

        Returns:
            點集陣列，形狀 (N, 2)，每行為 [y, x]
            如果圖為空或無可提取點，返回形狀 (0, 2) 的空陣列

        Raises:
            ValueError: 如果 include_nodes 和 include_edge_paths 都為 False
        """
        if not include_nodes and not include_edge_paths:
            raise ValueError("至少需要包含節點或邊路徑點")

        points = []
        stats = {
            "num_nodes": 0,
            "num_edges": 0,
            "edges_with_path": 0,
            "num_path_points": 0,
        }

        # 提取節點
        if include_nodes:
            nodes = list(graph.nodes())
            points.extend(nodes)
            stats["num_nodes"] = len(nodes)

        # 提取邊路徑點
        if include_edge_paths:
            path_points, edge_stats = self._extract_edge_paths(graph)
            points.extend(path_points)
            stats["num_edges"] = edge_stats["total_edges"]
            stats["edges_with_path"] = edge_stats["edges_with_path"]
            stats["num_path_points"] = edge_stats["num_path_points"]

        # 記錄統計資訊
        self._log_extraction_stats(stats)

        # 轉換為 numpy 陣列
        if len(points) == 0:
            return np.array([]).reshape(0, 2)

        points_array = np.array(points, dtype=np.float64)

        # 去重
        if self.remove_duplicates:
            points_array = self._deduplicate_points(points_array)

        return points_array

    def _extract_edge_paths(
        self, graph: nx.Graph
    ) -> Tuple[List[Tuple[float, float]], dict]:
        """
        提取所有邊上的路徑點

        Returns:
            (路徑點列表, 統計資訊字典)
        """
        path_points = []
        edges_with_path = 0
        num_path_points = 0

        for u, v, edge_data in graph.edges(data=True):
            # 嘗試所有支援的屬性名稱
            path = self._get_edge_path(edge_data)

            if path is not None and len(path) > 0:
                path_points.extend(path)
                num_path_points += len(path)
                edges_with_path += 1

        stats = {
            "total_edges": graph.number_of_edges(),
            "edges_with_path": edges_with_path,
            "num_path_points": num_path_points,
        }

        return path_points, stats

    def _get_edge_path(self, edge_data: dict) -> Optional[List]:
        """
        從邊資料中獲取路徑點

        嘗試所有支援的屬性名稱，返回第一個找到的
        """
        for attr_name in self.EDGE_PATH_ATTRIBUTES:
            path = edge_data.get(attr_name)
            if path is not None:
                return path
        return None

    def _deduplicate_points(self, points: np.ndarray) -> np.ndarray:
        """
        去除重複點

        例如節點可能與邊的端點重複，這可以減少計算量並提高效率
        """
        points_unique = np.unique(points, axis=0)

        num_duplicates = len(points) - len(points_unique)
        if num_duplicates > 0:
            self.logger.debug(
                f"去除 {num_duplicates} 個重複點，剩餘 {len(points_unique)} 個唯一點"
            )

        return points_unique

    def _log_extraction_stats(self, stats: dict) -> None:
        """記錄提取統計資訊"""
        num_nodes = stats["num_nodes"]
        num_edges = stats["num_edges"]
        edges_with_path = stats["edges_with_path"]
        num_path_points = stats["num_path_points"]
        total_points = num_nodes + num_path_points

        if num_edges > 0:
            self.logger.debug(
                f"提取點集: {num_nodes} 節點 + {num_path_points} 邊路徑點 "
                f"({edges_with_path}/{num_edges} 條邊包含路徑) = {total_points} 總點數"
            )
        else:
            self.logger.debug(f"提取點集: {num_nodes} 節點（無邊）")


def extract_graph_points(graph: nx.Graph, remove_duplicates: bool = True) -> np.ndarray:
    """
    便捷函數：從圖中提取所有點

    Args:
        graph: NetworkX 圖
        remove_duplicates: 是否去除重複點

    Returns:
        點集陣列，形狀 (N, 2)

    Examples:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_node((10, 20))
        >>> G.add_node((30, 40))
        >>> points = extract_graph_points(G)
        >>> print(points.shape)
        (2, 2)
    """
    extractor = GraphPointExtractor(remove_duplicates=remove_duplicates)
    return extractor.extract_points(graph)
