"""
資料型別定義模組 (Data Types Module)

定義神經重建分析所需的資料結構：
- SeedPoint: 種子點
- TopologyNode: 拓樸節點（端點/分支點）
- TopologyEdge: 拓樸邊
- TopologyResult: 完整拓樸結構
- ComponentAnalysisResult: 元件分析結果
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx


@dataclass
class SeedPoint:
    """
    種子點資料結構

    Attributes:
        position: (y, x) 座標（局部座標）
        seed_type: 種子類型 ('edge', 'endpoint', 'branchpoint', 'centroid')
        component_id: 所屬元件 ID
        edge_id: 所屬邊 ID（僅 edge 類型有值）
    """

    position: Tuple[int, int]
    seed_type: str
    component_id: int
    edge_id: Optional[int] = None


@dataclass
class TopologyNode:
    """
    拓樸節點資料結構

    Attributes:
        node_id: 節點 ID
        position: (y, x) 座標（局部座標）
        node_type: 節點類型 ('endpoint' or 'branchpoint')
    """

    node_id: int
    position: Tuple[int, int]
    node_type: str


@dataclass
class TopologyEdge:
    """
    拓樸邊資料結構

    Attributes:
        source_id: 起點節點 ID
        target_id: 終點節點 ID
        path: 路徑座標列表（局部座標）
        length: 路徑長度（考慮對角線距離）
    """

    source_id: int
    target_id: int
    path: List[Tuple[int, int]]
    length: float


@dataclass
class TopologyResult:
    """
    完整拓樸結構

    Attributes:
        nodes: 節點列表
        edges: 邊列表
    """

    nodes: List[TopologyNode] = field(default_factory=list)
    edges: List[TopologyEdge] = field(default_factory=list)


@dataclass
class ComponentAnalysisResult:
    """
    單一元件的完整分析結果

    Attributes:
        component_id: 元件 ID
        bbox: 邊界框 (minr, minc, maxr, maxc)，用於座標轉換
        skeleton: 骨架影像（局部座標）
        topology: 拓樸結構（局部座標）
        seeds: 種子點列表（局部座標）
    """

    component_id: int
    bbox: Tuple[int, int, int, int]
    skeleton: np.ndarray
    topology: TopologyResult
    seeds: List[SeedPoint] = field(default_factory=list)


@dataclass
class ConnectionGraphBuilderResult:
    """
    元件連接圖建構結果

    Attributes:
        graph: 連接圖 (networkx.Graph)
        component_id_to_node: 元件 ID 映射到圖節點的字典
    """

    nodes: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # 全局種子座標 (N, 2)
    component_ids: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # 對應的元件 ID (N,)
    edges: List[dict] = field(default_factory=list)  # 邊列表

    graph: nx.Graph = field(default_factory=nx.Graph)


@dataclass
class PathFindingResult:
    pass
