"""
純 MST 重建連接器 (Pure MST Reconstruction Linker)

主控制器，協調完整的神經網路 MST 重建流程：
1. 預處理：ROI 提取、背景減除、偽標註生成
2. 連通元件分析
3. 骨架化與種子切分（TopologyBuilder + SeedGraphBuilder）
4. 元件間連接路徑查找（PathFinder）
5. MST 骨架萃取
"""

import logging
from typing import Optional

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from skimage.measure import label

from neural_reconstruction.core.preprocessing import SkinAnalysisPipeline
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.pathfinding import PathFinder

logger = logging.getLogger(__name__)


class PureMstLinker:
    """
    純 MST 神經重建連接器（含預處理）

    協調完整的 MST 神經纖維重建流程。

    Examples:
        >>> linker = PureMstLinker(segment_length=5.0, search_radius=50.0)
        >>> mst_forest = linker.run(image, mask, annotation)
    """

    def __init__(
        self,
        # 預處理參數
        offset_px: int = 100,
        rolling_ball_radius: int = 2,
        sato_weight: float = 0.0,
        opening_kernel_size: int = 3,
        # 元件分析參數
        segment_length: float = 5.0,
        min_edge_length: Optional[float] = None,
        # 路徑查找參數
        search_radius: float = 50.0,
        max_cost_threshold: float = 0.98,
        intensity_weight: float = 2.0,
    ):
        # 預處理參數
        self.offset_px = offset_px
        self.rolling_ball_radius = rolling_ball_radius
        self.sato_weight = sato_weight
        self.opening_kernel_size = opening_kernel_size
        # 重建參數
        self.segment_length = segment_length
        self.min_edge_length = (
            min_edge_length if min_edge_length is not None else segment_length
        )
        self.search_radius = search_radius
        self.max_cost_threshold = max_cost_threshold
        self.intensity_weight = intensity_weight

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        annotation: np.ndarray,
    ) -> nx.Graph:
        """
        運行完整的 MST 神經纖維重建流程（含預處理）

        Args:
            image: 原始圖像 (H, W) 或 (H, W, 3)
            mask: 表皮遮罩 (H, W)
            annotation: 手工標註 (H, W)

        Returns:
            MST 森林 (nx.Graph)
        """
        logger.info("1. 圖像預處理...")

        preprocessing_config = {
            "morphology": {
                "closing_kernel": 0,
                "opening_kernel": self.opening_kernel_size,
            },
            "mask": {
                "dilate_offset": self.offset_px,
            },
            "background": {
                "method": "rolling_ball",
                "radius": self.rolling_ball_radius,
                "sato_weight": self.sato_weight,
                "sato_sigmas": (1.0, 2.0),
            },
            "threshold": {
                "use_full_roi": False,
            },
            "normalization": {
                "enabled": False,
            },
        }

        pipeline = SkinAnalysisPipeline(preprocessing_config)

        if len(image.shape) == 3:
            orig_img = image[:, :, 1]  # 綠色通道
        else:
            orig_img = image

        roi_annotation, roi_image = pipeline.run(annotation, mask, orig_img)

        logger.info("  ✓ 預處理完成")

        return self._run_reconstruction(roi_annotation, roi_image)

    def _run_reconstruction(
        self,
        annotation: np.ndarray,
        image: np.ndarray,
    ) -> nx.Graph:
        """
        運行 MST 重建（不含預處理）

        Args:
            annotation: 二值化標註影像 (0/255 或 0/1)
            image: 影像 (uint8, 0-255)

        Returns:
            MST 森林 (nx.Graph)
        """
        if annotation is None or annotation.size == 0:
            return nx.Graph()
        if image is None or image.size == 0:
            return nx.Graph()

        # 1. 連通元件標記（用於排除同元件連接，使用 8-connectivity）
        binary = (annotation > 0).astype(np.uint8)
        labeled = np.asarray(label(binary, connectivity=2))

        # 2. 骨架化 + 種子切分
        topology_builder = TopologyBuilder(segment_length=self.segment_length)
        global_graph = topology_builder.build_seed_graph(annotation, image)

        if global_graph.number_of_nodes() == 0:
            return nx.Graph()

        # 從 labeled image 補上 component_id
        for node in global_graph.nodes():
            y, x = node
            global_graph.nodes[node]["component_id"] = int(labeled[int(y), int(x)])

        # 元件內邊設低成本
        for u, v, data in global_graph.edges(data=True):
            data["weight"] = 1e-5

        # 3. 元件間路徑查找
        cost_map = (
            255 - image.astype(np.float64)
        ) ** self.intensity_weight / 255**self.intensity_weight
        self._add_inter_component_edges(global_graph, cost_map)

        # 4. MST
        return self._extract_mst_forest(global_graph)

    def _add_inter_component_edges(
        self, global_graph: nx.MultiGraph, cost_map: np.ndarray
    ) -> None:
        """使用 PathFinder 在不同元件的節點間建立連接邊"""
        path_finder = PathFinder(cost_map)
        topology_points = np.array(list(global_graph.nodes()))
        kdtree = KDTree(topology_points)
        component_ids = np.array(
            [global_graph.nodes[tuple(p)]["component_id"] for p in topology_points]
        )

        processed_pairs: set = set()
        for i, source in enumerate(topology_points):
            source_comp = component_ids[i]
            neighbor_indices = np.array(
                kdtree.query_ball_point(source, r=self.search_radius), dtype=np.int32
            )

            # 過濾：排除自己與同元件
            mask = (neighbor_indices != i) & (
                component_ids[neighbor_indices] != source_comp
            )
            valid_indices = neighbor_indices[mask]

            targets = []
            for j in valid_indices:
                pair = (min(i, j), max(i, j))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    targets.append(tuple(topology_points[j].astype(int)))

            if not targets:
                continue

            paths = path_finder.find_paths_from_source(
                tuple(source.astype(int)), targets
            )
            for target_pos, result in paths.items():
                if result is None:
                    continue
                path, cost = result
                global_graph.add_edge(
                    tuple(source.astype(int)), target_pos, weight=cost, path=path
                )

    def _extract_mst_forest(self, graph: nx.MultiGraph) -> nx.Graph:
        """對每個連通分量分別萃取 MST，合併為森林"""
        forest = nx.MultiGraph()
        for component_nodes in nx.connected_components(graph):
            subgraph = graph.subgraph(component_nodes)
            if subgraph.number_of_edges() == 0:
                forest.add_nodes_from(subgraph.nodes(data=True))
            else:
                mst = nx.minimum_spanning_tree(subgraph, weight="weight")
                forest = nx.compose(forest, mst)
        return forest
