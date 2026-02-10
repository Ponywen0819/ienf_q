"""
純 MST 重建連接器 (Pure MST Reconstruction Linker)

主控制器，協調完整的神經網路 MST 重建流程：
1. 連通元件分析
2. 骨架化與種子切分（TopologyBuilder + SeedGraphBuilder）
3. 元件間連接路徑查找（PathFinder）
4. MST 骨架萃取
"""

import logging
from typing import Optional

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from skimage.measure import label, regionprops

from neural_reconstruction.core.topology import TopologyBuilder, SeedGraphBuilder
from neural_reconstruction.core.pathfinding import PathFinder

logger = logging.getLogger(__name__)


class PureMstLinker:
    """
    純 MST 神經重建連接器

    協調完整的 MST 神經纖維重建流程。

    Examples:
        >>> linker = PureMstLinker(segment_length=5.0, search_radius=50.0)
        >>> mst_forest = linker.run(label_image, green_channel)
    """

    def __init__(
        self,
        # 連通元件參數
        connectivity: int = 4,
        min_area: int = 0,
        # 元件分析參數
        segment_length: float = 5.0,
        min_edge_length: Optional[float] = None,
        # 路徑查找參數
        search_radius: float = 50.0,
        max_cost_threshold: float = 0.98,
        intensity_weight: float = 0.6,
        shape_weight: float = 0.4,
    ):
        self.connectivity = connectivity
        self.min_area = min_area
        self.segment_length = segment_length
        self.min_edge_length = min_edge_length if min_edge_length is not None else segment_length
        self.search_radius = search_radius
        self.max_cost_threshold = max_cost_threshold
        self.intensity_weight = intensity_weight
        self.shape_weight = shape_weight

    def run(
        self,
        label_image: np.ndarray,
        green_channel: np.ndarray,
    ) -> nx.Graph:
        """
        運行完整的 MST 神經纖維重建流程

        Args:
            label_image: 二值化標註影像 (0/255 或 0/1)
            green_channel: 綠色通道影像 (uint8, 0-255)

        Returns:
            MST 森林 (nx.Graph)
        """
        if label_image is None or label_image.size == 0:
            return nx.Graph()
        if green_channel is None or green_channel.size == 0:
            return nx.Graph()

        # 1. 連通元件分析
        binary = (label_image > 0).astype(np.uint8)
        skimage_connectivity = 1 if self.connectivity == 4 else 2
        labeled = label(binary, connectivity=skimage_connectivity)
        regions = regionprops(labeled)

        if self.min_area > 0:
            regions = [r for r in regions if r.area >= self.min_area]

        if not regions:
            return nx.Graph()

        # 2. 骨架化 + 種子切分，轉換至全局座標
        global_graph = self._build_global_graph(regions)

        if global_graph.number_of_nodes() == 0:
            return nx.Graph()

        # 3. 元件間路徑查找
        cost_map = (
            (255 - green_channel.astype(np.float64)) ** self.intensity_weight
            / 255 ** self.intensity_weight
        )
        self._add_inter_component_edges(global_graph, cost_map)

        # 4. MST
        return self._extract_mst_forest(global_graph)

    def _build_global_graph(self, regions) -> nx.MultiGraph:
        """對每個元件骨架化、切分種子，合併到全局座標圖"""
        topology_builder = TopologyBuilder()
        seed_builder = SeedGraphBuilder(segment_length=self.segment_length)
        global_graph = nx.MultiGraph()

        for region in regions:
            minr, minc, _, _ = region.bbox
            component_mask = region.image.astype(np.uint8) * 255
            component_id = region.label

            skeleton = topology_builder.build_skeleton_graph(component_mask)
            if skeleton.number_of_nodes() == 0:
                continue

            seed_graph = seed_builder.build(skeleton)

            # 轉換至全局座標，標記 component_id
            for node in seed_graph.nodes():
                global_node = (node[0] + minr, node[1] + minc)
                global_graph.add_node(global_node, component_id=component_id)

            for u, v, data in seed_graph.edges(data=True):
                gu = (u[0] + minr, u[1] + minc)
                gv = (v[0] + minr, v[1] + minc)
                gpath = [(p[0] + minr, p[1] + minc) for p in data.get("path", [])]
                global_graph.add_edge(gu, gv, path=gpath, weight=1e-5)

        return global_graph

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
