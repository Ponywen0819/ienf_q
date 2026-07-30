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

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from skimage.measure import label

from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.pathfinding import PathFinder
from neural_reconstruction.common.data_types import LinkerResult
from neural_reconstruction.core.preprocessing import PreprocessingPipeline
from neural_reconstruction.core.crosses_detection import run_crossing_analysis

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
        offset_px: int = 50,
        bg_kernel_size: int = 31,
        clahe_clip: float = 30.0,
        clahe_grid: tuple[int, int] = (1024, 1024),
        sato_sigmas_start: int = 3,
        sato_sigmas_stop: int = 8,
        sato_sigmas: list[float] | None = None,
        # 元件分析參數
        segment_length: float = 5.0,
        # 路徑查找參數
        search_radius: float = 20.0,
        min_component_length: float = 10.0,
        # 子樹過濾參數
        min_tree_components: int = 5,
    ):
        # 預處理參數
        self.offset_px = offset_px
        self.bg_kernel_size = bg_kernel_size
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.sato_sigmas_start = sato_sigmas_start
        self.sato_sigmas_stop = sato_sigmas_stop
        # 重建參數
        self.segment_length = segment_length
        self.search_radius = search_radius
        self.min_component_length = min_component_length
        self.min_tree_components = min_tree_components

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        annotation: np.ndarray,
    ) -> LinkerResult:
        """
        運行完整的 MST 神經纖維重建流程（含預處理）

        Args:
            image: 原始圖像 (H, W) 或 (H, W, 3)
            mask: 表皮遮罩 (H, W)
            annotation: 手工標註 (H, W)

        Returns:
            LinkerResult
        """
        logger.info("1. 圖像預處理...")

        # 保留 2D 原始表皮遮罩供交叉分析使用
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        pre = PreprocessingPipeline(
            offset_px=self.offset_px,
            bg_kernel_size=self.bg_kernel_size,
            clahe_clip=self.clahe_clip,
            clahe_grid=self.clahe_grid,
            sato_sigmas_start=self.sato_sigmas_start,
            sato_sigmas_stop=self.sato_sigmas_stop,
        ).run(image, mask, annotation)

        annot_labeled = np.asarray(
            label((pre.roi_annotation > 0).astype(np.uint8), connectivity=2)
        )
        reconstruction_graph = self._run_reconstruction(
            pre.roi_annotation, pre.roi_image, pre.cost_map
        )

        valid_count, labeled_graph = run_crossing_analysis(
            reconstruction_graph,
            mask,
            annot_labeled,
            min_tree_components=self.min_tree_components,
        )

        return LinkerResult(
            annotation=pre.roi_annotation,
            image=pre.roi_image,
            mask=pre.roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )

    def _run_reconstruction(
        self,
        annotation: np.ndarray,
        image: np.ndarray,
        cost_map: np.ndarray,
    ) -> nx.Graph:
        """
        運行 MST 重建（不含預處理）

        Args:
            annotation: 二值化標註影像 (0/255 或 0/1)
            image: 影像 (uint8, 0-255)
            cost_map: 由前處理產生的成本圖 (float32)

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
        for _, _, data in global_graph.edges(data=True):
            data["weight"] = 1e-5

        # 3. 元件間路徑查找（cost_map 由前處理階段提供）
        path_finder = PathFinder(cost_map)
        topology_points = np.array(list(global_graph.nodes()))

        kdtree = KDTree(topology_points)
        seed_map = np.zeros_like(cost_map, dtype=bool)
        for p in topology_points:
            seed_map[p[0], p[1]] = True

        path_finder = PathFinder(cost_map)
        topology_points = np.array(list(global_graph.nodes()))
        path_lookup = path_finder.find_paths_from_seeds(
            topology_points=topology_points,
            kdtree=kdtree,
            search_radius=self.search_radius,
            seed_map=seed_map,
            label_img=labeled,
        )

        for (source, target), (path, cost) in path_lookup.items():
            if global_graph.has_edge(tuple(source), tuple(target)):
                continue
            global_graph.add_edge(tuple(source), tuple(target), weight=cost, path=path)

        # 4. MST
        mst_forest = nx.minimum_spanning_tree(global_graph, weight="weight")

        nodes_to_remove = []
        for component_nodes in nx.connected_components(mst_forest):
            subgraph = mst_forest.subgraph(component_nodes)
            total_length = 0.0
            for u, v, data in subgraph.edges(data=True):
                path = data.get("path", [])
                if len(path) >= 2:
                    pts = np.array(path)
                    total_length += float(
                        np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                    )
                else:
                    total_length += float(np.linalg.norm(np.array(u) - np.array(v)))
            if total_length < self.min_component_length:
                nodes_to_remove.extend(component_nodes)

        filtered = mst_forest.copy()
        filtered.remove_nodes_from(nodes_to_remove)

        return filtered
