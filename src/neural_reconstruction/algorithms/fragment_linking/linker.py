"""
階層式片段連接器模組 (Hierarchical Fragment Linker Module)

主控制器，協調完整的片段連接流程：
1. 預處理：ROI 提取、背景減除、偽標註生成
2. 骨架圖構建
3. 種子圖生成
4. MCP 路徑查找
5. 階段1：高信心端點延伸（嚴格約束）
6. 階段2：MST 候選邊生成（寬鬆約束）
7. MST 優化
"""

import logging
import numpy as np
import cv2
import networkx as nx
from scipy.spatial import KDTree
from skimage.measure import label

from neural_reconstruction.core.preprocessing import SkinAnalysisPipeline
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.pathfinding import PathFinder
from neural_reconstruction.common.data_types import LinkerResult
from .endpoint_extension import extend_endpoints
from .mst_candidates import generate_mst_candidates
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.crosses_detection import (
    RegionLabeler,
    SegmentDetector,
    CrossingCounter,
)
from collections import defaultdict


logger = logging.getLogger(__name__)


class HierarchicalFragmentLinker:
    """
    階層式片段連接器（含預處理）

    協調完整的神經纖維片段連接流程。

    Examples:
        >>> linker = HierarchicalFragmentLinker()
        >>> result_graph = linker.run(image, mask, annotation)
    """

    def __init__(
        self,
        # 預處理參數
        offset_px: int = 50,
        rolling_ball_radius: int = 50,
        opening_kernel_size: int = 3,
        # 種子圖參數
        segment_length: float = 3.0,
        intensity_power: float = 2.0,
        # 階段1參數
        search_radius_endpoint_extension: float = 20.0,
        max_angle_endpoint_extension: float = 75.0,
        angle_penalty_endpoint_extension: float = 0.5,
        direction_threshold_endpoint_extension: float = 5.0,
        # 階段2參數
        search_radius_mst: float = 50.0,
        max_angle_mst: float = 90.0,
        angle_penalty_mst: float = 0.5,
        distance_weight_mst: float = 0.2,
        max_cost_threshold_mst: float = 0.75,
        # MST參數
        endpoint_extension_weight_discount: float = 0.5,
        min_component_length: float = 10.0,
    ):
        # 預處理參數
        self.offset_px = offset_px
        self.rolling_ball_radius = rolling_ball_radius
        self.opening_kernel_size = opening_kernel_size

        # 重建參數
        self.segment_length = segment_length
        self.intensity_power = intensity_power
        self.search_radius_pathfinding = max(
            search_radius_endpoint_extension, search_radius_mst
        )

        self.search_radius_endpoint_extension = search_radius_endpoint_extension
        self.max_angle_endpoint_extension = max_angle_endpoint_extension
        self.angle_penalty_endpoint_extension = angle_penalty_endpoint_extension
        self.direction_threshold_endpoint_extension = (
            direction_threshold_endpoint_extension
        )

        self.search_radius_mst = search_radius_mst
        self.max_angle_mst = max_angle_mst
        self.angle_penalty_mst = angle_penalty_mst
        self.distance_weight_mst = distance_weight_mst
        self.max_cost_threshold_mst = max_cost_threshold_mst

        self.endpoint_extension_weight_discount = endpoint_extension_weight_discount
        self.min_component_length = min_component_length

    def run(
        self, image: np.ndarray, mask: np.ndarray, annotation: np.ndarray
    ) -> LinkerResult:
        """
        運行完整的階層式片段連接算法（含預處理）

        Args:
            image: 原始圖像 (H, W) 或 (H, W, 3)
            mask: 表皮遮罩 (H, W)
            annotation: 手工標註 (H, W)

        Returns:
            LinkerResult
        """
        # 提取綠色通道
        if len(image.shape) == 3:
            image = image[:, :, 1]  # 綠色通道

        logger.info("=" * 60)
        logger.info("開始階層式片段連接（含預處理）")
        logger.info("=" * 60)

        # 1. 預處理
        logger.info("1. 圖像預處理...")
        roi_mask = dilate_epidermis_vertically(mask, offset_px=50)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.subtract(image, background)

        roi_image = cv2.bitwise_and(image, image, mask=roi_mask)
        roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)

        # apply opening to roi_annotation to remove small noise
        if self.opening_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.opening_kernel_size, self.opening_kernel_size)
            )
            roi_annotation = cv2.morphologyEx(roi_annotation, cv2.MORPH_OPEN, kernel)

            # apply closing to roi_annotation to fill small holes
            roi_annotation = cv2.morphologyEx(roi_annotation, cv2.MORPH_CLOSE, kernel)
        roi_annotation[roi_annotation > 0] = 255

        logger.info("  ✓ 預處理完成")

        topology_builder = TopologyBuilder(
            segment_length=self.segment_length,
        )
        seed_graph = topology_builder.build_seed_graph(roi_annotation, roi_image)

        # 為種子圖的邊設置低成本
        for u, v, data in seed_graph.edges(data=True):
            data["weight"] = 1e-5

        # 4. 準備路徑查找
        logger.info("4. 準備路徑查找...")

        topology_points = np.array(list(seed_graph.nodes()))
        kdtree = KDTree(topology_points)

        # 構建成本地圖和種子地圖
        cost_map = (
            (255 - roi_image.astype(np.float64)) / 255.0
        ) ** self.intensity_power

        seed_map = np.zeros_like(cost_map, dtype=np.uint8)
        for _, _, data in seed_graph.edges(data=True):
            path = data.get("path", [])
            for p in path:
                seed_map[int(p[0]), int(p[1])] = True

        # 構建 label 圖（用於排除同一連通分量）
        binary = (roi_annotation > 0).astype(np.uint8)
        label_img = np.asarray(label(binary, connectivity=2))

        # 5. 路徑查找
        logger.info("5. 路徑查找...")

        path_finder = PathFinder(cost_map=cost_map)

        path_lookup = path_finder.find_paths_from_seeds(
            topology_points=topology_points,
            kdtree=kdtree,
            search_radius=self.search_radius_pathfinding,
            seed_map=seed_map,
            label_img=label_img,
        )

        # 6. 端點延伸
        logger.info("6. 端點延伸...")

        phase1_edges = extend_endpoints(
            graph=seed_graph,
            topology_points=topology_points,
            kdtree=kdtree,
            path_lookup=path_lookup,
            search_radius=self.search_radius_endpoint_extension,
            max_angle_degrees=self.max_angle_endpoint_extension,
            max_angle_penalty=self.angle_penalty_endpoint_extension,
            direction_threshold=self.direction_threshold_endpoint_extension,
        )

        # 添加階段1的邊
        extended_graph = seed_graph.copy()
        for endpoint, target, cost, path in phase1_edges:
            if not extended_graph.has_edge(endpoint, target):
                extended_graph.add_edge(
                    endpoint,
                    target,
                    weight=cost * self.endpoint_extension_weight_discount,
                    path=path,
                    phase=1,
                )
        extended_graph = nx.minimum_spanning_tree(extended_graph, weight="weight")

        # 7. 生成 MST 候選邊
        logger.info("7. 生成 MST 候選邊...")

        phase2_candidates = generate_mst_candidates(
            graph=extended_graph,
            topology_points=topology_points,
            kdtree=kdtree,
            path_lookup=path_lookup,
            search_radius=self.search_radius_mst,
            max_angle_degrees=self.max_angle_mst,
            angle_penalty_weight=self.angle_penalty_mst,
            distance_weight=self.distance_weight_mst,
            max_cost_threshold=self.max_cost_threshold_mst,
        )

        # 8. 執行 MST 優化
        logger.info("8. 執行 MST 優化...")

        # 複製圖並對階段1的邊打折
        mst_tree = extended_graph.copy()
        # 添加階段2候選邊
        for endpoint, target, cost, path in phase2_candidates:
            if not mst_tree.has_edge(endpoint, target):
                mst_tree.add_edge(endpoint, target, weight=cost, path=path, phase=2)

        # 執行 MST
        mst_result = nx.minimum_spanning_tree(mst_tree, weight="weight")

        nodes_to_remove = []
        for component_nodes in nx.connected_components(mst_result):
            subgraph = mst_result.subgraph(component_nodes)
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

        filtered = mst_result.copy()
        filtered.remove_nodes_from(nodes_to_remove)

        valid_count, labeled_graph = self._run_crossing_analysis(mask, filtered)
        return LinkerResult(
            annotation=roi_annotation,
            image=roi_image,
            mask=roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )

    def _run_crossing_analysis(
        self,
        mask: np.ndarray,
        graph: nx.Graph,
    ) -> tuple[int, nx.Graph]:
        """
        交叉點分析

        Args:
            mask: 二值化遮罩影像 (0/255 或 0/1)
            graph: MST 重建圖 (nx.Graph)

        Returns:
            {component_id: crossing_count}
        """

        region_labeler = RegionLabeler()
        segment_detector = SegmentDetector()
        crossing_counter = CrossingCounter()

        # Step 1: Detect segments (must run before RegionLabeler — segment_id is needed by label_topology)
        segmented_graph = segment_detector.detect_segments(graph)

        # Step 1b: Remove short stub segments (total path length < 5 px, with at least one endpoint)
        def _path_length(data, u, v):
            path = data.get("path", [u, v])
            return len(path) - 1  # number of pixel steps

        seg_edges = defaultdict(list)
        for u, v, data in segmented_graph.edges(data=True):
            seg_id = data.get("segment_id")
            if seg_id is not None:
                seg_edges[seg_id].append((u, v, data))

        edges_to_remove = []
        for seg_id, edges in seg_edges.items():
            # Collect boundary nodes of this segment
            boundary_nodes = set()
            for u, v, _ in edges:
                if segmented_graph.nodes[u].get("node_type") in (
                    "endpoint",
                    "branchpoint",
                ):
                    boundary_nodes.add(u)
                if segmented_graph.nodes[v].get("node_type") in (
                    "endpoint",
                    "branchpoint",
                ):
                    boundary_nodes.add(v)

            # Only prune if at least one boundary node is an endpoint (dangling stub)
            has_endpoint = any(
                segmented_graph.nodes[n].get("node_type") == "endpoint"
                for n in boundary_nodes
            )
            if not has_endpoint:
                continue

            total_length = sum(_path_length(data, u, v) for u, v, data in edges)
            if total_length < 5:
                edges_to_remove.extend((u, v) for u, v, _ in edges)

        segmented_graph.remove_edges_from(edges_to_remove)
        segmented_graph.remove_nodes_from(list(nx.isolates(segmented_graph)))
        logger.info(
            f"Pruned {len(edges_to_remove)} stub edges → "
            f"{segmented_graph.number_of_nodes()} nodes, {segmented_graph.number_of_edges()} edges"
        )

        segmented_graph = segment_detector.detect_segments(segmented_graph)
        # Step 2: Label regions and mark crossing edges
        labeled_graph, _ = region_labeler.label_topology(segmented_graph, mask)

        # Step 3: Count effective crossings
        result = crossing_counter.count_effective_crossings(
            labeled_graph, epidermis_mask=mask
        )

        return result["effective_crossing_count"], labeled_graph
