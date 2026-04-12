import logging

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from skimage.measure import label
import cv2

from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.pathfinding import PathFinder
from neural_reconstruction.common.data_types import LinkerResult
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.crosses_detection import (
    RegionLabeler,
    SegmentDetector,
    CrossingCounter,
)
from collections import defaultdict
from skimage.feature import hessian_matrix

logger = logging.getLogger(__name__)


class CompositLinker:
    def __init__(
        self,
        # 預處理參數
        offset_px: int = 10,
        rolling_ball_radius: int = 50,
        opening_kernel_size: int = 3,
        # 元件分析參數
        segment_length: float = 5.0,
        # 路徑查找參數
        search_radius: float = 50.0,
        intensity_weight: float = 2.0,
        min_component_length: float = 10.0,
    ):
        # 預處理參數
        self.offset_px = offset_px
        self.rolling_ball_radius = rolling_ball_radius
        self.opening_kernel_size = opening_kernel_size
        # 重建參數
        self.segment_length = segment_length

        self.search_radius = search_radius
        self.intensity_weight = intensity_weight
        self.min_component_length = min_component_length

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

        if len(image.shape) == 3:
            image = image[:, :, 1]  # 綠色通道

        roi_mask = dilate_epidermis_vertically(mask, offset_px=self.offset_px)
        roi_mask[roi_mask > 0] = 255

        kernal_size = self.rolling_ball_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernal_size, kernal_size)
        )
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.subtract(image, background)

        roi_image = cv2.bitwise_and(image, image, mask=roi_mask)
        roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)

        # apply opening to roi_annotation to remove small noise
        if self.opening_kernel_size > 0:
            # roi_annotation = cv2.morphologyEx(roi_annotation, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.opening_kernel_size, self.opening_kernel_size)
            )
            # apply closing to roi_annotation to fill small holes
            roi_annotation = cv2.morphologyEx(roi_annotation, cv2.MORPH_CLOSE, kernel)

        reconstruction_graph = self._run_reconstruction(roi_annotation, roi_image)
        roi_annotation[roi_annotation > 0] = 255

        valid_count, labeled_graph = self._run_crossing_analysis(
            mask, reconstruction_graph
        )

        return LinkerResult(
            annotation=roi_annotation,
            image=roi_image,
            mask=roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )

    def extract_path_features(
        self, source, target, path, cost, cost_map, orient_x, orient_y
    ):
        """
        Group 1 – Cost features
            mean_cost   : cost / path_length  (average per-step cost)
            cost_std    : std of per-pixel cost_map values along path
            max_cost    : maximum per-pixel cost along path (darkest point)
            path_length : number of pixels in path

        Group 2 – Geometric features (Hessian-based local orientation)
            angle_A  : angle between fiber orientation at A and A→B direction [0,90]°
            angle_B  : angle between fiber orientation at B and B→A direction [0,90]°
            angle_AB : consistency of tangents at A and B – how "parallel" they are [0,90]°
                    0° = smooth curve, 90° = orientations are perpendicular
        """

        path_arr = np.array(path)
        path_length = len(path_arr)

        # --- Group 1: Cost features ---
        step_costs = cost_map[path_arr[:, 0], path_arr[:, 1]]
        mean_cost = cost / path_length
        cost_std = float(np.std(step_costs))
        max_cost = float(np.max(step_costs))

        # --- Group 2: Geometric features ---
        def get_orientation(pt):
            """Return unit fiber-direction vector (dy, dx) at seed point pt=(y,x)."""
            y, x = int(pt[0]), int(pt[1])
            return np.array([orient_y[y, x], orient_x[y, x]])

        o_A = get_orientation(source)  # local fiber direction at A
        o_B = get_orientation(target)  # local fiber direction at B

        # Connection direction A→B (unit vector)
        conn = np.array(target, dtype=float) - np.array(source, dtype=float)
        conn_unit = conn / (np.linalg.norm(conn) + 1e-10)

        # Tangent-connection angle: use abs(dot) because orientation is undirected
        def _angle_undirected(v, u):
            return float(np.degrees(np.arccos(np.clip(abs(np.dot(v, u)), 0.0, 1.0))))

        angle_A = _angle_undirected(o_A, conn_unit)  # A's orientation vs A→B
        angle_B = _angle_undirected(o_B, -conn_unit)  # B's orientation vs B→A

        # Tangent consistency: align o_B to o_A's half-space, then measure angle
        o_B_aligned = o_B if np.dot(o_A, o_B) >= 0 else -o_B
        angle_AB = float(
            np.degrees(np.arccos(np.clip(np.dot(o_A, o_B_aligned), -1.0, 1.0)))
        )

        return dict(
            source=source,
            target=target,
            mean_cost=mean_cost,
            cost_std=cost_std,
            max_cost=max_cost,
            path_length=path_length,
            angle_A=angle_A,
            angle_B=angle_B,
            angle_AB=angle_AB,
        )

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
        for _, _, data in global_graph.edges(data=True):
            data["weight"] = 1e-5

        # 3. 元件間路徑查找
        cost_map = ((255 - image.astype(np.float64)) / 255) ** self.intensity_weight

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

        SIGMA = 2
        img_float = image.astype(np.float64)
        Hrr, Hrc, Hcc = hessian_matrix(img_float, sigma=SIGMA, order="rc")

        discriminant = np.sqrt(((Hrr - Hcc) / 2) ** 2 + Hrc**2)
        lam2 = (Hrr + Hcc) / 2 + discriminant  # larger eigenvalue

        fiber_vy = Hrc
        fiber_vx = lam2 - Hrr
        fiber_norm = np.sqrt(fiber_vy**2 + fiber_vx**2)
        fiber_norm = np.where(fiber_norm < 1e-10, 1.0, fiber_norm)

        orient_y = fiber_vy / fiber_norm  # (H, W) unit-vector y-component
        orient_x = fiber_vx / fiber_norm  # (H, W) unit-vector x-component

        for (source, target), (path, cost) in path_lookup.items():
            if global_graph.has_edge(tuple(source), tuple(target)):
                continue

            feat = self.extract_path_features(
                source, target, path, cost, cost_map, orient_x, orient_y
            )
            new_cost = (
                0.30 * feat["mean_cost"]
                + 0.20 * feat["max_cost"]
                + 0.25 * (feat["angle_A"] / 90.0)
                + 0.25 * (feat["angle_B"] / 90.0)
            )

            global_graph.add_edge(
                tuple(source), tuple(target), weight=new_cost, path=path
            )

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
