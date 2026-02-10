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

import numpy as np
import cv2
import networkx as nx
from scipy.spatial import KDTree
from skimage.measure import label

from neural_reconstruction.core.preprocessing import SkinAnalysisPipeline

from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.pathfinding import PathFinder
from .endpoint_extension import extend_endpoints
from .mst_candidates import generate_mst_candidates


class HierarchicalFragmentLinker:
    """
    階層式片段連接器（含預處理）

    協調完整的神經纖維片段連接流程。

    Examples:
        >>> linker = HierarchicalFragmentLinker(verbose=True)
        >>> result_graph = linker.run(image, mask, annotation)
    """

    def __init__(
        self,
        # 預處理參數
        offset_px: int = 100,
        rolling_ball_radius: int = 2,
        sato_weight: float = 0.0,
        opening_kernel_size: int = 3,
        # 種子圖參數
        segment_length: float = 3.0,
        # 路徑查找參數
        search_radius_pathfinding: float = 50.0,
        # 階段1參數
        search_radius_phase1: float = 10.0,
        max_angle_phase1: float = 75.0,
        angle_penalty_phase1: float = 0.5,
        direction_threshold_phase1: float = 5.0,
        # 階段2參數
        search_radius_phase2: float = 20.0,
        max_angle_phase2: float = 90.0,
        angle_penalty_phase2: float = 0.5,
        distance_weight_phase2: float = 0.2,
        max_cost_threshold_phase2: float = 0.75,
        # MST參數
        phase1_weight_discount: float = 0.5,
        verbose: bool = False,
    ):
        # 預處理參數
        self.offset_px = offset_px
        self.rolling_ball_radius = rolling_ball_radius
        self.sato_weight = sato_weight
        self.opening_kernel_size = opening_kernel_size

        # 重建參數
        self.segment_length = segment_length
        self.search_radius_pathfinding = search_radius_pathfinding

        self.search_radius_phase1 = search_radius_phase1
        self.max_angle_phase1 = max_angle_phase1
        self.angle_penalty_phase1 = angle_penalty_phase1
        self.direction_threshold_phase1 = direction_threshold_phase1

        self.search_radius_phase2 = search_radius_phase2
        self.max_angle_phase2 = max_angle_phase2
        self.angle_penalty_phase2 = angle_penalty_phase2
        self.distance_weight_phase2 = distance_weight_phase2
        self.max_cost_threshold_phase2 = max_cost_threshold_phase2

        self.phase1_weight_discount = phase1_weight_discount

        self.verbose = verbose

    def run(
        self, image: np.ndarray, mask: np.ndarray, annotation: np.ndarray
    ) -> nx.Graph:
        """
        運行完整的階層式片段連接算法（含預處理）

        Args:
            image: 原始圖像 (H, W) 或 (H, W, 3)
            mask: 表皮遮罩 (H, W)
            annotation: 手工標註 (H, W)

        Returns:
            MST 圖
        """
        if self.verbose:
            print("=" * 60)
            print("開始階層式片段連接（含預處理）")
            print("=" * 60)

        # 1. 預處理
        if self.verbose:
            print("\n1. 圖像預處理...")

        preprocessing_config = {
            'morphology': {
                'closing_kernel': 0,
                'opening_kernel': self.opening_kernel_size,
            },
            'mask': {
                'dilate_offset': self.offset_px,
            },
            'background': {
                'method': 'rolling_ball',
                'radius': self.rolling_ball_radius,
                'sato_weight': self.sato_weight,
                'sato_sigmas': (1.0, 2.0),
            },
            'threshold': {
                'use_full_roi': False,
            },
            'normalization': {
                'enabled': False,
            }
        }

        pipeline = SkinAnalysisPipeline(preprocessing_config)

        # 提取綠色通道
        if len(image.shape) == 3:
            orig_img = image[:, :, 1]  # 綠色通道
        else:
            orig_img = image

        if self.verbose:
            print("  - 提取綠色通道")
            print("  - 構建 ROI mask")
            print("  - 背景減除")
            print("  - 提取 ROI")
            print("  - 生成偽標註")
            print("  - 形態學處理")

        roi_annotation, roi_image = pipeline.run(annotation, mask, orig_img)

        if self.verbose:
            print("  ✓ 預處理完成")

        # CLAHE 均衡化用於種子點提取
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_img = clahe.apply(roi_image)

        # 2 + 3. 構建骨架圖並生成種子圖
        if self.verbose:
            print("\n2. 構建骨架圖...")
            print("\n3. 構建種子圖...")

        topology_builder = TopologyBuilder(
            segment_length=self.segment_length, verbose=self.verbose
        )
        seed_graph = topology_builder.build_seed_graph(roi_annotation, equalized_img)

        # 為種子圖的邊設置低成本
        for u, v, data in seed_graph.edges(data=True):
            data["weight"] = 1e-5

        # 4. 準備路徑查找
        if self.verbose:
            print("\n4. 準備路徑查找...")

        topology_points = np.array(list(seed_graph.nodes()))
        kdtree = KDTree(topology_points)

        # 構建成本地圖和種子地圖
        cost_map = ((255 - roi_image.astype(np.float64)) / 255.0) ** 1.5
        seed_map = np.zeros_like(cost_map, dtype=np.uint8)
        for p in topology_points:
            seed_map[p[0], p[1]] = True

        # 構建 label 圖（用於排除同一連通分量）
        binary = (roi_annotation > 0).astype(np.uint8)
        label_img = label(binary, connectivity=2)

        # 5. 路徑查找
        if self.verbose:
            print("\n5. 路徑查找...")

        path_finder = PathFinder(cost_map=cost_map)

        path_lookup = path_finder.find_paths_from_seeds(
            topology_points=topology_points,
            kdtree=kdtree,
            search_radius=self.search_radius_pathfinding,
            seed_map=seed_map,
            label_img=label_img,
            verbose=self.verbose,
        )

        # 6. 階段1：端點延伸
        if self.verbose:
            print("\n6. 階段1：端點延伸...")

        phase1_edges = extend_endpoints(
            graph=seed_graph,
            topology_points=topology_points,
            kdtree=kdtree,
            path_lookup=path_lookup,
            search_radius=self.search_radius_phase1,
            max_angle_degrees=self.max_angle_phase1,
            max_angle_penalty=self.angle_penalty_phase1,
            direction_threshold=self.direction_threshold_phase1,
            verbose=self.verbose,
        )

        # 添加階段1的邊
        extended_graph = seed_graph.copy()
        for endpoint, target, cost, path in phase1_edges:
            if not extended_graph.has_edge(endpoint, target):
                extended_graph.add_edge(
                    endpoint, target, weight=cost, path=path, phase=1
                )

        if self.verbose:
            endpoints_after_phase1 = sum(
                1 for n in extended_graph.nodes() if extended_graph.degree(n) == 1
            )
            print(f"  延伸後端點數: {endpoints_after_phase1}")

        # 7. 階段2：生成 MST 候選邊
        if self.verbose:
            print("\n7. 階段2：生成 MST 候選邊...")

        phase2_candidates = generate_mst_candidates(
            graph=extended_graph,
            topology_points=topology_points,
            kdtree=kdtree,
            path_lookup=path_lookup,
            search_radius=self.search_radius_phase2,
            max_angle_degrees=self.max_angle_phase2,
            angle_penalty_weight=self.angle_penalty_phase2,
            distance_weight=self.distance_weight_phase2,
            max_cost_threshold=self.max_cost_threshold_phase2,
            verbose=self.verbose,
        )

        # 8. 構建候選圖並執行 MST
        if self.verbose:
            print("\n8. 執行 MST 優化...")

        # 複製圖並對階段1的邊打折
        mst_tree = extended_graph.copy()
        for u, v, data in mst_tree.edges(data=True):
            data["weight"] = data["weight"] * self.phase1_weight_discount
            data["phase"] = 1

        # 添加階段2候選邊
        for endpoint, target, cost, path in phase2_candidates:
            if not mst_tree.has_edge(endpoint, target):
                mst_tree.add_edge(endpoint, target, weight=cost, path=path, phase=2)

        # 執行 MST
        mst_result = nx.minimum_spanning_tree(mst_tree, weight="weight")

        if self.verbose:
            phase1_in_mst = sum(
                1 for u, v in mst_result.edges() if mst_result[u][v].get("phase") == 1
            )
            phase2_in_mst = sum(
                1 for u, v in mst_result.edges() if mst_result[u][v].get("phase") == 2
            )
            final_endpoints = sum(
                1 for n in mst_result.nodes() if mst_result.degree(n) == 1
            )

            print(f"\n✓ MST 結果:")
            print(f"  - 總邊數: {mst_result.number_of_edges()}")
            print(f"  - 來自階段1: {phase1_in_mst}")
            print(f"  - 來自階段2: {phase2_in_mst}")
            print(f"  - 最終端點數: {final_endpoints}")
            print(f"  - 連通分量數: {nx.number_connected_components(mst_result)}")

        return mst_result
