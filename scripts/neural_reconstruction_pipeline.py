#!/usr/bin/env python3
"""
神經纖維重建完整流程腳本
整合預處理 (preprocess.ipynb) 和 MST 重建 (mst_pipeline.ipynb)

不依賴 neural_reconstruction 模塊，使用 hierarchical_fragment_linking.ipynb 中的代碼

使用方式:
    python neural_reconstruction_pipeline.py --image_id S1585-2_a --data_dir ../data

或作為模塊導入:
    from neural_reconstruction_pipeline import NeuralReconstructionPipeline
    pipeline = NeuralReconstructionPipeline()
    result = pipeline.run(image_path, mask_path, annotation_path)
"""

import numpy as np
import cv2
import skimage as ski
from skimage.filters import sato
from skimage.measure import label, regionprops
from skimage import morphology
from scipy.spatial import KDTree
import networkx as nx
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass

# Skan 用於骨架分析
from skan import Skeleton, summarize
from skan.csr import skeleton_to_nx


@dataclass
class PreprocessingConfig:
    """預處理配置參數"""
    # Dermis 遮罩參數
    dermis_offset_px: int = 100

    # 背景減除參數
    rolling_ball_radius: int = 2
    sato_weight: float = 0.0  # 0 表示不使用 Sato 濾波
    sato_sigmas: Tuple[int, int] = (1, 3)

    # 分割參數
    chan_vese_mu: float = 0.0
    chan_vese_lambda1: float = 10.0
    chan_vese_lambda2: float = 1.0
    chan_vese_tol: float = 1e-4
    chan_vese_max_iter: int = 100
    chan_vese_dt: float = 0.5

    # 形態學操作參數
    morphology_kernel_size: int = 3


@dataclass
class ReconstructionConfig:
    """重建配置參數"""
    # 種子提取參數
    segment_length: float = 3.0  # 種子間距

    # 路徑查找參數
    search_radius: float = 50.0
    path_finding_bbox_padding: int = 10


@dataclass
class ReconstructionResult:
    """重建結果"""
    # 預處理結果
    roi_image: np.ndarray  # 處理後的 ROI 圖像
    roi_annotation: np.ndarray  # 處理後的標註
    roi_mask: np.ndarray  # ROI 遮罩

    # 重建結果
    mst_trees: List[nx.Graph]  # MST 森林
    seed_graph: nx.MultiGraph  # 種子圖
    topology_points: np.ndarray  # 所有種子點
    pairings: Dict  # 所有配對及其路徑

    # 統計信息
    num_mst_trees: int
    num_edges: int
    num_seeds: int


class NeuralReconstructionPipeline:
    """神經纖維重建完整流程"""

    def __init__(
        self,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        reconstruction_config: Optional[ReconstructionConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        初始化流程

        Args:
            preprocessing_config: 預處理配置，None 使用默認值
            reconstruction_config: 重建配置，None 使用默認值
            progress_callback: 進度回調函數 (message: str, progress: float [0-1])
        """
        self.preproc_config = preprocessing_config or PreprocessingConfig()
        self.recon_config = reconstruction_config or ReconstructionConfig()
        self.progress_callback = progress_callback or (lambda msg, prog: None)

    def run(
        self,
        image_path: str,
        mask_path: str,
        annotation_path: str,
        output_dir: Optional[str] = None
    ) -> ReconstructionResult:
        """
        執行完整的重建流程

        Args:
            image_path: 原始圖像路徑
            mask_path: 遮罩圖像路徑
            annotation_path: 標註圖像路徑
            output_dir: 可選的輸出目錄（保存中間結果）

        Returns:
            ReconstructionResult: 重建結果
        """
        self.progress_callback("正在加載圖像...", 0.0)

        # 1. 加載圖像
        orig_img = cv2.imread(image_path)
        if len(orig_img.shape) == 3:
            orig_img = orig_img[:, :, 1]  # 取綠色通道
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        annotation_img = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

        self.progress_callback("執行預處理...", 0.1)

        # 2. 預處理
        roi_image, roi_annotation, roi_mask = self._preprocess(
            orig_img, mask_img, annotation_img
        )

        # 保存預處理結果（可選）
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path / "roi_image.png"), roi_image)
            cv2.imwrite(str(output_path / "roi_annotation.png"), roi_annotation * 255)

        self.progress_callback("提取骨架與種子...", 0.3)

        # 3. 骨架與種子提取
        seed_graph = self._extract_skeleton_and_seeds(roi_image, roi_annotation)

        self.progress_callback("計算種子對連接...", 0.5)

        # 4. A* 路徑查找
        topology_points, pairings = self._find_connections(roi_image, seed_graph)

        self.progress_callback("構建 MST...", 0.8)

        # 5. 構建 MST
        mst_trees = self._build_mst(pairings)

        self.progress_callback("完成！", 1.0)

        # 6. 返回結果
        num_edges = sum(tree.number_of_edges() for tree in mst_trees)

        return ReconstructionResult(
            roi_image=roi_image,
            roi_annotation=roi_annotation,
            roi_mask=roi_mask,
            mst_trees=mst_trees,
            seed_graph=seed_graph,
            topology_points=topology_points,
            pairings=pairings,
            num_mst_trees=len(mst_trees),
            num_edges=num_edges,
            num_seeds=len(topology_points)
        )

    def _preprocess(
        self,
        orig_img: np.ndarray,
        mask_img: np.ndarray,
        annotation_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        預處理流程（來自 preprocess.ipynb）

        Returns:
            (roi_image, roi_annotation, roi_mask)
        """
        # 1. 創建 dermis 遮罩
        mask_uint8 = mask_img.astype(np.uint8) * 255
        offset_px = self.preproc_config.dermis_offset_px

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, offset_px * 6))
        anchor = (0, offset_px * 6 - 1)
        dilated = cv2.dilate(mask_uint8, kernel, anchor=anchor, iterations=1)

        dermis_available_mask = dilated - mask_uint8
        dermis_mask = cv2.bitwise_and(
            dermis_available_mask,
            cv2.dilate(
                mask_uint8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset_px, offset_px)),
                iterations=1
            )
        )
        roi_mask = cv2.bitwise_or(dermis_mask.astype(np.uint8), mask_uint8.astype(np.uint8))

        # 2. Rolling ball 背景減除
        bg = ski.restoration.rolling_ball(
            orig_img,
            radius=self.preproc_config.rolling_ball_radius
        )
        rolling_fg = (orig_img.astype(np.uint8) - bg.astype(np.uint8)).astype(np.uint8)

        # 3. 可選的 Sato 濾波
        if self.preproc_config.sato_weight > 0:
            sato_result = sato(
                rolling_fg.astype(np.float32),
                sigmas=range(*self.preproc_config.sato_sigmas),
                black_ridges=False,
                mode='reflect'
            )
            sato_normalized = (sato_result - sato_result.min()) / (sato_result.max() - sato_result.min())
            fg_img = ((1 - self.preproc_config.sato_weight) * rolling_fg +
                     self.preproc_config.sato_weight * sato_normalized)
            fg_img = fg_img.clip(0, 255).astype(np.uint8)
        else:
            fg_img = rolling_fg

        # 4. ROI 提取
        roi_image = cv2.bitwise_and(fg_img, fg_img, mask=roi_mask)

        # 5. Multi-Otsu 閾值 + Chan-Vese 分割
        thresholds = ski.filters.threshold_multiotsu(roi_image)
        regions = np.digitize(roi_image, bins=thresholds)
        toplevel_mask = (regions == len(thresholds)).astype(np.uint8)

        cv_result = ski.segmentation.chan_vese(
            roi_image.astype(np.float32),
            mu=self.preproc_config.chan_vese_mu,
            lambda1=self.preproc_config.chan_vese_lambda1,
            lambda2=self.preproc_config.chan_vese_lambda2,
            tol=self.preproc_config.chan_vese_tol,
            max_num_iter=self.preproc_config.chan_vese_max_iter,
            dt=self.preproc_config.chan_vese_dt,
            init_level_set=toplevel_mask.astype(np.float32),
            extended_output=True,
        )

        chanvese_mask = (cv_result[0] > 0).astype(np.uint8)
        dermis_annotation = cv2.bitwise_and(chanvese_mask, chanvese_mask, mask=dermis_mask)

        # 6. 合併標註
        roi_annotation = cv2.bitwise_or(
            dermis_annotation,
            (annotation_img > 0).astype(np.uint8)
        )

        # 7. 形態學 opening
        kernel_size = self.preproc_config.morphology_kernel_size
        roi_annotation = cv2.morphologyEx(
            roi_annotation.astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1
        )

        return roi_image, roi_annotation, roi_mask

    def _extract_skeleton_and_seeds(
        self,
        roi_image: np.ndarray,
        roi_annotation: np.ndarray
    ) -> nx.MultiGraph:
        """
        骨架與種子提取（來自 hierarchical_fragment_linking.ipynb cells 5-6）

        Returns:
            seed_graph: 包含種子點和路徑的圖
        """
        # 1. 骨架化
        binary = (roi_annotation > 0).astype(np.uint8)
        skeleton = morphology.skeletonize(binary).astype(np.uint8)
        skel_obj = Skeleton(skeleton, keep_images=False)
        summary = summarize(skel_obj)
        skeleton_graph = skeleton_to_nx(skel_obj, summary=summary)

        # 2. 過濾短邊
        filtered_skeleton_graph = nx.MultiGraph()
        for u, v, data in skeleton_graph.edges(data=True):
            path = data['path']
            if len(path) > 2 or (skeleton_graph.degree(u) != 1 and skeleton_graph.degree(v) != 1):
                filtered_skeleton_graph.add_edge(u, v, **data)

        # 將節點 id 轉回座標
        mapping = {i: tuple(skel_obj.coordinates[i].astype(int)) for i in filtered_skeleton_graph.nodes()}
        filtered_skeleton_graph = nx.relabel_nodes(filtered_skeleton_graph, mapping)

        # 3. 合併中間點
        middle_points = [
            point for point in filtered_skeleton_graph.nodes()
            if len(list(filtered_skeleton_graph.neighbors(point))) == 2
        ]

        for mp in middle_points:
            neighbors = list(filtered_skeleton_graph.neighbors(mp))
            u, v = neighbors
            path1 = filtered_skeleton_graph[u][mp][0]['path']
            path2 = filtered_skeleton_graph[mp][v][0]['path']
            u_y, u_x = u
            mp_y, mp_x = mp
            v_y, v_x = v

            result_path = []
            if tuple(path1[-1]) == (mp_y, mp_x) and tuple(path1[0]) == (u_y, u_x):
                result_path.extend(path1)
            else:
                result_path.extend(path1[::-1])

            if tuple(path2[0]) == (mp_y, mp_x) and tuple(path2[-1]) == (v_y, v_x):
                result_path.extend(path2[1:])
            else:
                result_path.extend(path2[-2::-1])

            filtered_skeleton_graph.remove_node(mp)
            filtered_skeleton_graph.add_edge(u, v, path=result_path)

        # 4. 補齊缺失的節點
        label_img = label(binary, connectivity=2)
        regions = regionprops(label_img)

        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            bbox_nodes = [
                node for node in filtered_skeleton_graph.nodes()
                if node[0] >= min_row and node[0] < max_row and node[1] >= min_col and node[1] < max_col
            ]
            if len(bbox_nodes) != 0:
                continue

            brightest_pixel = None
            brightest_value = -1

            for r in range(min_row, max_row):
                for c in range(min_col, max_col):
                    if roi_image[r, c] > brightest_value:
                        brightest_value = roi_image[r, c]
                        brightest_pixel = (r, c)

            filtered_skeleton_graph.add_node(brightest_pixel)

        # 5. 生成種子圖
        seed_graph = nx.MultiGraph()
        segment_length = self.recon_config.segment_length

        for u in filtered_skeleton_graph.nodes():
            seed_graph.add_node(u)

        for u, v, data in filtered_skeleton_graph.edges(data=True):
            path = data['path']
            # 擺正方向
            corrected_path = path[:] if tuple(path[0]) == u and tuple(path[-1]) == v else path[::-1]
            path_arr = np.array(corrected_path)
            diffs = np.diff(path_arr, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
            path_length = cumulative_distances[-1]
            num_segments = int(path_length // segment_length)

            if num_segments <= 0:
                seed_graph.add_edge(u, v, path=path)
                continue

            last_index = 0
            for i in range(num_segments):
                target_distance = (i + 1) * path_length / num_segments
                segment_end_index = 0
                for idx, cumulative_distance in enumerate(cumulative_distances[last_index:]):
                    if cumulative_distance >= target_distance:
                        segment_end_index = idx + last_index
                        break
                segment_path = corrected_path[last_index:segment_end_index + 1]
                if len(segment_path) == 0:
                    continue
                seed_graph.add_edge(tuple(segment_path[0]), tuple(segment_path[-1]), path=segment_path)
                last_index = segment_end_index

            if last_index < len(corrected_path) - 1:
                final_segment_path = corrected_path[last_index:]
                seed_graph.add_edge(tuple(final_segment_path[0]), tuple(final_segment_path[-1]), path=final_segment_path)

        return seed_graph

    def _find_connections(
        self,
        roi_image: np.ndarray,
        seed_graph: nx.MultiGraph
    ) -> Tuple[np.ndarray, Dict]:
        """
        A* 路徑查找（來自 hierarchical_fragment_linking.ipynb cell 9）

        Returns:
            (topology_points, pairings)
        """
        # 建立 KDTree
        topology_points = np.array(list(seed_graph.nodes()))
        kdtree = KDTree(topology_points)

        # 建立成本地圖
        cost_map = ((255 - roi_image.astype(np.float64)) / 255.0) ** 1.5
        cost_map_h, cost_map_w = cost_map.shape

        # 建立種子地圖（用於過濾）
        seed_map = np.zeros_like(cost_map, dtype=np.uint8)
        for p in topology_points:
            seed_map[p[0], p[1]] = True

        # 為 seed graph 中的邊添加低成本
        path_lookup = {}
        for u, v, data in seed_graph.edges(data=True):
            path = data['path']
            path_lookup[(u, v)] = (path, 1e-5)

        # A* 搜索
        search_radius = self.recon_config.search_radius
        bbox_padding = self.recon_config.path_finding_bbox_padding

        # 獲取 annotation 的連通分量（用於排除同一 component 的點）
        binary = (roi_image > 0).astype(np.uint8)
        label_img = label(binary, connectivity=2)

        total = len(topology_points)
        for u_idx in range(total):
            # 進度更新
            if u_idx % 50 == 0:
                progress = 0.5 + 0.3 * (u_idx / total)
                self.progress_callback(f"路徑查找: {u_idx}/{total}", progress)

            u = topology_points[u_idx]

            # KDTree 查詢
            neighbor_indices = kdtree.query_ball_point(u, r=search_radius)
            targets = [topology_points[v_idx] for v_idx in neighbor_indices]

            # 排除自己
            targets = [target for target in targets if tuple(target) != tuple(u)]

            # 排除屬於同一個 component 的點
            current_component_id = label_img[u[0], u[1]]
            targets = [
                target for target in targets
                if label_img[target[0], target[1]] != current_component_id
            ]

            # 排除已存在的配對
            targets = [
                target for target in targets
                if (tuple(u), tuple(target)) not in path_lookup
                and (tuple(target), tuple(u)) not in path_lookup
            ]

            if len(targets) == 0:
                continue

            # 計算 bbox
            all_points = [u] + targets
            all_y = [p[0] for p in all_points]
            all_x = [p[1] for p in all_points]

            min_y = max(0, min(all_y) - bbox_padding)
            max_y = min(cost_map_h - 1, max(all_y) + bbox_padding)
            min_x = max(0, min(all_x) - bbox_padding)
            max_x = min(cost_map_w - 1, max(all_x) + bbox_padding)

            cropped_cost_map = cost_map[min_y:max_y + 1, min_x:max_x + 1]

            # A* 路徑查找
            local_points = [
                (pos_global[0] - min_y, pos_global[1] - min_x)
                for pos_global in all_points
            ]

            mcp = ski.graph.MCP_Geometric(cropped_cost_map, fully_connected=True)
            cumulative_costs, traceback = mcp.find_costs(
                starts=local_points[:1],
                ends=local_points[1:]
            )

            for target in local_points[1:]:
                if np.isinf(cumulative_costs[target]):
                    continue

                path = mcp.traceback(target)
                cost = cumulative_costs[target]

                # 轉換回全局座標
                global_path = [(p[0] + min_y, p[1] + min_x) for p in path]
                global_start = (u[0], u[1])
                global_target = (target[0] + min_y, target[1] + min_x)

                normalized_cost = cost / 1

                # 過濾路徑中經過其他種子的連接
                middle_points = np.array(path[1:-1])
                if len(middle_points) > 0:
                    middle_points_global = middle_points + np.array([min_y, min_x])
                    if np.any(seed_map[middle_points_global[:, 0], middle_points_global[:, 1]]):
                        continue

                # 更新 path_lookup
                if (global_target, global_start) in path_lookup:
                    min_cost = min(path_lookup[(global_target, global_start)][1], normalized_cost)
                    path_lookup[(global_target, global_start)] = (global_path, min_cost)
                else:
                    path_lookup[(global_start, global_target)] = (global_path, normalized_cost)

        return topology_points, path_lookup

    def _build_mst(self, pairings: Dict) -> List[nx.Graph]:
        """
        構建 MST（來自 mst_pipeline.ipynb cells 7-8）

        Returns:
            List of MST trees
        """
        # 構建圖
        graph = nx.Graph()
        for (p1, p2), (path, cost) in pairings.items():
            graph.add_edge(p1, p2, weight=cost, path=path)

        # 找連通分量
        sub_trees = list(nx.connected_components(graph))

        # 計算每個連通分量的 MST
        mst_trees = []
        for component in sub_trees:
            subgraph = graph.subgraph(component)
            mst = nx.minimum_spanning_tree(subgraph, weight="weight")
            mst_trees.append(mst)

        return mst_trees


def visualize_result(
    result: ReconstructionResult,
    roi_x: int = 0,
    roi_y: int = 0,
    roi_w: Optional[int] = None,
    roi_h: Optional[int] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    可視化重建結果

    Args:
        result: 重建結果
        roi_x, roi_y, roi_w, roi_h: ROI 區域（用於裁剪顯示）
        save_path: 保存路徑（可選）

    Returns:
        可視化圖像 (BGR)
    """
    if roi_w is None:
        roi_w = result.roi_image.shape[1]
    if roi_h is None:
        roi_h = result.roi_image.shape[0]

    # 創建彩色圖像
    viz_img = cv2.cvtColor(result.roi_image.copy(), cv2.COLOR_GRAY2BGR)

    # 繪製 MST
    for mst_tree in result.mst_trees:
        for u, v, data in mst_tree.edges(data=True):
            if "path" in data and data["path"]:
                path_points = data["path"]
                path_array = np.array(path_points)[:, [1, 0]]  # (y, x) -> (x, y)
                cv2.polylines(viz_img, [path_array.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

    # 裁剪 ROI
    viz_roi = viz_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    if save_path:
        cv2.imwrite(save_path, viz_roi)

    return viz_roi


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='神經纖維重建流程')
    parser.add_argument('--image_id', type=str, required=True, help='圖像 ID')
    parser.add_argument('--data_dir', type=str, default='../data', help='數據目錄')
    parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    parser.add_argument('--search_radius', type=float, default=50.0, help='搜索半徑')
    parser.add_argument('--segment_length', type=float, default=3.0, help='種子間距')

    args = parser.parse_args()

    # 構建路徑
    base_path = Path(args.data_dir) / args.image_id
    image_path = str(base_path / "image.png")
    mask_path = str(base_path / "mask.png")
    annotation_path = str(base_path / "annotation.png")

    # 配置
    recon_config = ReconstructionConfig(
        search_radius=args.search_radius,
        segment_length=args.segment_length
    )

    # 進度回調
    def progress_callback(msg: str, progress: float):
        print(f"[{progress*100:.1f}%] {msg}")

    # 運行流程
    pipeline = NeuralReconstructionPipeline(
        reconstruction_config=recon_config,
        progress_callback=progress_callback
    )

    result = pipeline.run(
        image_path, mask_path, annotation_path,
        output_dir=args.output_dir
    )

    # 打印統計
    print("\n=== 重建結果 ===")
    print(f"MST 樹數: {result.num_mst_trees}")
    print(f"總邊數: {result.num_edges}")
    print(f"總種子數: {result.num_seeds}")

    # 保存可視化
    if args.output_dir:
        viz_path = str(Path(args.output_dir) / f"{args.image_id}_mst.png")
        visualize_result(result, save_path=viz_path)
        print(f"\n可視化已保存: {viz_path}")


if __name__ == "__main__":
    main()
