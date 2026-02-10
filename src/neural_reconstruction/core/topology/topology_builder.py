"""
拓樸建構模組 (Topology Builder)

統一的骨架圖建構器，整合兩個管線的功能：
- fragment_linking 的完整骨架圖建構流程（作為主要邏輯）
- component_analyzer 的空骨架保護與 branch-distance 屬性

建構流程：
1. 空骨架 / 錯誤保護
2. 二值化和骨架化（Zhang-Suen）
3. skan 建圖
4. 過濾短邊
5. 節點重標籤為 (y, x) 座標
6. 合併 degree-2 中間點
7. 補齊缺失的孤立節點（equalized_img=None 時降級為質心）
8. 計算 branch-distance 邊屬性
"""

import logging
from typing import Optional

import numpy as np
import networkx as nx
from skimage import morphology
from skimage.measure import label, regionprops
from skan import Skeleton, summarize
from skan.csr import skeleton_to_nx

logger = logging.getLogger(__name__)


class TopologyBuilder:
    """
    統一的骨架圖建構器

    可用於完整 annotation 圖像（fragment_linking 管線）或
    單個元件 mask（component_analyzer 管線）。

    提供兩層 API：
    - build_skeleton_graph(): 只建構骨架圖
    - build_seed_graph(): 骨架圖 + 種子圖，一次完成

    Examples:
        >>> builder = TopologyBuilder(segment_length=3.0)

        >>> # 僅需骨架圖
        >>> skeleton = builder.build_skeleton_graph(annotation, equalized_img)

        >>> # 骨架圖 + 種子圖（一步完成）
        >>> seed_graph = builder.build_seed_graph(annotation, equalized_img)

        >>> # 不提供 equalized_img 時，缺失節點改用質心
        >>> skeleton = builder.build_skeleton_graph(component_mask)
    """

    def __init__(self, segment_length: float = 3.0, verbose: bool = False):
        """
        Args:
            segment_length: 種子圖的分段長度（像素），供 build_seed_graph 使用
            verbose: 是否輸出建構過程資訊
        """
        self.segment_length = segment_length
        self.verbose = verbose

    def build_skeleton_graph(
        self,
        annotation: np.ndarray,
        equalized_img: Optional[np.ndarray] = None,
    ) -> nx.MultiGraph:
        """
        從標注構建骨架圖

        Args:
            annotation: 二值標注圖像
            equalized_img: 均衡化後的原始圖像（用於補齊缺失節點找最亮像素）；
                          None 時改用區域質心作為降級方案

        Returns:
            NetworkX MultiGraph，節點為 (y, x) 座標，邊包含 path 與 branch-distance 屬性
        """
        # 1. 二值化
        binary = (annotation > 0).astype(np.uint8)

        # 2. 骨架化前保護：空或過小的骨架
        skeleton_pixels = np.sum(binary)
        if skeleton_pixels < 2:
            logger.debug(f"二值圖太小（{skeleton_pixels} 像素），返回空圖")
            return nx.MultiGraph()

        # 3. 骨架化（Zhang-Suen）
        skeleton = morphology.skeletonize(binary).astype(np.uint8)

        if np.sum(skeleton) < 2:
            logger.debug("骨架化後像素太少，返回空圖")
            return nx.MultiGraph()

        # 4. 使用 skan 建圖
        try:
            skel_obj = Skeleton(skeleton, keep_images=False)
        except (ValueError, IndexError) as e:
            logger.debug(f"Skeleton 建立失敗: {e}，返回空圖")
            return nx.MultiGraph()

        summary = summarize(skel_obj)
        skeleton_graph = skeleton_to_nx(skel_obj, summary=summary)

        # 5. 過濾短邊（保留長邊或端點邊）
        filtered_graph = nx.MultiGraph()
        for u, v, data in skeleton_graph.edges(data=True):
            path = data.get("path", [])
            if len(path) > 2 or (
                skeleton_graph.degree(u) != 1 and skeleton_graph.degree(v) != 1
            ):
                filtered_graph.add_edge(u, v, **data)

        # 6. 節點重標籤為 (y, x) 座標
        mapping = {
            i: tuple(skel_obj.coordinates[i].astype(int))
            for i in filtered_graph.nodes()
        }
        filtered_graph = nx.relabel_nodes(filtered_graph, mapping)

        # 7. 合併 degree-2 中間點
        filtered_graph = self._merge_middle_points(filtered_graph)

        # 8. 補齊缺失節點
        filtered_graph = self._fill_missing_nodes(filtered_graph, binary, equalized_img)

        # 9. 計算並加入 branch-distance 屬性
        filtered_graph = self._compute_branch_distances(filtered_graph)

        if self.verbose:
            print(
                f"✓ 骨架圖構建完成: {filtered_graph.number_of_nodes()} 節點, "
                f"{filtered_graph.number_of_edges()} 邊"
            )

        return filtered_graph

    def build_seed_graph(
        self,
        annotation: np.ndarray,
        equalized_img: Optional[np.ndarray] = None,
    ) -> nx.MultiGraph:
        """
        從標注一步建構種子圖（骨架圖 → 種子圖）

        組合 build_skeleton_graph() 與 SeedGraphBuilder，
        適合只需要種子圖而不需要中間骨架圖的使用場景。

        Args:
            annotation: 二值標注圖像
            equalized_img: 均衡化後的原始圖像；None 時缺失節點改用質心

        Returns:
            種子圖 MultiGraph，節點為 (y, x) 座標，邊按 segment_length 切分
        """
        from .seed_graph import SeedGraphBuilder

        skeleton_graph = self.build_skeleton_graph(annotation, equalized_img)
        seed_builder = SeedGraphBuilder(
            segment_length=self.segment_length, verbose=self.verbose
        )
        return seed_builder.build(skeleton_graph)

    # ========== 私有輔助方法 ==========

    def _merge_middle_points(self, graph: nx.MultiGraph) -> nx.MultiGraph:
        """合併 degree-2 中間點，將兩條邊合為一條"""
        middle_points = [
            point for point in graph.nodes() if len(list(graph.neighbors(point))) == 2
        ]

        for mp in middle_points:
            neighbors = list(graph.neighbors(mp))
            if len(neighbors) != 2:
                continue
            u, v = neighbors

            path1 = graph[u][mp][0]["path"]
            path2 = graph[mp][v][0]["path"]

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

            graph.remove_node(mp)
            graph.add_edge(u, v, path=result_path)

        return graph

    def _fill_missing_nodes(
        self,
        graph: nx.MultiGraph,
        binary: np.ndarray,
        equalized_img: Optional[np.ndarray] = None,
    ) -> nx.MultiGraph:
        """補齊缺失的節點（孤立的連通分量）

        Args:
            graph: 已建構的骨架圖
            binary: 二值化後的圖像
            equalized_img: 均衡化圖像（找最亮像素）；None 時改用質心
        """
        label_img = label(binary, connectivity=2)
        regions = regionprops(label_img)

        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox

            bbox_nodes = [
                node
                for node in graph.nodes()
                if min_row <= node[0] < max_row and min_col <= node[1] < max_col
            ]

            if len(bbox_nodes) != 0:
                continue

            if equalized_img is not None:
                # 找區域中最亮的像素
                brightest_pixel = None
                brightest_value = -1
                for r in range(min_row, max_row):
                    for c in range(min_col, max_col):
                        if equalized_img[r, c] > brightest_value:
                            brightest_value = equalized_img[r, c]
                            brightest_pixel = (r, c)
            else:
                # 降級：使用區域質心
                brightest_pixel = (
                    (min_row + max_row) // 2,
                    (min_col + max_col) // 2,
                )

            if brightest_pixel is not None:
                graph.add_node(brightest_pixel)

        return graph

    def _compute_branch_distances(self, graph: nx.MultiGraph) -> nx.MultiGraph:
        """從 path 屬性計算並加入 branch-distance 邊屬性"""
        for u, v, key, data in graph.edges(keys=True, data=True):
            path = data.get("path", [])
            if len(path) >= 2:
                pts = np.array(path)
                dist = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            else:
                dist = 0.0
            graph[u][v][key]["branch-distance"] = dist
        return graph
