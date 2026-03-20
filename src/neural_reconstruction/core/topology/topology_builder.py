"""
拓樸建構模組 (Topology Builder)

統一的骨架圖與種子圖建構器，整合兩個管線的功能：
- fragment_linking 的完整骨架圖建構流程（作為主要邏輯）
- component_analyzer 的空骨架保護與 branch-distance 屬性

建構流程：
1. 空骨架 / 錯誤保護
2. 二值化
3. Zhang-Suen 骨架化
4. skan 建圖 → 節點重標籤為 (y, x) 座標
5. MultiGraph → Graph（自環跳過；平行邊：短邊過濾 + 中點拆分）
6. 合併 degree-2 中間點（消除骨架線段上的冗餘中繼節點）
7. 補齊缺失的孤立節點（equalized_img=None 時降級為質心）
8. 計算 branch-distance 邊屬性（沿 path 的累積歐式距離）

主要類別：
    TopologyBuilder: 骨架圖 / 種子圖建構入口

典型用法::

    builder = TopologyBuilder(segment_length=3.0)
    skeleton = builder.build_skeleton_graph(annotation, equalized_img)
    seed_graph = builder.build_seed_graph(annotation, equalized_img)
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
    """統一的骨架圖建構器。

    可用於完整 annotation 圖像（fragment_linking 管線）或
    單個元件 mask（component_analyzer 管線）。

    提供兩層 API：

    - :meth:`build_skeleton_graph`: 從二值標注建構骨架圖（節點為交叉/端點，
      邊含完整像素路徑與 branch-distance）。
    - :meth:`build_seed_graph`: 在骨架圖基礎上按 ``segment_length`` 切分，
      一步取得種子圖。

    Attributes:
        segment_length (float): 種子圖分段長度（像素）。

    Examples:
        >>> builder = TopologyBuilder(segment_length=3.0)

        >>> # 僅需骨架圖
        >>> skeleton = builder.build_skeleton_graph(annotation, equalized_img)

        >>> # 骨架圖 + 種子圖（一步完成）
        >>> seed_graph = builder.build_seed_graph(annotation, equalized_img)

        >>> # 不提供 equalized_img 時，缺失的孤立節點改用區域質心
        >>> skeleton = builder.build_skeleton_graph(component_mask)
    """

    def __init__(self, segment_length: float = 3.0):
        """初始化 TopologyBuilder。

        Args:
            segment_length (float): 種子圖的分段長度（像素），
                供 :meth:`build_seed_graph` 使用。預設值為 ``3.0``。
        """
        self.segment_length = segment_length

    def build_skeleton_graph(
        self,
        annotation: np.ndarray,
        equalized_img: Optional[np.ndarray] = None,
    ) -> nx.Graph:
        """從二值標注建構骨架圖。

        執行流程：二值化 → Zhang-Suen 骨架化 → skan 建圖 → 節點重標籤
        → MultiGraph 轉 Graph → degree-2 中間點合併 → 孤立節點補齊
        → branch-distance 計算。

        Args:
            annotation (np.ndarray): 二值標注圖像，shape ``(H, W)``，
                非零像素視為神經纖維區域。
            equalized_img (np.ndarray, optional): 均衡化後的灰階圖像，shape ``(H, W)``，
                用於在孤立連通分量中定位最亮像素作為補齊節點。
                傳入 ``None`` 時降級為使用區域 bounding-box 質心。

        Returns:
            nx.Graph: 骨架圖，節點為 ``(y, x)`` 整數座標元組；
            邊資料包含：

            - ``path`` (list[tuple[int, int]]): 沿骨架的像素座標序列。
            - ``branch-distance`` (float): 沿 path 的累積歐式距離（像素）。

            若輸入圖像過小或骨架化失敗，返回空的 ``nx.Graph()``。
        """
        # 1. 空骨架 / 錯誤保護
        binary = (annotation > 0).astype(np.uint8)
        if np.sum(binary) < 2:
            logger.debug(f"二值圖太小（{np.sum(binary)} 像素），返回空圖")
            return nx.Graph()

        # 2 & 3. 二值化 → Zhang-Suen 骨架化
        skeleton = morphology.skeletonize(binary).astype(np.uint8)
        if np.sum(skeleton) < 2:
            logger.debug("骨架化後像素太少，返回空圖")
            return nx.Graph()

        # 4. skan 建圖 → 節點重標籤為 (y, x) 座標
        try:
            skel_obj = Skeleton(skeleton, keep_images=False)
        except (ValueError, IndexError) as e:
            logger.debug(f"Skeleton 建立失敗: {e}，返回空圖")
            return nx.Graph()

        summary = summarize(skel_obj)
        skeleton_graph = skeleton_to_nx(skel_obj, summary=summary)
        mapping = {
            i: tuple(skel_obj.coordinates[i].astype(int))
            for i in skeleton_graph.nodes()
        }
        coordinate_graph = nx.relabel_nodes(skeleton_graph, mapping)

        # 5. MultiGraph → Graph（自環跳過；平行邊：短邊過濾 + 中點拆分）
        graph = self.to_simple_graph(coordinate_graph)

        # 6. 合併 degree-2 中間點
        graph = self._merge_middle_points(graph)

        # 7. 補齊缺失節點
        graph = self._fill_missing_nodes(graph, binary, equalized_img)

        # 8. 計算並加入 branch-distance 屬性
        graph = self._compute_branch_distances(graph)

        logger.debug(
            f"骨架圖構建完成: {graph.number_of_nodes()} 節點, "
            f"{graph.number_of_edges()} 邊"
        )

        return graph

    def build_seed_graph(
        self,
        annotation: np.ndarray,
        equalized_img: Optional[np.ndarray] = None,
    ) -> nx.Graph:
        """從二值標注一步建構種子圖（骨架圖 → 種子圖）。

        內部先呼叫 :meth:`build_skeleton_graph` 取得骨架圖，再按
        :attr:`segment_length` 均勻切分每條邊，在骨架路徑上插入種子節點。

        適合只需要種子圖、不需要保留中間骨架圖的使用場景。

        Args:
            annotation (np.ndarray): 二值標注圖像，shape ``(H, W)``。
            equalized_img (np.ndarray, optional): 均衡化灰階圖像；
                傳入 ``None`` 時孤立節點以質心補齊。

        Returns:
            nx.Graph: 種子圖，節點為 ``(y, x)`` 座標元組，
            邊按 :attr:`segment_length` 切分，各邊含 ``path`` 屬性。
        """
        skeleton_graph = self.build_skeleton_graph(annotation, equalized_img)
        seed_graph = nx.Graph()
        seed_graph.add_nodes_from(skeleton_graph.nodes())

        for u, v, data in skeleton_graph.edges(data=True):
            path = data["path"]

            # 擺正方向：確保 path[0] == u、path[-1] == v
            corrected_path = (
                path[:] if tuple(path[0]) == u and tuple(path[-1]) == v else path[::-1]
            )

            # 計算路徑累積距離
            path_arr = np.array(corrected_path)
            cumulative_distances = np.concatenate(
                ([0], np.cumsum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1)))
            )
            path_length = cumulative_distances[-1]

            num_segments = int(path_length // self.segment_length)

            if num_segments <= 0:
                seed_graph.add_edge(u, v, path=path)
                continue

            # 按距離切分
            last_index = 0
            for i in range(num_segments):
                target_distance = (i + 1) * path_length / num_segments
                segment_end_index = 0
                for idx, cum_dist in enumerate(cumulative_distances[last_index:]):
                    if cum_dist >= target_distance:
                        segment_end_index = idx + last_index
                        break

                segment_path = corrected_path[last_index : segment_end_index + 1]

                if len(segment_path) < 2:
                    continue

                start_node = tuple(segment_path[0])
                end_node = tuple(segment_path[-1])

                if start_node == end_node:
                    continue
                seed_graph.add_edge(start_node, end_node, path=segment_path)
                last_index = segment_end_index

            # 最後一段
            if last_index < len(corrected_path) - 1:
                final_path = corrected_path[last_index:]
                start_node = tuple(final_path[0])
                end_node = tuple(final_path[-1])
                if start_node != end_node:
                    seed_graph.add_edge(start_node, end_node, path=final_path)

        logger.debug(
            f"種子圖構建完成: {seed_graph.number_of_nodes()} 節點, "
            f"{seed_graph.number_of_edges()} 邊"
        )

        return seed_graph

    def to_simple_graph(self, multigraph: nx.MultiGraph) -> nx.Graph:
        """將 MultiGraph 轉換為 Graph，平行邊以路徑中點節點拆分。

        對於兩節點間只有單一邊的情況，直接複製到新圖；
        若兩節點間存在複數平行邊，先捨棄 path 長度 < 5 的短邊（全捨棄時保留最長者），
        剩餘每條邊於其 ``path`` 中點插入一個新節點，並將原邊拆為兩段，消除平行邊。

        Args:
            multigraph (nx.MultiGraph): 來源骨架圖（skan 輸出）。

        Returns:
            nx.Graph: 不含平行邊與自環的簡單圖，節點為 ``(y, x)`` 整數座標元組；
            邊資料保留 ``path`` 與 ``branch-distance`` 屬性。
        """
        graph = nx.Graph()
        graph.add_nodes_from(multigraph.nodes(data=True))

        processed_pairs: set = set()

        for u, v, _key, data in multigraph.edges(keys=True, data=True):
            if u == v:
                continue  # 跳過自環（骨架閉合環路）
            pair = (u, v) if u <= v else (v, u)
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            parallel_edges = list(multigraph[u][v].values())

            if len(parallel_edges) == 1:
                graph.add_edge(u, v, **parallel_edges[0])
            else:
                # 複數平行邊：捨棄 path 長度 < 5 的短邊
                long_edges = [
                    e
                    for e in parallel_edges
                    if self._path_length(e.get("path", [])) >= 5
                ]
                if len(long_edges) == 0:
                    # 全部都是短邊：保留最長的一條
                    long_edges = [
                        max(
                            parallel_edges,
                            key=lambda e: self._path_length(e.get("path", [])),
                        )
                    ]
                if len(long_edges) == 1:
                    graph.add_edge(u, v, **long_edges[0])
                    continue
                for edge_data in long_edges:
                    path = edge_data.get("path", [])
                    if len(path) >= 3:
                        mid_idx = len(path) // 2
                        mid_node = tuple(int(c) for c in path[mid_idx])
                        path1 = path[: mid_idx + 1]
                        path2 = path[mid_idx:]
                    else:
                        mid_node = ((u[0] + v[0]) // 2, (u[1] + v[1]) // 2)
                        path1 = [u, mid_node]
                        path2 = [mid_node, v]

                    graph.add_node(mid_node)
                    graph.add_edge(
                        u,
                        mid_node,
                        path=path1,
                        **{"branch-distance": self._path_length(path1)},
                    )
                    graph.add_edge(
                        mid_node,
                        v,
                        path=path2,
                        **{"branch-distance": self._path_length(path2)},
                    )

        return graph

    # ========== 私有輔助方法 ==========

    @staticmethod
    def _path_length(path) -> float:
        """計算座標序列的累積歐式距離。"""
        if len(path) < 2:
            return 0.0
        pts = np.array(path)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def _merge_middle_points(self, graph: nx.Graph) -> nx.Graph:
        """合併 degree-2 中間點，將相鄰兩條邊合併為一條並串接 path。

        骨架中 degree-2 的節點是純粹的中繼點（非端點、非交叉），
        移除後將左右兩條 path 串接，可減少不必要的節點數。

        Args:
            graph (nx.Graph): 已重標籤節點的骨架圖。

        Returns:
            nx.Graph: 移除所有 degree-2 中繼節點後的圖。
        """
        middle_points = [
            point for point in graph.nodes() if len(list(graph.neighbors(point))) == 2
        ]

        for mp in middle_points:
            neighbors = list(graph.neighbors(mp))
            if len(neighbors) != 2:
                continue
            u, v = neighbors

            path1 = graph[u][mp]["path"]
            path2 = graph[mp][v]["path"]

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
            if u != v:
                graph.add_edge(u, v, path=result_path)

        return graph

    def _fill_missing_nodes(
        self,
        graph: nx.Graph,
        binary: np.ndarray,
        equalized_img: Optional[np.ndarray] = None,
    ) -> nx.Graph:
        """補齊骨架圖中未被覆蓋的孤立連通分量。

        以 ``regionprops`` 掃描所有連通分量的 bounding-box；
        若某分量的 bbox 範圍內完全沒有圖節點，則插入一個代表性節點：

        - 提供 ``equalized_img`` 時：取該分量 bbox 內灰度值最高的像素座標。
        - 未提供時（降級）：取 bbox 中心作為質心節點。

        Args:
            graph (nx.Graph): 待補齊的骨架圖。
            binary (np.ndarray): 二值化圖像，shape ``(H, W)``，值為 0/1。
            equalized_img (np.ndarray, optional): 均衡化灰階圖像，shape ``(H, W)``；
                ``None`` 時改用質心。

        Returns:
            nx.Graph: 補齊孤立節點後的圖（原地修改並回傳）。
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
                roi = equalized_img[min_row:max_row, min_col:max_col]
                local_idx = np.unravel_index(np.argmax(roi), roi.shape)
                brightest_pixel = (min_row + local_idx[0], min_col + local_idx[1])
            else:
                # 降級：使用區域質心
                brightest_pixel = (
                    (min_row + max_row) // 2,
                    (min_col + max_col) // 2,
                )

            graph.add_node(brightest_pixel)

        return graph

    def _compute_branch_distances(self, graph: nx.Graph) -> nx.Graph:
        """計算每條邊的 branch-distance 並寫入邊資料。

        ``branch-distance`` 定義為沿 ``path`` 座標序列逐段歐式距離的總和，
        反映骨架分支的實際曲線長度（而非端點直線距離）。
        path 長度不足 2 點時距離設為 ``0.0``。

        Args:
            graph (nx.Graph): 含 ``path`` 屬性的骨架圖。

        Returns:
            nx.Graph: 各邊新增 ``branch-distance`` (float) 後的圖（原地修改並回傳）。
        """
        for u, v, data in graph.edges(data=True):
            graph[u][v]["branch-distance"] = self._path_length(data.get("path", []))
        return graph
