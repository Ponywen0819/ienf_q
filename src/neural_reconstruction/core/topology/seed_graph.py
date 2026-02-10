"""
種子圖生成模組 (Seed Graph Builder)

將骨架圖轉換為種子圖：
沿每條邊按 segment_length 均勻切分，生成更密集的種子點。
"""

import numpy as np
import networkx as nx


class SeedGraphBuilder:
    """
    將骨架圖轉換為種子圖

    沿骨架邊按 segment_length 均勻插入種子點，
    使每段距離不超過 segment_length 像素。

    Examples:
        >>> builder = SeedGraphBuilder(segment_length=3.0)
        >>> seed_graph = builder.build(skeleton_graph)
    """

    def __init__(self, segment_length: float = 3.0, verbose: bool = False):
        self.segment_length = segment_length
        self.verbose = verbose

    def build(self, skeleton_graph: nx.MultiGraph) -> nx.MultiGraph:
        """
        構建種子圖

        Args:
            skeleton_graph: 骨架圖

        Returns:
            種子圖，沿邊切分為更小的片段
        """
        seed_graph = nx.MultiGraph()

        # 添加所有節點
        for u in skeleton_graph.nodes():
            seed_graph.add_node(u)

        # 處理每條邊
        for u, v, data in skeleton_graph.edges(data=True):
            path = data["path"]

            # 擺正方向
            corrected_path = (
                path[:] if tuple(path[0]) == u and tuple(path[-1]) == v else path[::-1]
            )

            # 計算路徑長度
            path_arr = np.array(corrected_path)
            diffs = np.diff(path_arr, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
            path_length = cumulative_distances[-1]

            # 計算分段數
            num_segments = int(path_length // self.segment_length)

            if num_segments <= 0:
                seed_graph.add_edge(u, v, path=path)
                continue

            # 按距離切分
            last_index = 0
            for i in range(num_segments):
                target_distance = (i + 1) * path_length / num_segments
                segment_end_index = 0

                for idx, cumulative_distance in enumerate(
                    cumulative_distances[last_index:]
                ):
                    if cumulative_distance >= target_distance:
                        segment_end_index = idx + last_index
                        break

                segment_path = corrected_path[last_index : segment_end_index + 1]
                if len(segment_path) == 0:
                    continue

                seed_graph.add_edge(
                    tuple(segment_path[0]), tuple(segment_path[-1]), path=segment_path
                )
                last_index = segment_end_index

            # 添加最後一段
            if last_index < len(corrected_path) - 1:
                final_segment_path = corrected_path[last_index:]
                seed_graph.add_edge(
                    tuple(final_segment_path[0]),
                    tuple(final_segment_path[-1]),
                    path=final_segment_path,
                )

        if self.verbose:
            print(
                f"✓ 種子圖構建完成: {seed_graph.number_of_nodes()} 節點, "
                f"{seed_graph.number_of_edges()} 邊"
            )

        return seed_graph
