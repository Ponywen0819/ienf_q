"""
片段連接模組 (Fragment Linking Module)

實作階層式神經纖維片段連接算法：

主控制器：
    - HierarchicalFragmentLinker: 完整流程控制器

算法元件：
    - TopologyBuilder: 從標注圖像構建骨架圖
    - SeedGraphBuilder: 將骨架圖轉換為種子圖
    - extend_endpoints: 階段1 高信心端點延伸（嚴格約束）
    - generate_mst_candidates: 階段2 MST 候選邊生成（寬鬆約束）

路徑查找：
    - PathFinder: 統一路徑查找器（來自 neural_reconstruction.core.pathfinding）

工具函數：
    - compute_vector_angle: 計算兩向量夾角
    - is_direction_too_similar: 檢查方向相似度

使用範例：
---------
from neural_reconstruction.algorithms.fragment_linking import HierarchicalFragmentLinker

linker = HierarchicalFragmentLinker(
    segment_length=3.0,
    search_radius_pathfinding=50.0,
    verbose=True,
)
result_graph = linker.run(image, mask, annotation)
"""

from .linker import HierarchicalFragmentLinker
from neural_reconstruction.core.topology import TopologyBuilder, SeedGraphBuilder
from .endpoint_extension import extend_endpoints
from .mst_candidates import generate_mst_candidates
from .utils import compute_vector_angle, is_direction_too_similar

__all__ = [
    # 主控制器
    'HierarchicalFragmentLinker',
    # 算法元件
    'TopologyBuilder',
    'SeedGraphBuilder',
    'extend_endpoints',
    'generate_mst_candidates',
    # 工具函數
    'compute_vector_angle',
    'is_direction_too_similar',
]

__version__ = '1.0.0'
