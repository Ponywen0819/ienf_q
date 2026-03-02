"""
評估模組 (Evaluation Module)

提供神經纖維拓樸的評估與比對功能：

核心度量：
    - compute_average_hausdorff_distance: 計算平均 Hausdorff 距離
    - compute_directed_hausdorff_distance: 計算單向 Hausdorff 距離

拓樸比對：
    - TopologyComparator: 高階比對器
    - GraphPointExtractor: 點集提取器

檔案處理：
    - TopologyLoader: 多格式拓樸載入/儲存

資料型別：
    - ComparisonResult: 比對結果
    - EvaluationMetrics: 度量集合

使用範例：
---------
# 簡單比對
from neural_reconstruction.core.evaluation import TopologyComparator

comparator = TopologyComparator()
result = comparator.compare(graph_pred, graph_gt)
print(f"Distance: {result.hausdorff_distance:.4f}")

# 直接計算度量
from neural_reconstruction.core.evaluation import (
    compute_average_hausdorff_distance,
    extract_graph_points
)

points_a = extract_graph_points(graph_a)
points_b = extract_graph_points(graph_b)
distance = compute_average_hausdorff_distance(points_a, points_b)

# 載入拓樸
from neural_reconstruction.core.evaluation import TopologyLoader

loader = TopologyLoader()
graph = loader.load(Path("topology.graphml"))
"""

# 核心度量函數
from .metrics import (
    compute_point_min_distances,
    compute_average_hausdorff_distance,
    compute_directed_hausdorff_distance,
)

# 點集提取
from .point_extractor import (
    GraphPointExtractor,
    extract_graph_points,
)

# 拓樸載入器
from .topology_loader import TopologyLoader

# 比對器
from .comparator import TopologyComparator

# 資料型別
from .data_types import (
    ComparisonResult,
    EvaluationMetrics,
)

__all__ = [
    # 度量函數
    'compute_point_min_distances',
    'compute_average_hausdorff_distance',
    'compute_directed_hausdorff_distance',
    # 點集提取
    'GraphPointExtractor',
    'extract_graph_points',
    # 拓樸載入
    'TopologyLoader',
    # 比對器
    'TopologyComparator',
    # 資料型別
    'ComparisonResult',
    'EvaluationMetrics',
]

__version__ = '1.0.0'
