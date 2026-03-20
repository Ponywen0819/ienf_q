"""
拓樸建構模組 (Topology Module)

提供統一的骨架圖與種子圖建構功能。

核心類別：
    - TopologyBuilder: 骨架圖建構器，同時提供 build_seed_graph() 組合方法

使用範例：
---------
from neural_reconstruction.core.topology import TopologyBuilder

# 兩層 API：
builder = TopologyBuilder(segment_length=3.0)

# 1. 僅骨架圖
skeleton = builder.build_skeleton_graph(annotation, equalized_img)

# 2. 骨架圖 + 種子圖（一步完成，fragment_linking 管線推薦）
seed_graph = builder.build_seed_graph(annotation, equalized_img)

# 3. 單一元件 mask（component_analyzer 管線，equalized_img=None 降級為質心）
skeleton = builder.build_skeleton_graph(component_mask)
"""

from .topology_builder import TopologyBuilder

__all__ = ['TopologyBuilder']

__version__ = '1.0.0'
