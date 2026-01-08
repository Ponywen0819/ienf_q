"""
跨越偵測模組 (Crosses Detection Module)

提供神經纖維穿越表皮/真皮邊界的偵測與計算功能：

1. RegionLabeler - 為拓樸節點標注區域屬性（表皮/真皮）
2. SegmentDetector - 識別神經纖維的獨立分段
3. CrossingCounter - 計算有效跨越數量（每個分段最多算一次）

使用範例：
---------
from crosses_detection import RegionLabeler, SegmentDetector, CrossingCounter

# 初始化
region_labeler = RegionLabeler()
segment_detector = SegmentDetector()
crossing_counter = CrossingCounter()

# Step 1: 標注區域
labeled_topology = region_labeler.label_topology(topology, epidermis_mask)

# Step 2: 偵測分段
segments = segment_detector.detect_segments(labeled_topology)

# Step 3: 計算有效跨越
result = crossing_counter.count_effective_crossings(segments, labeled_topology)

print(f"有效跨越數: {result['effective_crossing_count']}")
"""

from .region_labeler import RegionLabeler
from .segment_detector import SegmentDetector
from .crossing_counter import CrossingCounter

__all__ = [
    'RegionLabeler',
    'SegmentDetector',
    'CrossingCounter',
]
