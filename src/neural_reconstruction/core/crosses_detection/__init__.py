"""
跨越偵測模組 (Crosses Detection Module)

提供神經纖維穿越表皮/真皮邊界的偵測與計算功能：

1. SegmentDetector - 識別神經纖維的獨立分段
2. MainTrunkExtractor - 找出每個 component 的主幹並合併為單一 segment_id
3. RegionLabeler - 為拓樸節點標注區域屬性（表皮/真皮）
4. CrossingCounter - 計算有效跨越數量（每個分段最多算一次）

使用範例：
---------
from crosses_detection import (
    SegmentDetector,
    MainTrunkExtractor,
    RegionLabeler,
    CrossingCounter,
)

segmented = SegmentDetector.detect_segments(topology)
trunked = MainTrunkExtractor.extract(segmented)
labeled, _ = RegionLabeler().label_topology(trunked, epidermis_mask)
result = CrossingCounter().count_effective_crossings(labeled, epidermis_mask)

print(f"有效跨越數: {result['effective_crossing_count']}")
"""

from .region_labeler import RegionLabeler
from .segment_detector import SegmentDetector
from .crossing_counter import CrossingCounter
from .main_trunk_extractor import MainTrunkExtractor
from .pipeline import run_crossing_analysis

__all__ = [
    'RegionLabeler',
    'SegmentDetector',
    'CrossingCounter',
    'MainTrunkExtractor',
    'run_crossing_analysis',
]
