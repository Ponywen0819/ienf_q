"""
Label baseline linker (skip reconstruction).

直接對 GT label 做骨架化，不使用原始影像，也不執行任何 pathfinding 或 MST
重建。作為其他重建演算法的上界對照 baseline 使用。

注意 ``annot_labeled`` 用「真實 annotation (weka.png)」, 不用 label 本身 —
這樣 ``min_tree_components`` 過濾條件才能與重建版 linker 對齊
(skeleton-CC 要覆蓋足夠多 annotation 元件才計入 ``valid_count``)。
"""

import logging

import cv2
import numpy as np
from skimage.measure import label as cc_label

from neural_reconstruction.common.data_types import LinkerResult
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.crosses_detection import run_crossing_analysis


logger = logging.getLogger(__name__)


class LabelLinker:
    """
    Baseline：直接對 GT label 做骨架化, 但仍以真實 annotation 作為計數過濾依據。

    流程：
      1. 依 ``offset_px`` 將表皮遮罩向下擴張取得 ROI mask
      2. 將 label 與 annotation 限制在 ROI 內
      3. 用 ``TopologyBuilder`` 對 ``roi_label`` 做骨架化 (Zhang-Suen → skan)
      4. 以 ``roi_annotation`` 計算 ``annot_labeled`` 做為小子樹過濾依據
      5. 執行共用 ``run_crossing_analysis`` 取得 valid_count

    Args:
        offset_px: 表皮往下擴張像素數
        min_tree_components: 子樹覆蓋 annotation 元件下限,低於此值的子樹
            「保留在圖上但不計入交叉數」。預設 ``5``,與
            ``AnnotationGrowLinker`` / ``PureMstLinker`` 一致。
    """

    def __init__(self, offset_px: int = 50, min_tree_components: int = 5):
        self.offset_px = offset_px
        self.min_tree_components = min_tree_components

    def run(
        self,
        mask: np.ndarray,
        label: np.ndarray,
        annotation: np.ndarray,
    ) -> LinkerResult:
        """
        Args:
            mask:       表皮遮罩 (H, W)
            label:      GT 標註影像 (H, W) — skeleton 來源
            annotation: 真實 annotation / weka 輸出 (H, W) — 過濾條件來源
        """
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if label.ndim == 3:
            label = label[:, :, 0]
        if annotation.ndim == 3:
            annotation = annotation[:, :, 0]

        roi_mask = dilate_epidermis_vertically(mask, offset_px=self.offset_px)
        roi_label = cv2.bitwise_and(label, label, mask=roi_mask)
        roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)

        topology_builder = TopologyBuilder()
        skeleton_graph = topology_builder.build_skeleton_graph(roi_label)
        logger.info(
            f"Skeleton graph: {skeleton_graph.number_of_nodes()} nodes, "
            f"{skeleton_graph.number_of_edges()} edges"
        )

        annot_labeled = np.asarray(
            cc_label((roi_annotation > 0).astype(np.uint8), connectivity=2)
        )

        valid_count, labeled_graph = run_crossing_analysis(
            skeleton_graph,
            mask,
            annot_labeled,
            min_tree_components=self.min_tree_components,
        )

        return LinkerResult(
            annotation=roi_label,
            image=roi_label,
            mask=roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )
