"""
SkeletonLinker - no-op baseline from ROI annotation to topology.

This linker does not perform reconstruction, path finding, growing, or MST
connection. It only clips the input annotation to the ROI mask, skeletonizes it
with TopologyBuilder, and runs the standard crossing analysis.
"""

import logging

import cv2
import numpy as np
from skimage.measure import label as cc_label

from neural_reconstruction.common.data_types import LinkerResult
from neural_reconstruction.core.crosses_detection import run_crossing_analysis
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.topology import TopologyBuilder


logger = logging.getLogger(__name__)


class SkeletonLinker:
    """
    Baseline linker: directly converts ``roi_annotation`` into a topology graph.

    Flow:
      1. Build ROI mask from epidermis mask using ``offset_px``
      2. Clip annotation to ROI, producing ``roi_annotation``
      3. Build skeleton topology with ``TopologyBuilder``
      4. Run the shared crossing analysis pipeline

    Args:
        offset_px:            表皮往下擴張像素數
        segment_length:       TopologyBuilder seed spacing (pixels)
        min_tree_components:  子樹覆蓋 annotation 元件下限,低於此值的子樹
            「保留在圖上但不計入交叉數」。預設 ``0`` 表示永不排除 —
            因為此 baseline 不做 MST 合併,每個 annotation CC 在輸出圖中
            各自獨立 (覆蓋 1 個元件),套用 > 1 的門檻會把全部排除掉。
            如需與重建版 linker 同步門檻請手動設定。
    """

    def __init__(
        self,
        offset_px: int = 50,
        segment_length: float = 3.0,
        min_tree_components: int =  5,
    ):
        self.offset_px = offset_px
        self.segment_length = segment_length
        self.min_tree_components = min_tree_components

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        annotation: np.ndarray,
    ) -> LinkerResult:
        """
        Args:
            image: RGB or grayscale original image (H, W, 3) or (H, W)
            mask: epidermis mask (H, W)
            annotation: binary annotation / weka output (H, W)
        """
        if image.ndim == 3:
            image = image[:, :, 1]
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if annotation.ndim == 3:
            annotation = annotation[:, :, 0]

        roi_mask = dilate_epidermis_vertically(mask, offset_px=self.offset_px)
        roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)

        topology_builder = TopologyBuilder(segment_length=self.segment_length)
        skeleton_graph = topology_builder.build_skeleton_graph(roi_annotation)
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
            annotation=roi_annotation,
            image=cv2.bitwise_and(image, image, mask=roi_mask),
            mask=roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )
