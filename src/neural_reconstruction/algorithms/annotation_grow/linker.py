"""
AnnotationGrowLinker — fiber reconstruction via annotation expansion.

Algorithm:
  1. Preprocess: green channel → background removal → CLAHE → Sato → cost map
  2. Dijkstra:   multi-source expansion with per-component adaptive stopping
  3. Meeting:    find minimum-cost touching point for each component pair
  4. Graph:      build component graph → prune high-cost edges → MST
  5. Skeleton:   per-CC dist-threshold corridor → skeletonize → pixel graph
"""

import logging

import cv2
import numpy as np
import networkx as nx

from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.common.data_types import LinkerResult

from .cost_map import build_enhanced_image, build_cost_map
from .dijkstra import (
    get_components,
    multi_source_dijkstra,
)
from .graph_builder import (
    find_meeting_points,
    build_component_graph,
    prune_edges,
    minimum_spanning_forest,
)
from .skeleton import build_result_graph

from neural_reconstruction.core.crosses_detection import run_crossing_analysis


logger = logging.getLogger(__name__)


class AnnotationGrowLinker:
    """
    Reconstruct neural fiber network by expanding annotation components
    via multi-source Dijkstra and connecting them through a minimum spanning tree.
    """

    def __init__(
        self,
        # Preprocessing
        offset_px: int = 50,
        bg_kernel_size: int = 51,
        clahe_clip: float = 20.0,
        clahe_grid: tuple[int, int] = (16, 16),
        sato_sigmas_start: int = 3,
        sato_sigmas_stop: int = 8,
        # Dijkstra
        connectivity: int = 8,
        # Edge pruning
        prune_threshold: float = 20.0,
        # Subtree filtering
        min_tree_components: int = 5,
        # Skeletonization
        segment_length: float = 100.0,
    ):
        self.offset_px = offset_px
        self.bg_kernel_size = bg_kernel_size
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.sato_sigmas_start = sato_sigmas_start
        self.sato_sigmas_stop = sato_sigmas_stop
        self.connectivity = connectivity
        self.prune_threshold = prune_threshold
        self.min_tree_components = min_tree_components
        self.segment_length = segment_length

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        annotation: np.ndarray,
    ) -> LinkerResult:
        """
        Args:
            image:      RGB or grayscale original image (H, W, 3) or (H, W)
            mask:       epidermis mask (H, W)
            annotation: binary annotation / weka output (H, W)
        """
        # ── 1. Extract green channel ──────────────────────────────────────
        if image.ndim == 3:
            green = image[:, :, 1]
        else:
            green = image

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if annotation.ndim == 3:
            annotation = annotation[:, :, 0]

        logger.info(f"Input image: {green.shape}")

        # ── 2. Preprocess ─────────────────────────────────────────────────
        roi_mask = dilate_epidermis_vertically(mask, offset_px=self.offset_px)

        roi_image = build_enhanced_image(
            green=green,
            roi_mask=roi_mask,
            bg_kernel_size=self.bg_kernel_size,
            clahe_clip=self.clahe_clip,
            clahe_grid=self.clahe_grid,
            sato_sigmas=range(self.sato_sigmas_start, self.sato_sigmas_stop),
        )

        roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
        cost_map = build_cost_map(roi_image)

        logger.info("Preprocessing done")

        # ── 3. Connected components ───────────────────────────────────────
        annotation_bin = (roi_annotation > 127).astype(np.uint8)
        annot_labeled = get_components(annotation_bin)
        n_components = int(annot_labeled.max())
        logger.info(f"Annotation components: {n_components}")

        # ── 4. Multi-source Dijkstra ──────────────────────────────────────

        owner_map, dist_map, prev_y, prev_x = multi_source_dijkstra(
            cost_map,
            annot_labeled,
            connectivity=self.connectivity,
            roi_mask=(roi_mask > 127),
        )
        logger.info(
            f"Dijkstra done — pixels reached: {(owner_map > 0).sum():,} "
            f"({(owner_map > 0).mean() * 100:.1f}%)"
        )

        # ── 5. Meeting points → component graph ───────────────────────────
        connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
        logger.info(f"Meeting points found: {len(connections)} component pairs")

        G = build_component_graph(connections, n_components)

        # ── 6. Prune + MST ────────────────────────────────────────────────
        G_pruned = prune_edges(G, threshold=self.prune_threshold)
        logger.info(
            f"After pruning: {G_pruned.number_of_edges()} edges remain "
            f"(removed {G.number_of_edges() - G_pruned.number_of_edges()})"
        )

        mst = minimum_spanning_forest(G_pruned)
        logger.info(
            f"MST: {mst.number_of_edges()} edges, "
            f"{nx.number_connected_components(mst)} trees"
        )

        # ── 7. Per-CC skeleton → pixel-level graph ───────────────────────
        result_graph = build_result_graph(
            mst,
            annotation_bin,
            segment_length=self.segment_length,
        )
        logger.info(
            f"Result graph: {result_graph.number_of_nodes()} nodes, "
            f"{result_graph.number_of_edges()} edges"
        )
        valid_count, labeled_graph = run_crossing_analysis(
            result_graph,
            mask,
            annot_labeled,
            min_tree_components=self.min_tree_components,
        )
        return LinkerResult(
            annotation=roi_annotation,
            image=roi_image,
            mask=roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )
