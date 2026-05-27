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

import numpy as np
import networkx as nx

from neural_reconstruction.core.preprocessing import PreprocessingPipeline
from neural_reconstruction.common.data_types import LinkerResult

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
        stub_length_threshold: int = 5,
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
        self.stub_length_threshold = stub_length_threshold

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
        # ── 1. Shared preprocessing ───────────────────────────────────────
        # Keep a 2-D copy of the raw epidermis mask for crossing analysis.
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        pre = PreprocessingPipeline(
            offset_px=self.offset_px,
            bg_kernel_size=self.bg_kernel_size,
            clahe_clip=self.clahe_clip,
            clahe_grid=self.clahe_grid,
            sato_sigmas_start=self.sato_sigmas_start,
            sato_sigmas_stop=self.sato_sigmas_stop,
        ).run(image, mask, annotation)
        logger.info("Preprocessing done")

        # ── 2. Connected components ───────────────────────────────────────
        annotation_bin = (pre.roi_annotation > 127).astype(np.uint8)
        annot_labeled = get_components(annotation_bin)
        n_components = int(annot_labeled.max())
        logger.info(f"Annotation components: {n_components}")

        # ── 3. Multi-source Dijkstra ──────────────────────────────────────

        owner_map, dist_map, prev_y, prev_x = multi_source_dijkstra(
            pre.cost_map,
            annot_labeled,
            connectivity=self.connectivity,
            roi_mask=(pre.roi_mask > 127),
        )
        logger.info(
            f"Dijkstra done — pixels reached: {(owner_map > 0).sum():,} "
            f"({(owner_map > 0).mean() * 100:.1f}%)"
        )

        # ── 4. Meeting points → component graph ───────────────────────────
        connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
        logger.info(f"Meeting points found: {len(connections)} component pairs")

        G = build_component_graph(connections, n_components)

        # ── 5. Prune + MST ────────────────────────────────────────────────
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

        # ── 6. Per-CC skeleton → pixel-level graph ───────────────────────
        result_graph = build_result_graph(
            mst,
            annotation_bin,
            segment_length=500,
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
            stub_length_threshold=self.stub_length_threshold

        )
        return LinkerResult(
            annotation=pre.roi_annotation,
            image=pre.roi_image,
            mask=pre.roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )
