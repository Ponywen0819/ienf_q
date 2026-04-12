"""
DBSCAN-based Fragment Linker

Pipeline:
  1. Preprocessing  — background subtraction, ROI masking, CLAHE, Sato vesselness
  2. Seed graph     — TopologyBuilder skeletonizes annotation, assigns component_id
  3. DBSCAN         — clusters seed nodes by spatial proximity
  4. Intra-cluster MST — path-finding within each cluster, keep MST edges
  5. Endpoint extension — connect endpoints across clusters with cost = base + angle + distance
  6. Global MST     — prune the extended graph to a forest

Output: LinkerResult whose .graph is the final reconstructed network.
"""

import logging
from typing import Optional

import cv2
import networkx as nx
import numpy as np
import skimage as ski
from scipy.spatial import KDTree
from skimage.measure import label
from sklearn.cluster import DBSCAN

from neural_reconstruction.algorithms.fragment_linking.utils import compute_vector_angle
from neural_reconstruction.common.data_types import LinkerResult
from neural_reconstruction.core.crosses_detection import (
    CrossingCounter,
    RegionLabeler,
    SegmentDetector,
)
from neural_reconstruction.core.pathfinding import PathFinder
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.topology import TopologyBuilder

logger = logging.getLogger(__name__)


class DbscanLinker:
    """
    DBSCAN-based neural fiber reconstruction linker.

    Clusters seed nodes with DBSCAN, builds an intra-cluster MST for each
    cluster, then connects cluster endpoints across clusters to form the
    final network.

    Args:
        offset_px:               Vertical epidermis mask dilation (px).
        rolling_ball_radius:     Morphological opening radius for background
                                 subtraction (px).
        opening_kernel_size:     Closing kernel applied to annotation (px).
        use_sato:                Apply CLAHE + Sato vesselness to enhance
                                 fiber contrast before cost-map generation.
        sato_sigmas:             Scale range for Sato filter (iterable of ints).
        segment_length:          Seed spacing for TopologyBuilder (px).
        eps:                     DBSCAN neighborhood radius (px).
        min_samples:             DBSCAN minimum points to form a core point.
        intra_search_radius:     Path-finding search radius inside each cluster (px).
        extend_radius:           Endpoint-extension search radius (px).
        distance_penalty_weight: Weight of distance penalty in endpoint cost.
        angle_penalty_weight:    Weight of angle penalty in endpoint cost.
        base_cost_weight:        Weight of path base cost in endpoint cost.
        min_component_length:    Remove reconstructed components shorter than
                                 this (px).

    Examples:
        >>> linker = DbscanLinker(eps=20, min_samples=10)
        >>> result = linker.run(image, mask, annotation)
        >>> db_graph = result.graph
    """

    def __init__(
        self,
        # Preprocessing
        offset_px: int = 50,
        rolling_ball_radius: int = 50,
        opening_kernel_size: int = 3,
        use_sato: bool = True,
        sato_sigmas: tuple = (4, 5, 6, 7),
        # Topology
        segment_length: float = 3.0,
        # DBSCAN
        eps: float = 20.0,
        min_samples: int = 10,
        # Intra-cluster path finding
        intra_search_radius: float = 50.0,
        # Endpoint extension
        extend_radius: float = 30.0,
        distance_penalty_weight: float = 0.25,
        angle_penalty_weight: float = 0.25,
        base_cost_weight: float = 0.5,
        # Output filtering
        min_component_length: float = 10.0,
    ):
        self.offset_px = offset_px
        self.rolling_ball_radius = rolling_ball_radius
        self.opening_kernel_size = opening_kernel_size
        self.use_sato = use_sato
        self.sato_sigmas = sato_sigmas

        self.segment_length = segment_length

        self.eps = eps
        self.min_samples = min_samples

        self.intra_search_radius = intra_search_radius

        self.extend_radius = extend_radius
        self.distance_penalty_weight = distance_penalty_weight
        self.angle_penalty_weight = angle_penalty_weight
        self.base_cost_weight = base_cost_weight

        self.min_component_length = min_component_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        annotation: np.ndarray,
    ) -> LinkerResult:
        """
        Run the full DBSCAN linking pipeline.

        Args:
            image:      Original image (H, W) or (H, W, 3).
            mask:       Epidermis mask (H, W).
            annotation: Manual annotation (H, W), binary 0/255.

        Returns:
            LinkerResult with .graph = final reconstructed network.
        """
        logger.info("Step 1: Preprocessing")
        roi_image, roi_annotation, roi_mask = self._preprocess(image, mask, annotation)

        logger.info("Step 2–4: Reconstruction")
        graph = self._reconstruct(roi_annotation, roi_image)

        logger.info("Step 5: Crossing analysis")
        valid_count, labeled_graph = self._run_crossing_analysis(mask, graph)

        return LinkerResult(
            annotation=roi_annotation,
            image=roi_image,
            mask=roi_mask,
            graph=labeled_graph,
            valid_count=valid_count,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        annotation: np.ndarray,
    ):
        if image.ndim == 3:
            image = image[:, :, 1]  # green channel

        roi_mask = dilate_epidermis_vertically(mask, offset_px=self.offset_px)

        # Background subtraction
        kernel_size = self.rolling_ball_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.subtract(image, background)

        roi_image = cv2.bitwise_and(image, image, mask=roi_mask)
        roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)

        # Morphological closing on annotation
        if self.opening_kernel_size > 0:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.opening_kernel_size, self.opening_kernel_size),
            )
            roi_annotation = cv2.morphologyEx(
                roi_annotation, cv2.MORPH_CLOSE, close_kernel, iterations=3
            )
        roi_annotation[roi_annotation > 0] = 255

        # CLAHE + Sato vesselness
        if self.use_sato:
            clahe = cv2.createCLAHE()
            roi_image = clahe.apply(roi_image)
            roi_image = ski.filters.sato(
                roi_image, sigmas=self.sato_sigmas, black_ridges=False
            )
            roi_image = (
                (roi_image - roi_image.min())
                / (roi_image.max() - roi_image.min() + 1e-8)
                * 255
            )
            roi_image = roi_image.astype(np.uint8)

        return roi_image, roi_annotation, roi_mask

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct(
        self,
        annotation: np.ndarray,
        image: np.ndarray,
    ) -> nx.Graph:
        if annotation is None or annotation.size == 0:
            return nx.Graph()

        # --- Connected component labels (for intra-cluster path filter) ---
        binary = (annotation > 0).astype(np.uint8)
        annotation_component = np.asarray(label(binary, connectivity=2))

        # --- Seed graph ---
        tb = TopologyBuilder(segment_length=self.segment_length)
        seed_graph = tb.build_seed_graph(annotation, image)

        if seed_graph.number_of_nodes() == 0:
            return nx.Graph()

        for node in seed_graph.nodes():
            y, x = node
            seed_graph.nodes[node]["component_id"] = int(
                annotation_component[int(y), int(x)]
            )
        for _, _, data in seed_graph.edges(data=True):
            data["weight"] = 1e-5

        # --- Cost map & path finder ---
        cost_map = ((255 - image.astype(np.float64)) / 255) ** 2
        path_finder = PathFinder(cost_map)

        topology_points = np.array(list(seed_graph.nodes()))
        seed_map = np.zeros_like(cost_map, dtype=bool)
        for p in topology_points:
            seed_map[int(p[0]), int(p[1])] = True

        # --- DBSCAN ---
        node_positions = np.array([[y, x] for y, x in seed_graph.nodes()])
        cluster_labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(
            node_positions
        )

        cluster_nodes: dict[int, list] = {}
        for node, cid in zip(seed_graph.nodes(), cluster_labels):
            seed_graph.nodes[node]["cluster_id"] = int(cid)
            cluster_nodes.setdefault(int(cid), []).append(node)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(
            f"DBSCAN: {len(seed_graph.nodes())} nodes → "
            f"{n_clusters} clusters, "
            f"{int(np.sum(cluster_labels == -1))} noise"
        )

        # --- Intra-cluster MST ---
        db_graph = self._build_intra_cluster_msts(
            seed_graph,
            cluster_nodes,
            path_finder,
            seed_map,
            annotation_component,
        )

        # --- Endpoint extension ---
        extend_graph = self._extend_endpoints(
            db_graph,
            seed_graph,
            topology_points,
            path_finder,
            seed_map,
            annotation_component,
        )

        # --- Filter short components ---
        extend_graph = self._filter_short_components(extend_graph)

        return extend_graph

    def _build_intra_cluster_msts(
        self,
        seed_graph: nx.Graph,
        cluster_nodes: dict,
        path_finder: PathFinder,
        seed_map: np.ndarray,
        annotation_component: np.ndarray,
    ) -> nx.Graph:
        """Build an MST inside each DBSCAN cluster, merge into one graph."""
        db_graph = nx.Graph()

        for cid, nodes in cluster_nodes.items():
            if cid == -1:
                continue  # skip noise

            cluster_graph = seed_graph.subgraph(nodes).copy()
            cluster_arr = np.array(nodes)
            kdtree = KDTree(cluster_arr)

            path_lookup = path_finder.find_paths_from_seeds(
                topology_points=nodes,
                kdtree=kdtree,
                search_radius=self.intra_search_radius,
                seed_map=seed_map,
                label_img=annotation_component,
            )

            for (u, v), (path, cost) in path_lookup.items():
                cluster_graph.add_edge(u, v, weight=cost, path=path)

            mst = nx.minimum_spanning_tree(cluster_graph, weight="weight")
            db_graph.add_edges_from(mst.edges(data=True))

        return db_graph

    def _extend_endpoints(
        self,
        db_graph: nx.Graph,
        seed_graph: nx.Graph,
        topology_points: np.ndarray,
        path_finder: PathFinder,
        seed_map: np.ndarray,
        annotation_component: np.ndarray,
    ) -> nx.Graph:
        """
        Connect degree-1 endpoints across clusters.

        For each endpoint, find nearby endpoints, compute paths, score by
        base cost + angle penalty + distance penalty, add the edge, then
        take a global MST of the entire extended graph.
        """
        extend_graph = db_graph.copy()
        endpoints = [
            node for node in extend_graph.nodes() if extend_graph.degree(node) == 1
        ]
        logger.info(f"Endpoint extension: {len(endpoints)} endpoints")

        if not endpoints:
            return extend_graph

        endpoint_arr = np.array(endpoints)
        endpoint_kdtree = KDTree(endpoint_arr)

        path_lookup = path_finder.find_paths_from_seeds(
            topology_points=endpoint_arr,
            kdtree=endpoint_kdtree,
            search_radius=self.extend_radius,
            seed_map=seed_map,
            label_img=annotation_component,
        )

        for endpoint in endpoints:
            neighbors = list(extend_graph.neighbors(endpoint))
            if len(neighbors) != 1:
                continue
            neighbor = neighbors[0]
            extend_vector = np.array(endpoint) - np.array(neighbor)
            ep_arr = np.array(endpoint)

            for (u, v), (path, base_cost) in path_lookup.items():
                if u != endpoint and v != endpoint:
                    continue
                target = v if u == endpoint else u
                if target == neighbor:
                    continue
                if extend_graph.has_edge(endpoint, target):
                    continue
                # Skip same cluster
                cid_ep = seed_graph.nodes.get(endpoint, {}).get("cluster_id")
                cid_tg = seed_graph.nodes.get(target, {}).get("cluster_id")
                if cid_ep is not None and cid_ep == cid_tg:
                    continue

                ac_vector = np.array(target) - ep_arr
                distance = float(np.linalg.norm(ac_vector))
                angle = compute_vector_angle(extend_vector, ac_vector)

                distance_penalty = (
                    distance / self.extend_radius * self.distance_penalty_weight
                )
                angle_penalty = angle / 180.0 * self.angle_penalty_weight
                final_cost = (
                    self.base_cost_weight * base_cost + angle_penalty + distance_penalty
                )

                extend_graph.add_edge(
                    endpoint,
                    target,
                    weight=final_cost,
                    path=path,
                    base_cost=base_cost,
                )

        extend_graph = nx.minimum_spanning_tree(extend_graph, weight="weight")
        return extend_graph

    def _filter_short_components(self, graph: nx.Graph) -> nx.Graph:
        """Remove connected components whose total path length is too short."""
        if self.min_component_length <= 0:
            return graph

        nodes_to_remove = []
        for component_nodes in nx.connected_components(graph):
            sub = graph.subgraph(component_nodes)
            total_length = 0.0
            for u, v, data in sub.edges(data=True):
                path = data.get("path", [])
                if len(path) >= 2:
                    pts = np.array(path)
                    total_length += float(
                        np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                    )
                else:
                    total_length += float(np.linalg.norm(np.array(u) - np.array(v)))
            if total_length < self.min_component_length:
                nodes_to_remove.extend(component_nodes)

        graph = graph.copy()
        graph.remove_nodes_from(nodes_to_remove)
        return graph

    # ------------------------------------------------------------------
    # Crossing analysis (same pattern as PureMstLinker)
    # ------------------------------------------------------------------

    def _run_crossing_analysis(self, mask: np.ndarray, graph: nx.Graph):
        try:
            segment_detector = SegmentDetector()
            region_labeler = RegionLabeler()
            crossing_counter = CrossingCounter()

            segments = segment_detector.detect(graph)
            labeled_regions = region_labeler.label(mask)
            valid_count = crossing_counter.count(segments, labeled_regions)
            return valid_count, graph
        except Exception as e:
            logger.warning(f"Crossing analysis failed: {e}")
            return 0, graph
