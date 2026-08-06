"""
electron_worker.py — stateful, handle-based RPC worker for the Electron labeler.

Protocol: one JSON object per stdin line; one JSON object per stdout line.
  Request:  {"id": <str>, "method": <str>, "params": {<str>: <any>}}
  Success:  {"id": <str>, "result": <any>}
  Failure:  {"id": <str>, "error": {"type": <str>, "message": <str>, "traceback": <str>}}

Large numpy arrays / networkx graphs are held in-process and referenced by
opaque string handles; only handles flow over the protocol. The TS orchestrator
owns the cache key → handle mapping and the stage DAG.

Stdout discipline: ALL logs go to stderr. Stdout is reserved for protocol frames.
"""

# ── Lock stdout BEFORE importing anything that might print ──────────────────
import sys as _sys

_STDOUT = _sys.stdout
_sys.stdout = _sys.stderr  # any stray print() now lands on stderr instead

# Imports below are deliberately placed after the stdout swap so that any
# module-level print() inside transitively-loaded dependencies cannot leak
# onto the protocol channel.
import base64  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from io import BytesIO  # noqa: E402
from typing import Any  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="[worker] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("electron_worker")

import cv2  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from neural_reconstruction.algorithms.annotation_grow.cost_map import (  # noqa: E402
    apply_within_mask_strips,
    build_cost_map,
)
from neural_reconstruction.algorithms.annotation_grow.dijkstra import (  # noqa: E402
    get_components,
    multi_source_dijkstra,
)
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (  # noqa: E402
    build_component_graph,
    find_meeting_points,
    minimum_spanning_forest,
    prune_edges,
)
from collections import defaultdict  # noqa: E402

from neural_reconstruction.algorithms.annotation_grow.skeleton import build_result_graph  # noqa: E402
from neural_reconstruction.core.crosses_detection import run_crossing_analysis  # noqa: E402
from neural_reconstruction.core.crosses_detection.crossing_counter import (  # noqa: E402
    CrossingCounter,
)
from neural_reconstruction.core.crosses_detection.pipeline import (  # noqa: E402
    _exclude_small_subtrees_from_count,
)
from neural_reconstruction.core.crosses_detection.region_labeler import RegionLabeler  # noqa: E402
from neural_reconstruction.core.crosses_detection.segment_detector import (  # noqa: E402
    SegmentDetector,
)
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically  # noqa: E402
from neural_reconstruction.core.topology import TopologyBuilder  # noqa: E402


def _send(frame: dict) -> None:
    _STDOUT.write(json.dumps(frame) + "\n")
    _STDOUT.flush()


def _coerce_value(v: Any) -> Any:
    """Numpy → plain Python so values survive json.dumps in `extract_graph`."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_coerce_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce_value(x) for k, x in v.items()}
    return v


def _coerce(attrs: dict) -> dict:
    return {str(k): _coerce_value(v) for k, v in attrs.items()}


def _path_length(path) -> float:
    """Cumulative Euclidean distance along a path's pixel coordinates.

    Not the edge's `weight` attribute — that's an A* search cost, not a
    real-world length.
    """
    if path is None or len(path) < 2:
        return 0.0
    pts = np.array(path)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _tag_and_measure_subtrees(graph: nx.Graph) -> list:
    """Tag every node/edge with `tree_id` (connected-component index) and
    return each tree's total path length.

    One independent nerve fiber = one connected component (the MST forest
    is already disjoint per-tree). Tagging `tree_id` directly onto the graph
    means it survives `extract_graph` serialization, so the frontend can
    join a `subtree_lengths` list entry back to its on-canvas nodes/edges by
    `tree_id` without any separate lookup.

    Ids run left to right: `nx.connected_components` yields in node-insertion
    order, which is arbitrary and reshuffles whenever an edit re-imports the
    graph. Ordering by the leftmost node instead keeps the id tied to where the
    fiber actually is on screen.
    """
    node_tree_id: dict = {}
    tree_node_counts: list = []
    # Nodes are (y, x); sort on (x, y) so a shared leftmost column is stable.
    components = sorted(
        nx.connected_components(graph),
        key=lambda nodes: min((n[1], n[0]) for n in nodes),
    )
    for tree_id, nodes in enumerate(components):
        for n in nodes:
            node_tree_id[n] = tree_id
            graph.nodes[n]["tree_id"] = tree_id
        tree_node_counts.append(len(nodes))

    tree_lengths = [0.0] * len(tree_node_counts)
    tree_edge_counts = [0] * len(tree_node_counts)
    for u, v, data in graph.edges(data=True):
        tid = node_tree_id[u]
        data["tree_id"] = tid
        tree_lengths[tid] += _path_length(data.get("path", [u, v]))
        tree_edge_counts[tid] += 1

    return [
        {
            "tree_id": tid,
            "num_nodes": tree_node_counts[tid],
            "num_edges": tree_edge_counts[tid],
            "total_length": tree_lengths[tid],
        }
        for tid in range(len(tree_node_counts))
    ]


class StageWorker:
    """Handle-based stateful pipeline. Each stage owns no caching; the TS
    orchestrator decides when to call. `stats()` exposes per-stage call counts
    for cache-correctness assertions."""

    def __init__(self) -> None:
        self.handles: dict[str, Any] = {}
        self.stage_calls: dict[str, int] = {}

    # ── Internals ────────────────────────────────────────────────────────
    def _new_handle(self, obj: Any) -> str:
        h = uuid.uuid4().hex
        self.handles[h] = obj
        return h

    def _bump(self, stage: str) -> None:
        self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1

    # ── Sample I/O ───────────────────────────────────────────────────────
    def load_sample(
        self, image_path: str, mask_path: str, annotation_path: str
    ) -> dict:
        """Load (image, mask, annotation) PNGs; return three handles + shape."""
        img = np.array(Image.open(image_path))
        green = img[:, :, 1] if img.ndim == 3 else img
        mask = np.array(Image.open(mask_path).convert("L"))
        annotation = np.array(Image.open(annotation_path).convert("L"))
        return {
            "green": self._new_handle(green),
            "mask": self._new_handle(mask),
            "annotation": self._new_handle(annotation),
            "shape": list(green.shape),
        }

    # ── Stage methods ───────────────────────────────────────────────────
    def stage_roi_mask(self, mask: str, offset_px: int) -> str:
        self._bump("roi_mask")
        return self._new_handle(
            dilate_epidermis_vertically(self.handles[mask], offset_px=int(offset_px))
        )

    def stage_annot_comp(self, annotation: str, roi_mask: str) -> dict:
        self._bump("annot_comp")
        ann = self.handles[annotation]
        rm = self.handles[roi_mask]
        roi_annotation = cv2.bitwise_and(ann, ann, mask=rm)
        annotation_bin = (roi_annotation > 127).astype(np.uint8)
        annot_labeled = get_components(annotation_bin)
        return {
            "annot_labeled": self._new_handle(annot_labeled),
            "annotation_bin": self._new_handle(annotation_bin),
            "n_components": int(annot_labeled.max()),
        }

    def stage_bg_removed(
        self, green: str, roi_mask: str, bg_kernel_size: int
    ) -> str:
        self._bump("bg_removed")
        g = self.handles[green]
        rm = self.handles[roi_mask]
        bks = int(bg_kernel_size)
        if bks > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bks, bks))

            def _op(patch: np.ndarray) -> np.ndarray:
                bg = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
                return cv2.subtract(patch, bg)

            corrected = apply_within_mask_strips(g, rm, _op, pad=bks)
        else:
            corrected = g
        return self._new_handle(cv2.bitwise_and(corrected, corrected, mask=rm))

    def stage_clahe_applied(
        self, bg_removed: str, clahe_clip: float, clahe_grid: Any
    ) -> str:
        self._bump("clahe_applied")
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid)
        )
        return self._new_handle(clahe.apply(self.handles[bg_removed]))

    def stage_sato_per_sigma(
        self, clahe_applied: str, roi_mask: str, sigma: int
    ) -> str:
        self._bump("sato_per_sigma")
        import skimage as ski

        s = int(sigma)
        pad = int(np.ceil(4 * s)) + 4
        out = apply_within_mask_strips(
            self.handles[clahe_applied],
            self.handles[roi_mask],
            lambda patch: ski.filters.sato(
                patch, sigmas=range(s, s + 1), black_ridges=False
            ),
            pad=pad,
        )
        return self._new_handle(out)

    def stage_roi_image(self, per_sigma: list) -> str:
        self._bump("roi_image")
        responses = [self.handles[h] for h in per_sigma]
        result = np.maximum.reduce(responses)
        vmin, vmax = result.min(), result.max()
        if vmax > vmin:
            result = (result - vmin) / (vmax - vmin) * 255
        return self._new_handle(result.astype(np.uint8))

    def stage_cost_map(self, roi_image: str) -> str:
        self._bump("cost_map")
        return self._new_handle(build_cost_map(self.handles[roi_image]))

    def stage_dijkstra(
        self,
        cost_map: str,
        annot_labeled: str,
        roi_mask: str,
        connectivity: int,
    ) -> str:
        self._bump("dijkstra")
        owner_map, dist_map, prev_y, prev_x = multi_source_dijkstra(
            self.handles[cost_map],
            self.handles[annot_labeled],
            connectivity=int(connectivity),
            roi_mask=(self.handles[roi_mask] > 127),
        )
        return self._new_handle((owner_map, dist_map, prev_y, prev_x))

    def stage_comp_graph(self, dijkstra: str, n_components: int) -> str:
        self._bump("comp_graph")
        owner_map, dist_map, prev_y, prev_x = self.handles[dijkstra]
        connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
        return self._new_handle(build_component_graph(connections, int(n_components)))

    def stage_pruned_graph(self, comp_graph: str, prune_threshold: float) -> str:
        self._bump("pruned_graph")
        return self._new_handle(
            prune_edges(self.handles[comp_graph], threshold=float(prune_threshold))
        )

    def stage_mst(self, pruned_graph: str) -> str:
        self._bump("mst")
        return self._new_handle(minimum_spanning_forest(self.handles[pruned_graph]))

    def stage_result_graph(
        self, mst: str, annotation_bin: str
    ) -> str:
        self._bump("result_graph")
        return self._new_handle(
            build_result_graph(
                self.handles[mst],
                self.handles[annotation_bin],
            )
        )

    def stage_reconstructed_graph(
        self, result_graph: str, stub_length_threshold: int
    ) -> str:
        """Reconstruction-side post-processing: segment detect → stub trim →
        re-segment. Mirrors steps 1-1c of run_crossing_analysis.

        Output is the graph the user sees in the editor (and may modify before
        counting). No region labelling or counting is performed here.
        """
        self._bump("reconstructed_graph")
        seg_detector = SegmentDetector()
        graph = self.handles[result_graph].copy()
        graph = seg_detector.detect_segments(graph)

        # Stub trim: drop segments shorter than threshold that touch an endpoint.
        seg_edges: dict = defaultdict(list)
        for u, v, data in graph.edges(data=True):
            sid = data.get("segment_id")
            if sid is not None:
                seg_edges[sid].append((u, v, data))

        thresh = int(stub_length_threshold)
        edges_to_remove: list = []
        for _, edges in seg_edges.items():
            boundary_nodes: set = set()
            for u, v, _data in edges:
                if graph.nodes[u].get("node_type") in ("endpoint", "branchpoint"):
                    boundary_nodes.add(u)
                if graph.nodes[v].get("node_type") in ("endpoint", "branchpoint"):
                    boundary_nodes.add(v)
            has_endpoint = any(
                graph.nodes[n].get("node_type") == "endpoint" for n in boundary_nodes
            )
            if not has_endpoint:
                continue
            total_length = sum(
                len(data.get("path", [u, v])) - 1 for u, v, data in edges
            )
            if total_length < thresh:
                edges_to_remove.extend((u, v) for u, v, _ in edges)

        graph.remove_edges_from(edges_to_remove)
        graph.remove_nodes_from(list(nx.isolates(graph)))

        # Trimming one branch off a junction leaves a degree-2 node behind, and
        # the skeleton builder's own merge ran long before this point. Re-merge
        # so the editor only ever shows endpoints and branch points — a degree-2
        # node is just a waypoint in the middle of a segment.
        graph = TopologyBuilder().merge_middle_points(graph)

        # Re-detect segments since topology changed.
        graph = seg_detector.detect_segments(graph)
        return self._new_handle(graph)

    def stage_count(
        self,
        reconstructed_graph: str,
        mask: str,
        annot_labeled: str,
        min_tree_components: int,
    ) -> dict:
        """Counting stage: region label → exclude small subtrees → count.

        Designed to be runnable on a USER-EDITED graph (use ``import_graph`` to
        register the edited graph as a handle first). Segments are re-detected
        defensively in case the input graph lacks segment_id attributes.

        Tags each edge with ``is_effective_segment`` so the UI can render
        effective crossing segments in green.
        """
        self._bump("count")
        graph = self.handles[reconstructed_graph].copy()
        mask_arr = self.handles[mask]
        annot_arr = self.handles[annot_labeled]

        # Defensive re-segmentation: edits in the UI invalidate segment_ids.
        graph = SegmentDetector().detect_segments(graph)

        labeled, _ = RegionLabeler().label_topology(graph, mask_arr)

        _exclude_small_subtrees_from_count(
            labeled,
            annot_arr,
            min_tree_components=int(min_tree_components),
            neighborhood=3,
        )

        details = CrossingCounter().count_effective_crossings(
            labeled, epidermis_mask=mask_arr
        )
        valid_seg_ids = {
            d["segment_id"] for d in details["segment_details"] if d["is_valid"]
        }
        for _u, _v, data in labeled.edges(data=True):
            sid = data.get("segment_id")
            data["is_effective_segment"] = sid is not None and sid in valid_seg_ids

        subtree_lengths = _tag_and_measure_subtrees(labeled)

        return {
            "labeled_graph": self._new_handle(labeled),
            "pred_count": int(details["effective_crossing_count"]),
            "subtree_lengths": subtree_lengths,
        }

    def import_graph(self, nodes: list, edges: list) -> str:
        """Reconstruct an nx.Graph from the JSON form emitted by extract_graph.

        Used to ingest a user-edited graph for re-counting. Node identity is
        ``(y, x)`` to match the rest of the pipeline; per-edge ``path`` is
        restored if provided, otherwise a 2-point fallback is used.
        """
        g = nx.Graph()
        for n in nodes:
            key = (int(n["y"]), int(n["x"]))
            g.add_node(key, **(n.get("attrs") or {}))
        for e in edges:
            u = (int(e["u"][0]), int(e["u"][1]))
            v = (int(e["v"][0]), int(e["v"][1]))
            if u not in g:
                g.add_node(u)
            if v not in g:
                g.add_node(v)
            attrs = dict(e.get("attrs") or {})
            raw_path = e.get("path")
            if raw_path:
                attrs["path"] = [(int(p[0]), int(p[1])) for p in raw_path]
            else:
                attrs["path"] = [u, v]
            g.add_edge(u, v, **attrs)
        return self._new_handle(g)

    def stage_labeled_graph(
        self,
        result_graph: str,
        mask: str,
        annot_labeled: str,
        min_tree_components: int,
        stub_length_threshold: int,
    ) -> dict:
        """Crossing analysis. Copies result_graph because the function mutates.

        Also tags each edge with ``is_effective_segment``: True iff the edge's
        segment passed the full validity test (≥1 crossing edge + epidermis and
        dermis lengths meet the threshold). This drives the green colouring of
        effective crossing segments in the Electron UI, mirroring
        ``effective_ids`` in ``tools/viz/viz_crossing.py``.
        """
        self._bump("labeled_graph")
        rg = self.handles[result_graph].copy()
        mask_arr = self.handles[mask]
        pred_count, labeled = run_crossing_analysis(
            rg,
            mask_arr,
            self.handles[annot_labeled],
            min_tree_components=int(min_tree_components),
            stub_length_threshold=int(stub_length_threshold),
        )
        # Re-run the per-segment validity check to recover which segment_ids
        # were counted as effective; run_crossing_analysis discards the
        # per-segment details and only returns the aggregate count.
        details = CrossingCounter().count_effective_crossings(
            labeled, epidermis_mask=mask_arr
        )
        valid_seg_ids = {
            d["segment_id"] for d in details["segment_details"] if d["is_valid"]
        }
        for _u, _v, data in labeled.edges(data=True):
            sid = data.get("segment_id")
            data["is_effective_segment"] = sid is not None and sid in valid_seg_ids

        subtree_lengths = _tag_and_measure_subtrees(labeled)

        return {
            "labeled_graph": self._new_handle(labeled),
            "pred_count": int(pred_count),
            "subtree_lengths": subtree_lengths,
        }

    # ── Inspection / lifecycle ──────────────────────────────────────────
    def stats(self) -> dict:
        return {"calls": dict(self.stage_calls), "handles": len(self.handles)}

    def reset_stats(self) -> int:
        n = sum(self.stage_calls.values())
        self.stage_calls.clear()
        return n

    def free(self, handles: list) -> int:
        n = 0
        for h in handles:
            if h in self.handles:
                del self.handles[h]
                n += 1
        return n

    def summary(self, handle: str) -> dict:
        obj = self.handles[handle]
        if isinstance(obj, np.ndarray):
            return {
                "kind": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "min": float(obj.min()) if obj.size else None,
                "max": float(obj.max()) if obj.size else None,
            }
        if isinstance(obj, nx.Graph):
            return {
                "kind": "graph",
                "nodes": obj.number_of_nodes(),
                "edges": obj.number_of_edges(),
            }
        if isinstance(obj, tuple):
            return {"kind": "tuple", "len": len(obj)}
        return {"kind": type(obj).__name__}

    def extract_graph(self, handle: str) -> dict:
        """Serialize a NetworkX graph for transport over the RPC protocol.

        Graph convention (see tools/viz/viz_crossing.py):
          - Nodes are (y, x) integer tuples
          - Edges may carry a `path` attribute = list of (y, x) pixel tuples

        Returned shape:
          {
            "nodes": [{"y": int, "x": int, "attrs": {...}}, ...],
            "edges": [{"u": [y, x], "v": [y, x], "path": [[y, x], ...], "attrs": {...}}, ...],
          }
        """
        obj = self.handles[handle]
        if not isinstance(obj, nx.Graph):
            raise TypeError(
                f"handle {handle!r} is not a Graph (kind={type(obj).__name__})"
            )

        nodes = [
            {"y": int(n[0]), "x": int(n[1]), "attrs": _coerce(attrs)}
            for n, attrs in obj.nodes(data=True)
        ]

        edges = []
        for u, v, data in obj.edges(data=True):
            path = data.get("path", [])
            path_out = [[int(p[0]), int(p[1])] for p in path]
            other = {k: _coerce_value(val) for k, val in data.items() if k != "path"}
            edges.append(
                {
                    "u": [int(u[0]), int(u[1])],
                    "v": [int(v[0]), int(v[1])],
                    "path": path_out,
                    "attrs": other,
                }
            )

        return {"nodes": nodes, "edges": edges}

    def render_handle_png(self, handle: str) -> str:
        """Serialize a 2D ndarray handle as a base64 PNG data URL for display.

        Used by the Electron UI to show intermediate stage outputs (currently
        the ROI mask). Non-uint8 arrays are min-max stretched to 0..255.
        """
        arr = self.handles[handle]
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"handle {handle!r} is not an ndarray")
        if arr.ndim != 2:
            raise ValueError(f"expected 2D array, got shape {tuple(arr.shape)}")
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            amin, amax = float(a.min()), float(a.max())
            if amax > amin:
                a = (a - amin) / (amax - amin) * 255.0
            arr = a.astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def ping(self) -> str:
        return "pong"


def main() -> None:
    worker = StageWorker()
    # Public methods (not underscore-prefixed) are RPC-callable
    methods = {
        name: getattr(worker, name)
        for name in dir(worker)
        if not name.startswith("_") and callable(getattr(worker, name))
    }
    log.info("worker ready, %d methods", len(methods))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            if method not in methods:
                _send(
                    {
                        "id": req_id,
                        "error": {
                            "type": "MethodNotFound",
                            "message": f"unknown method {method!r}",
                        },
                    }
                )
                continue
            try:
                result = methods[method](**params)
                _send({"id": req_id, "result": result})
            except Exception as e:  # noqa: BLE001 — RPC barrier
                _send(
                    {
                        "id": req_id,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    }
                )
        except Exception as e:  # noqa: BLE001 — parse/dispatch barrier
            _send(
                {
                    "id": req_id,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                }
            )


if __name__ == "__main__":
    main()
