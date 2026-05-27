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
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
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
from neural_reconstruction.algorithms.annotation_grow.skeleton import build_result_graph  # noqa: E402
from neural_reconstruction.core.crosses_detection import run_crossing_analysis  # noqa: E402
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically  # noqa: E402


def _send(frame: dict) -> None:
    _STDOUT.write(json.dumps(frame) + "\n")
    _STDOUT.flush()


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
        self, mst: str, annotation_bin: str, segment_length: float
    ) -> str:
        self._bump("result_graph")
        return self._new_handle(
            build_result_graph(
                self.handles[mst],
                self.handles[annotation_bin],
                segment_length=float(segment_length),
            )
        )

    def stage_labeled_graph(
        self,
        result_graph: str,
        mask: str,
        annot_labeled: str,
        min_tree_components: int,
        stub_length_threshold: int,
    ) -> dict:
        """Crossing analysis. Copies result_graph because the function mutates."""
        self._bump("labeled_graph")
        rg = self.handles[result_graph].copy()
        pred_count, labeled = run_crossing_analysis(
            rg,
            self.handles[mask],
            self.handles[annot_labeled],
            min_tree_components=int(min_tree_components),
            stub_length_threshold=int(stub_length_threshold),
        )
        return {
            "labeled_graph": self._new_handle(labeled),
            "pred_count": int(pred_count),
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
