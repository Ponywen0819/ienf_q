"""
tools/grid_search/staged_grid_search.py

Annotation-Grow grid search with per-stage in-memory caching.

Each pipeline stage output is cached by (sample_id, stage, *param_values).
When iterating over parameter combinations, upstream stage outputs are reused
if their governing parameters have not changed.

Stage dependency chain:
  roi_mask       ← offset_px
  annot_comp     ← offset_px
  bg_removed     ← offset_px, bg_kernel_size
  clahe_applied  ← offset_px, bg_kernel_size, clahe_clip, clahe_grid
  sato_per_sigma ← clahe_p + sigma   (raw per-sigma vesselness, reused across ranges)
  roi_image      ← offset_px, bg_kernel_size, clahe_clip, clahe_grid, sato_sigmas_{start,stop}
  cost_map       ← (same as roi_image)
  dijkstra       ← (roi_image params) + connectivity
  comp_graph     ← (same as dijkstra)
  pruned_graph   ← (dijkstra params) + prune_threshold
  mst            ← (same as pruned_graph)
  result_graph   ← (pruned_graph params) + segment_length
  roi_gt         ← offset_px  (GT label masked to ROI, for metrics)
  gt_points      ← offset_px  (extracted GT topology points, for HD95)

Example:
  python tools/grid_search/staged_grid_search.py \\
    --data-dir data/ \\
    --output-dir output/staged_gs \\
    --param-grid '{"clahe_grid": [[16,16],[32,32],[64,64]], "clahe_clip": [10.0, 20.0, 40.0]}' \\
    --sort-by hd95_mean
"""

import argparse
import csv
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from PIL import Image
from tqdm import tqdm

from neural_reconstruction.algorithms.annotation_grow.cost_map import (
    apply_within_mask_strips,
    build_cost_map,
)
from neural_reconstruction.algorithms.annotation_grow.dijkstra import (
    get_components,
    multi_source_dijkstra,
)
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (
    find_meeting_points,
    build_component_graph,
    prune_edges,
    minimum_spanning_forest,
)
from neural_reconstruction.algorithms.annotation_grow.skeleton import build_result_graph
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.evaluation import (
    extract_graph_points,
    compute_hd95,
    compute_cldice,
)
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.dataset import DatasetLoader, SampleFiles


# ── Default parameters ───────────────────────────────────────────────────────

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    # "bg_kernel_size": [0,3,5,7,9,11],
    # "clahe_grid": [(s, s) for s in [704, 736, 768, 800, 832]],
    # "clahe_clip": [10.0, 20.0, 30.0, 40.0, 50.0],
    # "sato_sigmas_start": [i for i in range(1, 6) ],
    # "sato_sigmas_stop": [i+1 for i in range(1, 6)],
    "prune_threshold": [10.0, 20.0, 30.0, 40.0, 50.0],
}

FIXED_PARAMS: Dict[str, Any] = {
    "offset_px": 50,
    "bg_kernel_size": 5,
    "clahe_clip": 30.0,
    "clahe_grid": (768, 768),
    "sato_sigmas_start": 1,
    "sato_sigmas_stop": 4,
    "connectivity": 8,
    "prune_threshold": 20.0,
    "segment_length": 100.0,
}

CLDICE_TOLERANCE_PX = 1

SORT_KEYS = ("hd95_mean", "cldice_mean", "tprec_mean", "tsens_mean")
ASC_METRICS = {"hd95_mean"}


def _is_valid_combination(params: Dict[str, Any]) -> bool:
    """Return False for parameter combos that should be skipped entirely.

    Constraints (extend as needed):
      - sato_sigmas_start < sato_sigmas_stop
        (sigmas = range(start, stop), so start ≥ stop yields an empty range)
    """
    start = params.get("sato_sigmas_start")
    stop = params.get("sato_sigmas_stop")
    if start is not None and stop is not None and start >= stop:
        return False
    return True


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class SampleData:
    sample_id: str
    green: np.ndarray
    mask: np.ndarray
    annotation: np.ndarray
    gt_label: Optional[np.ndarray]
    px_um_ratio: Optional[float] = None  # um per pixel; if set, HD95 is scaled to um


@dataclass
class SampleMetrics:
    hd95: Optional[float]
    cldice: Optional[float]
    tprec: Optional[float]
    tsens: Optional[float]


@dataclass
class GridSearchResult:
    params: Dict[str, Any]
    hd95_mean: Optional[float]
    hd95_median: Optional[float]
    hd95_std: Optional[float]
    hd95_min: Optional[float]
    hd95_max: Optional[float]
    hd95_n: int
    cldice_mean: Optional[float]
    cldice_median: Optional[float]
    cldice_std: Optional[float]
    cldice_min: Optional[float]
    cldice_max: Optional[float]
    cldice_n: int
    tprec_mean: Optional[float]
    tsens_mean: Optional[float]
    num_success: int
    num_skipped: int
    num_failed: int


# ── Stage cache ──────────────────────────────────────────────────────────────


class StageCache:
    """
    In-memory cache keyed by (sample_id, stage_name, *param_scalars).
    Tracks hits and misses per stage for reporting.
    """

    def __init__(self):
        self._data: Dict[tuple, Any] = {}
        self._hits: Dict[str, int] = {}
        self._misses: Dict[str, int] = {}

    def get(self, key: tuple) -> Tuple[bool, Any]:
        stage = key[1]
        if key in self._data:
            self._hits[stage] = self._hits.get(stage, 0) + 1
            return True, self._data[key]
        self._misses[stage] = self._misses.get(stage, 0) + 1
        return False, None

    def put(self, key: tuple, value: Any) -> None:
        self._data[key] = value

    def report(self) -> str:
        stages = sorted(set(list(self._hits) + list(self._misses)))
        lines = [f"Stage cache ({len(self._data)} entries):"]
        for stage in stages:
            h = self._hits.get(stage, 0)
            m = self._misses.get(stage, 0)
            total = h + m
            pct = h / total * 100 if total > 0 else 0.0
            lines.append(f"  {stage:<16s}: {h:>5}/{total:>5} hits ({pct:5.1f}%)")
        return "\n".join(lines)


def _cached(cache: StageCache, key: tuple, fn) -> Any:
    hit, value = cache.get(key)
    if hit:
        return value
    value = fn()
    cache.put(key, value)
    return value


# ── Pipeline stages ──────────────────────────────────────────────────────────


def _run_staged_pipeline(
    cache: StageCache,
    sample: SampleData,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, nx.Graph]:
    """
    Execute all pipeline stages for one sample, reusing cached intermediate
    outputs wherever the governing parameters match a prior computation.

    Returns:
        (roi_mask, result_graph)
    """
    sid = sample.sample_id
    offset_px: int = params["offset_px"]
    bg_kernel_size: int = params["bg_kernel_size"]
    clahe_clip: float = params["clahe_clip"]
    clahe_grid: Tuple[int, int] = tuple(params["clahe_grid"])
    sato_start: int = params["sato_sigmas_start"]
    sato_stop: int = params["sato_sigmas_stop"]
    connectivity: int = params["connectivity"]
    prune_threshold: float = params["prune_threshold"]
    segment_length: float = params["segment_length"]

    # Cumulative param tuples for each stage boundary
    mask_p = (offset_px,)
    bg_p = (offset_px, bg_kernel_size)
    clahe_p = bg_p + (clahe_clip, clahe_grid[0], clahe_grid[1])
    preproc_p = clahe_p + (sato_start, sato_stop)
    dijkstra_p = preproc_p + (connectivity,)
    pruned_p = dijkstra_p + (prune_threshold,)
    result_p = pruned_p + (segment_length,)

    # Stage: roi_mask
    roi_mask: np.ndarray = _cached(
        cache,
        (sid, "roi_mask") + mask_p,
        lambda: dilate_epidermis_vertically(sample.mask, offset_px=offset_px),
    )

    # Stage: annot_comp — (annot_labeled, annotation_bin, n_components)
    annot_labeled, annotation_bin, n_components = _cached(
        cache,
        (sid, "annot_comp") + mask_p,
        lambda: _compute_annot_comp(sample.annotation, roi_mask),
    )

    # Stage: bg_removed — background subtraction, masked to ROI
    bg_removed: np.ndarray = _cached(
        cache,
        (sid, "bg_removed") + bg_p,
        lambda: _apply_bg_removal(sample.green, roi_mask, bg_kernel_size),
    )

    # Stage: clahe_applied — contrast enhancement
    clahe_applied: np.ndarray = _cached(
        cache,
        (sid, "clahe_applied") + clahe_p,
        lambda: _apply_clahe(bg_removed, clahe_clip, clahe_grid),
    )

    # Stage: sato_per_sigma — raw (unnormalised) Sato response per single sigma.
    # Cached individually so that overlapping (start, stop) ranges reuse work:
    # range(1,6), range(1,7), range(2,6)… all share their per-sigma responses.
    per_sigma_responses = [
        _cached(
            cache,
            (sid, "sato_per_sigma") + clahe_p + (sigma,),
            lambda s=sigma: _apply_sato_single(clahe_applied, roi_mask, s),
        )
        for sigma in range(sato_start, sato_stop)
    ]

    # Stage: roi_image — element-wise max across sigmas + min-max → uint8
    roi_image: np.ndarray = _cached(
        cache,
        (sid, "roi_image") + preproc_p,
        lambda: _combine_and_normalise_sato(per_sigma_responses),
    )

    # Stage: cost_map
    cost_map: np.ndarray = _cached(
        cache,
        (sid, "cost_map") + preproc_p,
        lambda: build_cost_map(roi_image),
    )

    # Stage: dijkstra — (owner_map, dist_map, prev_y, prev_x)
    owner_map, dist_map, prev_y, prev_x = _cached(
        cache,
        (sid, "dijkstra") + dijkstra_p,
        lambda: multi_source_dijkstra(
            cost_map,
            annot_labeled,
            connectivity=connectivity,
            roi_mask=(roi_mask > 127),
        ),
    )

    # Stage: comp_graph
    G: nx.Graph = _cached(
        cache,
        (sid, "comp_graph") + dijkstra_p,
        lambda: _compute_comp_graph(owner_map, dist_map, prev_y, prev_x, n_components),
    )

    # Stage: pruned_graph
    G_pruned: nx.Graph = _cached(
        cache,
        (sid, "pruned_graph") + pruned_p,
        lambda: prune_edges(G, threshold=prune_threshold),
    )

    # Stage: mst
    mst: nx.Graph = _cached(
        cache,
        (sid, "mst") + pruned_p,
        lambda: minimum_spanning_forest(G_pruned),
    )

    # Stage: result_graph
    result_graph: nx.Graph = _cached(
        cache,
        (sid, "result_graph") + result_p,
        lambda: build_result_graph(mst, annotation_bin, segment_length=segment_length),
    )

    return roi_mask, result_graph


def _apply_bg_removal(
    green: np.ndarray, roi_mask: np.ndarray, bg_kernel_size: int
) -> np.ndarray:
    """Background subtraction via morphological opening, masked to ROI.

    Morphology is strip-cropped to ROI: opening = erosion → dilation, so the
    halo is 2× the kernel radius (= bg_kernel_size).
    """
    if bg_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size)
        )

        def _op(patch: np.ndarray) -> np.ndarray:
            bg = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
            return cv2.subtract(patch, bg)

        corrected = apply_within_mask_strips(green, roi_mask, _op, pad=bg_kernel_size)
    else:
        corrected = green
    return cv2.bitwise_and(corrected, corrected, mask=roi_mask)


def _apply_clahe(
    bg_removed: np.ndarray, clahe_clip: float, clahe_grid: Tuple[int, int]
) -> np.ndarray:
    """CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    return clahe.apply(bg_removed)


def _apply_sato_single(
    img: np.ndarray, roi_mask: np.ndarray, sigma: int
) -> np.ndarray:
    """Raw Sato vesselness response at one sigma (no normalisation).

    Strip-cropped to the ROI: per-sigma padding 4·σ + 4 keeps inside-ROI
    pixels byte-identical to a full-image call after uint8 quantisation.
    """
    import skimage as ski

    pad = int(np.ceil(4 * sigma)) + 4
    return apply_within_mask_strips(
        img,
        roi_mask,
        lambda patch: ski.filters.sato(
            patch, sigmas=range(sigma, sigma + 1), black_ridges=False
        ),
        pad=pad,
    )


def _combine_and_normalise_sato(per_sigma_responses: List[np.ndarray]) -> np.ndarray:
    """Element-wise max across per-sigma Sato responses, then min-max → uint8."""
    result = np.maximum.reduce(per_sigma_responses)
    vmin, vmax = result.min(), result.max()
    if vmax > vmin:
        result = (result - vmin) / (vmax - vmin) * 255
    return result.astype(np.uint8)


def _compute_annot_comp(
    annotation: np.ndarray, roi_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
    annotation_bin = (roi_annotation > 127).astype(np.uint8)
    annot_labeled = get_components(annotation_bin)
    return annot_labeled, annotation_bin, int(annot_labeled.max())


def _compute_comp_graph(
    owner_map: np.ndarray,
    dist_map: np.ndarray,
    prev_y: np.ndarray,
    prev_x: np.ndarray,
    n_components: int,
) -> nx.Graph:
    connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
    return build_component_graph(connections, n_components)


# ── GT topology helpers ──────────────────────────────────────────────────────


def _get_roi_gt(
    cache: StageCache,
    sid: str,
    gt_label: np.ndarray,
    roi_mask: np.ndarray,
    offset_px: int,
) -> np.ndarray:
    """GT label restricted to ROI (cached per sample × offset_px)."""
    return _cached(
        cache,
        (sid, "roi_gt", offset_px),
        lambda: cv2.bitwise_and(
            gt_label[:, :, 0] if gt_label.ndim == 3 else gt_label,
            gt_label[:, :, 0] if gt_label.ndim == 3 else gt_label,
            mask=roi_mask,
        ),
    )


def _get_gt_points(
    cache: StageCache, sid: str, roi_gt: np.ndarray, offset_px: int
) -> np.ndarray:
    """Extracted GT skeleton points for HD95 (cached per sample × offset_px)."""

    def compute():
        graph_gt = TopologyBuilder().build_seed_graph(roi_gt)
        return extract_graph_points(graph_gt)

    return _cached(cache, (sid, "gt_points", offset_px), compute)


# ── Per-sample evaluation ────────────────────────────────────────────────────


def _evaluate_sample(
    cache: StageCache,
    sample: SampleData,
    params: Dict[str, Any],
    cldice_tolerance_px: int,
    logger: logging.Logger,
) -> Optional[SampleMetrics]:
    """
    Returns None if the sample should be skipped (missing files or GT label).
    Returns SampleMetrics with None fields on per-metric failure.
    """
    if sample.gt_label is None:
        return None

    try:
        roi_mask, result_graph = _run_staged_pipeline(cache, sample, params)
    except Exception as e:
        logger.debug(f"Sample {sample.sample_id} pipeline failed: {e}", exc_info=True)
        return SampleMetrics(hd95=None, cldice=None, tprec=None, tsens=None)

    offset_px: int = params["offset_px"]
    roi_gt = _get_roi_gt(cache, sample.sample_id, sample.gt_label, roi_mask, offset_px)

    hd95: Optional[float] = None
    try:
        gt_pts = _get_gt_points(cache, sample.sample_id, roi_gt, offset_px)
        pred_pts = extract_graph_points(result_graph)
        hd95_val, _, _ = compute_hd95(pred_pts, gt_pts)
        hd95 = (
            hd95_val * sample.px_um_ratio
            if sample.px_um_ratio is not None
            else hd95_val
        )
    except Exception as e:
        logger.debug(f"Sample {sample.sample_id} HD95 failed: {e}", exc_info=True)

    cldice: Optional[float] = None
    tprec: Optional[float] = None
    tsens: Optional[float] = None
    try:
        cld, tp, ts = compute_cldice(
            result_graph, roi_gt, tolerance_px=cldice_tolerance_px
        )
        cldice, tprec, tsens = cld, tp, ts
    except Exception as e:
        logger.debug(f"Sample {sample.sample_id} clDice failed: {e}", exc_info=True)

    return SampleMetrics(hd95=hd95, cldice=cldice, tprec=tprec, tsens=tsens)


# ── Grid search runner ───────────────────────────────────────────────────────


class GridSearchRunner:
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        param_grid: Dict[str, List[Any]],
        fixed_params: Dict[str, Any],
        sort_by: str = "hd95_mean",
        sample_ids: Optional[List[str]] = None,
        cldice_tolerance_px: int = CLDICE_TOLERANCE_PX,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.param_grid = param_grid
        self.fixed_params = dict(fixed_params)
        self.sort_by = sort_by
        self.cldice_tolerance_px = cldice_tolerance_px
        self.logger = logging.getLogger(__name__)

        self.px_um_ratios = self._load_px_um_ratios(data_dir)
        raw_samples = DatasetLoader(data_dir).load_samples(sample_ids)
        self.samples = self._load_samples(raw_samples)

        n_with_label = sum(1 for s in self.samples if s.gt_label is not None)
        self.logger.info(
            f"Loaded {len(self.samples)} samples ({n_with_label} with GT label)"
        )
        if n_with_label == 0:
            raise RuntimeError(
                "No samples with label.png — HD95 and clDice both require GT."
            )

    @staticmethod
    def _load_px_um_ratios(data_dir: Path) -> Dict[str, float]:
        px_um_path = Path(data_dir) / "px_um.json"
        if not px_um_path.exists():
            logging.getLogger(__name__).warning(
                f"px_um.json not found at {px_um_path} — HD95 will remain in pixels"
            )
            return {}
        with open(px_um_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_samples(self, raw: List[SampleFiles]) -> List[SampleData]:
        samples = []
        for s in raw:
            ok, reason = s.is_complete()
            if not ok:
                self.logger.warning(f"Skipping {s.sample_id}: {reason}")
                continue
            image = np.array(Image.open(s.image_path))
            green = image[:, :, 1] if image.ndim == 3 else image
            mask = np.array(Image.open(s.mask_path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            annotation = np.array(Image.open(s.annotation_path))
            if annotation.ndim == 3:
                annotation = annotation[:, :, 0]
            gt_label = None
            if s.label_path and s.label_path.exists():
                gt_label = np.array(Image.open(s.label_path))
            ratio = self.px_um_ratios.get(s.sample_id)
            if ratio is None and self.px_um_ratios:
                self.logger.warning(
                    f"Sample {s.sample_id} not in px_um.json — HD95 will be in pixels"
                )
            samples.append(
                SampleData(
                    sample_id=s.sample_id,
                    green=green,
                    mask=mask,
                    annotation=annotation,
                    gt_label=gt_label,
                    px_um_ratio=ratio,
                )
            )
        return samples

    def run(self) -> List[GridSearchResult]:
        combinations = list(self._iter_combinations())
        n_combos = len(combinations)
        self.logger.info(
            f"Grid search: {len(self.samples)} samples × {n_combos} combinations"
        )

        # combo_metrics[i] collects one SampleMetrics per sample for combination i
        combo_metrics: List[List[Optional[SampleMetrics]]] = [
            [] for _ in range(n_combos)
        ]

        # Accumulated cache stats across all per-sample caches
        total_hits: Dict[str, int] = {}
        total_misses: Dict[str, int] = {}

        for sample in tqdm(self.samples, desc="Samples"):
            # Fresh cache per sample — discarded when this iteration ends
            cache = StageCache()

            for i, params in enumerate(combinations):
                metrics = _evaluate_sample(
                    cache, sample, params, self.cldice_tolerance_px, self.logger
                )
                combo_metrics[i].append(metrics)

            for stage, h in cache._hits.items():
                total_hits[stage] = total_hits.get(stage, 0) + h
            for stage, m in cache._misses.items():
                total_misses[stage] = total_misses.get(stage, 0) + m

        # Aggregate per-combination metrics collected above
        results = [
            self._aggregate_metrics(combinations[i], combo_metrics[i])
            for i in range(n_combos)
        ]

        self._save_per_combo_results(combinations, combo_metrics)

        for r in results:
            self._log_combination(r)

        results.sort(key=self._sort_key)
        self._save_results(results)
        self._print_top_results(results)
        self.logger.info("\n" + self._format_cache_report(total_hits, total_misses))
        return results

    @staticmethod
    def _format_cache_report(hits: Dict[str, int], misses: Dict[str, int]) -> str:
        stages = sorted(set(list(hits) + list(misses)))
        total_entries = sum(hits.values()) + sum(misses.values())
        lines = [f"Cache stats (all samples combined, {total_entries} total calls):"]
        for stage in stages:
            h = hits.get(stage, 0)
            m = misses.get(stage, 0)
            total = h + m
            pct = h / total * 100 if total > 0 else 0.0
            lines.append(f"  {stage:<16s}: {h:>6}/{total:>6} hits ({pct:5.1f}%)")
        return "\n".join(lines)

    def _save_per_combo_results(
        self,
        combinations: List[Dict[str, Any]],
        combo_metrics: List[List[Optional[SampleMetrics]]],
    ) -> None:
        """Save one JSON per parameter combination containing per-sample results."""
        combo_dir = self.output_dir / "per_combo"
        combo_dir.mkdir(exist_ok=True)

        for i, (params, metrics_list) in enumerate(zip(combinations, combo_metrics)):
            sample_rows = []
            for sample, metrics in zip(self.samples, metrics_list):
                if metrics is None:
                    row: Dict[str, Any] = {
                        "sample_id": sample.sample_id,
                        "status": "skipped",
                    }
                elif metrics.hd95 is None and metrics.cldice is None:
                    row = {"sample_id": sample.sample_id, "status": "failed"}
                else:
                    row = {
                        "sample_id": sample.sample_id,
                        "status": "success",
                        "hd95": metrics.hd95,
                        "cldice": metrics.cldice,
                        "tprec": metrics.tprec,
                        "tsens": metrics.tsens,
                    }
                sample_rows.append(row)

            out = {
                "combo_index": i,
                "params": self._jsonable(params),
                "samples": sample_rows,
            }
            path = combo_dir / f"combo_{i:04d}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, default=str)

        self.logger.info(f"Per-combo results saved to {combo_dir}/")

    def _aggregate_metrics(
        self, params: Dict[str, Any], metrics_list: List[Optional[SampleMetrics]]
    ) -> GridSearchResult:
        hd95_vals: List[float] = []
        cldice_vals: List[float] = []
        tprec_vals: List[float] = []
        tsens_vals: List[float] = []
        num_success = num_skipped = num_failed = 0

        for metrics in metrics_list:
            if metrics is None:
                num_skipped += 1
                continue
            if metrics.hd95 is None and metrics.cldice is None:
                num_failed += 1
                continue
            num_success += 1
            if metrics.hd95 is not None:
                hd95_vals.append(metrics.hd95)
            if metrics.cldice is not None:
                cldice_vals.append(metrics.cldice)
            if metrics.tprec is not None:
                tprec_vals.append(metrics.tprec)
            if metrics.tsens is not None:
                tsens_vals.append(metrics.tsens)

        def _stats(vals: List[float]):
            if not vals:
                return None, None, None, None, None
            a = np.array(vals)
            return (
                float(a.mean()),
                float(np.median(a)),
                float(a.std()),
                float(a.min()),
                float(a.max()),
            )

        h_mn, h_md, h_sd, h_mi, h_mx = _stats(hd95_vals)
        c_mn, c_md, c_sd, c_mi, c_mx = _stats(cldice_vals)

        return GridSearchResult(
            params=params,
            hd95_mean=h_mn,
            hd95_median=h_md,
            hd95_std=h_sd,
            hd95_min=h_mi,
            hd95_max=h_mx,
            hd95_n=len(hd95_vals),
            cldice_mean=c_mn,
            cldice_median=c_md,
            cldice_std=c_sd,
            cldice_min=c_mi,
            cldice_max=c_mx,
            cldice_n=len(cldice_vals),
            tprec_mean=float(np.mean(tprec_vals)) if tprec_vals else None,
            tsens_mean=float(np.mean(tsens_vals)) if tsens_vals else None,
            num_success=num_success,
            num_skipped=num_skipped,
            num_failed=num_failed,
        )

    def _iter_combinations(self):
        keys = list(self.param_grid.keys())
        skipped = 0
        for values in itertools.product(*self.param_grid.values()):
            params = dict(self.fixed_params)
            params.update(dict(zip(keys, values)))
            # Normalise clahe_grid to tuple so it is hashable
            if "clahe_grid" in params and isinstance(params["clahe_grid"], list):
                params["clahe_grid"] = tuple(params["clahe_grid"])
            if not _is_valid_combination(params):
                skipped += 1
                continue
            yield params
        if skipped:
            self.logger.info(
                f"Skipped {skipped} invalid combination(s) "
                f"(e.g. sato_sigmas_start >= sato_sigmas_stop)"
            )

    def _sort_key(self, r: GridSearchResult):
        v = getattr(r, self.sort_by)
        if v is None:
            return float("inf")
        return v if self.sort_by in ASC_METRICS else -v

    def _log_combination(self, result: GridSearchResult):
        grid_keys = list(self.param_grid.keys())
        param_str = ", ".join(f"{k}={result.params[k]}" for k in grid_keys)
        h = f"{result.hd95_mean:.4f}" if result.hd95_mean is not None else "N/A"
        c = f"{result.cldice_mean:.4f}" if result.cldice_mean is not None else "N/A"
        self.logger.debug(f"  [{param_str}] → hd95={h}  cldice={c}")

    # ── Output ───────────────────────────────────────────────────────────────

    @staticmethod
    def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @classmethod
    def _result_to_dict(cls, r: GridSearchResult) -> Dict[str, Any]:
        row = cls._jsonable(r.params)
        row.update(
            hd95_mean=r.hd95_mean,
            hd95_median=r.hd95_median,
            hd95_std=r.hd95_std,
            hd95_min=r.hd95_min,
            hd95_max=r.hd95_max,
            hd95_n=r.hd95_n,
            cldice_mean=r.cldice_mean,
            cldice_median=r.cldice_median,
            cldice_std=r.cldice_std,
            cldice_min=r.cldice_min,
            cldice_max=r.cldice_max,
            cldice_n=r.cldice_n,
            tprec_mean=r.tprec_mean,
            tsens_mean=r.tsens_mean,
            num_success=r.num_success,
            num_skipped=r.num_skipped,
            num_failed=r.num_failed,
        )
        return row

    def _save_results(self, results: List[GridSearchResult]):
        json_path = self.output_dir / "grid_search_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "param_grid": self._jsonable(self.param_grid),
                    "fixed_params": self._jsonable(self.fixed_params),
                    "sort_by": self.sort_by,
                    "num_samples": len(self.samples),
                    "cldice_tolerance_px": self.cldice_tolerance_px,
                    "results": [self._result_to_dict(r) for r in results],
                },
                f,
                indent=2,
                default=str,
            )
        self.logger.info(f"Saved: {json_path}")

        csv_path = self.output_dir / "grid_search_results.csv"
        if results:
            fieldnames = list(self._result_to_dict(results[0]).keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(self._result_to_dict(r))
        self.logger.info(f"Saved: {csv_path}")

    def _print_top_results(self, results: List[GridSearchResult], top_n: int = 10):
        valid = [r for r in results if getattr(r, self.sort_by) is not None]
        order = "asc" if self.sort_by in ASC_METRICS else "desc"
        print("\n" + "=" * 110)
        print(
            f"Staged Annotation-Grow Grid Search — Top {min(top_n, len(valid))} "
            f"(sorted by {self.sort_by}, {order})"
        )
        print("=" * 110)
        grid_keys = list(self.param_grid.keys())
        print(
            f"{'Rank':>4}  {'HD95':>8}  {'clDice':>8}  {'Tprec':>8}  {'Tsens':>8}  "
            f"{'n_h':>4}  {'n_c':>4}  Params"
        )
        print("-" * 110)

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A "

        for rank, r in enumerate(valid[:top_n], start=1):
            param_str = "  ".join(f"{k}={r.params[k]}" for k in grid_keys)
            print(
                f"{rank:>4}  {_fmt(r.hd95_mean):>8}  {_fmt(r.cldice_mean):>8}  "
                f"{_fmt(r.tprec_mean):>8}  {_fmt(r.tsens_mean):>8}  "
                f"{r.hd95_n:>4}  {r.cldice_n:>4}  {param_str}"
            )

        if valid:
            best = valid[0]
            print("\nBest configuration:")
            for k, v in best.params.items():
                print(f"  {k}: {v}")
            if best.hd95_mean is not None:
                print(f"  hd95_mean:   {best.hd95_mean:.4f}")
            if best.cldice_mean is not None:
                print(f"  cldice_mean: {best.cldice_mean:.4f}")

        print("=" * 110 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def setup_logging(output_dir: Path, verbose: bool):
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    fh = logging.FileHandler(output_dir / "staged_grid_search.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Staged Annotation-Grow grid search — reuses intermediate pipeline "
            "outputs across parameter combinations for faster evaluation."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-ids", nargs="+", help="Specific sample IDs (default: all)"
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        default=None,
        help=(
            'JSON param grid, e.g. \'{"clahe_grid": [[16,16],[32,32]], "clahe_clip": [10.0, 20.0]}\''
        ),
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default=None,
        help='JSON fixed params, e.g. \'{"offset_px": 50, "bg_kernel_size": 51}\'',
    )
    parser.add_argument(
        "--sort-by",
        choices=list(SORT_KEYS),
        default="hd95_mean",
    )
    parser.add_argument(
        "--cldice-tolerance",
        type=int,
        default=CLDICE_TOLERANCE_PX,
        help=f"clDice tolerance radius in pixels (default: {CLDICE_TOLERANCE_PX})",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    param_grid = json.loads(args.param_grid) if args.param_grid else DEFAULT_PARAM_GRID
    fixed_params = json.loads(args.fixed_params) if args.fixed_params else FIXED_PARAMS

    # Normalise any list-form clahe_grid in the param grid
    if "clahe_grid" in param_grid:
        param_grid["clahe_grid"] = [
            tuple(v) if isinstance(v, list) else v for v in param_grid["clahe_grid"]
        ]
    if "clahe_grid" in fixed_params and isinstance(fixed_params["clahe_grid"], list):
        fixed_params["clahe_grid"] = tuple(fixed_params["clahe_grid"])

    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)

    logger.info("=" * 80)
    logger.info("Staged Annotation-Grow Grid Search")
    logger.info("=" * 80)
    logger.info(f"Data dir:         {args.data_dir}")
    logger.info(f"Output dir:       {args.output_dir}")
    logger.info(f"Sort by:          {args.sort_by}")
    logger.info(f"clDice tolerance: {args.cldice_tolerance} px")
    logger.info(f"Param grid:       {param_grid}")
    logger.info(f"Fixed params:     {fixed_params}")
    logger.info(f"Combinations:     {n_combinations}")

    runner = GridSearchRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        param_grid=param_grid,
        fixed_params=fixed_params,
        sort_by=args.sort_by,
        sample_ids=args.sample_ids,
        cldice_tolerance_px=args.cldice_tolerance,
    )
    runner.run()

    logger.info("Done.")


if __name__ == "__main__":
    main()
