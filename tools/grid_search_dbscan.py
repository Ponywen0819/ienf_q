"""
DBSCAN Clustering Parameter Grid Search

Evaluates DBSCAN (eps, min_samples) combinations across all dataset samples.
For each combination, builds the annotation seed graph, clusters nodes, then
compares clusters against GT label components using:
  - V-measure  (harmonic mean of homogeneity + completeness)
  - Homogeneity
  - Completeness
  - Weighted cluster purity

Usage:
    # Default parameter grid
    python tools/grid_search_dbscan.py \
        --data-dir data_0331 \
        --output-dir output/grid_search_dbscan

    # Custom parameter grid (JSON)
    python tools/grid_search_dbscan.py \
        --data-dir data_0331 \
        --output-dir output/grid_search_dbscan \
        --param-grid '{"eps": [5, 10, 20], "min_samples": [2, 3, 5]}'

    # Specific samples
    python tools/grid_search_dbscan.py \
        --data-dir data_0331 \
        --output-dir output/grid_search_dbscan \
        --sample-ids S222-2_a S222-2_b

    # Sort by purity instead of v_measure
    python tools/grid_search_dbscan.py \
        --data-dir data_0331 \
        --output-dir output/grid_search_dbscan \
        --sort-by purity_mean
"""

import argparse
import csv
import itertools
import json
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import skimage as ski
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from tqdm import tqdm

from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.dataset import DatasetLoader, SampleFiles


# ============================================================================
# Default parameter grid
# ============================================================================

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "eps": [3, 5, 10, 15, 20],
    "min_samples": [2, 3, 5, 7, 10],
}

SORT_KEYS = (
    "v_measure_mean",
    "homogeneity_mean",
    "completeness_mean",
    "purity_mean",
    "count_mae_mean",
)


# ============================================================================
# Preprocessing helpers
# ============================================================================


def preprocess_sample(
    image_path: Path,
    mask_path: Path,
    annotation_path: Path,
    label_path: Path,
    offset_px: int = 50,
    opening_kernel_size: int = 51,
    closing_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicate the notebook preprocessing pipeline.

    Returns:
        roi_image       - background-subtracted ROI image (H, W) uint8
        roi_annotation  - cleaned annotation within ROI (H, W) uint8
        roi_label       - GT label within ROI (H, W) uint8
        roi_mask        - dilated epidermis mask (H, W) uint8
    """
    image = np.array(Image.open(image_path).convert("RGB"))[:, :, 1]  # green channel
    mask = np.array(Image.open(mask_path).convert("L"))
    annotation = np.array(Image.open(annotation_path).convert("L"))
    label = np.array(Image.open(label_path).convert("L"))

    roi_mask = dilate_epidermis_vertically(mask, offset_px=offset_px)

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size)
    )
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
    image = cv2.subtract(image, background)

    roi_image = cv2.bitwise_and(image, image, mask=roi_mask)
    roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_annotation = cv2.morphologyEx(
        roi_annotation, cv2.MORPH_CLOSE, kernel_close, iterations=closing_iterations
    )
    roi_annotation[roi_annotation > 0] = 255

    roi_label = cv2.bitwise_and(label, label, mask=roi_mask)
    roi_label = cv2.morphologyEx(
        roi_label, cv2.MORPH_CLOSE, kernel_close, iterations=closing_iterations
    )

    return roi_image, roi_annotation, roi_label, roi_mask


# ============================================================================
# Evaluation helpers
# ============================================================================


def evaluate_dbscan(
    nodes: List[Tuple[float, float]],
    cluster_labels: np.ndarray,
    gt_fiber_label: np.ndarray,
    n_gt_fibers: int,
) -> Dict[str, float]:
    """
    Compute cluster quality metrics against GT fiber labels.

    Args:
        nodes:          List of (y, x) node coordinates.
        cluster_labels: DBSCAN cluster assignments (-1 = noise).
        gt_fiber_label: Labeled GT image (H, W), 0 = background.
        n_gt_fibers:    Number of connected components in GT label image.

    Returns:
        Dict with keys: purity, homogeneity, completeness, v_measure,
                        count_mae, n_clusters, n_noise, noise_ratio.

    Notes:
        - Purity includes ALL clustered nodes (including those on background).
        - Homogeneity/completeness/V-measure exclude nodes that fall on
          GT background (GT ID = 0) — these nodes don't correspond to any
          fiber so including them creates a spurious majority class and
          inflates/deflates the information-theoretic scores.
        - count_mae = |n_clusters - n_gt_fibers| is the most direct metric
          since we have the GT fiber count from label.png.
    """
    node_gt_ids = np.array([int(gt_fiber_label[int(y), int(x)]) for y, x in nodes])
    valid_mask = cluster_labels != -1  # non-noise
    n_noise = int(np.sum(~valid_mask))
    n_total = len(nodes)
    n_clusters = int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))

    # Weighted purity (excluding noise; background nodes included as GT class 0)
    if valid_mask.sum() > 0:
        valid_cids = cluster_labels[valid_mask]
        valid_gt = node_gt_ids[valid_mask]
        pure_sum = 0.0
        for cid in set(valid_cids):
            mask_c = valid_cids == cid
            counter = Counter(valid_gt[mask_c])
            majority_count = counter.most_common(1)[0][1]
            pure_sum += majority_count
        purity = pure_sum / valid_mask.sum()
    else:
        purity = 0.0

    # sklearn metrics: exclude DBSCAN noise AND GT background nodes.
    # Background nodes (GT ID = 0) are annotation pixels that don't overlap
    # any GT fiber — treating them as a class pollutes the scores.
    on_fiber_mask = node_gt_ids > 0
    eval_mask = valid_mask & on_fiber_mask
    if eval_mask.sum() > 1 and len(set(cluster_labels[eval_mask])) > 1:
        hom = float(homogeneity_score(node_gt_ids[eval_mask], cluster_labels[eval_mask]))
        com = float(completeness_score(node_gt_ids[eval_mask], cluster_labels[eval_mask]))
        vme = float(v_measure_score(node_gt_ids[eval_mask], cluster_labels[eval_mask]))
    else:
        hom = com = vme = 0.0

    return {
        "purity": purity,
        "homogeneity": hom,
        "completeness": com,
        "v_measure": vme,
        "count_mae": float(abs(n_clusters - n_gt_fibers)),
        "n_clusters": n_clusters,
        "n_gt_fibers": n_gt_fibers,
        "n_noise": n_noise,
        "noise_ratio": n_noise / n_total if n_total > 0 else 0.0,
    }


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class SampleMetrics:
    """Per-sample metrics for one DBSCAN configuration."""

    purity: Optional[float]
    homogeneity: Optional[float]
    completeness: Optional[float]
    v_measure: Optional[float]
    count_mae: Optional[float]
    n_clusters: Optional[int]
    noise_ratio: Optional[float]


@dataclass
class GridSearchResult:
    """Aggregated result for one (eps, min_samples) combination."""

    params: Dict[str, Any]
    # aggregated metrics
    purity_mean: Optional[float]
    purity_std: Optional[float]
    homogeneity_mean: Optional[float]
    homogeneity_std: Optional[float]
    completeness_mean: Optional[float]
    completeness_std: Optional[float]
    v_measure_mean: Optional[float]
    v_measure_std: Optional[float]
    n_clusters_mean: Optional[float]
    noise_ratio_mean: Optional[float]
    # counts
    num_success: int
    num_skipped: int
    num_failed: int


# ============================================================================
# Per-combination evaluator
# ============================================================================


class CombinationEvaluator:
    """Evaluate one (eps, min_samples) combination across all samples."""

    def __init__(
        self,
        samples: List[SampleFiles],
        segment_length: float = 3.0,
        num_workers: Optional[int] = None,
    ):
        self.samples = samples
        self.segment_length = segment_length
        self.num_workers = num_workers
        self._local = threading.local()
        self.logger = logging.getLogger(__name__)

        # Pre-process and cache all samples (images + graphs)
        self.logger.info(f"Pre-processing {len(samples)} samples...")
        self._cache: Dict[str, Optional[Tuple]] = {}
        for sample in tqdm(samples, desc="Preprocessing"):
            self._cache[sample.sample_id] = self._load_sample(sample)
        loaded = sum(1 for v in self._cache.values() if v is not None)
        self.logger.info(f"Cached {loaded}/{len(samples)} samples")

    def _get_topology_builder(self) -> TopologyBuilder:
        if not hasattr(self._local, "tb"):
            self._local.tb = TopologyBuilder(segment_length=self.segment_length)
        return self._local.tb

    def _load_sample(self, sample: SampleFiles) -> Optional[Tuple[List, np.ndarray, int]]:
        """
        Preprocess one sample and build the annotation seed graph.

        Returns:
            (nodes, gt_fiber_label) or None if sample should be skipped.
        """
        ok, _ = sample.is_complete()
        if not ok:
            return None
        if sample.label_path is None or not sample.label_path.exists():
            return None

        try:
            roi_image, roi_annotation, roi_label, _ = preprocess_sample(
                image_path=sample.image_path,
                mask_path=sample.mask_path,
                annotation_path=sample.annotation_path,
                label_path=sample.label_path,
            )

            tb = TopologyBuilder(segment_length=self.segment_length)
            global_graph = tb.build_seed_graph(roi_annotation, roi_image)

            if global_graph.number_of_nodes() == 0:
                return None

            nodes = list(global_graph.nodes())
            gt_fiber_label: np.ndarray = np.array(
                ski.measure.label(roi_label > 0, connectivity=2)
            )
            n_gt_fibers = int(gt_fiber_label.max())

            return nodes, gt_fiber_label, n_gt_fibers

        except Exception as e:
            self.logger.debug(
                f"Sample {sample.sample_id} load failed: {e}", exc_info=True
            )
            return None

    def evaluate(self, params: Dict[str, Any]) -> GridSearchResult:
        eps = params["eps"]
        min_samples = params["min_samples"]

        metrics_list: List[SampleMetrics] = []
        num_skipped = 0
        num_failed = 0

        def _run(sample: SampleFiles) -> Optional[SampleMetrics]:
            cached = self._cache.get(sample.sample_id)
            if cached is None:
                return None
            nodes, gt_fiber_label, n_gt_fibers = cached
            try:
                positions = np.array([[y, x] for y, x in nodes])
                cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
                    positions
                )
                m = evaluate_dbscan(nodes, cluster_labels, gt_fiber_label, n_gt_fibers)
                return SampleMetrics(
                    purity=m["purity"],
                    homogeneity=m["homogeneity"],
                    completeness=m["completeness"],
                    v_measure=m["v_measure"],
                    count_mae=m["count_mae"],
                    n_clusters=int(m["n_clusters"]),
                    noise_ratio=m["noise_ratio"],
                )
            except Exception as e:
                self.logger.debug(
                    f"DBSCAN failed on {sample.sample_id}: {e}", exc_info=True
                )
                return SampleMetrics(
                    purity=None,
                    homogeneity=None,
                    completeness=None,
                    v_measure=None,
                    count_mae=None,
                    n_clusters=None,
                    noise_ratio=None,
                )

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_run, sample): sample for sample in self.samples}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    num_skipped += 1
                elif result.v_measure is None:
                    num_failed += 1
                else:
                    metrics_list.append(result)

        def _stats(values):
            if not values:
                return None, None
            arr = np.array(values, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        purities = [m.purity for m in metrics_list if m.purity is not None]
        homs = [m.homogeneity for m in metrics_list if m.homogeneity is not None]
        coms = [m.completeness for m in metrics_list if m.completeness is not None]
        vmes = [m.v_measure for m in metrics_list if m.v_measure is not None]
        ncls = [m.n_clusters for m in metrics_list if m.n_clusters is not None]
        nrs = [m.noise_ratio for m in metrics_list if m.noise_ratio is not None]

        p_mean, p_std = _stats(purities)
        h_mean, h_std = _stats(homs)
        c_mean, c_std = _stats(coms)
        v_mean, v_std = _stats(vmes)
        nc_mean, _ = _stats(ncls)
        nr_mean, _ = _stats(nrs)

        return GridSearchResult(
            params=params,
            purity_mean=p_mean,
            purity_std=p_std,
            homogeneity_mean=h_mean,
            homogeneity_std=h_std,
            completeness_mean=c_mean,
            completeness_std=c_std,
            v_measure_mean=v_mean,
            v_measure_std=v_std,
            n_clusters_mean=nc_mean,
            noise_ratio_mean=nr_mean,
            num_success=len(metrics_list),
            num_skipped=num_skipped,
            num_failed=num_failed,
        )


# ============================================================================
# Grid search runner
# ============================================================================


class GridSearchRunner:
    """Run DBSCAN parameter grid search over the full dataset."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        param_grid: Dict[str, List[Any]],
        sort_by: str = "v_measure_mean",
        sample_ids: Optional[List[str]] = None,
        segment_length: float = 3.0,
        num_workers: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.param_grid = param_grid
        self.sort_by = sort_by
        self.logger = logging.getLogger(__name__)

        loader = DatasetLoader(data_dir)
        samples = loader.load_samples(sample_ids)

        self.evaluator = CombinationEvaluator(
            samples=samples,
            segment_length=segment_length,
            num_workers=num_workers,
        )

    def run(self) -> List[GridSearchResult]:
        combinations = list(self._iter_combinations())
        self.logger.info(
            f"Grid search: {len(combinations)} combinations  (sort_by={self.sort_by})"
        )

        results: List[GridSearchResult] = []
        for params in tqdm(combinations, desc="Grid search"):
            result = self.evaluator.evaluate(params)
            results.append(result)
            self._log_combination(result)

        results.sort(
            key=lambda r: (
                -(getattr(r, self.sort_by) or 0.0)  # descending (higher = better)
            )
        )
        self._save_results(results)
        self._print_top_results(results)
        return results

    def _iter_combinations(self):
        keys = list(self.param_grid.keys())
        for values in itertools.product(*self.param_grid.values()):
            yield dict(zip(keys, values))

    def _log_combination(self, r: GridSearchResult):
        grid_keys = list(self.param_grid.keys())
        param_str = ", ".join(f"{k}={r.params[k]}" for k in grid_keys)
        v = f"{r.v_measure_mean:.4f}" if r.v_measure_mean is not None else "N/A"
        p = f"{r.purity_mean:.4f}" if r.purity_mean is not None else "N/A"
        self.logger.debug(
            f"  [{param_str}] v_measure={v}  purity={p}  (n={r.num_success})"
        )

    def _save_results(self, results: List[GridSearchResult]):
        json_path = self.output_dir / "grid_search_dbscan_results.json"
        json_data = {
            "param_grid": self.param_grid,
            "sort_by": self.sort_by,
            "results": [self._to_dict(r) for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        self.logger.info(f"Results saved: {json_path}")

        csv_path = self.output_dir / "grid_search_dbscan_results.csv"
        if results:
            grid_keys = list(self.param_grid.keys())
            metric_keys = [
                "v_measure_mean",
                "v_measure_std",
                "homogeneity_mean",
                "homogeneity_std",
                "completeness_mean",
                "completeness_std",
                "purity_mean",
                "purity_std",
                "n_clusters_mean",
                "noise_ratio_mean",
                "num_success",
                "num_skipped",
                "num_failed",
            ]
            fieldnames = grid_keys + metric_keys
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(self._to_dict(r))
        self.logger.info(f"Results saved: {csv_path}")

    @staticmethod
    def _to_dict(r: GridSearchResult) -> Dict[str, Any]:
        row = dict(r.params)
        row.update(
            {
                "v_measure_mean": r.v_measure_mean,
                "v_measure_std": r.v_measure_std,
                "homogeneity_mean": r.homogeneity_mean,
                "homogeneity_std": r.homogeneity_std,
                "completeness_mean": r.completeness_mean,
                "completeness_std": r.completeness_std,
                "purity_mean": r.purity_mean,
                "purity_std": r.purity_std,
                "n_clusters_mean": r.n_clusters_mean,
                "noise_ratio_mean": r.noise_ratio_mean,
                "num_success": r.num_success,
                "num_skipped": r.num_skipped,
                "num_failed": r.num_failed,
            }
        )
        return row

    def _print_top_results(self, results: List[GridSearchResult], top_n: int = 15):
        valid = [r for r in results if getattr(r, self.sort_by) is not None]
        print("\n" + "=" * 110)
        print(
            f"DBSCAN Grid Search — Top {min(top_n, len(valid))} Configurations"
            f"  (sorted by {self.sort_by}, higher is better)"
        )
        print("=" * 110)
        print(
            f"{'Rank':>4}  {'eps':>5}  {'min_s':>5}  "
            f"{'V-measure':>10}  {'Purity':>8}  {'Homog':>8}  {'Compl':>8}  "
            f"{'Clusters':>8}  {'Noise%':>7}  {'n':>4}"
        )
        print("-" * 110)

        for rank, r in enumerate(valid[:top_n], start=1):
            eps = r.params.get("eps", "?")
            min_s = r.params.get("min_samples", "?")
            fmt = lambda v: f"{v:.4f}" if v is not None else "   N/A"  # noqa: E731
            nc = (
                f"{r.n_clusters_mean:.1f}" if r.n_clusters_mean is not None else "  N/A"
            )
            nr = (
                f"{r.noise_ratio_mean * 100:.1f}"
                if r.noise_ratio_mean is not None
                else " N/A"
            )
            print(
                f"{rank:>4}  {eps:>5}  {min_s:>5}  "
                f"{fmt(r.v_measure_mean):>10}  {fmt(r.purity_mean):>8}  "
                f"{fmt(r.homogeneity_mean):>8}  {fmt(r.completeness_mean):>8}  "
                f"{nc:>8}  {nr:>7}  {r.num_success:>4}"
            )

        if valid:
            best = valid[0]
            print("\nBest configuration:")
            for k, v in best.params.items():
                print(f"  {k}: {v}")
            if best.v_measure_mean is not None:
                print(
                    f"  v_measure_mean:   {best.v_measure_mean:.4f} ± {best.v_measure_std:.4f}"
                )
            if best.purity_mean is not None:
                print(
                    f"  purity_mean:      {best.purity_mean:.4f} ± {best.purity_std:.4f}"
                )
            if best.homogeneity_mean is not None:
                print(
                    f"  homogeneity_mean: {best.homogeneity_mean:.4f} ± {best.homogeneity_std:.4f}"
                )
            if best.completeness_mean is not None:
                print(
                    f"  completeness_mean:{best.completeness_mean:.4f} ± {best.completeness_std:.4f}"
                )

        print("=" * 110 + "\n")


# ============================================================================
# CLI
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool):
    log_path = output_dir / "grid_search_dbscan.log"
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(
        description="DBSCAN parameter grid search — finds best eps and min_samples"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--sample-ids", nargs="+", help="Specific sample IDs (default: all)"
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        default=None,
        help=(
            'JSON parameter grid, e.g. \'{"eps": [10, 15, 20], "min_samples": [2, 3]}\''
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=list(SORT_KEYS),
        default="v_measure_mean",
        help="Metric to sort by (default: v_measure_mean)",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=3.0,
        help="Seed spacing for TopologyBuilder (default: 3.0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker threads (default: cpu_count)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    param_grid = json.loads(args.param_grid) if args.param_grid else DEFAULT_PARAM_GRID

    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)

    logger.info("=" * 80)
    logger.info("DBSCAN Grid Search")
    logger.info("=" * 80)
    logger.info(f"Data dir:           {args.data_dir}")
    logger.info(f"Output dir:         {args.output_dir}")
    logger.info(f"Sort by:            {args.sort_by}")
    logger.info(f"Segment length:     {args.segment_length}")
    logger.info(f"Num workers:        {args.num_workers or 'default'}")
    logger.info(f"Param grid:         {param_grid}")
    logger.info(f"Total combinations: {total_combinations}")

    runner = GridSearchRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        param_grid=param_grid,
        sort_by=args.sort_by,
        sample_ids=args.sample_ids,
        segment_length=args.segment_length,
        num_workers=args.num_workers,
    )
    runner.run()

    logger.info("Grid search complete.")


if __name__ == "__main__":
    main()
