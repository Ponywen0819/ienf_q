"""
Pure MST 演算法參數 Grid Search 工具

對整個資料集執行參數網格搜索，找出最佳 PureMstLinker 參數組合。
同時評估兩個指標：
  - Average Hausdorff Distance（拓樸相似度，需要 label.png）
  - Count MAE（神經穿越數預測誤差，需要 count.json）

使用範例:
    # 使用預設參數網格
    python tools/grid_search_pure_mst.py \
        --data-dir data \
        --output-dir output/grid_search

    # 指定部分樣本加速搜索
    python tools/grid_search_pure_mst.py \
        --data-dir data \
        --output-dir output/grid_search \
        --sample-ids S1585-2_a S1585-2_b

    # 自訂參數網格（JSON 格式）
    python tools/grid_search_pure_mst.py \
        --data-dir data \
        --output-dir output/grid_search \
        --param-grid '{"segment_length": [3.0, 5.0], "search_radius": [20.0, 50.0]}'

    # 按 count MAE 排序
    python tools/grid_search_pure_mst.py \
        --data-dir data \
        --output-dir output/grid_search \
        --sort-by count_mae_mean

    # 指定平行處理線程數
    python tools/grid_search_pure_mst.py \
        --data-dir data \
        --output-dir output/grid_search \
        --num-workers 8
"""

import argparse
import csv
import itertools
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from neural_reconstruction.algorithms.pure_mst.linker import PureMstLinker
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.evaluation import (
    extract_graph_points,
    compute_average_hausdorff_distance,
)
from neural_reconstruction.dataset import DatasetLoader, SampleFiles


# ============================================================================
# Default parameter grid
# ============================================================================

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "segment_length": [3.0, 5.0, 8.0],
    "search_radius": [20.0, 50.0, 100.0],
    "max_cost_threshold": [0.90, 0.95, 0.98],
    "intensity_weight": [1.0, 2.0, 4.0],
    "rolling_ball_radius": [20, 50, 100],
}

# Parameters not in the grid will use these fixed defaults
FIXED_PARAMS: Dict[str, Any] = {
    "offset_px": 50,
    "opening_kernel_size": 3,
}

SORT_KEYS = ("hausdorff_mean", "count_mae_mean")


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class SampleMetrics:
    """Per-sample evaluation metrics for one parameter combination."""

    hausdorff: Optional[float]  # None if label.png missing or failed
    count_error: Optional[float]  # |pred - gt|, None if gt count missing or failed


@dataclass
class GridSearchResult:
    """Aggregated result for one parameter combination."""

    params: Dict[str, Any]
    # Hausdorff metrics (samples with label.png)
    hausdorff_mean: Optional[float]
    hausdorff_median: Optional[float]
    hausdorff_std: Optional[float]
    hausdorff_min: Optional[float]
    hausdorff_max: Optional[float]
    hausdorff_n: int  # number of samples with valid hausdorff
    # Count metrics (samples with GT count in count.json)
    count_mae_mean: Optional[float]
    count_mae_median: Optional[float]
    count_mae_std: Optional[float]
    count_mae_min: Optional[float]
    count_mae_max: Optional[float]
    count_mae_n: int  # number of samples with valid count comparison
    # Overall
    num_success: int
    num_skipped: int
    num_failed: int


# ============================================================================
# Single-combination evaluator
# ============================================================================


class CombinationEvaluator:
    """Evaluate one parameter combination across all samples."""

    def __init__(
        self,
        samples: List[SampleFiles],
        gt_counts: Dict[str, int],
        num_workers: Optional[int] = None,
    ):
        self.samples = samples
        self.gt_counts = gt_counts
        self.num_workers = num_workers
        self._local = threading.local()  # thread-local storage for TopologyBuilder
        self.logger = logging.getLogger(__name__)

    def _get_topology_builder(self) -> TopologyBuilder:
        """Return a per-thread TopologyBuilder instance."""
        if not hasattr(self._local, "topology_builder"):
            self._local.topology_builder = TopologyBuilder()
        return self._local.topology_builder

    def evaluate(self, params: Dict[str, Any]) -> GridSearchResult:
        linker = PureMstLinker(**params)

        hausdorff_list: List[float] = []
        count_error_list: List[float] = []
        num_success = 0
        num_skipped = 0
        num_failed = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._evaluate_sample, sample, linker): sample
                for sample in self.samples
            }
            for future in as_completed(futures):
                metrics = future.result()

                if metrics is None:
                    num_skipped += 1
                    continue

                if metrics.hausdorff is None and metrics.count_error is None:
                    num_failed += 1
                    continue

                num_success += 1
                if metrics.hausdorff is not None:
                    hausdorff_list.append(metrics.hausdorff)
                if metrics.count_error is not None:
                    count_error_list.append(metrics.count_error)

        def _stats(values: List[float]):
            if not values:
                return None, None, None, None, None
            arr = np.array(values)
            return (
                float(np.mean(arr)),
                float(np.median(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
            )

        h_mean, h_med, h_std, h_min, h_max = _stats(hausdorff_list)
        c_mean, c_med, c_std, c_min, c_max = _stats(count_error_list)

        return GridSearchResult(
            params=params,
            hausdorff_mean=h_mean,
            hausdorff_median=h_med,
            hausdorff_std=h_std,
            hausdorff_min=h_min,
            hausdorff_max=h_max,
            hausdorff_n=len(hausdorff_list),
            count_mae_mean=c_mean,
            count_mae_median=c_med,
            count_mae_std=c_std,
            count_mae_min=c_min,
            count_mae_max=c_max,
            count_mae_n=len(count_error_list),
            num_success=num_success,
            num_skipped=num_skipped,
            num_failed=num_failed,
        )

    def _evaluate_sample(
        self, sample: SampleFiles, linker: PureMstLinker
    ) -> Optional[SampleMetrics]:
        """
        Returns None if sample should be skipped (missing required files).
        Returns SampleMetrics with individual fields None on partial failure.
        """
        is_complete, _ = sample.is_complete()
        if not is_complete:
            return None

        # Check if this sample contributes to at least one metric
        has_hausdorff_gt = sample.label_path and sample.label_path.exists()
        has_count_gt = sample.sample_id in self.gt_counts

        if not has_hausdorff_gt and not has_count_gt:
            return None  # nothing to evaluate → skip

        try:
            image = np.array(Image.open(sample.image_path))
            mask = np.array(Image.open(sample.mask_path))
            annotation = np.array(Image.open(sample.annotation_path))

            extract_result = linker.run(image, mask, annotation)

            # --- Hausdorff ---
            hausdorff: Optional[float] = None
            if has_hausdorff_gt and sample.label_path is not None:
                gt_label = np.array(Image.open(sample.label_path))
                roi_label = cv2.bitwise_and(
                    gt_label, gt_label, mask=extract_result.mask
                )
                graph_gt = self._get_topology_builder().build_seed_graph(roi_label)
                pred_points = extract_graph_points(extract_result.graph)
                gt_points = extract_graph_points(graph_gt)
                avg_dist, _, _ = compute_average_hausdorff_distance(
                    pred_points, gt_points
                )
                hausdorff = avg_dist

            # --- Count error ---
            count_error: Optional[float] = None
            if has_count_gt:
                gt_count = self.gt_counts[sample.sample_id]
                pred_count = extract_result.valid_count
                count_error = abs(pred_count - gt_count)

            return SampleMetrics(hausdorff=hausdorff, count_error=count_error)

        except Exception as e:
            self.logger.debug(f"Sample {sample.sample_id} failed: {e}", exc_info=True)
            return SampleMetrics(hausdorff=None, count_error=None)


# ============================================================================
# Grid search runner
# ============================================================================


class GridSearchRunner:
    """Run grid search over PureMstLinker parameters."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        param_grid: Dict[str, List[Any]],
        fixed_params: Dict[str, Any],
        sort_by: str = "hausdorff_mean",
        sample_ids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.param_grid = param_grid
        self.fixed_params = fixed_params
        self.sort_by = sort_by
        self.logger = logging.getLogger(__name__)

        loader = DatasetLoader(data_dir)
        self.samples = loader.load_samples(sample_ids)

        gt_counts = self._load_gt_counts(data_dir)
        self.logger.info(
            f"Loaded GT counts for {len(gt_counts)} samples from count.json"
        )

        self.evaluator = CombinationEvaluator(self.samples, gt_counts, num_workers=num_workers)

    @staticmethod
    def _load_gt_counts(data_dir: Path) -> Dict[str, int]:
        count_path = Path(data_dir) / "count.json"
        if not count_path.exists():
            logging.getLogger(__name__).warning(
                f"count.json not found at {count_path} — count MAE will not be computed"
            )
            return {}
        with open(count_path, encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> List[GridSearchResult]:
        combinations = list(self._iter_combinations())
        self.logger.info(
            f"Grid search: {len(combinations)} combinations × {len(self.samples)} samples"
            f"  (sort_by={self.sort_by}, num_workers={self.evaluator.num_workers or 'default'})"
        )

        results: List[GridSearchResult] = []
        for params in tqdm(combinations, desc="Grid search"):
            result = self.evaluator.evaluate(params)
            results.append(result)
            self._log_combination(result)

        results.sort(
            key=lambda r: (
                getattr(r, self.sort_by)
                if getattr(r, self.sort_by) is not None
                else float("inf")
            )
        )
        self._save_results(results)
        self._print_top_results(results)
        return results

    def _iter_combinations(self):
        keys = list(self.param_grid.keys())
        for values in itertools.product(*self.param_grid.values()):
            params = dict(self.fixed_params)
            params.update(dict(zip(keys, values)))
            yield params

    def _log_combination(self, result: GridSearchResult):
        grid_keys = list(self.param_grid.keys())
        param_str = ", ".join(f"{k}={result.params[k]}" for k in grid_keys)
        h = (
            f"{result.hausdorff_mean:.4f}"
            if result.hausdorff_mean is not None
            else "N/A"
        )
        c = (
            f"{result.count_mae_mean:.4f}"
            if result.count_mae_mean is not None
            else "N/A"
        )
        self.logger.debug(
            f"  [{param_str}] → hausdorff={h}  count_mae={c}"
            f"  (n_h={result.hausdorff_n}, n_c={result.count_mae_n})"
        )

    def _save_results(self, results: List[GridSearchResult]):
        json_path = self.output_dir / "grid_search_results.json"
        json_data = {
            "param_grid": self.param_grid,
            "fixed_params": self.fixed_params,
            "sort_by": self.sort_by,
            "num_samples": len(self.samples),
            "results": [self._result_to_dict(r) for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        self.logger.info(f"Results saved: {json_path}")

        csv_path = self.output_dir / "grid_search_results.csv"
        if results:
            all_param_keys = list(results[0].params.keys())
            metric_keys = [
                "hausdorff_mean",
                "hausdorff_median",
                "hausdorff_std",
                "hausdorff_min",
                "hausdorff_max",
                "hausdorff_n",
                "count_mae_mean",
                "count_mae_median",
                "count_mae_std",
                "count_mae_min",
                "count_mae_max",
                "count_mae_n",
                "num_success",
                "num_skipped",
                "num_failed",
            ]
            fieldnames = all_param_keys + metric_keys
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(self._result_to_dict(r))
        self.logger.info(f"Results saved: {csv_path}")

    @staticmethod
    def _result_to_dict(r: GridSearchResult) -> Dict[str, Any]:
        row = dict(r.params)
        row.update(
            {
                "hausdorff_mean": r.hausdorff_mean,
                "hausdorff_median": r.hausdorff_median,
                "hausdorff_std": r.hausdorff_std,
                "hausdorff_min": r.hausdorff_min,
                "hausdorff_max": r.hausdorff_max,
                "hausdorff_n": r.hausdorff_n,
                "count_mae_mean": r.count_mae_mean,
                "count_mae_median": r.count_mae_median,
                "count_mae_std": r.count_mae_std,
                "count_mae_min": r.count_mae_min,
                "count_mae_max": r.count_mae_max,
                "count_mae_n": r.count_mae_n,
                "num_success": r.num_success,
                "num_skipped": r.num_skipped,
                "num_failed": r.num_failed,
            }
        )
        return row

    def _print_top_results(self, results: List[GridSearchResult], top_n: int = 10):
        valid = [r for r in results if getattr(r, self.sort_by) is not None]
        print("\n" + "=" * 100)
        print(
            f"Grid Search Results — Top {min(top_n, len(valid))} Configurations"
            f"  (sorted by {self.sort_by})"
        )
        print("=" * 100)

        grid_keys = list(self.param_grid.keys())
        print(
            f"{'Rank':>4}  {'Hausdorff':>10}  {'Count MAE':>10}  "
            f"{'n_h':>4}  {'n_c':>4}  Params"
        )
        print("-" * 100)

        for rank, r in enumerate(valid[:top_n], start=1):
            h_str = (
                f"{r.hausdorff_mean:.4f}" if r.hausdorff_mean is not None else "  N/A  "
            )
            c_str = (
                f"{r.count_mae_mean:.4f}" if r.count_mae_mean is not None else "  N/A  "
            )
            param_str = "  ".join(f"{k}={r.params[k]}" for k in grid_keys)
            print(
                f"{rank:>4}  {h_str:>10}  {c_str:>10}  "
                f"{r.hausdorff_n:>4}  {r.count_mae_n:>4}  {param_str}"
            )

        if valid:
            best = valid[0]
            print("\nBest configuration:")
            for k, v in best.params.items():
                print(f"  {k}: {v}")
            if best.hausdorff_mean is not None:
                print(f"  hausdorff_mean:  {best.hausdorff_mean:.4f}")
            if best.count_mae_mean is not None:
                print(f"  count_mae_mean:  {best.count_mae_mean:.4f}")

        print("=" * 100 + "\n")


# ============================================================================
# CLI
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool):
    log_path = output_dir / "grid_search.log"
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
        description="Pure MST parameter grid search — finds optimal PureMstLinker config"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--sample-ids", nargs="+", help="Specific sample IDs to use (default: all)"
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        default=None,
        help=(
            "JSON string defining parameter grid, e.g. "
            '\'{"segment_length": [3.0, 5.0], "search_radius": [20.0, 50.0]}\''
        ),
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default=None,
        help=(
            "JSON string for fixed (non-searched) params, e.g. "
            '\'{"offset_px": 1, "rolling_ball_radius": 2}\''
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=list(SORT_KEYS),
        default="hausdorff_mean",
        help="Metric to sort results by (default: hausdorff_mean)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of threads for parallel sample processing (default: os.cpu_count())",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    param_grid = json.loads(args.param_grid) if args.param_grid else DEFAULT_PARAM_GRID
    fixed_params = json.loads(args.fixed_params) if args.fixed_params else FIXED_PARAMS

    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)

    logger.info("=" * 80)
    logger.info("Pure MST Grid Search")
    logger.info("=" * 80)
    logger.info(f"Data dir:           {args.data_dir}")
    logger.info(f"Num workers:        {args.num_workers or 'default'}")
    logger.info(f"Output dir:         {args.output_dir}")
    logger.info(f"Sort by:            {args.sort_by}")
    logger.info(f"Param grid:         {param_grid}")
    logger.info(f"Fixed params:       {fixed_params}")
    logger.info(f"Total combinations: {total_combinations}")

    runner = GridSearchRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        param_grid=param_grid,
        fixed_params=fixed_params,
        sort_by=args.sort_by,
        sample_ids=args.sample_ids,
        num_workers=args.num_workers,
    )
    runner.run()

    logger.info("Grid search complete.")


if __name__ == "__main__":
    main()
