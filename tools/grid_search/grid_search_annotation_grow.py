"""
Annotation-Grow 演算法參數 Grid Search 工具

對整個資料集執行參數網格搜索，找出最佳 AnnotationGrowLinker 參數組合。
同時評估兩個指標（皆需要 label.png）：
  - HD95     (95th Percentile Hausdorff Distance) — 越小越好
  - clDice   (Centerline Dice)                    — 越大越好

使用範例:
    # 使用預設參數網格
    uv run python tools/grid_search/grid_search_annotation_grow.py \
        --data-dir data_0331 \
        --output-dir output/grid_search_annotation_grow

    # 指定部分樣本加速搜索
    uv run python tools/grid_search/grid_search_annotation_grow.py \
        --data-dir data \
        --output-dir output/grid_search_annotation_grow \
        --sample-ids S1585-2_a S1585-2_b

    # 自訂參數網格（JSON 格式）
    uv run python tools/grid_search/grid_search_annotation_grow.py \
        --data-dir data \
        --output-dir output/grid_search_annotation_grow \
        --param-grid '{"prune_threshold": [10.0, 20.0, 30.0], "segment_length": [50.0, 100.0]}'

    # 按 clDice 排序（預設按 hd95 排序）
    uv run python tools/grid_search/grid_search_annotation_grow.py \
        --data-dir data \
        --output-dir output/grid_search_annotation_grow \
        --sort-by cldice_mean

    # 指定平行處理線程數
    uv run python tools/grid_search/grid_search_annotation_grow.py \
        --data-dir data \
        --output-dir output/grid_search_annotation_grow \
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
from typing import Any, Dict, List, Optional
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from neural_reconstruction.algorithms.annotation_grow.linker import AnnotationGrowLinker
from neural_reconstruction.core.evaluation import (
    extract_graph_points,
    compute_hd95,
    compute_cldice,
)
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.dataset import DatasetLoader, SampleFiles


# ============================================================================
# Default parameter grid
# ============================================================================

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "bg_kernel_size": [16, 31, 51, 71, 101],
    # "clahe_clip": [10.0, 20.0, 40.0],
    # "sato_sigmas_start": [2, 3],
    # "sato_sigmas_stop": [6, 8],
    # "prune_threshold": [10.0, 20.0, 30.0],
}

# Parameters not in the grid will use these fixed defaults
FIXED_PARAMS: Dict[str, Any] = {
    "offset_px": 50,
    "clahe_grid": (16, 16),
    "clahe_clip": 20,
    "sato_sigmas_start": 3,
    "sato_sigmas_stop": 8,
    "prune_threshold": 20.0,
    "connectivity": 8,
    "segment_length": 100.0,
    "bg_kernel_size": 51,
}

# clDice 容忍半徑（像素）
CLDICE_TOLERANCE_PX = 1

# 排序鍵：hd95 越小越好（升序），clDice 越大越好（降序）
SORT_KEYS = (
    "hd95_mean",
    "cldice_mean",
    "tprec_mean",
    "tsens_mean",
)
# 預設升序的指標；其他則使用降序
ASC_METRICS = {"hd95_mean"}


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class SampleMetrics:
    """Per-sample evaluation metrics for one parameter combination."""

    hd95: Optional[float]  # 越小越好
    cldice: Optional[float]  # 越大越好
    tprec: Optional[float]  # clDice precision 分量
    tsens: Optional[float]  # clDice sensitivity 分量


@dataclass
class GridSearchResult:
    """Aggregated result for one parameter combination."""

    params: Dict[str, Any]
    # HD95 metrics
    hd95_mean: Optional[float]
    hd95_median: Optional[float]
    hd95_std: Optional[float]
    hd95_min: Optional[float]
    hd95_max: Optional[float]
    hd95_n: int
    # clDice metrics
    cldice_mean: Optional[float]
    cldice_median: Optional[float]
    cldice_std: Optional[float]
    cldice_min: Optional[float]
    cldice_max: Optional[float]
    cldice_n: int
    # clDice 分量
    tprec_mean: Optional[float]
    tsens_mean: Optional[float]
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
        num_workers: Optional[int] = None,
        cldice_tolerance_px: int = CLDICE_TOLERANCE_PX,
    ):
        self.samples = samples
        self.num_workers = num_workers
        self.cldice_tolerance_px = cldice_tolerance_px
        self._local = threading.local()
        self.logger = logging.getLogger(__name__)

    def _get_topology_builder(self) -> TopologyBuilder:
        if not hasattr(self._local, "topology_builder"):
            self._local.topology_builder = TopologyBuilder()
        return self._local.topology_builder

    def evaluate(self, params: Dict[str, Any]) -> GridSearchResult:
        # AnnotationGrowLinker 需要 tuple 而非 list，從 JSON 來的可能是 list
        normalized = dict(params)
        if "clahe_grid" in normalized and isinstance(normalized["clahe_grid"], list):
            normalized["clahe_grid"] = tuple(normalized["clahe_grid"])

        linker = AnnotationGrowLinker(**normalized)

        hd95_list: List[float] = []
        cldice_list: List[float] = []
        tprec_list: List[float] = []
        tsens_list: List[float] = []
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

                if metrics.hd95 is None and metrics.cldice is None:
                    num_failed += 1
                    continue

                num_success += 1
                if metrics.hd95 is not None:
                    hd95_list.append(metrics.hd95)
                if metrics.cldice is not None:
                    cldice_list.append(metrics.cldice)
                if metrics.tprec is not None:
                    tprec_list.append(metrics.tprec)
                if metrics.tsens is not None:
                    tsens_list.append(metrics.tsens)

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

        h_mean, h_med, h_std, h_min, h_max = _stats(hd95_list)
        c_mean, c_med, c_std, c_min, c_max = _stats(cldice_list)
        tp_mean = float(np.mean(tprec_list)) if tprec_list else None
        ts_mean = float(np.mean(tsens_list)) if tsens_list else None

        return GridSearchResult(
            params=params,
            hd95_mean=h_mean,
            hd95_median=h_med,
            hd95_std=h_std,
            hd95_min=h_min,
            hd95_max=h_max,
            hd95_n=len(hd95_list),
            cldice_mean=c_mean,
            cldice_median=c_med,
            cldice_std=c_std,
            cldice_min=c_min,
            cldice_max=c_max,
            cldice_n=len(cldice_list),
            tprec_mean=tp_mean,
            tsens_mean=ts_mean,
            num_success=num_success,
            num_skipped=num_skipped,
            num_failed=num_failed,
        )

    def _evaluate_sample(
        self, sample: SampleFiles, linker: AnnotationGrowLinker
    ) -> Optional[SampleMetrics]:
        """
        Returns None if sample should be skipped (missing image/mask/annotation
        or missing GT label — both metrics need label.png).
        """
        is_complete, _ = sample.is_complete()
        if not is_complete:
            return None

        # 兩個指標都需要 GT label，沒有 label 直接 skip
        if sample.label_path is None or not sample.label_path.exists():
            return None

        try:
            image = np.array(Image.open(sample.image_path))
            mask = np.array(Image.open(sample.mask_path))
            annotation = np.array(Image.open(sample.annotation_path))
            gt_label = np.array(Image.open(sample.label_path))

            extract_result = linker.run(image, mask, annotation)

            # GT label 限制在 ROI 內，與 pred graph 的範圍對齊
            if gt_label.ndim == 3:
                gt_label = gt_label[:, :, 0]
            roi_label = cv2.bitwise_and(gt_label, gt_label, mask=extract_result.mask)

            # ── HD95 ───────────────────────────────────────────────────────
            hd95: Optional[float] = None
            try:
                graph_gt = self._get_topology_builder().build_seed_graph(roi_label)
                pred_points = extract_graph_points(extract_result.graph)
                gt_points = extract_graph_points(graph_gt)
                hd95_val, _, _ = compute_hd95(pred_points, gt_points)
                hd95 = hd95_val
            except Exception as e:
                self.logger.debug(
                    f"Sample {sample.sample_id} HD95 failed: {e}", exc_info=True
                )

            # ── clDice ─────────────────────────────────────────────────────
            cldice: Optional[float] = None
            tprec: Optional[float] = None
            tsens: Optional[float] = None
            try:
                cld, tp, ts = compute_cldice(
                    extract_result.graph,
                    roi_label,
                    tolerance_px=self.cldice_tolerance_px,
                )
                cldice, tprec, tsens = cld, tp, ts
            except Exception as e:
                self.logger.debug(
                    f"Sample {sample.sample_id} clDice failed: {e}", exc_info=True
                )

            return SampleMetrics(hd95=hd95, cldice=cldice, tprec=tprec, tsens=tsens)

        except Exception as e:
            self.logger.debug(
                f"Sample {sample.sample_id} pipeline failed: {e}", exc_info=True
            )
            return SampleMetrics(hd95=None, cldice=None, tprec=None, tsens=None)


# ============================================================================
# Grid search runner
# ============================================================================


class GridSearchRunner:
    """Run grid search over AnnotationGrowLinker parameters."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        param_grid: Dict[str, List[Any]],
        fixed_params: Dict[str, Any],
        sort_by: str = "hd95_mean",
        sample_ids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        cldice_tolerance_px: int = CLDICE_TOLERANCE_PX,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.param_grid = param_grid
        self.fixed_params = fixed_params
        self.sort_by = sort_by
        self.logger = logging.getLogger(__name__)

        loader = DatasetLoader(data_dir)
        self.samples = loader.load_samples(sample_ids)

        n_with_label = sum(
            1 for s in self.samples if s.label_path and s.label_path.exists()
        )
        self.logger.info(
            f"Samples loaded: {len(self.samples)} (with label.png: {n_with_label})"
        )
        if n_with_label == 0:
            raise RuntimeError(
                "No samples with label.png found — both HD95 and clDice require GT."
            )

        self.evaluator = CombinationEvaluator(
            self.samples,
            num_workers=num_workers,
            cldice_tolerance_px=cldice_tolerance_px,
        )

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

        results.sort(key=self._sort_key)
        self._save_results(results)
        self._print_top_results(results)
        return results

    def _sort_key(self, r: GridSearchResult):
        v = getattr(r, self.sort_by)
        if v is None:
            # 缺值放最後
            return float("inf")
        # hd95: 升序；其他（clDice / tprec / tsens）: 降序 → 取負
        return v if self.sort_by in ASC_METRICS else -v

    def _iter_combinations(self):
        keys = list(self.param_grid.keys())
        for values in itertools.product(*self.param_grid.values()):
            params = dict(self.fixed_params)
            params.update(dict(zip(keys, values)))
            yield params

    def _log_combination(self, result: GridSearchResult):
        grid_keys = list(self.param_grid.keys())
        param_str = ", ".join(f"{k}={result.params[k]}" for k in grid_keys)
        h = f"{result.hd95_mean:.4f}" if result.hd95_mean is not None else "N/A"
        c = f"{result.cldice_mean:.4f}" if result.cldice_mean is not None else "N/A"
        self.logger.debug(
            f"  [{param_str}] → hd95={h}  cldice={c}"
            f"  (n_h={result.hd95_n}, n_c={result.cldice_n})"
        )

    def _save_results(self, results: List[GridSearchResult]):
        json_path = self.output_dir / "grid_search_results.json"
        json_data = {
            "param_grid": self.param_grid,
            "fixed_params": self._jsonable(self.fixed_params),
            "sort_by": self.sort_by,
            "num_samples": len(self.samples),
            "cldice_tolerance_px": self.evaluator.cldice_tolerance_px,
            "results": [self._result_to_dict(r) for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, default=str)
        self.logger.info(f"Results saved: {json_path}")

        csv_path = self.output_dir / "grid_search_results.csv"
        if results:
            all_param_keys = list(results[0].params.keys())
            metric_keys = [
                "hd95_mean",
                "hd95_median",
                "hd95_std",
                "hd95_min",
                "hd95_max",
                "hd95_n",
                "cldice_mean",
                "cldice_median",
                "cldice_std",
                "cldice_min",
                "cldice_max",
                "cldice_n",
                "tprec_mean",
                "tsens_mean",
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
    def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            if isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out

    @classmethod
    def _result_to_dict(cls, r: GridSearchResult) -> Dict[str, Any]:
        row = cls._jsonable(r.params)
        row.update(
            {
                "hd95_mean": r.hd95_mean,
                "hd95_median": r.hd95_median,
                "hd95_std": r.hd95_std,
                "hd95_min": r.hd95_min,
                "hd95_max": r.hd95_max,
                "hd95_n": r.hd95_n,
                "cldice_mean": r.cldice_mean,
                "cldice_median": r.cldice_median,
                "cldice_std": r.cldice_std,
                "cldice_min": r.cldice_min,
                "cldice_max": r.cldice_max,
                "cldice_n": r.cldice_n,
                "tprec_mean": r.tprec_mean,
                "tsens_mean": r.tsens_mean,
                "num_success": r.num_success,
                "num_skipped": r.num_skipped,
                "num_failed": r.num_failed,
            }
        )
        return row

    def _print_top_results(self, results: List[GridSearchResult], top_n: int = 10):
        valid = [r for r in results if getattr(r, self.sort_by) is not None]
        print("\n" + "=" * 110)
        order = "asc" if self.sort_by in ASC_METRICS else "desc"
        print(
            f"Annotation-Grow Grid Search — Top {min(top_n, len(valid))} Configurations"
            f"  (sorted by {self.sort_by}, {order})"
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
                print(f"  hd95_mean:    {best.hd95_mean:.4f}")
            if best.cldice_mean is not None:
                print(f"  cldice_mean:  {best.cldice_mean:.4f}")
            if best.tprec_mean is not None:
                print(f"  tprec_mean:   {best.tprec_mean:.4f}")
            if best.tsens_mean is not None:
                print(f"  tsens_mean:   {best.tsens_mean:.4f}")

        print("=" * 110 + "\n")


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
        description=(
            "Annotation-Grow parameter grid search — finds optimal "
            "AnnotationGrowLinker config using HD95 and clDice"
        )
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
            '\'{"prune_threshold": [10.0, 20.0], "segment_length": [50.0, 100.0]}\''
        ),
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default=None,
        help=(
            "JSON string for fixed (non-searched) params, e.g. "
            '\'{"offset_px": 50, "connectivity": 8}\''
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=list(SORT_KEYS),
        default="hd95_mean",
        help="Metric to sort results by (default: hd95_mean, ascending)",
    )
    parser.add_argument(
        "--cldice-tolerance",
        type=int,
        default=CLDICE_TOLERANCE_PX,
        help=f"clDice 容忍半徑 (px, default: {CLDICE_TOLERANCE_PX})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
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
    logger.info("Annotation-Grow Grid Search (HD95 + clDice)")
    logger.info("=" * 80)
    logger.info(f"Data dir:           {args.data_dir}")
    logger.info(f"Num workers:        {args.num_workers or 'default'}")
    logger.info(f"Output dir:         {args.output_dir}")
    logger.info(f"Sort by:            {args.sort_by}")
    logger.info(f"clDice tolerance:   {args.cldice_tolerance} px")
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
        cldice_tolerance_px=args.cldice_tolerance,
    )
    runner.run()

    logger.info("Grid search complete.")


if __name__ == "__main__":
    main()
