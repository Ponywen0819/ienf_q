"""
CLAHE + Sato Enhancement Parameter Grid Search

Evaluates combinations of background subtraction (bg_kernel_size), CLAHE
(clip_limit, tile_size), and Sato vesselness (sigma_min, sigma_max) parameters
across all dataset samples.

Metric: Fisher Score — measures how well the enhanced image separates fiber
pixels (label > 0 within ROI) from background pixels (label == 0 within ROI):

    Fisher Score = (μ_fiber - μ_bg)² / (σ_fiber² + σ_bg²)

Higher Fisher Score → better fiber/background separation → better parameters.

Usage:
    # Default grid
    python tools/grid_search_enhancement.py \
        --data-dir data_0331 \
        --output-dir output/grid_search_enhancement

    # Custom grid (JSON)
    python tools/grid_search_enhancement.py \
        --data-dir data_0331 \
        --output-dir output/grid_search_enhancement \
        --param-grid '{"bg_kernel_size":[31,51],"clip_limit":[10.0,40.0],"tile_size":[8],"sigma_min":[2,4],"sigma_max":[8,12]}'

    # Specific samples
    python tools/grid_search_enhancement.py \
        --data-dir data_0331 \
        --output-dir output/grid_search_enhancement \
        --sample-ids S222-2_a S222-2_b
"""

import argparse
import csv
import itertools
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import skimage as ski
from PIL import Image
from tqdm import tqdm

from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.dataset import DatasetLoader, SampleFiles


# ============================================================================
# Default parameter grid
# ============================================================================

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    # "clip_limit": [10.0, 20.0, 30.0, 40.0, 50.0],
    # "tile_size": [256, 512, 768, 1024, 1536, 2048],
    # "sigma_min": [1, 2, 3, 4, 5],
    "sigma_max": [6, 7, 8, 9, 10],
    "bg_kernel_size": [31],
    "clip_limit": [20.0],
    "tile_size": [768],
    "sigma_min": [3],
    # "sigma_max": [8],
}


# ============================================================================
# Fisher Score
# ============================================================================


def compute_fisher_score(
    enhanced: np.ndarray,
    roi_mask: np.ndarray,
    roi_label: np.ndarray,
) -> Optional[float]:
    """
    Fisher Score = (μ_fiber - μ_bg)² / (σ_fiber² + σ_bg²)

    Args:
        enhanced:  Enhanced ROI image (H, W), float or uint8.
        roi_mask:  Dilated epidermis mask (H, W) uint8.
        roi_label: GT label clipped to ROI (H, W) uint8, 0 = background.

    Returns:
        Fisher Score, or None if either class has no pixels.
    """
    fiber_mask = roi_label > 0
    bg_mask = (roi_label == 0) & (roi_mask > 0)

    if fiber_mask.sum() == 0 or bg_mask.sum() == 0:
        return None

    vals_f = enhanced[fiber_mask].astype(np.float64)
    vals_b = enhanced[bg_mask].astype(np.float64)

    mu_f, sig_f = vals_f.mean(), vals_f.std()
    mu_b, sig_b = vals_b.mean(), vals_b.std()

    denom = sig_f**2 + sig_b**2
    if denom < 1e-8:
        return None

    return float((mu_f - mu_b) ** 2 / denom)


# ============================================================================
# Preprocessing helpers
# ============================================================================


def base_preprocess(
    image_path: Path,
    mask_path: Path,
    label_path: Path,
    offset_px: int = 50,
    bg_kernel_size: int = 51,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fixed preprocessing (done once per sample and bg_kernel_size, then cached):
      green channel → ROI masking
      green channel → background subtraction → ROI masking

    Returns:
        (roi_image_raw, roi_image_base, roi_mask, roi_label) or None on failure.
    """
    try:
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, 1]
        mask = np.array(Image.open(mask_path).convert("L"))
        label = np.array(Image.open(label_path).convert("L"))
    except Exception:
        return None

    roi_mask = dilate_epidermis_vertically(mask, offset_px=offset_px)
    roi_image_raw = cv2.bitwise_and(image, image, mask=roi_mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size)
    )
    bg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.subtract(image, bg)

    roi_image_base = cv2.bitwise_and(image, image, mask=roi_mask)
    roi_label = cv2.bitwise_and(label, label, mask=roi_mask)

    return roi_image_raw, roi_image_base, roi_mask, roi_label


def apply_enhancement(
    roi_image_base: np.ndarray,
    clip_limit: float,
    tile_size: int,
    sigma_min: int,
    sigma_max: int,
) -> np.ndarray:
    """
    Apply CLAHE + Sato vesselness filter.

    Returns:
        Enhanced image (H, W) uint8, normalised to [0, 255].
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(roi_image_base)

    # enhanced = ski.filters.meijering(
    #     enhanced,
    #     sigmas=range(sigma_min, sigma_max),
    #     black_ridges=False,
    # )
    enhanced = ski.filters.sato(
        enhanced,
        sigmas=range(sigma_min, sigma_max),
        black_ridges=False,
    )

    lo, hi = enhanced.min(), enhanced.max()
    if hi - lo < 1e-8:
        return np.zeros_like(roi_image_base, dtype=np.uint8)

    enhanced = (enhanced - lo) / (hi - lo) * 255.0
    return enhanced.astype(np.uint8)


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class GridSearchResult:
    params: Dict[str, Any]
    fisher_mean: Optional[float]
    fisher_std: Optional[float]
    fisher_min: Optional[float]
    fisher_max: Optional[float]
    baseline_mean: Optional[float]  # Fisher on raw green channel with ROI mask only
    improvement_mean: Optional[float]  # fisher_mean - baseline_mean
    num_success: int
    num_skipped: int
    num_failed: int


# ============================================================================
# Evaluator
# ============================================================================


class EnhancementEvaluator:
    """
    Caches base-preprocessed data per bg_kernel_size, then applies CLAHE+Sato
    for each parameter combination (fast inner loop).
    """

    def __init__(
        self,
        samples: List[SampleFiles],
        offset_px: int = 50,
        bg_kernel_size: int = 51,
        num_workers: Optional[int] = None,
    ):
        self.samples = samples
        self.offset_px = offset_px
        self.default_bg_kernel_size = bg_kernel_size
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self._cache_by_bg_kernel: Dict[int, Dict[str, Optional[Tuple]]] = {}
        self._baselines_by_bg_kernel: Dict[int, Dict[str, Optional[float]]] = {}

    def _ensure_base_cache(
        self, bg_kernel_size: int
    ) -> Tuple[Dict[str, Optional[Tuple]], Dict[str, Optional[float]]]:
        if bg_kernel_size in self._cache_by_bg_kernel:
            return (
                self._cache_by_bg_kernel[bg_kernel_size],
                self._baselines_by_bg_kernel[bg_kernel_size],
            )

        self.logger.info(
            f"Base-preprocessing {len(self.samples)} samples "
            f"(bg_kernel_size={bg_kernel_size})..."
        )
        cache: Dict[str, Optional[Tuple]] = {}
        for sample in tqdm(
            self.samples, desc=f"Base preprocessing bg={bg_kernel_size}"
        ):
            if sample.label_path is None or not sample.label_path.exists():
                cache[sample.sample_id] = None
                continue
            ok, _ = sample.is_complete()
            if not ok:
                cache[sample.sample_id] = None
                continue
            cache[sample.sample_id] = base_preprocess(
                sample.image_path,
                sample.mask_path,
                sample.label_path,
                offset_px=self.offset_px,
                bg_kernel_size=bg_kernel_size,
            )

        loaded = sum(1 for v in cache.values() if v is not None)
        self.logger.info(
            f"Cached {loaded}/{len(self.samples)} samples "
            f"(bg_kernel_size={bg_kernel_size})"
        )

        # Pre-compute baseline Fisher Score (raw green channel + ROI masking only)
        baselines: Dict[str, Optional[float]] = {}
        for sid, cached in cache.items():
            if cached is None:
                baselines[sid] = None
            else:
                roi_raw, _, roi_mask, roi_label = cached
                baselines[sid] = compute_fisher_score(roi_raw, roi_mask, roi_label)

        valid_baselines = [v for v in baselines.values() if v is not None]
        if valid_baselines:
            self.logger.info(
                f"Baseline Fisher Score (bg_kernel_size={bg_kernel_size}): "
                f"raw ROI only  "
                f"mean={np.mean(valid_baselines):.4f}  "
                f"std={np.std(valid_baselines):.4f}"
            )

        self._cache_by_bg_kernel[bg_kernel_size] = cache
        self._baselines_by_bg_kernel[bg_kernel_size] = baselines
        return cache, baselines

    def evaluate(self, params: Dict[str, Any]) -> GridSearchResult:
        bg_kernel_size = int(params.get("bg_kernel_size", self.default_bg_kernel_size))
        clip_limit = params["clip_limit"]
        tile_size = params["tile_size"]
        sigma_min = params["sigma_min"]
        sigma_max = params["sigma_max"]
        cache, baselines = self._ensure_base_cache(bg_kernel_size)

        if sigma_min >= sigma_max:
            return GridSearchResult(
                params=params,
                fisher_mean=None,
                fisher_std=None,
                fisher_min=None,
                fisher_max=None,
                baseline_mean=None,
                improvement_mean=None,
                num_success=0,
                num_skipped=0,
                num_failed=0,
            )

        fisher_list: List[float] = []
        baseline_list: List[float] = []
        num_skipped = 0
        num_failed = 0

        def _run(sid: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
            cached = cache.get(sid)
            if cached is None:
                return None  # skipped
            _, roi_base, roi_mask, roi_label = cached
            try:
                enhanced = apply_enhancement(
                    roi_base, clip_limit, tile_size, sigma_min, sigma_max
                )
                score = compute_fisher_score(enhanced, roi_mask, roi_label)
                return score, baselines.get(sid)
            except Exception as e:
                self.logger.debug(f"Enhancement failed on {sid}: {e}", exc_info=True)
                return (None, None)  # failed

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_run, sid): sid for sid in cache}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    num_skipped += 1
                elif result[0] is None:
                    num_failed += 1
                else:
                    score, baseline = result
                    fisher_list.append(score)
                    if baseline is not None:
                        baseline_list.append(baseline)

        def _stats(values):
            if not values:
                return None, None, None, None
            arr = np.array(values, dtype=float)
            return (
                float(arr.mean()),
                float(arr.std()),
                float(arr.min()),
                float(arr.max()),
            )

        f_mean, f_std, f_min, f_max = _stats(fisher_list)
        b_mean, *_ = _stats(baseline_list)
        improvement = (
            float(f_mean - b_mean)
            if f_mean is not None and b_mean is not None
            else None
        )

        return GridSearchResult(
            params=params,
            fisher_mean=f_mean,
            fisher_std=f_std,
            fisher_min=f_min,
            fisher_max=f_max,
            baseline_mean=b_mean,
            improvement_mean=improvement,
            num_success=len(fisher_list),
            num_skipped=num_skipped,
            num_failed=num_failed,
        )


# ============================================================================
# Grid search runner
# ============================================================================


def normalize_param_grid(param_grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Keep the public grid flexible while using bg_kernel_size internally.

    Some notes and older ad-hoc configs use the misspelled bg_kernal_size. Accept
    it as an alias so those JSON grids still work.
    """
    normalized = dict(param_grid)
    if "bg_kernal_size" in normalized:
        if "bg_kernel_size" in normalized:
            raise ValueError(
                "Use only one of bg_kernel_size or bg_kernal_size in --param-grid"
            )
        normalized["bg_kernel_size"] = normalized.pop("bg_kernal_size")
    return normalized


class GridSearchRunner:
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        param_grid: Dict[str, List[Any]],
        sample_ids: Optional[List[str]] = None,
        offset_px: int = 50,
        bg_kernel_size: int = 51,
        num_workers: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.param_grid = normalize_param_grid(param_grid)
        self.logger = logging.getLogger(__name__)

        loader = DatasetLoader(data_dir)
        samples = loader.load_samples(sample_ids)

        self.evaluator = EnhancementEvaluator(
            samples=samples,
            offset_px=offset_px,
            bg_kernel_size=bg_kernel_size,
            num_workers=num_workers,
        )

    def run(self) -> List[GridSearchResult]:
        combinations = [
            p for p in self._iter_combinations() if p["sigma_min"] < p["sigma_max"]
        ]
        self.logger.info(f"Grid search: {len(combinations)} valid combinations")

        results: List[GridSearchResult] = []
        for params in tqdm(combinations, desc="Grid search"):
            result = self.evaluator.evaluate(params)
            results.append(result)
            self._log_combination(result)

        results.sort(key=lambda r: -(r.fisher_mean or 0.0))
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
        f = f"{r.fisher_mean:.4f}" if r.fisher_mean is not None else "N/A"
        imp = f"{r.improvement_mean:+.4f}" if r.improvement_mean is not None else "N/A"
        self.logger.debug(f"  [{param_str}] fisher={f}  improvement={imp}")

    def _save_results(self, results: List[GridSearchResult]):
        grid_keys = list(self.param_grid.keys())
        metric_keys = [
            "fisher_mean",
            "fisher_std",
            "fisher_min",
            "fisher_max",
            "baseline_mean",
            "improvement_mean",
            "num_success",
            "num_skipped",
            "num_failed",
        ]

        json_path = self.output_dir / "grid_search_enhancement_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "param_grid": self.param_grid,
                    "results": [self._to_dict(r) for r in results],
                },
                f,
                indent=2,
            )
        self.logger.info(f"Results saved: {json_path}")

        csv_path = self.output_dir / "grid_search_enhancement_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=grid_keys + metric_keys)
            writer.writeheader()
            for r in results:
                writer.writerow(self._to_dict(r))
        self.logger.info(f"Results saved: {csv_path}")

    @staticmethod
    def _to_dict(r: GridSearchResult) -> Dict[str, Any]:
        row = dict(r.params)
        row.update(
            {
                "fisher_mean": r.fisher_mean,
                "fisher_std": r.fisher_std,
                "fisher_min": r.fisher_min,
                "fisher_max": r.fisher_max,
                "baseline_mean": r.baseline_mean,
                "improvement_mean": r.improvement_mean,
                "num_success": r.num_success,
                "num_skipped": r.num_skipped,
                "num_failed": r.num_failed,
            }
        )
        return row

    def _print_top_results(self, results: List[GridSearchResult], top_n: int = 15):
        valid = [r for r in results if r.fisher_mean is not None]
        print("\n" + "=" * 122)
        print(
            f"Enhancement Grid Search — Top {min(top_n, len(valid))} Configurations"
            "  (sorted by Fisher Score, higher is better)"
        )
        print("=" * 122)
        print(
            f"{'Rank':>4}  {'bg':>5}  {'clip':>6}  {'tile':>4}  "
            f"{'σ_min':>5}  {'σ_max':>5}  "
            f"{'Fisher':>8}  {'±std':>7}  {'Baseline':>9}  {'Δ':>8}  {'n':>4}"
        )
        print("-" * 122)

        for rank, r in enumerate(valid[:top_n], start=1):
            fmt = lambda v: f"{v:.4f}" if v is not None else "   N/A"  # noqa
            fmtd = lambda v: f"{v:+.4f}" if v is not None else "   N/A"  # noqa
            print(
                f"{rank:>4}  "
                f"{r.params.get('bg_kernel_size', '?'):>5}  "
                f"{r.params.get('clip_limit', '?'):>6}  "
                f"{r.params.get('tile_size', '?'):>4}  "
                f"{r.params.get('sigma_min', '?'):>5}  "
                f"{r.params.get('sigma_max', '?'):>5}  "
                f"{fmt(r.fisher_mean):>8}  "
                f"{fmt(r.fisher_std):>7}  "
                f"{fmt(r.baseline_mean):>9}  "
                f"{fmtd(r.improvement_mean):>8}  "
                f"{r.num_success:>4}"
            )

        if valid:
            best = valid[0]
            print("\nBest configuration:")
            for k, v in best.params.items():
                print(f"  {k}: {v}")
            print(f"  fisher_mean:   {best.fisher_mean:.4f} ± {best.fisher_std:.4f}")
            if best.baseline_mean is not None:
                print(f"  baseline_mean: {best.baseline_mean:.4f}")
                print(f"  improvement:   {best.improvement_mean:+.4f}")

        print("=" * 122 + "\n")


# ============================================================================
# CLI
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool):
    log_path = output_dir / "grid_search_enhancement.log"
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    fh = logging.FileHandler(log_path, encoding="utf-8")
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
            "CLAHE + Sato enhancement grid search — "
            "finds best parameters using Fisher Score (fiber vs background)"
        )
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
            'JSON parameter grid, e.g. \'{"bg_kernel_size":[31,51],'
            '"clip_limit":[10.0,40.0],'
            '"tile_size":[8],"sigma_min":[2,4],"sigma_max":[8,12]}\''
        ),
    )
    parser.add_argument(
        "--offset-px",
        type=int,
        default=50,
        help="Epidermis mask dilation (default: 50)",
    )
    parser.add_argument(
        "--bg-kernel-size",
        "--bg-kernal-size",
        type=int,
        default=51,
        dest="bg_kernel_size",
        help=(
            "Background subtraction kernel size used only when bg_kernel_size "
            "is omitted from --param-grid (default: 51)"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel threads (default: cpu_count)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    param_grid = normalize_param_grid(
        json.loads(args.param_grid) if args.param_grid else DEFAULT_PARAM_GRID
    )

    total = 1
    for v in param_grid.values():
        total *= len(v)

    logger.info("=" * 80)
    logger.info("CLAHE + Sato Enhancement Grid Search")
    logger.info("=" * 80)
    logger.info(f"Data dir:           {args.data_dir}")
    logger.info(f"Output dir:         {args.output_dir}")
    logger.info(f"Param grid:         {param_grid}")
    logger.info(f"Total combinations: {total} (invalid sigma combos skipped)")
    logger.info(f"Num workers:        {args.num_workers or 'default'}")

    GridSearchRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        param_grid=param_grid,
        sample_ids=args.sample_ids,
        offset_px=args.offset_px,
        bg_kernel_size=args.bg_kernel_size,
        num_workers=args.num_workers,
    ).run()

    logger.info("Grid search complete.")


if __name__ == "__main__":
    main()
