"""
資料集評測腳本 (Dataset Evaluation Script)

對整個資料集進行批次處理與評估：
1. 驗證資料集完整性
2. 執行神經重建 Pipeline
3. 計算 Hausdorff 距離（預留 GT 比較）
4. 生成統計報告

使用範例:
    python tools/evaluate_dataset.py \
        --data-dir data/ \
        --output-dir output/evaluation

作者: Claude Code
日期: 2026-01-15
"""

import argparse
import copy
import concurrent.futures
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import csv
import cv2
from scipy.stats import pearsonr, spearmanr

import numpy as np
import networkx as nx
from PIL import Image
from tqdm import tqdm

from neural_reconstruction.dataset import SampleFiles, DatasetLoader
from neural_reconstruction.algorithms.pure_mst.linker import PureMstLinker
from neural_reconstruction.algorithms.fragment_linking.linker import (
    HierarchicalFragmentLinker,
)
from neural_reconstruction.algorithms.xgb_mst.linker import XgbMstLinker
from neural_reconstruction.algorithms.unet_linker import UnetLinker

from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.common.data_types import LinkerResult


# 使用新的評估模組
from neural_reconstruction.core.evaluation import (
    GraphPointExtractor,
    extract_graph_points,
    TopologyComparator,
    compute_average_hausdorff_distance,
    compute_point_min_distances,
)


# ============================================================================
# 資料結構定義
# ============================================================================


@dataclass
class SampleResult:
    """單一樣本的評測結果"""

    sample_id: str
    status: str  # success, skipped, failed
    hausdorff_distance: Optional[float] = None
    hausdorff_distance_pred_to_gt: Optional[float] = None
    hausdorff_distance_gt_to_pred: Optional[float] = None
    num_nodes_pred: Optional[int] = None
    num_nodes_gt: Optional[int] = None
    num_edges_pred: Optional[int] = None
    num_edges_gt: Optional[int] = None
    num_components_pred: Optional[int] = None
    valid_count_pred: Optional[int] = None
    gt_count: Optional[int] = None
    count_error: Optional[float] = None  # |pred - gt|
    error_message: Optional[str] = None


@dataclass
class EvaluationSummary:
    """評測統計摘要"""

    total_samples: int
    processed: int
    skipped: int
    failed: int
    hausdorff_mean: Optional[float] = None
    hausdorff_median: Optional[float] = None
    hausdorff_std: Optional[float] = None
    hausdorff_min: Optional[float] = None
    hausdorff_max: Optional[float] = None
    count_mae_mean: Optional[float] = None
    count_mae_median: Optional[float] = None
    count_mae_std: Optional[float] = None
    count_pearson_r: Optional[float] = None
    count_pearson_p: Optional[float] = None
    count_spearman_r: Optional[float] = None
    count_spearman_p: Optional[float] = None
    count_n: int = 0


# ============================================================================
# 拓樸萃取器
# ============================================================================


class TopologyExtractor:
    """
    拓樸萃取器

    統一處理 GT 和 Pipeline 結果的拓樸建構。
    """

    def __init__(self, linker: Any):
        """
        Args:
            linker: 實作 run(image, mask, annotation) -> nx.Graph 的 linker 實例
        """
        self.linker = linker
        self.gt_builder = TopologyBuilder()
        self.logger = logging.getLogger(__name__)

    def extract_from_pipeline(
        self, image: np.ndarray, mask: np.ndarray, annotation: np.ndarray
    ) -> Optional[LinkerResult]:
        """
        從 Pipeline 萃取拓樸

        Args:
            image: 原始影像
            mask: 遮罩影像
            annotation: 標註影像

        Returns:
            NetworkX Graph，失敗則返回 None
        """
        try:
            return self.linker.run(image, mask, annotation)
        except Exception as e:
            self.logger.error(f"Pipeline 執行失敗: {e}")
            return None

    def extract_from_gt(self, gt_label: np.ndarray) -> Optional[nx.Graph]:
        """
        從 GT 標註萃取拓樸

        直接對整張 GT 影像呼叫 build_seed_graph，無需逐連通分量處理。

        Args:
            gt_label: GT 標註影像（二值，0 或 255）

        Returns:
            NetworkX Graph，失敗則返回 None
        """
        try:
            graph = self.gt_builder.build_seed_graph(gt_label)
            self.logger.debug(
                f"GT 拓樸建構完成: {graph.number_of_nodes()} 節點, "
                f"{graph.number_of_edges()} 邊"
            )
            return graph
        except Exception as e:
            self.logger.error(f"GT 拓樸建構失敗: {e}", exc_info=True)
            return None


# ============================================================================
# 評測報告器
# ============================================================================


class EvaluationReporter:
    """
    評測報告器

    彙整結果並生成報告（JSON、CSV、終端輸出）。
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_report(
        self, results: List[SampleResult], config: Dict[str, Any]
    ) -> EvaluationSummary:
        """
        生成完整評測報告

        Args:
            results: 所有樣本的評測結果
            config: 評測配置

        Returns:
            評測統計摘要
        """
        self.logger.info("生成評測報告...")

        # 計算統計摘要
        summary = self._compute_summary(results)

        # 生成 JSON 報告
        self._save_json_report(results, summary, config)

        # 生成 CSV 報告
        self._save_csv_report(results)

        # 終端輸出摘要
        self._print_summary(summary)

        return summary

    def _compute_summary(self, results: List[SampleResult]) -> EvaluationSummary:
        """計算統計摘要"""
        total = len(results)
        processed = sum(1 for r in results if r.status == "success")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")

        # Hausdorff
        valid_distances = [
            r.hausdorff_distance
            for r in results
            if r.status == "success" and r.hausdorff_distance is not None
        ]
        if valid_distances:
            hausdorff_mean = float(np.mean(valid_distances))
            hausdorff_median = float(np.median(valid_distances))
            hausdorff_std = float(np.std(valid_distances))
            hausdorff_min = float(np.min(valid_distances))
            hausdorff_max = float(np.max(valid_distances))
        else:
            hausdorff_mean = hausdorff_median = hausdorff_std = None
            hausdorff_min = hausdorff_max = None

        # Count MAE + correlation
        count_pairs = [
            (r.valid_count_pred, r.gt_count)
            for r in results
            if r.status == "success"
            and r.valid_count_pred is not None
            and r.gt_count is not None
        ]
        count_mae_mean = count_mae_median = count_mae_std = None
        count_pearson_r = count_pearson_p = None
        count_spearman_r = count_spearman_p = None
        count_n = len(count_pairs)

        if count_pairs:
            preds, gts = zip(*count_pairs)
            errors = np.abs(np.array(preds, dtype=float) - np.array(gts, dtype=float))
            count_mae_mean = float(np.mean(errors))
            count_mae_median = float(np.median(errors))
            count_mae_std = float(np.std(errors))

            if count_n >= 2:
                pr, pp = pearsonr(preds, gts)
                sr, sp = spearmanr(preds, gts)
                count_pearson_r = float(pr)
                count_pearson_p = float(pp)
                count_spearman_r = float(sr)
                count_spearman_p = float(sp)

        return EvaluationSummary(
            total_samples=total,
            processed=processed,
            skipped=skipped,
            failed=failed,
            hausdorff_mean=hausdorff_mean,
            hausdorff_median=hausdorff_median,
            hausdorff_std=hausdorff_std,
            hausdorff_min=hausdorff_min,
            hausdorff_max=hausdorff_max,
            count_mae_mean=count_mae_mean,
            count_mae_median=count_mae_median,
            count_mae_std=count_mae_std,
            count_pearson_r=count_pearson_r,
            count_pearson_p=count_pearson_p,
            count_spearman_r=count_spearman_r,
            count_spearman_p=count_spearman_p,
            count_n=count_n,
        )

    def _save_json_report(
        self,
        results: List[SampleResult],
        summary: EvaluationSummary,
        config: Dict[str, Any],
    ):
        """儲存 JSON 報告"""
        json_path = self.output_dir / "results.json"

        # 轉換為字典格式
        samples_dict = {r.sample_id: asdict(r) for r in results}
        skipped_samples = [r.sample_id for r in results if r.status == "skipped"]

        report = {
            "summary": asdict(summary),
            "samples": samples_dict,
            "skipped_samples": skipped_samples,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"JSON 報告已儲存: {json_path}")

    def _save_csv_report(self, results: List[SampleResult]):
        """儲存 CSV 報告"""
        csv_path = self.output_dir / "results.csv"

        fieldnames = [
            "sample_id",
            "status",
            "hausdorff_distance",
            "hausdorff_distance_gt_to_pred",
            "hausdorff_distance_pred_to_gt",
            "num_nodes_pred",
            "num_nodes_gt",
            "num_edges_pred",
            "num_edges_gt",
            "num_components_pred",
            "valid_count_pred",
            "gt_count",
            "count_error",
            "error_message",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

        self.logger.info(f"CSV 報告已儲存: {csv_path}")

    def _print_summary(self, summary: EvaluationSummary):
        """終端輸出摘要"""
        print("\n" + "=" * 80)
        print("評測摘要 (Evaluation Summary)")
        print("=" * 80)
        print(f"總樣本數:     {summary.total_samples}")
        print(f"成功處理:     {summary.processed}")
        print(f"跳過:         {summary.skipped}")
        print(f"失敗:         {summary.failed}")
        print("-" * 80)

        if summary.hausdorff_mean is not None:
            print("Hausdorff Distance:")
            print(f"  Mean:       {summary.hausdorff_mean:.4f}")
            print(f"  Median:     {summary.hausdorff_median:.4f}")
            print(f"  Std:        {summary.hausdorff_std:.4f}")
            print(f"  Min:        {summary.hausdorff_min:.4f}")
            print(f"  Max:        {summary.hausdorff_max:.4f}")
        else:
            print("Hausdorff Distance: no valid data")

        print("-" * 80)

        if summary.count_n > 0:
            print(f"Valid Count  (n={summary.count_n}):")
            print(f"  MAE Mean:   {summary.count_mae_mean:.4f}")
            print(f"  MAE Median: {summary.count_mae_median:.4f}")
            print(f"  MAE Std:    {summary.count_mae_std:.4f}")
            if summary.count_pearson_r is not None:
                print(
                    f"  Pearson r:  {summary.count_pearson_r:.4f}  (p={summary.count_pearson_p:.4f})"
                )
                print(
                    f"  Spearman r: {summary.count_spearman_r:.4f}  (p={summary.count_spearman_p:.4f})"
                )
        else:
            print("Valid Count: no GT data (count.json missing or no matching samples)")

        print("=" * 80 + "\n")


# ============================================================================
# 主評測器
# ============================================================================


class DatasetEvaluator:
    """
    資料集評測器

    整合所有元件，執行完整的評測流程。
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        linker: Any,
        num_workers: int = os.cpu_count() or 1,
    ):
        """
        Args:
            data_dir: 資料集目錄
            output_dir: 輸出目錄
            linker: 實作 run(image, mask, annotation) -> nx.Graph 的 linker 實例
            num_workers: 平行處理的執行緒數量（預設 1，即單線程）
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self._linker = linker

        # 建立元件
        self.loader = DatasetLoader(data_dir)
        self.comparator = TopologyComparator()
        self.reporter = EvaluationReporter(output_dir)

        # Thread-local storage：每個執行緒各自的 extractor / builder
        self._thread_local = threading.local()

        self.config = {"linker": type(linker).__name__, "params": vars(linker)}
        self.gt_counts = self._load_gt_counts(data_dir)

        self.logger = logging.getLogger(__name__)

    def _get_extractor(self) -> TopologyExtractor:
        """取得當前執行緒專用的 TopologyExtractor（含獨立 linker 副本）"""
        if not hasattr(self._thread_local, "extractor"):
            self._thread_local.extractor = TopologyExtractor(
                copy.deepcopy(self._linker)
            )
        return self._thread_local.extractor

    def _get_builder(self) -> TopologyBuilder:
        """取得當前執行緒專用的 TopologyBuilder"""
        if not hasattr(self._thread_local, "builder"):
            self._thread_local.builder = TopologyBuilder()
        return self._thread_local.builder

    @staticmethod
    def _load_gt_counts(data_dir: Path) -> Dict[str, int]:
        count_path = Path(data_dir) / "count.json"
        if not count_path.exists():
            logging.getLogger(__name__).warning(
                f"count.json not found at {count_path} — count metrics will be skipped"
            )
            return {}
        with open(count_path, encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, sample_ids: Optional[List[str]] = None) -> EvaluationSummary:
        """
        執行評測

        Args:
            sample_ids: 指定要評測的樣本 ID，None 則評測全部

        Returns:
            評測統計摘要
        """
        self.logger.info("開始資料集評測...")
        self.logger.info(f"平行執行緒數: {self.num_workers}")

        # 載入樣本
        samples = self.loader.load_samples(sample_ids)

        # 處理每個樣本
        results = []
        if self.num_workers <= 1:
            for sample in tqdm(samples, desc="評測進度"):
                result = self._evaluate_sample(sample)
                results.append(result)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = {
                    executor.submit(self._evaluate_sample, sample): sample
                    for sample in samples
                }
                with tqdm(total=len(futures), desc="評測進度") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                        pbar.update(1)

        # 生成報告
        summary = self.reporter.generate_report(results, self.config)

        self.logger.info("評測完成")
        return summary

    def _evaluate_sample(self, sample: SampleFiles) -> SampleResult:
        """
        評測單一樣本

        Args:
            sample: 樣本檔案

        Returns:
            評測結果
        """
        has_label = sample.label_path is not None and sample.label_path.exists()
        has_count = sample.sample_id in self.gt_counts

        # 至少需要一種 GT 來源才有意義執行
        if not has_label and not has_count:
            self.logger.info(
                f"樣本 {sample.sample_id} 跳過: 無 label.png 也無 count.json 條目"
            )
            return SampleResult(
                sample_id=sample.sample_id,
                status="skipped",
                error_message="missing_gt",
            )

        # 檢查輸入檔案完整性
        is_complete, missing_reason = sample.is_complete()
        if not is_complete:
            self.logger.warning(f"樣本 {sample.sample_id} 跳過: {missing_reason}")
            return SampleResult(
                sample_id=sample.sample_id,
                status="skipped",
                error_message=missing_reason,
            )

        try:
            # 載入影像
            image = np.array(Image.open(sample.image_path))
            mask = np.array(Image.open(sample.mask_path))
            annotation = np.array(Image.open(sample.annotation_path))

            # 萃取 Pipeline 拓樸（使用執行緒專用 extractor）
            extract_result = self._get_extractor().extract_from_pipeline(
                image, mask, annotation
            )

            if extract_result is None:
                return SampleResult(
                    sample_id=sample.sample_id,
                    status="failed",
                    error_message="pipeline_failed",
                )

            # --- Hausdorff ---
            avg_dist = d_pred_to_gt = d_gt_to_pred = None
            num_nodes_gt = num_edges_gt = None

            if has_label:
                gt_label = np.array(Image.open(sample.label_path))
                roi_label = cv2.bitwise_and(
                    gt_label, gt_label, mask=extract_result.mask
                )
                graph_gt = self._get_builder().build_seed_graph(roi_label)

                pred_points = extract_graph_points(extract_result.graph)
                gt_points = extract_graph_points(graph_gt)

                avg_dist, d_pred_to_gt, d_gt_to_pred = (
                    compute_average_hausdorff_distance(pred_points, gt_points)
                )
                num_nodes_gt = graph_gt.number_of_nodes()
                num_edges_gt = graph_gt.number_of_edges()

            # --- Count ---
            gt_count = self.gt_counts.get(sample.sample_id)
            valid_count_pred = extract_result.valid_count
            count_error = (
                float(abs(valid_count_pred - gt_count))
                if gt_count is not None
                else None
            )

            return SampleResult(
                sample_id=sample.sample_id,
                status="success",
                hausdorff_distance=avg_dist,
                hausdorff_distance_pred_to_gt=d_pred_to_gt,
                hausdorff_distance_gt_to_pred=d_gt_to_pred,
                num_nodes_pred=extract_result.graph.number_of_nodes(),
                num_nodes_gt=num_nodes_gt,
                num_edges_pred=extract_result.graph.number_of_edges(),
                num_edges_gt=num_edges_gt,
                num_components_pred=nx.number_connected_components(
                    extract_result.graph
                ),
                valid_count_pred=valid_count_pred,
                gt_count=gt_count,
                count_error=count_error,
            )

        except Exception as e:
            self.logger.error(f"樣本 {sample.sample_id} 處理失敗: {e}", exc_info=True)
            return SampleResult(
                sample_id=sample.sample_id, status="failed", error_message=str(e)
            )


# ============================================================================
# 命令列介面
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool):
    """設定日誌"""
    log_path = output_dir / "evaluation.log"

    # 設定日誌格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 設定檔案 handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # 設定終端 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # 設定 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="資料集評測腳本 - 批次處理並評估神經重建結果"
    )

    parser.add_argument("--data-dir", type=Path, required=True, help="資料集根目錄")

    parser.add_argument("--output-dir", type=Path, required=True, help="輸出目錄")

    parser.add_argument(
        "--sample-ids", nargs="+", help="指定要評測的樣本 ID（可選，預設評測全部）"
    )

    parser.add_argument(
        "--algorithm",
        choices=["pure_mst", "hierarchical", "xgb_mst", "unet"],
        default="pure_mst",
        help="使用的重建演算法 (預設: pure_mst)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help=f"平行處理的執行緒數量（預設: CPU 核心數 {os.cpu_count()}）",
    )

    parser.add_argument("--verbose", action="store_true", help="啟用詳細日誌輸出")

    args = parser.parse_args()

    # 建立輸出目錄
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 設定日誌
    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("資料集評測腳本")
    logger.info("=" * 80)
    logger.info(f"資料集目錄: {args.data_dir}")
    logger.info(f"輸出目錄: {args.output_dir}")
    logger.info(f"演算法: {args.algorithm}")

    # 建立 linker
    if args.algorithm == "pure_mst":
        linker = PureMstLinker(
            offset_px=50,
            rolling_ball_radius=50,
            opening_kernel_size=3,
            segment_length=5.0,
            search_radius=50.0,
            intensity_weight=2,
        )
    elif args.algorithm == "hierarchical":  # hierarchical
        linker = HierarchicalFragmentLinker(
            offset_px=50,
            rolling_ball_radius=50,
            opening_kernel_size=3,
            segment_length=3.0,
            search_radius_endpoint_extension=20.0,
            max_angle_endpoint_extension=75.0,
            search_radius_mst=50.0,
            max_angle_mst=90.0,
            max_cost_threshold_mst=0.5,
            distance_weight_mst=0.3,
            min_component_length=3.0,
        )
    elif args.algorithm == "xgb_mst":  # xgb_mst
        linker = XgbMstLinker(
            offset_px=1,
            rolling_ball_radius=2,
            opening_kernel_size=3,
            segment_length=3.0,
            search_radius=100.0,
            xgb_proba_threshold=0.5,
        )
    elif args.algorithm == "unet":  # unet
        linker = UnetLinker(
            checkpoint_path="/home/pony/projects/ienf_q/output/unet_0320/unet_best.pth",
            base_channels=32,
            patch_size=512,
            overlap=64,
            threshold=0.5,
            device="auto",
        )
    else:
        raise ValueError(f"未知的演算法選項: {args.algorithm}")
    # 建立評測器
    evaluator = DatasetEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        linker=linker,
        num_workers=args.workers,
    )

    # 執行評測
    summary = evaluator.evaluate(sample_ids=args.sample_ids)

    logger.info("=" * 80)
    logger.info("評測完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
